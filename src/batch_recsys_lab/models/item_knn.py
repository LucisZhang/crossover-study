"""Memory-blocked item-kNN recommender — cosine co-occurrence with shrinkage
(Phase 2, T6; UPGRADE_PLAN.md §8 "Key specs" → item-kNN).

The similarity matrix ``S`` (I×I) is the item-item cosine co-occurrence matrix

    sim(i, j) = c_ij / (sqrt(n_i * n_j) + shrinkage)

where ``c_ij`` is the co-occurrence count (the ``(i, j)`` entry of ``BᵀB`` for the
binary TRAIN interaction matrix ``B`` of shape U×I) and ``n_i`` is item ``i``'s
TRAIN interaction count (the column sum of ``B``; equal to ``c_ii`` for binary
``B``). The diagonal is removed (``sim(i, i) = 0``). A user's score vector is
``user_history_row @ S`` — the sum, over each candidate item's retained
neighbors, of the user's interactions with those neighbors.

Memory discipline (never materialize the full untruncated I×I product): ``BᵀB``
is computed one column-block at a time (``S_block = Bᵀ @ B[:, block]``, a sparse
× sparse product yielding a sparse I×block_size result), cosine-normalized,
diagonal-zeroed, and immediately truncated to each column's top-``top_n`` entries
by value before the next block is touched. Peak memory is one block's
untruncated result plus the accumulated truncated CSR (≈0.6–0.9GB at full scale,
top_n=200). The computation is fully deterministic: identical input yields a
byte-identical ``S`` (scipy's sparse matmul sums in a fixed order, and every
per-column top-``top_n`` selection breaks ties by ascending row index).

Truncation honesty
------------------
Every catalog item still receives a score. Items outside the union of the
querying user's items' neighbor lists score exactly 0 and are ranked below all
positive-scored items by the harness's deterministic index tie-break. This
remains full-catalog-valid scoring — every item is scored, none is sampled
away; the only approximation introduced by truncation is that similarity mass
below each item's top_n-th neighbor is floored to 0. ``top_n`` is recorded in
the run record's model params. (CLAUDE.md invariant #4 context: this
documentation is what keeps the exhibit honest.)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_similarity(
    train_csr: sp.csr_matrix,
    top_n: int,
    shrinkage: float = 0.0,
    block_size: int = 8192,
) -> sp.csr_matrix:
    """Build the truncated item-item cosine co-occurrence matrix ``S`` (I×I, float32).

    ``sim(i, j) = c_ij / (sqrt(n_i * n_j) + shrinkage)`` with ``c_ij`` the
    co-occurrence count (``BᵀB``), ``n_i`` the item interaction count (column sum
    of ``B``); the diagonal is zeroed. Each column ``j`` is truncated to its
    top-``top_n`` entries by value immediately after normalization — ties broken
    by ascending row index for determinism — so the full untruncated I×I product
    is never materialized. Column ``j`` therefore holds item ``j``'s (up to)
    ``top_n`` nearest neighbors, which is what ``user_row @ S`` sums over.

    Truncation honesty
    ------------------
    Every catalog item still receives a score. Items outside the union of the
    querying user's items' neighbor lists score exactly 0 and are ranked below
    all positive-scored items by the harness's deterministic index tie-break.
    This remains full-catalog-valid scoring — every item is scored, none is
    sampled away; the only approximation introduced by truncation is that
    similarity mass below each item's top_n-th neighbor is floored to 0.
    ``top_n`` is recorded in the run record's model params. (CLAUDE.md invariant
    #4 context: this documentation is what keeps the exhibit honest.)

    Parameters
    ----------
    train_csr : scipy.sparse.csr_matrix
        U×I binary TRAIN interaction matrix ``B``.
    top_n : int
        Neighbors retained per item column.
    shrinkage : float
        Additive denominator shrinkage ``λ`` (recorded as a model param).
    block_size : int
        Number of item columns per ``Bᵀ @ B[:, block]`` block (memory knob;
        does not change the result — see ``test_item_knn`` blocking invariance).

    Returns
    -------
    scipy.sparse.csr_matrix
        I×I float32 similarity matrix, each column with at most ``top_n``
        nonzeros and a zero diagonal.
    """
    B = train_csr.tocsr()
    n_items = B.shape[1]

    # n_i = item interaction count (column sums of B). For binary B this equals
    # the co-occurrence diagonal c_ii. float64 for a stable sqrt; the final S is
    # cast to float32.
    n = np.asarray(B.sum(axis=0), dtype=np.float64).ravel()
    sqrt_n = np.sqrt(n)

    Bt = B.T.tocsr()  # I×U, the left factor Bᵀ (reused across all blocks)
    Bcsc = B.tocsc()  # U×I, efficient contiguous column slicing per block

    # Per-column accumulators, filled in strict column order (block by block).
    col_rows: list[np.ndarray] = [None] * n_items  # type: ignore[list-item]
    col_vals: list[np.ndarray] = [None] * n_items  # type: ignore[list-item]
    col_counts = np.zeros(n_items, dtype=np.int64)

    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        # One block's untruncated co-occurrence counts: Bᵀ @ B[:, start:end].
        # Sparse × sparse -> sparse I×b result; never the full I×I product.
        block = Bcsc[:, start:end]
        C = (Bt @ block).tocsc()
        if C.nnz == 0:
            for jj in range(start, end):
                col_rows[jj] = np.empty(0, dtype=np.int32)
                col_vals[jj] = np.empty(0, dtype=np.float64)
            continue

        # Vectorized cosine normalization over the whole block.
        rows = C.indices  # row index i of each nonzero
        col_ids = np.repeat(np.arange(start, end), np.diff(C.indptr))  # col index j
        denom = sqrt_n[rows] * sqrt_n[col_ids] + shrinkage
        C.data = C.data / denom  # float64 during selection; cast to f32 at the end

        indptr = C.indptr
        indices = C.indices
        vals = C.data
        for jj in range(start, end):
            lo, hi = indptr[jj - start], indptr[jj - start + 1]
            r = indices[lo:hi]
            v = vals[lo:hi]
            # Zero the diagonal: drop the (j, j) self-entry.
            keep = r != jj
            r = r[keep]
            v = v[keep]
            if r.size > top_n:
                # r is ascending (csc sorted indices); a stable argsort on -v
                # keeps the smaller row index first on value ties, so [:top_n]
                # is exactly "top_n by value, ties by ascending row index".
                sel = np.argsort(-v, kind="stable")[:top_n]
                sel.sort()  # restore ascending row order for canonical CSC/CSR
                r = r[sel]
                v = v[sel]
            col_rows[jj] = r.astype(np.int32, copy=False)
            col_vals[jj] = v
            col_counts[jj] = r.size

    indptr = np.zeros(n_items + 1, dtype=np.int64)
    np.cumsum(col_counts, out=indptr[1:])
    indices = np.concatenate(col_rows).astype(np.int32, copy=False)
    data = np.concatenate(col_vals).astype(np.float32, copy=False)

    S = sp.csc_matrix((data, indices, indptr), shape=(n_items, n_items))
    return S.tocsr()


class ItemKNNRecommender:
    """Item-kNN recommender: ``score = user_history @ S`` (no exclusion here).

    Implements the structural ``Recommender`` protocol (``name``, ``params``,
    ``fit``, ``score_batch``) without importing ``models.base`` — the harness
    masks TRAIN-seen items after scoring, so ``score_batch`` returns raw scores
    for the full catalog.
    """

    name = "item_knn"

    def __init__(self, top_n: int, shrinkage: float = 0.0, block_size: int = 8192):
        self.top_n = top_n
        self.shrinkage = shrinkage
        self.block_size = block_size
        # block_size is an implementation knob but recorded for reproducibility.
        self.params = {
            "top_n": top_n,
            "shrinkage": shrinkage,
            "block_size": block_size,
        }
        self.ds = None
        self.S: sp.csr_matrix | None = None

    def fit(self, ds) -> "ItemKNNRecommender":
        """Keep the dataset reference and build the truncated similarity matrix."""
        self.ds = ds
        self.S = build_similarity(
            ds.train_csr,
            top_n=self.top_n,
            shrinkage=self.shrinkage,
            block_size=self.block_size,
        )
        return self

    def score_batch(self, user_idx) -> np.ndarray:
        """Score every catalog item for a batch of users -> (B, I) float32.

        ``scores = train_csr[user_idx] @ S``. No exclusion: the harness masks
        TRAIN-seen items afterward. A user with empty TRAIN history scores all
        zeros; items outside the neighbor union of the user's items score exactly
        0 (see the module "Truncation honesty" note).
        """
        if self.S is None:
            raise RuntimeError("ItemKNNRecommender.score_batch called before fit().")
        user_idx = np.asarray(user_idx)
        rows = self.ds.train_csr[user_idx]  # (B, I) csr
        scores = rows @ self.S  # (B, I) sparse
        return np.asarray(scores.toarray(), dtype=np.float32)
