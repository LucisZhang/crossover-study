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

Trailing window (Phase 8, T8-2)
-------------------------------
``train_window_days`` restricts the interactions that build ``S`` to TRAIN pairs
younger than the window at the train cutoff (``age_days < window``, the same
strict-lower/inclusive-upper boundary the frozen pop-t12m build uses). It
defaults to 0 = all history, which leaves every pre-Phase-8 code path and record
untouched. The user profile vectors used at scoring time are always the FULL
``train_csr``: per the T8-2 preregistration the recency treatment is on the
item-side co-occurrence only.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from batch_recsys_lab.eval.dataset import attach_train_pairs


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


def build_windowed_csr(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    age_days: np.ndarray,
    window_days: float,
    shape: tuple[int, int],
) -> sp.csr_matrix:
    """Binary U×I TRAIN matrix restricted to pairs newer than ``window_days``
    (Phase 8, T8-2).

    The retention rule is ``age_days < window_days`` — STRICT, on fractional
    days. Given ``age_days = train_end − ts`` this is exactly the frozen
    popularity-window semantics ``ts > as_of − window`` at ``as_of ==
    train_end`` (strict lower bound, inclusive upper bound at ``ts ==
    train_end`` ⇔ ``age == 0``), so kNN-t12m and pop-t12m see the same window
    of history rather than two nearly-equal ones.

    Construction mirrors ``eval.dataset.load_dataset``'s TRAIN CSR exactly
    (COO → CSR, data clipped to 1.0, zeros eliminated), so with a window wide
    enough to keep every pair the result is the full ``train_csr``.
    """
    user_idx = np.asarray(user_idx)
    item_idx = np.asarray(item_idx)
    age_days = np.asarray(age_days)
    if not (len(user_idx) == len(item_idx) == len(age_days)):
        raise ValueError(
            f"TRAIN pair columns disagree in length: user {len(user_idx)}, item "
            f"{len(item_idx)}, age {len(age_days)}"
        )
    keep = age_days < float(window_days)
    coo = sp.coo_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.float32),
            (user_idx[keep], item_idx[keep]),
        ),
        shape=shape,
        dtype=np.float32,
    )
    csr = coo.tocsr()
    csr.data[:] = 1.0
    csr.eliminate_zeros()
    return csr


class ItemKNNRecommender:
    """Item-kNN recommender: ``score = user_history @ S`` (no exclusion here).

    Implements the structural ``Recommender`` protocol (``name``, ``params``,
    ``fit``, ``score_batch``) without importing ``models.base`` — the harness
    masks TRAIN-seen items after scoring, so ``score_batch`` returns raw scores
    for the full catalog.

    ``train_window_days`` (Phase 8, T8-2; default 0 == all history, byte-identical
    to the Phase 2/3 arm) restricts the CO-OCCURRENCE matrix to TRAIN pairs newer
    than the window. Scoring still uses the user's FULL TRAIN profile: the
    preregistered recency treatment is item-side only, mirroring how pop-t12m's
    recency lives on the item side. It is recorded in the run record's model
    params only when it is in force, so every pre-T8-2 kNN record stays
    reproducible field-for-field.
    """

    name = "item_knn"

    def __init__(
        self,
        top_n: int,
        shrinkage: float = 0.0,
        block_size: int = 8192,
        train_window_days: int = 0,
    ):
        self.top_n = top_n
        self.shrinkage = shrinkage
        self.block_size = block_size
        self.train_window_days = int(train_window_days)
        if self.train_window_days < 0:
            raise ValueError(
                f"train_window_days must be >= 0 (0 == all history), got "
                f"{train_window_days!r}"
            )
        # block_size is an implementation knob but recorded for reproducibility.
        self.params = {
            "top_n": top_n,
            "shrinkage": shrinkage,
            "block_size": block_size,
        }
        if self.train_window_days > 0:
            self.params["train_window_days"] = self.train_window_days
        self.ds = None
        self.S: sp.csr_matrix | None = None

    def fit(self, ds) -> "ItemKNNRecommender":
        """Keep the dataset reference and build the truncated similarity matrix.

        With ``train_window_days > 0`` the similarity is built from the windowed
        TRAIN matrix (which needs the cache's ``train_age_days.npy``; a missing
        age array is a hard error naming ``make extract-age``, never a silent
        fall-back to all history). ``self.ds`` — and therefore scoring — keeps
        the full ``train_csr``.
        """
        self.ds = ds
        if self.train_window_days > 0:
            attach_train_pairs(ds, require_age=True)
            fit_csr = build_windowed_csr(
                ds.train_user_idx,
                ds.train_item_idx,
                ds.train_age_days,
                self.train_window_days,
                ds.train_csr.shape,
            )
        else:
            fit_csr = ds.train_csr
        self.S = build_similarity(
            fit_csr,
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
