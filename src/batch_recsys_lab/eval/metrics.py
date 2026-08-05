"""Full-catalog ranking metrics (Phase 2, T2; UPGRADE_PLAN.md §8, "Key specs" → Metrics).

The eval harness scores users in batches, producing a dense ``float32`` score
matrix ``S`` of shape ``(B, I)`` in which **TRAIN-seen items are already masked to
``-inf``** (exclusion happens upstream, once, model-agnostically). Ground-truth
(GT) items are *never* masked — silver dedup guarantees ``GT ∩ TRAIN-seen = ∅`` —
so a GT item always carries a finite score. Metrics are binary-relevance.

**The single ordering rule.** The catalog index order is itself the deterministic
tie-break: items are ranked by **descending score, ties broken by ascending item
index**. Equivalently, ``i`` outranks ``j`` iff ``s_i > s_j`` or (``s_i == s_j``
and ``i < j``). This one rule is implemented ONCE (see :func:`topk_indices`, which
:func:`pop_share_at_k` reuses) and is mirrored exactly by the closed-form rank in
:func:`gt_ranks`::

    rank(g) = 1 + #{j : s_j > s_g} + #{j : s_j == s_g and j < g}

``-inf`` (masked) items therefore sort strictly after every finitely-scored item,
and among themselves by ascending index; they can never outrank or tie-affect a
finite GT score. ``NaN`` must not appear in ``scores`` (undefined ordering) — this
is a documented precondition, not checked per-element for speed.

All accuracy metrics derive purely from GT ranks, so MRR is an exact full-catalog
reciprocal rank (no cutoff). These docstrings pin the exact formulas and are cited
by the case study; do not paraphrase them without updating the citation.
"""

from __future__ import annotations

import numpy as np


# --- the ordering rule + top-K extraction ------------------------------------


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` catalog indices per user under the global ordering rule.

    ``scores`` is ``(B, I)`` ``float32`` (TRAIN-seen already masked to ``-inf``).
    Returns ``(B, min(k, I))`` ``int32``: for each row, the indices ordered by
    **descending score, ties broken by ascending item index**. When ``k >= I`` the
    full row is returned (``min(k, I)`` columns); ``-inf`` items sort last, among
    themselves by ascending index.

    Exactness at the cut: ``argpartition`` selects a ``k``-item candidate set whose
    membership is correct for every item *strictly* better than the ``k``-th score,
    then the boundary-tied items are resolved by **ascending index** (smallest
    indices win the last slots) before a final ``lexsort``. This makes the
    tie-break exact even when many items share the boundary score (popularity's
    tied tail, kNN's zero mass). Precondition: no ``NaN`` in ``scores``.
    """
    scores = np.asarray(scores)
    B, I = scores.shape
    kk = min(int(k), I)
    out = np.empty((B, max(kk, 0)), dtype=np.int32)
    if kk <= 0:
        return out
    # We want the kk items with the SMALLEST ``neg`` (= largest score); among ties
    # in ``neg`` (= ties in score) the smallest index wins. ``-inf`` scores become
    # ``+inf`` here and therefore sort last, exactly as required.
    neg = -scores.astype(np.float64, copy=False)
    for b in range(B):
        row = neg[b]
        if kk == I:
            sel = np.arange(I)
        else:
            part = np.argpartition(row, kk - 1)[:kk]  # kk smallest ``neg`` by value
            bval = row[part].max()                    # kk-th smallest ``neg`` (the cut)
            strictly = part[row[part] < bval]         # all items strictly above the cut
            n_need = kk - strictly.size               # remaining slots at the cut
            tied = np.flatnonzero(row == bval)        # boundary ties, ascending index
            sel = np.concatenate([strictly, tied[:n_need]])
        # Order the kk-item candidate set by (neg asc, index asc). ``lexsort`` takes
        # the LAST key as primary: primary = row[sel] (score desc), secondary = sel.
        order = np.lexsort((sel, row[sel]))
        out[b] = sel[order][:kk]
    return out


def gt_ranks(
    scores: np.ndarray, gt_indptr: np.ndarray, gt_item_idx: np.ndarray
) -> np.ndarray:
    """Exact 1-based full-catalog rank of each GT item under the ordering rule.

    ``rank(g) = 1 + #{j : s_j > s_g} + #{j : s_j == s_g and j < g}`` — the closed
    form of "descending score, ties broken by ascending index". GT scores are
    finite (never masked), so ``-inf`` items (``s_j == s_g`` false, ``s_j > s_g``
    false) never contribute to any GT rank, including a masked item whose index is
    lower than the GT item's. ``rank == 1`` means best-ranked; the GT item never
    counts itself (the tie term is strict ``j < g``).

    ``gt_indptr`` (length ``B + 1``) and ``gt_item_idx`` are the CSR-ragged GT sets
    per user in the batch: user ``b``'s GT item indices are
    ``gt_item_idx[gt_indptr[b] : gt_indptr[b + 1]]``. Returns an ``int32`` array
    aligned element-for-element with ``gt_item_idx``.
    """
    scores = np.asarray(scores)
    B, I = scores.shape
    gt_indptr = np.asarray(gt_indptr)
    gt_item_idx = np.asarray(gt_item_idx)
    ranks = np.empty(gt_item_idx.shape[0], dtype=np.int32)
    ar = np.arange(I)
    for b in range(B):
        lo, hi = int(gt_indptr[b]), int(gt_indptr[b + 1])
        if hi <= lo:
            continue
        row = scores[b]
        g = gt_item_idx[lo:hi]           # (m,) GT item indices for this user
        sg = row[g]                      # (m,) their finite scores
        greater = (row[None, :] > sg[:, None]).sum(axis=1)          # #{s_j > s_g}
        eq_lower = (
            (row[None, :] == sg[:, None]) & (ar[None, :] < g[:, None])
        ).sum(axis=1)                                               # #{==, j < g}
        ranks[lo:hi] = 1 + greater + eq_lower
    return ranks


# --- accuracy metrics (derived purely from ranks) ----------------------------


def accuracy_metrics(
    gt_indptr: np.ndarray,
    ranks: np.ndarray,
    k_list: tuple[int, ...] = (10, 20, 50),
) -> dict[str, np.ndarray]:
    """Per-user accuracy metric vectors (``float64``, length ``B``) from GT ranks.

    With the default ``k_list`` the keys are exactly
    ``'recall@10', 'recall@20', 'recall@50', 'ndcg@10', 'ndcg@20', 'mrr',
    'hitrate@10'``. In general: ``recall@k`` for every ``k`` in ``k_list``;
    ``ndcg@k`` for ``k`` in ``{10, 20} ∩ k_list`` (the plan reports NDCG@{10,20}
    only); ``mrr`` and ``hitrate@10`` always. Formulas (pinned; ``|GT|`` = number
    of GT items for the user, ``r`` a GT item's 1-based rank)::

        recall@k   = |{g : rank(g) <= k}| / |GT|
        hitrate@10 = 1.0 if min_g rank(g) <= 10 else 0.0
        mrr        = 1 / min_g rank(g)        (full-catalog, no cutoff)
        DCG@k      = sum over GT hits with rank r <= k of 1 / log2(r + 1)
        IDCG@k     = sum_{j=1..min(|GT|, k)} 1 / log2(j + 1)
        ndcg@k     = DCG@k / IDCG@k

    A user with ``|GT|`` == 1 whose sole GT item ranks 1 has ``ndcg@k == 1.0`` for
    every ``k >= 1``. Users with empty GT (should not occur for test users, whose
    ``n_test > 0``) get ``0.0`` in every vector.
    """
    gt_indptr = np.asarray(gt_indptr)
    ranks = np.asarray(ranks)
    B = gt_indptr.shape[0] - 1

    ndcg_ks = tuple(k for k in (10, 20) if k in k_list)
    recall = {k: np.zeros(B, dtype=np.float64) for k in k_list}
    ndcg = {k: np.zeros(B, dtype=np.float64) for k in ndcg_ks}
    mrr = np.zeros(B, dtype=np.float64)
    hit10 = np.zeros(B, dtype=np.float64)

    for b in range(B):
        lo, hi = int(gt_indptr[b]), int(gt_indptr[b + 1])
        m = hi - lo
        if m == 0:
            continue
        rk = ranks[lo:hi].astype(np.float64)
        mn = rk.min()
        mrr[b] = 1.0 / mn
        hit10[b] = 1.0 if mn <= 10 else 0.0
        for k in k_list:
            recall[k][b] = np.count_nonzero(rk <= k) / m
        for k in ndcg_ks:
            hits = rk[rk <= k]
            dcg = float(np.sum(1.0 / np.log2(hits + 1.0)))
            idcg = float(
                np.sum(1.0 / np.log2(np.arange(1, min(m, k) + 1) + 1.0))
            )
            ndcg[k][b] = dcg / idcg if idcg > 0.0 else 0.0

    out: dict[str, np.ndarray] = {}
    for k in k_list:
        out[f"recall@{k}"] = recall[k]
    for k in ndcg_ks:
        out[f"ndcg@{k}"] = ndcg[k]
    out["mrr"] = mrr
    out["hitrate@10"] = hit10
    return out


# --- beyond-accuracy metrics -------------------------------------------------


def coverage_at_k(topk: np.ndarray, catalog_size: int) -> float:
    """Catalog coverage: ``|union of all users' top-k lists| / catalog_size``.

    ``topk`` is the ``(B, k)`` integer array from :func:`topk_indices`.
    """
    return float(np.unique(np.asarray(topk)).size) / float(catalog_size)


def novelty_per_user(topk: np.ndarray, item_train_counts: np.ndarray) -> np.ndarray:
    """Per-user novelty: mean of ``-log2 p(i)`` over the user's top-k list.

    ``p(i) = (c_i + 1) / (sum(c) + I)`` — Laplace-smoothed TRAIN-interaction
    frequency, where ``c_i = item_train_counts[i]`` and ``I = len(item_train_counts)``
    (smoothing is required because some catalog items have zero TRAIN
    interactions). Higher = the user is shown rarer items. Returns ``(B,)``
    ``float64``.
    """
    counts = np.asarray(item_train_counts, dtype=np.float64)
    I = counts.shape[0]
    p = (counts + 1.0) / (counts.sum() + I)
    self_info = -np.log2(p)                       # (I,) surprisal per item
    return self_info[np.asarray(topk)].mean(axis=1).astype(np.float64)


def pop_share_at_k(
    topk: np.ndarray, item_train_counts: np.ndarray, top_frac: float = 0.01
) -> float:
    """Fraction of all top-k slots occupied by the head of the popularity distribution.

    The "head" is the ``ceil(top_frac * I)`` most-popular items by TRAIN count,
    with popularity ties broken by **ascending item index** — the same global
    ordering rule as ranking (reused via :func:`topk_indices` on the count vector).
    Returns ``(# top-k slots in the head) / (B * k)``.
    """
    counts = np.asarray(item_train_counts)
    I = counts.shape[0]
    n_top = int(np.ceil(top_frac * I))
    n_top = max(min(n_top, I), 0)
    if n_top == 0:
        return 0.0
    # Reuse the ONE ordering rule: descending count, ascending index on ties.
    head = topk_indices(counts.astype(np.float32)[None, :], n_top)[0]
    topk = np.asarray(topk)
    if topk.size == 0:
        return 0.0
    in_head = int(np.isin(topk, head).sum())
    return in_head / float(topk.size)


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of the item TRAIN-interaction count distribution.

    Uses the standard sorted-order formula (mean of absolute differences,
    normalised): with counts sorted ascending as ``x_1 <= ... <= x_n``,

        G = (2 * sum_{i=1..n} i * x_i) / (n * sum_i x_i) - (n + 1) / n

    Consequences: a **uniform** distribution (all counts equal) gives ``G == 0``;
    a **one-hot** distribution (a single nonzero count) gives ``G == 1 - 1/n``.
    When ``sum_i x_i == 0`` (all counts zero → perfectly equal) ``G`` is defined
    to be ``0.0``.
    """
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.shape[0]
    total = x.sum()
    if n == 0 or total == 0.0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1.0) / n)
