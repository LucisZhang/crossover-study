"""Reference-verified tests for the ranking metrics module (Phase 2, T2).

Three layers:
  (a) hand-computed micro-cases (3 users x 6 items) with every expected value
      worked out by hand in comments — covers the tie-break, a masked-adjacent
      GT item, |GT| > k, and |GT| == 1 with rank 1 (NDCG must be exactly 1.0);
  (b) a naive full-``argsort`` reference (independent Python implementation of
      EVERY public function) fuzz-compared on 50 seeded random instances with
      deliberate score ties and random -inf masking of non-GT items;
  (c) edge cases: all-zero row, k > I, single-item catalog, and hand-computed
      coverage / novelty / pop-share / Gini.

No Spark / JAVA required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from batch_recsys_lab.eval.metrics import (
    accuracy_metrics,
    coverage_at_k,
    gini,
    gt_ranks,
    novelty_per_user,
    pop_share_at_k,
    topk_indices,
)

NINF = float("-inf")


# =============================================================================
# (a) Hand-computed micro-cases — 3 users x 6 items
# =============================================================================
#
# Items are indices 0..5. Ordering rule: descending score, ties -> ascending idx.
#
# User 0 — distinct descending scores; |GT| == 1 with rank 1 (NDCG must be 1.0).
#   scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
#   full order = [0,1,2,3,4,5];  GT = {0} -> rank 1.
#
# User 1 — TIE between a GT item and a non-GT item; index tie-break decides.
#   scores = [0.5, 0.9, 0.9, 0.2, 0.1, 0.0]
#   items 1 and 2 both score 0.9; lower index (1, non-GT) outranks 2 (GT).
#   full order = [1,2,0,3,4,5];  GT = {2} -> rank 2.
#
# User 2 — masked-adjacent GT (item 0 is -inf, index 0 < GT indices, must NOT
#   affect any GT rank) AND |GT| == 2 (used with k=1 to exercise |GT| > k).
#   scores = [-inf, 0.3, 0.7, 0.5, 0.9, 0.2]
#   full order = [4,2,3,1,5,0];  GT = {2,4} -> rank(2)=2, rank(4)=1.
#   The -inf item 0 sorts last and is never counted in a GT rank.

MICRO_SCORES = np.array(
    [
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        [0.5, 0.9, 0.9, 0.2, 0.1, 0.0],
        [NINF, 0.3, 0.7, 0.5, 0.9, 0.2],
    ],
    dtype=np.float32,
)
# CSR GT: user0=[0], user1=[2], user2=[2,4]
MICRO_INDPTR = np.array([0, 1, 2, 4], dtype=np.int32)
MICRO_ITEMS = np.array([0, 2, 2, 4], dtype=np.int32)


def test_micro_topk():
    got = topk_indices(MICRO_SCORES, 3)
    expected = np.array([[0, 1, 2], [1, 2, 0], [4, 2, 3]], dtype=np.int32)
    assert np.array_equal(got, expected)
    assert got.dtype == np.int32


def test_micro_gt_ranks():
    got = gt_ranks(MICRO_SCORES, MICRO_INDPTR, MICRO_ITEMS)
    # user0 item0 -> 1 ; user1 item2 -> 2 (tie w/ item1, idx tie-break)
    # user2 item2 -> 2, item4 -> 1 (masked item0 never counted)
    expected = np.array([1, 2, 2, 1], dtype=np.int32)
    assert np.array_equal(got, expected)
    assert got.dtype == np.int32


def test_micro_accuracy():
    # k_list=(1,2,10): recall@1 exercises |GT| > k for user 2 (|GT|=2, k=1).
    m = accuracy_metrics(MICRO_INDPTR, gt_ranks(MICRO_SCORES, MICRO_INDPTR, MICRO_ITEMS),
                         k_list=(1, 2, 10))
    assert set(m) == {"recall@1", "recall@2", "recall@10", "ndcg@10", "mrr", "hitrate@10"}

    # recall@1: u0 rank1 -> 1/1 ; u1 rank2 -> 0/1 ; u2 ranks{2,1} -> 1/2
    assert np.allclose(m["recall@1"], [1.0, 0.0, 0.5])
    # recall@2: u0 1 ; u1 rank2<=2 -> 1 ; u2 both<=2 -> 1
    assert np.allclose(m["recall@2"], [1.0, 1.0, 1.0])
    assert np.allclose(m["recall@10"], [1.0, 1.0, 1.0])
    # mrr = 1/min rank
    assert np.allclose(m["mrr"], [1.0, 0.5, 1.0])
    assert np.allclose(m["hitrate@10"], [1.0, 1.0, 1.0])

    # ndcg@10:
    #  u0: |GT|1 rank1 -> DCG=1, IDCG=1 -> 1.0 (the exact-1.0 case)
    #  u1: DCG=1/log2(3), IDCG=1 -> 1/log2(3)
    #  u2: DCG=1/log2(2)+1/log2(3), IDCG=1/log2(2)+1/log2(3) -> 1.0
    exp_ndcg = [1.0, 1.0 / math.log2(3), 1.0]
    assert np.allclose(m["ndcg@10"], exp_ndcg)


def test_micro_ndcg_gt1_rank1_is_exactly_one():
    # |GT|==1, rank 1 -> NDCG@k is EXACTLY 1.0 for every k.
    m = accuracy_metrics(np.array([0, 1]), np.array([1], dtype=np.int32),
                         k_list=(10, 20, 50))
    assert m["ndcg@10"][0] == 1.0
    assert m["ndcg@20"][0] == 1.0


def test_default_key_set():
    m = accuracy_metrics(np.array([0, 1]), np.array([3], dtype=np.int32))
    assert set(m) == {
        "recall@10", "recall@20", "recall@50",
        "ndcg@10", "ndcg@20", "mrr", "hitrate@10",
    }
    for v in m.values():
        assert v.dtype == np.float64 and v.shape == (1,)


# =============================================================================
# (b) Naive independent reference + seeded fuzz over EVERY public function
# =============================================================================


def _naive_topk(scores, k):
    B, I = scores.shape
    kk = min(int(k), I)
    out = np.empty((B, kk), dtype=np.int32)
    for b in range(B):
        # full sort by the ordering rule: (-score, index); -inf -> +inf sorts last.
        order = sorted(range(I), key=lambda j: (-float(scores[b, j]), j))
        out[b] = order[:kk]
    return out


def _naive_ranks(scores, indptr, items):
    B, I = scores.shape
    out = []
    for b in range(B):
        for g in items[indptr[b]:indptr[b + 1]]:
            g = int(g)
            sg = float(scores[b, g])
            greater = sum(1 for j in range(I) if float(scores[b, j]) > sg)
            eq_lower = sum(1 for j in range(I) if float(scores[b, j]) == sg and j < g)
            out.append(1 + greater + eq_lower)
    return np.array(out, dtype=np.int32)


def _naive_accuracy(indptr, ranks, k_list):
    B = len(indptr) - 1
    ndcg_ks = tuple(k for k in (10, 20) if k in k_list)
    res = {f"recall@{k}": np.zeros(B) for k in k_list}
    for k in ndcg_ks:
        res[f"ndcg@{k}"] = np.zeros(B)
    res["mrr"] = np.zeros(B)
    res["hitrate@10"] = np.zeros(B)
    for b in range(B):
        rk = [int(r) for r in ranks[indptr[b]:indptr[b + 1]]]
        m = len(rk)
        if m == 0:
            continue
        mn = min(rk)
        res["mrr"][b] = 1.0 / mn
        res["hitrate@10"][b] = 1.0 if mn <= 10 else 0.0
        for k in k_list:
            res[f"recall@{k}"][b] = sum(1 for r in rk if r <= k) / m
        for k in ndcg_ks:
            dcg = sum(1.0 / math.log2(r + 1) for r in rk if r <= k)
            idcg = sum(1.0 / math.log2(j + 1) for j in range(1, min(m, k) + 1))
            res[f"ndcg@{k}"][b] = dcg / idcg if idcg > 0 else 0.0
    return res


def _naive_coverage(topk, catalog_size):
    seen = set(int(x) for row in topk for x in row)
    return len(seen) / catalog_size


def _naive_novelty(topk, counts):
    counts = np.asarray(counts, dtype=np.float64)
    I = len(counts)
    total = counts.sum()
    p = [(counts[i] + 1.0) / (total + I) for i in range(I)]
    si = [-math.log2(pi) for pi in p]
    return np.array([np.mean([si[int(j)] for j in row]) for row in topk])


def _naive_pop_share(topk, counts, top_frac=0.01):
    counts = np.asarray(counts)
    I = len(counts)
    n_top = int(math.ceil(top_frac * I))
    order = sorted(range(I), key=lambda j: (-int(counts[j]), j))
    head = set(order[:n_top])
    slots = sum(len(row) for row in topk)
    if slots == 0 or n_top == 0:
        return 0.0
    hits = sum(1 for row in topk for x in row if int(x) in head)
    return hits / slots


def _naive_gini(counts):
    x = sorted(float(c) for c in counts)
    n = len(x)
    total = sum(x)
    if n == 0 or total == 0:
        return 0.0
    s = sum((i + 1) * x[i] for i in range(n))
    return (2.0 * s) / (n * total) - (n + 1.0) / n


def _random_instance(rng):
    """Random scores with frequent ties + random -inf masking of non-GT items."""
    B = int(rng.integers(1, 21))
    I = int(rng.integers(5, 201))
    # quantize to 1 decimal so score ties are frequent
    scores = np.round(rng.normal(0.0, 1.0, size=(B, I)), 1).astype(np.float32)

    indptr = [0]
    items = []
    for b in range(B):
        m = int(rng.integers(1, min(5, I) + 1))
        gt = rng.choice(I, size=m, replace=False)
        gt_set = set(int(x) for x in gt)
        items.extend(int(x) for x in gt)
        indptr.append(len(items))
        # mask a random subset of NON-GT items to -inf
        non_gt = [j for j in range(I) if j not in gt_set]
        if non_gt:
            n_mask = int(rng.integers(0, len(non_gt) + 1))
            if n_mask:
                masked = rng.choice(non_gt, size=n_mask, replace=False)
                scores[b, masked] = NINF
    return scores, np.array(indptr, dtype=np.int32), np.array(items, dtype=np.int32), I


def test_fuzz_all_functions_vs_naive():
    """Compare EVERY public function against the naive reference on 50 instances."""
    rng = np.random.default_rng(20260805)
    for _ in range(50):
        scores, indptr, items, I = _random_instance(rng)
        k = int(rng.integers(1, I + 5))  # sometimes k > I

        # topk_indices — exact equality
        assert np.array_equal(topk_indices(scores, k), _naive_topk(scores, k))

        # gt_ranks — exact equality
        ranks = gt_ranks(scores, indptr, items)
        assert np.array_equal(ranks, _naive_ranks(scores, indptr, items))

        # accuracy_metrics — allclose (rtol 1e-12)
        k_list = (10, 20, 50)
        got = accuracy_metrics(indptr, ranks, k_list=k_list)
        exp = _naive_accuracy(indptr, ranks, k_list=k_list)
        assert set(got) == set(exp)
        for key in got:
            assert np.allclose(got[key], exp[key], rtol=1e-12, atol=0.0), key

        # beyond-accuracy: build a top-k list + random TRAIN counts (with zeros)
        topk = topk_indices(scores, min(10, I))
        counts = rng.integers(0, 50, size=I)

        assert math.isclose(
            coverage_at_k(topk, I), _naive_coverage(topk, I), rel_tol=1e-12
        )
        assert np.allclose(
            novelty_per_user(topk, counts), _naive_novelty(topk, counts),
            rtol=1e-12, atol=0.0,
        )
        assert math.isclose(
            pop_share_at_k(topk, counts, top_frac=0.1),
            _naive_pop_share(topk, counts, top_frac=0.1),
            rel_tol=1e-12,
        )
        assert math.isclose(gini(counts), _naive_gini(counts), rel_tol=1e-12)


# =============================================================================
# (c) Edge cases
# =============================================================================


def test_all_zero_row_ranks_are_index_order():
    # everything tied -> full order is ascending index; rank(g) = g + 1.
    scores = np.zeros((1, 6), dtype=np.float32)
    items = np.array([0, 3, 5], dtype=np.int32)
    indptr = np.array([0, 3], dtype=np.int32)
    ranks = gt_ranks(scores, indptr, items)
    assert np.array_equal(ranks, np.array([1, 4, 6], dtype=np.int32))
    assert np.array_equal(topk_indices(scores, 4), np.array([[0, 1, 2, 3]]))


def test_k_greater_than_catalog():
    scores = np.array([[0.1, 0.9, 0.5]], dtype=np.float32)
    got = topk_indices(scores, 10)
    assert got.shape == (1, 3)  # clamped to I
    assert np.array_equal(got, np.array([[1, 2, 0]]))


def test_single_item_catalog():
    scores = np.array([[0.7], [0.0]], dtype=np.float32)
    assert np.array_equal(topk_indices(scores, 5), np.array([[0], [0]]))
    ranks = gt_ranks(scores, np.array([0, 1, 2]), np.array([0, 0], dtype=np.int32))
    assert np.array_equal(ranks, np.array([1, 1], dtype=np.int32))


def test_coverage_hand():
    topk = np.array([[0, 1], [1, 2]])
    assert coverage_at_k(topk, 6) == pytest.approx(3 / 6)


def test_novelty_hand():
    # counts=[3,1,0], I=3, sum=4 -> p=(c+1)/7. top-k=[0,2].
    # mean(-log2(4/7), -log2(1/7)) = mean(log2(7/4), log2(7))
    counts = np.array([3, 1, 0])
    topk = np.array([[0, 2]])
    expected = np.mean([math.log2(7 / 4), math.log2(7)])
    assert novelty_per_user(topk, counts)[0] == pytest.approx(expected, rel=1e-12)


def test_pop_share_hand():
    # counts=[10,5,5,0,0], I=5, top_frac=0.4 -> n_top=ceil(2)=2.
    # head by (-count, idx) = [0, 1] (items 1,2 tie at 5 -> idx tie-break picks 1).
    # topk slots = [0,3,1,2]; in-head = {0,1} -> 2/4 = 0.5.
    counts = np.array([10, 5, 5, 0, 0])
    topk = np.array([[0, 3], [1, 2]])
    assert pop_share_at_k(topk, counts, top_frac=0.4) == pytest.approx(0.5)


def test_gini_uniform_is_zero():
    assert gini(np.array([4, 4, 4, 4])) == pytest.approx(0.0, abs=1e-12)


def test_gini_one_hot_is_one_minus_inv_n():
    n = 4
    assert gini(np.array([0, 0, 0, 1])) == pytest.approx(1 - 1 / n)


def test_gini_known_value():
    # counts [1,2,3,4]: G = 2*30/(4*10) - 5/4 = 0.25
    assert gini(np.array([1, 2, 3, 4])) == pytest.approx(0.25)


def test_gini_all_zero_defined_zero():
    assert gini(np.array([0, 0, 0])) == 0.0
