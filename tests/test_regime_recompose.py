"""Brute-force reference tests for the T8-1 restricted-metric recomposition
(Phase 8; UPGRADE_PLAN.md §8b).

The regime map's whole claim to exactness rests on one vectorized kernel:
``regime_map.recompose_restricted`` recomposes recall@K (K <= 50) and NDCG@10
*restricted to an arbitrary GT subset* from the persisted per-user top-50 lists.
Here that kernel is graded against a naive per-user, per-bucket Python loop
transcribed straight from the formulas in ``eval/metrics.accuracy_metrics`` —
independent code, same definitions — over randomized small catalogs, plus the
edge cases that would silently produce plausible-but-wrong numbers:

* every GT item outside the top-K (all metrics 0, but the user still *belongs* to
  the cell — the count must not vanish);
* ``|GT_b(u)| > 10``, where IDCG@10 saturates at 10 ideal positions;
* buckets with zero GT for a user (must stay 0.0, never NaN from 0/0);
* a user with no GT at all (empty CSR row).

Pure numpy — no Spark, no cache, no artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from batch_recsys_lab.eval.protocol import DEEP_BUCKET_LABELS, deep_bucket_of, segment_of
from batch_recsys_lab.eval.regime_map import (
    first_seen_codes,
    recompose_restricted,
    recency_codes,
    support_codes,
    topk_ranks,
)

MISSING_MS = np.iinfo(np.int64).min
# 2022-06-30T23:59:59.999Z, the frozen train_end, in epoch milliseconds.
TRAIN_END_MS = 1656633599999
DAY_MS = 86_400_000


# --- the naive reference ------------------------------------------------------


def _reference(
    topk: np.ndarray,
    gt_indptr: np.ndarray,
    gt_items: np.ndarray,
    item_code: np.ndarray,
    n_buckets: int,
    k_list: tuple[int, ...],
    ndcg_k: int = 10,
) -> dict[str, np.ndarray]:
    """Per-(user, bucket) restricted metrics by explicit loops. No numpy tricks."""
    n_users = len(gt_indptr) - 1
    gt_count = np.zeros((n_users, n_buckets), dtype=np.int64)
    recall = {k: np.zeros((n_users, n_buckets)) for k in k_list}
    ndcg = np.zeros((n_users, n_buckets))

    for u in range(n_users):
        lo, hi = int(gt_indptr[u]), int(gt_indptr[u + 1])
        row = list(topk[u])
        for b in range(n_buckets):
            members = [int(g) for g in gt_items[lo:hi] if int(item_code[int(g)]) == b]
            m = len(members)
            gt_count[u, b] = m
            if m == 0:
                continue
            ranks = []
            for g in members:
                ranks.append(row.index(g) + 1 if g in row else None)
            for k in k_list:
                n_hit = sum(1 for r in ranks if r is not None and r <= k)
                recall[k][u, b] = n_hit / m
            dcg = sum(
                1.0 / np.log2(r + 1.0) for r in ranks if r is not None and r <= ndcg_k
            )
            idcg = sum(1.0 / np.log2(j + 1.0) for j in range(1, min(m, ndcg_k) + 1))
            ndcg[u, b] = dcg / idcg if idcg > 0 else 0.0

    out = {"gt_count": gt_count, f"ndcg@{ndcg_k}": ndcg}
    for k in k_list:
        out[f"recall@{k}"] = recall[k]
    return out


def _assert_matches(got: dict, want: dict) -> None:
    assert set(want) <= set(got), f"missing keys: {set(want) - set(got)}"
    for key, ref in want.items():
        if ref.dtype.kind == "i":
            np.testing.assert_array_equal(got[key], ref, err_msg=key)
        else:
            np.testing.assert_allclose(got[key], ref, rtol=0, atol=1e-12, err_msg=key)


def _random_case(rng: np.random.Generator, n_users: int, n_items: int, k: int, n_buckets: int):
    """(topk, gt_indptr, gt_items, item_code) with per-user GT sizes 0..14."""
    topk = np.stack(
        [rng.permutation(n_items)[: min(k, n_items)] for _ in range(n_users)]
    ).astype(np.int32)
    sizes = rng.integers(0, 15, size=n_users)
    gt_indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(sizes, out=gt_indptr[1:])
    gt_items = np.concatenate(
        [
            rng.choice(n_items, size=int(s), replace=False).astype(np.int32)
            if s
            else np.zeros(0, dtype=np.int32)
            for s in sizes
        ]
    ).astype(np.int32)
    item_code = rng.integers(0, n_buckets, size=n_items).astype(np.int8)
    return topk, gt_indptr, gt_items, item_code


# --- randomized equivalence ---------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_recompose_matches_bruteforce_random(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_items = int(rng.integers(20, 120))
    n_users = int(rng.integers(3, 25))
    k = int(rng.integers(5, 15))
    n_buckets = int(rng.integers(2, 5))
    topk, gt_indptr, gt_items, item_code = _random_case(rng, n_users, n_items, k, n_buckets)
    k_list = (1, 3, k)

    got = recompose_restricted(
        topk, gt_indptr, gt_items, item_code, n_buckets, k_list=k_list, ndcg_k=10
    )
    want = _reference(topk, gt_indptr, gt_items, item_code, n_buckets, k_list, ndcg_k=10)
    _assert_matches(got, want)


def test_block_size_is_irrelevant() -> None:
    """The row-blocked rank scan must be bit-identical to a single-block scan."""
    rng = np.random.default_rng(99)
    topk, gt_indptr, gt_items, item_code = _random_case(rng, 30, 80, 12, 3)
    big = recompose_restricted(topk, gt_indptr, gt_items, item_code, 3, block=10**9)
    tiny = recompose_restricted(topk, gt_indptr, gt_items, item_code, 3, block=1)
    for key in big:
        np.testing.assert_array_equal(big[key], tiny[key])


# --- edge cases ---------------------------------------------------------------


def test_all_gt_outside_topk() -> None:
    """A user whose every GT item misses the top-K: metrics 0, membership kept."""
    topk = np.array([[0, 1, 2]], dtype=np.int32)
    gt_items = np.array([7, 8, 9], dtype=np.int32)
    gt_indptr = np.array([0, 3], dtype=np.int64)
    item_code = np.zeros(10, dtype=np.int8)

    got = recompose_restricted(topk, gt_indptr, gt_items, item_code, 1, k_list=(1, 3))
    want = _reference(topk, gt_indptr, gt_items, item_code, 1, (1, 3))
    _assert_matches(got, want)
    assert got["gt_count"][0, 0] == 3  # the user IS in the cell
    assert got["recall@3"][0, 0] == 0.0
    assert got["ndcg@10"][0, 0] == 0.0
    np.testing.assert_array_equal(topk_ranks(topk, gt_indptr, gt_items), np.zeros(3))


def test_gt_count_above_ten_saturates_idcg() -> None:
    """|GT_b(u)| = 12 > 10: IDCG@10 uses 10 ideal positions, so a user who hits
    exactly the top 10 of a 12-item bucket scores ndcg@10 == 1.0 while
    recall@10 == 10/12."""
    n_items = 40
    topk = np.arange(n_items, dtype=np.int32)[None, :]  # ranks item i at position i+1
    gt_items = np.arange(12, dtype=np.int32)  # exactly the top 12 positions
    gt_indptr = np.array([0, 12], dtype=np.int64)
    item_code = np.zeros(n_items, dtype=np.int8)

    got = recompose_restricted(topk, gt_indptr, gt_items, item_code, 1, k_list=(10, 20, 50))
    want = _reference(topk, gt_indptr, gt_items, item_code, 1, (10, 20, 50))
    _assert_matches(got, want)
    assert got["ndcg@10"][0, 0] == pytest.approx(1.0)
    assert got["recall@10"][0, 0] == pytest.approx(10 / 12)
    assert got["recall@20"][0, 0] == pytest.approx(1.0)


def test_empty_bucket_and_empty_user() -> None:
    """A bucket with no GT for a user stays exactly 0.0 (no 0/0 NaN), and a user
    with no GT at all contributes an all-zero row."""
    topk = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
    # user 0: GT {0, 5} both in bucket 0; user 1: no GT at all.
    gt_items = np.array([0, 5], dtype=np.int32)
    gt_indptr = np.array([0, 2, 2], dtype=np.int64)
    item_code = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)

    got = recompose_restricted(topk, gt_indptr, gt_items, item_code, 2, k_list=(4,))
    want = _reference(topk, gt_indptr, gt_items, item_code, 2, (4,))
    _assert_matches(got, want)
    assert got["gt_count"][0, 1] == 0
    assert got["recall@4"][0, 1] == 0.0
    assert got["ndcg@10"][0, 1] == 0.0
    assert np.all(np.isfinite(got["ndcg@10"]))
    assert got["gt_count"][1].sum() == 0


def test_gt_indptr_length_is_validated() -> None:
    with pytest.raises(ValueError, match="gt_indptr length"):
        topk_ranks(
            np.zeros((3, 5), dtype=np.int32),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
        )


def test_bucket_ordinal_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="bucket ordinal"):
        recompose_restricted(
            np.array([[0, 1]], dtype=np.int32),
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([5, 5], dtype=np.int8),
            2,
        )


# --- item-axis bucketing ------------------------------------------------------


def test_support_codes_match_preregistered_edges() -> None:
    support = np.array([0, 1, 3, 4, 5, 6, 1000], dtype=np.int64)
    np.testing.assert_array_equal(support_codes(support), [0, 1, 1, 1, 2, 2, 2])


def test_recency_codes_match_preregistered_edges() -> None:
    last = np.array(
        [
            TRAIN_END_MS,                    # 0 days  -> <=90d
            TRAIN_END_MS - 90 * DAY_MS,      # exactly 90 -> <=90d (inclusive)
            TRAIN_END_MS - 91 * DAY_MS,      # 91 -> 91-365d
            TRAIN_END_MS - 365 * DAY_MS,     # exactly 365 -> 91-365d (inclusive)
            TRAIN_END_MS - 366 * DAY_MS,     # 366 -> >365d
            MISSING_MS,                      # absent
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(
        recency_codes(last, TRAIN_END_MS, MISSING_MS), [0, 0, 1, 1, 2, 3]
    )


def test_first_seen_codes_match_preregistered_edges() -> None:
    def ms(year: int, month: int = 1, day: int = 15) -> int:
        from datetime import datetime, timezone

        return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)

    first = np.array(
        [
            ms(2014),
            ms(2019, 12, 31),
            ms(2020, 6),
            ms(2021, 11),
            ms(2022, 1),          # 2022 and <= train_end -> 2022-H1
            ms(2022, 6, 30),      # train_end day, before 23:59:59.999 -> 2022-H1
            ms(2022, 7, 1),       # just past train_end -> post-cutoff
            ms(2023, 5),
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(
        first_seen_codes(first, TRAIN_END_MS, MISSING_MS), [0, 0, 1, 2, 3, 3, 4, 4]
    )


# --- T8-3 depth buckets -------------------------------------------------------


def test_deep_buckets_refine_the_frozen_segments() -> None:
    n_train = np.array([0, 1, 4, 5, 9, 10, 19, 20, 49, 50, 99, 100, 5000])
    deep = [str(s) for s in deep_bucket_of(n_train)]
    assert deep == [
        "0", "1-4", "1-4", "5-9", "5-9", "10-19", "10-19",
        "20-49", "20-49", "50-99", "50-99", "100+", "100+",
    ]
    # The first four deep buckets must be exactly the frozen segments; the last
    # three must all live inside "20+" (that is what makes the self-check valid).
    seg = [str(s) for s in segment_of(n_train)]
    for d, s in zip(deep, seg):
        if d in ("0", "1-4", "5-9", "10-19"):
            assert d == s
        else:
            assert s == "20+"
    assert set(deep) <= set(DEEP_BUCKET_LABELS)
