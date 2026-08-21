"""Brute-force reference tests for the T8-1 restricted-metric recomposition
(Phase 8; docs/engineering-log/UPGRADE_PLAN.md §8b).

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

A second section at the bottom grades the T8-2 *input-equivalence exception* to
the module's snapshot-lineage guard, against a hand-built toy warehouse in
``tmp_path`` (still no Spark).
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


# --- T8-2 input-equivalence exception (the lineage guard) ---------------------
#
# `regime_map.build_regime_map` requires every source eval record to carry the
# same gold 5-core snapshot id as the eval cache. The T8-2 recomposition runs on
# a rebuilt Linux warehouse whose bytes are identical to the Mac's but whose
# Iceberg snapshot id is new, so ONE preregistered, frozen-TEST comparator record
# carries the old id. These tests grade the narrowly-scoped exception that
# authorizes exactly that: what it must still reject (unattested mismatch,
# tampered proof, an exception nobody needed, a second mismatching arm) and what
# it must disclose when it passes.
#
# Everything below is a toy warehouse in tmp_path — eight items, six users, two
# arms — assembled by hand. No Spark, no cache, no real artifacts.

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402

from batch_recsys_lab.eval import runlog  # noqa: E402
from batch_recsys_lab.eval import regime_map  # noqa: E402
from batch_recsys_lab.features import item_train_stats  # noqa: E402
from batch_recsys_lab.features.splits import load_splits  # noqa: E402

TABLE = "local.gold.interactions_5core"
CACHE_SID = 7217506217965106727      # rebuilt Linux warehouse (the box of record)
RECORD_SID = 8184397443787800955     # the Mac warehouse the comparator was scored on
POP_RUN = "20260805T172047Z-035042b"
ALT_RUN = "20260818T060704Z-109c271"
REF_RUN = "20260817T095926Z-633d454"
DUP_RUN = "20260817T100112Z-633d454"  # the disclosed duplicate: same content, other run_id

ASINS = [f"i{j}" for j in range(8)]
N_TRAIN = np.array([0, 2, 7, 12, 25, 3], dtype=np.int64)  # one user per frozen segment
GT_PAIRS = [(0, 1), (0, 5), (1, 2), (2, 3), (3, 0), (3, 6), (4, 7), (5, 4)]
TOPK = {
    "pop_t12m": [[1, 0, 2, 3], [2, 4, 5, 6], [3, 1, 0, 2], [0, 6, 7, 1], [7, 3, 2, 1], [4, 5, 6, 0]],
    "alt": [[5, 1, 0, 2], [3, 2, 7, 6], [2, 3, 4, 1], [6, 0, 5, 7], [1, 7, 4, 3], [0, 4, 2, 6]],
}
K_LIST = (10, 20, 50)


def _gt_csr() -> tuple[np.ndarray, np.ndarray]:
    users = np.array([u for u, _ in GT_PAIRS], dtype=np.int32)
    items = np.array([i for _, i in GT_PAIRS], dtype=np.int32)
    order = np.argsort(users, kind="stable")
    sizes = np.bincount(users[order], minlength=len(N_TRAIN))
    indptr = np.zeros(len(N_TRAIN) + 1, dtype=np.int64)
    np.cumsum(sizes, out=indptr[1:])
    return indptr, items[order]


def _write_artifact(path: Path, arm: str) -> str:
    """Per-user artifact whose metric columns are computed by the naive reference
    loop above — so the identity anchor passes for reasons independent of the
    kernel under test."""
    topk = np.asarray(TOPK[arm], dtype=np.int32)
    indptr, items = _gt_csr()
    whole = _reference(topk, indptr, items, np.zeros(len(ASINS), dtype=np.int8), 1, K_LIST)
    cols = {
        "user_id": [f"u{u}" for u in range(len(N_TRAIN))],
        "user_idx": np.arange(len(N_TRAIN), dtype=np.int64),
        "segment": [str(s) for s in segment_of(N_TRAIN)],
        "top50": [list(map(int, row)) for row in topk],
        "ndcg@10": whole["ndcg@10"][:, 0],
    }
    for k in K_LIST:
        cols[f"recall@{k}"] = whole[f"recall@{k}"][:, 0]
    pq.write_table(pa.table(cols), path)
    return runlog.sha256_file(path)


def _write_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(
            {
                "snapshot_ids": {TABLE: CACHE_SID},
                "contract_identities": {TABLE: {"name": "gold_interactions_5core", "version": "1"}},
            }
        )
    )
    pq.write_table(pa.table({"item_id": ASINS}), cache_dir / "item_ids.parquet")
    np.save(cache_dir / "test_user_idx.npy", np.array([u for u, _ in GT_PAIRS], dtype=np.int32))
    np.save(cache_dir / "test_item_idx.npy", np.array([i for _, i in GT_PAIRS], dtype=np.int32))
    np.save(cache_dir / "n_train.npy", N_TRAIN)


def _write_item_stats(stats_dir: Path) -> dict:
    stats_dir.mkdir(parents=True, exist_ok=True)
    support = np.array([0, 1, 4, 5, 9, 0, 7, 2], dtype=np.int64)
    last = [
        None if s == 0 else TRAIN_END_MS - d * DAY_MS
        for s, d in zip(support, [0, 10, 100, 400, 30, 0, 200, 5])
    ]
    first = [TRAIN_END_MS - d * DAY_MS for d in [1200, 900, 700, 500, 300, -30, 100, 50]]
    table = pa.table(
        {
            "parent_asin": ASINS,
            "n_train_support": support,
            "last_train_ts": pa.array(last, type=pa.timestamp("ms", tz="UTC")),
            "first_seen_ts": pa.array(first, type=pa.timestamp("ms", tz="UTC")),
        },
        schema=item_train_stats.STATS_SCHEMA,
    )
    stats_path = stats_dir / item_train_stats.STATS_FILENAME
    pq.write_table(table, stats_path)
    manifest = {
        "schema_version": 1,
        "created_ts": "2026-08-18T04:00:00.000000+00:00",
        "source_table": TABLE,
        "interactions_5core_snapshot_id": CACHE_SID,
        "train_end": load_splits(runlog.DEFAULT_SPLITS_PATH).train_end.isoformat(),
        "splits_version": 1,
        "n_items": len(ASINS),
        "stats_parquet": item_train_stats.STATS_FILENAME,
        "stats_parquet_sha256": runlog.sha256_file(stats_path),
    }
    (stats_dir / item_train_stats.MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def _exception(**overrides) -> dict:
    exc = {
        "schema_version": 1,
        "exception_id": "t8-2-linux-rebuild-20260818",
        "table": TABLE,
        "record_snapshot_id": RECORD_SID,
        "cache_snapshot_id": CACHE_SID,
        "applies_to": {"arm": "pop_t12m", "run_id": POP_RUN},
        "proof": {
            "kind": "item_train_stats_parquet_sha256",
            "reference_regime_map_run_id": REF_RUN,
            "expected_sha256": "PLACEHOLDER",
        },
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(exc.get(key), dict):
            exc[key] = {**exc[key], **val}
        else:
            exc[key] = val
    return exc


def _toy_env(
    tmp_path: Path,
    *,
    pop_snapshot_id: int = CACHE_SID,
    alt_snapshot_id: int = CACHE_SID,
    exception: dict | None = None,
    with_reference: bool = True,
    ref_patch=None,
    manifest_patch=None,
) -> tuple[dict, Path, Path]:
    """A complete toy input set for :func:`regime_map.build_regime_map`.

    ``exception`` is inserted verbatim (its ``proof.expected_sha256`` is filled
    in with the real parquet digest when left as ``"PLACEHOLDER"``, so a test
    that wants a *tampered* proof passes its own value).
    """
    cache_dir = tmp_path / "cache" / str(CACHE_SID)
    _write_cache(cache_dir)
    stats_root = tmp_path / "item_train_stats"
    manifest = _write_item_stats(stats_root / str(CACHE_SID))

    artifacts = {arm: tmp_path / f"{arm}.parquet" for arm in ("pop_t12m", "alt")}
    shas = {arm: _write_artifact(path, arm) for arm, path in artifacts.items()}

    if manifest_patch:
        manifest = {**manifest, **manifest_patch}
        (stats_root / str(CACHE_SID) / item_train_stats.MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2)
        )

    lines = [
        {
            "kind": "eval",
            "run_id": run_id,
            "per_user_artifact": str(artifacts[arm]),
            "model": {"name": arm},
            "protocol": {"eval_split": "test"},
            "iceberg_snapshots": {TABLE: sid},
        }
        for arm, run_id, sid in (
            ("pop_t12m", POP_RUN, pop_snapshot_id),
            ("alt", ALT_RUN, alt_snapshot_id),
        )
    ]
    if with_reference:
        ref = {
            "kind": "regime_map",
            "run_id": REF_RUN,
            "iceberg_snapshots": {TABLE: RECORD_SID},
            "source_run_ids": {"pop_t12m": POP_RUN, "als": "20260806T082441Z-2f2f26d"},
            "source_artifact_sha256s": {"pop_t12m": shas["pop_t12m"], "als": "sha256:00"},
            "item_stats": {
                "manifest": {
                    **manifest,
                    "created_ts": "2026-08-17T09:48:51.778123+00:00",
                    "interactions_5core_snapshot_id": RECORD_SID,
                }
            },
            "results": {"identity_check": {"passed": True}},
        }
        if ref_patch:
            ref = ref_patch(ref)
        lines.append(ref)
        # The disclosed duplicate of the reference run: same content, DIFFERENT
        # run_id. Filtering by run_id must still yield exactly one.
        lines.append({**ref, "run_id": DUP_RUN})

    results_path = tmp_path / "runs.jsonl"
    results_path.write_text("".join(json.dumps(rec) + "\n" for rec in lines))

    config = {
        "kind": "regime_map",
        "run_ids": {"pop_t12m": POP_RUN, "alt": ALT_RUN},
        "delta": {"minuend": "alt", "subtrahend": "pop_t12m"},
        "split": "test",
        "cell_metrics": ["ndcg@10", "recall@20"],
        "k_list": list(K_LIST),
        "bootstrap": {"n_resamples": 8, "seed": 20260805},
        "cache_dir": str(cache_dir),
        "item_stats_dir": str(stats_root),
        "five_core_table": TABLE,
        "splits_path": str(runlog.DEFAULT_SPLITS_PATH),
        "results_path": str(results_path),
    }
    if exception is not None:
        proof = dict(exception["proof"])
        if proof.get("expected_sha256") == "PLACEHOLDER":
            proof["expected_sha256"] = manifest["stats_parquet_sha256"]
        config[regime_map.INPUT_EQUIVALENCE_KEY] = {**exception, "proof": proof}

    config_path = tmp_path / "regime_map_toy.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config, config_path, results_path


# --- (i) the guard without an exception is untouched --------------------------


def test_unattested_snapshot_mismatch_still_raises(tmp_path: Path) -> None:
    config, _, results_path = _toy_env(tmp_path, pop_snapshot_id=RECORD_SID)
    with pytest.raises(RuntimeError, match=r"pop_t12m .* was scored on .* != cache snapshot"):
        regime_map.build_regime_map(config, results_path)


def test_toy_env_without_mismatch_is_clean(tmp_path: Path) -> None:
    """Control: the same toy warehouse with matching snapshots computes fine —
    so the failures below are about lineage, not about the fixture."""
    config, _, results_path = _toy_env(tmp_path)
    out = regime_map.build_regime_map(config, results_path)
    assert out["identity_check"]["passed"] is True
    assert out["input_equivalence"] is None


# --- (ii) a declared exception with a broken proof is still fatal -------------


def test_exception_with_wrong_expected_sha_raises(tmp_path: Path) -> None:
    config, _, results_path = _toy_env(
        tmp_path,
        pop_snapshot_id=RECORD_SID,
        exception=_exception(proof={"expected_sha256": "sha256:" + "0" * 64}),
    )
    with pytest.raises(RuntimeError, match="proof.expected_sha256"):
        regime_map.build_regime_map(config, results_path)


def test_exception_with_tampered_local_manifest_sha_raises(tmp_path: Path) -> None:
    config, _, results_path = _toy_env(
        tmp_path,
        pop_snapshot_id=RECORD_SID,
        exception=_exception(),
        manifest_patch={"stats_parquet_sha256": "sha256:" + "1" * 64},
    )
    with pytest.raises(RuntimeError, match="its local manifest records"):
        regime_map.build_regime_map(config, results_path)


def test_exception_with_tampered_reference_record_raises(tmp_path: Path) -> None:
    """The reference regime-map record is evidence, so every field the exception
    leans on is checked: parquet digest, comparator artifact digest, and the
    identity anchor that record itself passed."""
    def _bad_stats_sha(ref: dict) -> dict:
        ref["item_stats"]["manifest"]["stats_parquet_sha256"] = "sha256:" + "2" * 64
        return ref

    def _bad_artifact_sha(ref: dict) -> dict:
        ref["source_artifact_sha256s"]["pop_t12m"] = "sha256:" + "3" * 64
        return ref

    def _identity_failed(ref: dict) -> dict:
        ref["results"]["identity_check"]["passed"] = False
        return ref

    for patch, message in (
        (_bad_stats_sha, "item_stats parquet sha"),
        (_bad_artifact_sha, "comparator artifact"),
        (_identity_failed, "identity_check.passed"),
    ):
        config, _, results_path = _toy_env(
            tmp_path / message.replace(" ", "_").replace(".", "_"),
            pop_snapshot_id=RECORD_SID,
            exception=_exception(),
            ref_patch=patch,
        )
        with pytest.raises(RuntimeError, match=message):
            regime_map.build_regime_map(config, results_path)


def test_exception_with_missing_reference_record_raises(tmp_path: Path) -> None:
    config, _, results_path = _toy_env(
        tmp_path, pop_snapshot_id=RECORD_SID, exception=_exception(), with_reference=False
    )
    with pytest.raises(RuntimeError, match="expected exactly 1 kind=regime_map record"):
        regime_map.build_regime_map(config, results_path)


# --- (iii) an exception nobody needed is stale authorization ------------------


def test_unused_exception_raises(tmp_path: Path) -> None:
    config, _, results_path = _toy_env(tmp_path, exception=_exception())
    with pytest.raises(RuntimeError, match="already matches the cache snapshot"):
        regime_map.build_regime_map(config, results_path)


# --- (v) the exception covers ONE arm, not "mismatches in general" ------------


def test_second_mismatching_arm_still_raises(tmp_path: Path) -> None:
    config, _, results_path = _toy_env(
        tmp_path,
        pop_snapshot_id=RECORD_SID,
        alt_snapshot_id=123456789,
        exception=_exception(),
    )
    with pytest.raises(RuntimeError, match="does NOT cover"):
        regime_map.build_regime_map(config, results_path)


def test_exception_naming_a_different_run_id_raises(tmp_path: Path) -> None:
    config, _, results_path = _toy_env(
        tmp_path,
        pop_snapshot_id=RECORD_SID,
        exception=_exception(applies_to={"run_id": "20990101T000000Z-deadbee"}),
    )
    with pytest.raises(RuntimeError, match="applies_to names run_id"):
        regime_map.build_regime_map(config, results_path)


# --- (iv) the happy path, and what it must disclose ---------------------------


def test_valid_exception_is_honored_for_the_named_arm_only(tmp_path: Path) -> None:
    exc = _exception()
    config, config_path, results_path = _toy_env(
        tmp_path / "excepted", pop_snapshot_id=RECORD_SID, exception=exc
    )
    out = regime_map.build_regime_map(config, results_path)

    block = out["input_equivalence"]
    assert block["exception_used"] is True
    assert block["declaration"] == config[regime_map.INPUT_EQUIVALENCE_KEY]  # verbatim
    assert block["applied_to"] == {
        "arm": "pop_t12m",
        "run_id": POP_RUN,
        "record_snapshot_id": RECORD_SID,
        "cache_snapshot_id": CACHE_SID,
    }
    val = block["validation"]
    assert val["status"] == "passed"
    assert val["reference_regime_map_run_id"] == REF_RUN
    assert (
        val["computed_stats_parquet_sha256"]
        == val["local_manifest_stats_parquet_sha256"]
        == val["reference_stats_parquet_sha256"]
    )
    assert val["manifests_equal_except"] == ["created_ts", "interactions_5core_snapshot_id"]
    assert (
        val["local_computed_comparator_artifact_sha256"]
        == val["reference_comparator_artifact_sha256"]
    )
    # The unconditional guards still ran, and the analysis is the normal one.
    assert out["identity_check"]["passed"] is True
    assert out["n_users"] == len(N_TRAIN)

    record = regime_map.build_record(config, config_path, out)
    # The record keeps the BOX's actual snapshot ids; the exception is disclosed
    # alongside them, never substituted for them.
    assert record["iceberg_snapshots"][TABLE] == CACHE_SID
    assert record[regime_map.INPUT_EQUIVALENCE_KEY] == block

    # Without an exception the record shape is exactly what it has always been.
    clean_cfg, clean_path, clean_results = _toy_env(tmp_path / "clean")
    clean_out = regime_map.build_regime_map(clean_cfg, clean_results)
    clean_record = regime_map.build_record(clean_cfg, clean_path, clean_out)
    assert regime_map.INPUT_EQUIVALENCE_KEY not in clean_record
    assert set(record) - set(clean_record) == {regime_map.INPUT_EQUIVALENCE_KEY}
    assert set(clean_record) - set(record) == set()


# --- the block's own schema ---------------------------------------------------


def test_input_equivalence_block_schema_is_strict() -> None:
    key = regime_map.INPUT_EQUIVALENCE_KEY
    assert regime_map._parse_input_equivalence({}) is None
    with pytest.raises(RuntimeError, match="unknown key"):
        regime_map._parse_input_equivalence({key: {**_exception(), "enabled": True}})
    with pytest.raises(RuntimeError, match="missing required key"):
        regime_map._parse_input_equivalence(
            {key: {k: v for k, v in _exception().items() if k != "proof"}}
        )
    with pytest.raises(RuntimeError, match="record_snapshot_id must be int"):
        regime_map._parse_input_equivalence({key: _exception(record_snapshot_id="8184397443787800955")})
    with pytest.raises(RuntimeError, match="schema_version"):
        regime_map._parse_input_equivalence({key: _exception(schema_version=2)})
    with pytest.raises(RuntimeError, match="proof.kind"):
        regime_map._parse_input_equivalence({key: _exception(proof={"kind": "trust_me"})})
    with pytest.raises(RuntimeError, match="unknown key"):
        regime_map._parse_input_equivalence(
            {key: _exception(applies_to={"arm": "pop_t12m", "run_id": POP_RUN, "why": "x"})}
        )
