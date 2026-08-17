"""Tests for the TRAIN-pair age cache extension (Phase 8, T8-2).

The Spark half of ``eval/extract_age.py`` is a read + groupBy; the part that
carries the risk is the ALIGNMENT contract — the saved pair arrays' row order is
a Spark shuffle artifact, so ages must be matched to the cached arrays by
``(user_idx, item_idx)`` key and any drift in the pair multiset must abort the
job before it writes. That logic is pure numpy (``align_ages``) and is tested
here without a JVM; one ``spark``-marked round-trip then exercises the whole job
(time-travelled read → group-by → align → write) against a toy gold table, so the
14M-pair production run is not the first time it executes.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from batch_recsys_lab.eval.extract_age import _composite_keys, align_ages

# Cached TRAIN pairs, in a deliberately unsorted order (as Spark emits them).
CACHED_U = np.array([2, 0, 1, 0], dtype=np.int32)
CACHED_I = np.array([3, 1, 0, 2], dtype=np.int32)
N_ITEMS = 4
AGE_OF = {(2, 3): 5.0, (0, 1): 10.25, (1, 0): 400.0, (0, 2): 0.0}


def _shuffled(order: list[int]):
    """The same pairs/ages in a different row order."""
    u = np.array([CACHED_U[k] for k in order], dtype=np.int32)
    i = np.array([CACHED_I[k] for k in order], dtype=np.int32)
    a = np.array([AGE_OF[(int(CACHED_U[k]), int(CACHED_I[k]))] for k in order])
    return u, i, a


# --------------------------------------------------------------------------- #
# 1. alignment is by key, not by position                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]])
def test_ages_are_gathered_into_cached_row_order(order):
    u, i, a = _shuffled(order)
    ages = align_ages(CACHED_U, CACHED_I, u, i, a, N_ITEMS)
    assert ages.dtype == np.float32
    expected = [AGE_OF[(int(x), int(y))] for x, y in zip(CACHED_U, CACHED_I)]
    assert np.allclose(ages, expected)


def test_empty_train_universe_round_trips():
    empty = np.array([], dtype=np.int32)
    ages = align_ages(empty, empty, empty, empty, np.array([]), N_ITEMS)
    assert ages.shape == (0,) and ages.dtype == np.float32


# --------------------------------------------------------------------------- #
# 2. every drift mode aborts (the caller writes nothing on an exception)        #
# --------------------------------------------------------------------------- #
def test_pair_count_mismatch_aborts():
    u, i, a = _shuffled([0, 1, 2])
    with pytest.raises(ValueError, match="TRAIN pair count mismatch"):
        align_ages(CACHED_U, CACHED_I, u, i, a, N_ITEMS)


def test_substituted_pair_aborts():
    u, i, a = _shuffled([0, 1, 2, 3])
    i = i.copy()
    i[0] = (i[0] + 1) % N_ITEMS  # same length, one pair swapped for another
    with pytest.raises(ValueError, match="absent from the recomputation"):
        align_ages(CACHED_U, CACHED_I, u, i, a, N_ITEMS)


def test_duplicate_pair_aborts():
    u, i, a = _shuffled([0, 1, 2, 3])
    u, i = u.copy(), i.copy()
    u[3], i[3] = u[0], i[0]  # duplicate (user, item) => dedup invariant broken
    with pytest.raises(ValueError, match="duplicate"):
        align_ages(CACHED_U, CACHED_I, u, i, a, N_ITEMS)


def test_ragged_input_columns_abort():
    u, i, a = _shuffled([0, 1, 2, 3])
    with pytest.raises(ValueError, match="disagree in length"):
        align_ages(CACHED_U, CACHED_I, u, i, a[:-1], N_ITEMS)


@pytest.mark.parametrize("bad", [-0.001, np.nan, np.inf])
def test_impossible_ages_abort(bad):
    u, i, a = _shuffled([0, 1, 2, 3])
    a = a.copy()
    a[2] = bad
    with pytest.raises(ValueError):
        align_ages(CACHED_U, CACHED_I, u, i, a, N_ITEMS)


# --------------------------------------------------------------------------- #
# 3. the composite key is a bijection at lab scale                             #
# --------------------------------------------------------------------------- #
def test_composite_keys_are_unique_at_lab_scale():
    rng = np.random.default_rng(20260805)
    n_users, n_items = 1_641_058, 368_260  # the frozen 5-core universe
    users = rng.integers(0, n_users, size=50_000, dtype=np.int64)
    items = rng.integers(0, n_items, size=50_000, dtype=np.int64)
    keys = _composite_keys(users.astype(np.int32), items.astype(np.int32), n_items)
    pairs = {(int(u), int(i)) for u, i in zip(users, items)}
    assert len(np.unique(keys)) == len(pairs)
    assert keys.dtype == np.int64


# --------------------------------------------------------------------------- #
# 4. Spark round-trip on a toy gold table (the whole job, end to end)          #
# --------------------------------------------------------------------------- #
AGE_FIVE_CORE = "local.gold.five_core_age"
AGE_USER_STATS = "local.gold.user_stats_age"
AGE_ITEM_FEATURES = "local.gold.item_features_age"
AGE_POPULARITY = "local.gold.popularity_age"


def _cache_snapshot_id(cache_dir: Path, table: str) -> int:
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
    return int(manifest["snapshot_ids"][table])


@pytest.mark.spark
def test_extract_age_round_trip_against_a_toy_cache(spark, tmp_path):
    """Build a real cache with ``extract``, extend it with ``extract_age``, and
    check the ages against hand-computed day offsets from the frozen train_end."""
    from batch_recsys_lab.eval.dataset import TRAIN_AGE_FILE, load_dataset
    from batch_recsys_lab.eval.extract import extract
    from batch_recsys_lab.eval.extract_age import AGE_MANIFEST, extract_age
    from batch_recsys_lab.features.splits import load_splits
    from tests.test_eval_extract import (
        FIVE_CORE_DDL,
        ITEM_FEATURES_DDL,
        POPULARITY_DDL,
        USER_STATS_DDL,
        _stamp_contract,
        _write,
    )

    s = load_splits()
    day = timedelta(days=1)
    # (user, item) -> age in days before train_end. The 12-hour offset proves the
    # ages are fractional, not truncated to whole days.
    ages = {("U1", "P1"): 10.0, ("U1", "P2"): 0.5, ("U2", "P1"): 400.0}
    five_core_rows = [
        (u, p, s.train_end - timedelta(days=age), 4.0) for (u, p), age in ages.items()
    ]
    five_core_rows.append(("U2", "P2", s.val_end + day, 5.0))  # TEST row: no age
    _write(spark, five_core_rows, FIVE_CORE_DDL, AGE_FIVE_CORE)
    _write(
        spark,
        [
            ("U1", 2, 2, 0, 0, s.train_end - 10 * day, s.train_end, 10),
            ("U2", 2, 1, 0, 1, s.train_end - 400 * day, s.val_end + day, 401),
        ],
        USER_STATS_DDL,
        AGE_USER_STATS,
    )
    _write(
        spark,
        [
            ("P1", "t1", "acme", 9.99, "Cat A", ["c"], 4.0, 10),
            ("P2", "t2", "acme", 8.99, "Cat B", ["c"], 4.5, 20),
        ],
        ITEM_FEATURES_DDL,
        AGE_ITEM_FEATURES,
    )
    _write(spark, [(s.train_end, 0, "P1", 2, 2)], POPULARITY_DDL, AGE_POPULARITY)
    for t, name in (
        (AGE_FIVE_CORE, "gold_interactions_5core"),
        (AGE_USER_STATS, "gold_user_stats"),
        (AGE_ITEM_FEATURES, "gold_item_features"),
        (AGE_POPULARITY, "gold_popularity"),
    ):
        _stamp_contract(spark, t, name, "1")

    summary = extract(
        spark,
        out=tmp_path / "cache",
        five_core_table=AGE_FIVE_CORE,
        user_stats_table=AGE_USER_STATS,
        item_features_table=AGE_ITEM_FEATURES,
        popularity_table=AGE_POPULARITY,
    )
    cache_dir = Path(summary["cache_dir"])

    out = extract_age(spark, cache_dir, five_core_table=AGE_FIVE_CORE)
    assert out["status"] == "built"
    assert out["n_train_pairs"] == 3

    ds = load_dataset(cache_dir, with_train_pairs=True)
    got = {
        (str(ds.user_ids[u]), str(ds.item_ids[i])): float(a)
        for u, i, a in zip(ds.train_user_idx, ds.train_item_idx, ds.train_age_days)
    }
    assert set(got) == set(ages)
    for key, expected in ages.items():
        assert got[key] == pytest.approx(expected, abs=1e-3)

    man = json.loads((cache_dir / AGE_MANIFEST).read_text())
    assert man["five_core_snapshot_id"] == _cache_snapshot_id(cache_dir, AGE_FIVE_CORE)
    assert man["n_train_pairs"] == 3
    assert man["train_end"].startswith("2022-06-30")

    # Idempotent: a second call short-circuits and leaves the bytes alone.
    before = (cache_dir / TRAIN_AGE_FILE).read_bytes()
    again = extract_age(spark, cache_dir, five_core_table=AGE_FIVE_CORE)
    assert again["status"] == "up_to_date"
    assert (cache_dir / TRAIN_AGE_FILE).read_bytes() == before
