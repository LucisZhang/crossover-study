"""Eval cache extract + EvalDataset round-trip tests (Phase 2, T1).

Synthetic gold tables (distinct table names, passed into the extract functions —
never ``local.gold.*`` hardcoded) spanning TRAIN/VAL/TEST around the frozen
split boundaries in ``configs/splits.yaml``.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from batch_recsys_lab.eval import extract as extract_mod
from batch_recsys_lab.eval.extract import _read_table, _snapshot_id, extract
from batch_recsys_lab.eval.dataset import load_dataset
from batch_recsys_lab.features.splits import load_splits
from conftest import warehouse_of

pytestmark = pytest.mark.spark

UTC = timezone.utc
DAY = timedelta(days=1)

FIVE_CORE_DDL = "user_id string, parent_asin string, ts timestamp, rating double"
USER_STATS_DDL = (
    "user_id string, n_total long, n_train long, n_val long, n_test long, "
    "first_ts timestamp, last_ts timestamp, tenure_days long"
)
ITEM_FEATURES_DDL = (
    "parent_asin string, title string, brand_norm string, price_usd double, "
    "main_category string, categories array<string>, average_rating double, "
    "rating_number long"
)
POPULARITY_DDL = (
    "as_of timestamp, window_days int, parent_asin string, n_interactions long, "
    "n_unique_users long"
)

FIVE_CORE = "local.gold.five_core_extract"
USER_STATS = "local.gold.user_stats_extract"
ITEM_FEATURES = "local.gold.item_features_extract"
POPULARITY = "local.gold.popularity_extract"


def _write(spark, rows, ddl, table):
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.gold")
    spark.createDataFrame(rows, ddl).writeTo(table).createOrReplace()


def _stamp_contract(spark, table, name, version):
    spark.sql(
        f"ALTER TABLE {table} SET TBLPROPERTIES "
        f"('contracts.name'='{name}', 'contracts.version'='{version}')"
    )


@pytest.fixture()
def toy_gold(spark):
    s = load_splits()
    train_ts = s.train_end - 10 * DAY
    val_ts = s.train_end + DAY
    test_ts = s.val_end + DAY

    # Items: P1..P4, sorted catalog order is P1,P2,P3,P4.
    five_core_rows = [
        ("U1", "P1", train_ts, 5.0),
        ("U1", "P2", train_ts, 4.0),
        ("U1", "P3", val_ts, 3.0),   # U1's VAL GT
        ("U2", "P1", train_ts, 2.0),
        ("U2", "P4", test_ts, 1.0),  # U2's TEST GT
        ("U2", "P2", test_ts, 5.0),  # U2's second TEST GT item
    ]
    _write(spark, five_core_rows, FIVE_CORE_DDL, FIVE_CORE)

    user_stats_rows = [
        ("U1", 3, 2, 1, 0, train_ts, val_ts, 11),
        ("U2", 3, 1, 0, 2, train_ts, test_ts, 20),
    ]
    _write(spark, user_stats_rows, USER_STATS_DDL, USER_STATS)

    item_features_rows = [
        ("P1", "t1", "acme", 9.99, "Cat A", ["c"], 4.0, 10),
        ("P2", "t2", "acme", 8.99, "Cat B", ["c"], 4.5, 20),
        ("P3", "t3", "sony", 7.99, None, ["c"], 3.5, 5),
        ("P4", "t4", "sony", 6.99, "Cat A", ["c"], 4.2, 8),
    ]
    _write(spark, item_features_rows, ITEM_FEATURES_DDL, ITEM_FEATURES)

    popularity_rows = [
        (s.train_end, 0, "P1", 2, 2),
        (s.train_end, 0, "P2", 1, 1),
        (s.train_end, 365, "P1", 2, 2),
        (s.val_end, 0, "P1", 2, 2),
        (s.val_end, 365, "P3", 1, 1),
    ]
    _write(spark, popularity_rows, POPULARITY_DDL, POPULARITY)

    for t, name in (
        (FIVE_CORE, "gold_interactions_5core"),
        (USER_STATS, "gold_user_stats"),
        (ITEM_FEATURES, "gold_item_features"),
        (POPULARITY, "gold_popularity"),
    ):
        _stamp_contract(spark, t, name, "1")

    return s


def test_extract_and_dataset_round_trip(spark, toy_gold, tmp_path):
    out_dir = tmp_path / "cache"
    summary = extract(
        spark,
        out=out_dir,
        five_core_table=FIVE_CORE,
        user_stats_table=USER_STATS,
        item_features_table=ITEM_FEATURES,
        popularity_table=POPULARITY,
    )
    assert summary["status"] == "built"
    cache_dir = Path(summary["cache_dir"])
    assert cache_dir.exists()

    ds = load_dataset(cache_dir)

    # Catalog sorted ascending.
    assert list(ds.item_ids) == ["P1", "P2", "P3", "P4"]
    assert list(ds.user_ids) == ["U1", "U2"]

    # TRAIN CSR: U1->{P1,P2}, U2->{P1} => nnz == 3.
    assert ds.train_csr.nnz == 3
    assert ds.train_csr.shape == (2, 4)
    u1 = 0
    p1, p2 = 0, 1
    assert ds.train_csr[u1, p1] == 1.0
    assert ds.train_csr[u1, p2] == 1.0

    # TRAIN ratings cache: element-aligned with train_user_idx/train_item_idx.
    train_user_idx = np.load(cache_dir / "train_user_idx.npy")
    train_item_idx = np.load(cache_dir / "train_item_idx.npy")
    train_rating = np.load(cache_dir / "train_rating.npy")
    assert train_rating.dtype == np.float32
    assert len(train_rating) == len(train_user_idx) == len(train_item_idx)
    expected = {(0, p1): 5.0, (0, p2): 4.0, (1, p1): 2.0}
    for u, i, r in zip(train_user_idx, train_item_idx, train_rating):
        assert expected[(int(u), int(i))] == float(r)

    # GT: VAL — only U1 -> {P3}.
    val_gt = ds.gt["val"]
    assert list(val_gt.user_idx) == [0]
    assert list(val_gt.item_idx) == [2]  # P3 index
    assert list(val_gt.indptr) == [0, 1]

    # GT: TEST — only U2 -> {P2, P4} (sorted item order within user not required,
    # but both items present).
    test_gt = ds.gt["test"]
    assert list(test_gt.user_idx) == [1]
    u2_items = set(test_gt.item_idx[test_gt.indptr[0]:test_gt.indptr[1]].tolist())
    assert u2_items == {1, 3}  # P2, P4

    # Popularity vectors.
    pop_train_0 = ds.pop[("train_end", 0)]
    assert pop_train_0[p1] == 2.0
    assert pop_train_0[p2] == 1.0
    assert pop_train_0[2] == 0.0  # P3 absent -> 0
    pop_val_365 = ds.pop[("val_end", 365)]
    assert pop_val_365[2] == 1.0  # P3

    # Item categories: P3 has NULL main_category -> "__unknown__" code.
    assert ds.category_names[0] == "__unknown__"
    p3_code = ds.item_category_codes[2]
    assert ds.category_names[p3_code] == "__unknown__"
    p1_code = ds.item_category_codes[0]
    assert ds.category_names[p1_code] == "Cat A"

    # Manifest contents.
    manifest = ds.manifest
    assert manifest["schema_version"] == 2
    assert len(manifest["snapshot_ids"]) == 4
    assert manifest["catalog_size"] == 4
    assert manifest["n_users"] == 2
    assert "splits_file_sha256" in manifest
    assert manifest["split_pair_counts"] == {"train": 3, "val": 1, "test": 2}


TABLES = (FIVE_CORE, USER_STATS, ITEM_FEATURES, POPULARITY)


def _extract_kwargs():
    return {
        "five_core_table": FIVE_CORE,
        "user_stats_table": USER_STATS,
        "item_features_table": ITEM_FEATURES,
        "popularity_table": POPULARITY,
    }


def test_pinned_extract_time_travels_to_the_recorded_snapshot(spark, toy_gold, tmp_path):
    """Phase 5, T18: a pinned extract must read the snapshot the RECORD names, not
    whatever the table holds now.

    Snapshot A is captured, the 5-core table then gains a TRAIN row (snapshot B).
    A live read/extract sees B; a pinned read/extract at A still sees A — and the
    pinned cache is keyed by, and stamped with, A's IDs.
    """
    snap_a = {t: _snapshot_id(spark, t) for t in TABLES}
    live_a = extract(spark, out=tmp_path / "a", **_extract_kwargs())
    assert live_a["split_pair_counts"] == {"train": 3, "val": 1, "test": 2}

    # --- append one TRAIN row for an existing user/item -> snapshot B ---
    train_ts = toy_gold.train_end - 10 * DAY
    spark.createDataFrame([("U1", "P4", train_ts, 3.0)], FIVE_CORE_DDL).writeTo(
        FIVE_CORE
    ).append()
    snap_b = _snapshot_id(spark, FIVE_CORE)
    assert snap_b != snap_a[FIVE_CORE]

    # --- the read helper itself: pinned vs live ---
    assert _read_table(spark, FIVE_CORE, snap_a).count() == 6
    assert _read_table(spark, FIVE_CORE, None).count() == 7
    assert _read_table(spark, FIVE_CORE).count() == 7

    # A contract re-stamp after the fact must NOT leak into a pinned extract:
    # SHOW TBLPROPERTIES has no time-travel form, so identities come from the caller.
    _stamp_contract(spark, FIVE_CORE, "gold_interactions_5core", "9")

    pinned_contracts = {t: {"name": f"c_{i}", "version": "1"} for i, t in enumerate(TABLES)}
    pinned = extract(
        spark,
        out=tmp_path / "pinned",
        pinned_snapshot_ids=snap_a,
        pinned_contracts=pinned_contracts,
        **_extract_kwargs(),
    )
    assert pinned["status"] == "built"
    assert pinned["pinned"] is True
    cache_dir = Path(pinned["cache_dir"])
    # Keyed by the PINNED 5-core snapshot, not the live one.
    assert cache_dir.name == str(snap_a[FIVE_CORE])
    # Same content as the pre-append live extract.
    assert pinned["split_pair_counts"] == live_a["split_pair_counts"]

    manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
    assert manifest["snapshot_ids"] == snap_a
    assert manifest["contract_identities"] == pinned_contracts

    ds = load_dataset(cache_dir)
    assert ds.train_csr.nnz == 3  # U1->{P1,P2}, U2->{P1}; the appended U1->P4 is not here

    # --- a LIVE extract now sees snapshot B ---
    live_b = extract(spark, out=tmp_path / "b", **_extract_kwargs())
    assert live_b["split_pair_counts"]["train"] == 4
    assert Path(live_b["cache_dir"]).name == str(snap_b)


def test_pinned_extract_requires_complete_pins(spark, toy_gold, tmp_path):
    snap = {t: _snapshot_id(spark, t) for t in TABLES}
    contracts = {t: {"name": "c", "version": "1"} for t in TABLES}

    with pytest.raises(ValueError, match="requires pinned_contracts"):
        extract(
            spark,
            out=tmp_path / "p1",
            pinned_snapshot_ids=snap,
            **_extract_kwargs(),
        )
    with pytest.raises(ValueError, match="no snapshot id"):
        extract(
            spark,
            out=tmp_path / "p2",
            pinned_snapshot_ids={FIVE_CORE: snap[FIVE_CORE]},
            pinned_contracts=contracts,
            **_extract_kwargs(),
        )
    with pytest.raises(ValueError, match="no contract identity"):
        extract(
            spark,
            out=tmp_path / "p3",
            pinned_snapshot_ids=snap,
            pinned_contracts={FIVE_CORE: contracts[FIVE_CORE]},
            **_extract_kwargs(),
        )


class _SharedSession:
    """Proxy that forwards everything to the session-scoped Spark session but
    swallows ``stop()``.

    ``extract.main`` owns its session's lifecycle (``spark.stop()`` in a
    ``finally``); the fixture session is process-wide and shared with every other
    spark-marked test, so stopping it would tear down the JVM for the rest of the
    run. Recording the call instead lets the test assert main() did release it.
    """

    def __init__(self, session):
        self._session = session
        self.stopped = False

    def __getattr__(self, name):
        return getattr(self._session, name)

    def stop(self) -> None:
        self.stopped = True


def test_main_table_overrides_build_the_same_cache(spark, toy_gold, tmp_path, monkeypatch, capsys):
    """Phase 7, T-A2: the CLI must be able to point the extract at ANY four
    tables (the un-cored universe lives in *_uncored tables) and must hand
    ``spark.driver.maxResultSize`` to the session builder.

    get_spark is stubbed because maxResultSize is pre-JVM conf: a real call in
    this process returns the already-running fixture session and silently drops
    the setting, so what is verifiable here is that main() *requests* it — that
    ``get_spark`` then applies it to the builder is spark_session's contract.
    """
    captured: dict = {}
    proxy = _SharedSession(spark)

    def _fake_get_spark(**kwargs):
        captured.update(kwargs)
        return proxy

    monkeypatch.setattr("batch_recsys_lab.spark_session.get_spark", _fake_get_spark)

    out_dir = tmp_path / "cli_cache"
    rc = extract_mod.main(
        [
            "--warehouse", warehouse_of(spark),
            "--out", str(out_dir),
            "--master", "local[2]",
            "--driver-memory", "2g",
            "--max-result-size", "6g",
            "--five-core-table", FIVE_CORE,
            "--user-stats-table", USER_STATS,
            "--item-features-table", ITEM_FEATURES,
            "--popularity-table", POPULARITY,
        ]
    )
    assert rc == 0
    assert captured["extra_conf"] == {"spark.driver.maxResultSize": "6g"}
    assert captured["driver_memory"] == "2g"
    assert proxy.stopped, "main() must release the session it built"

    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["status"] == "built"
    assert summary["split_pair_counts"] == {"train": 3, "val": 1, "test": 2}

    # Same cache the in-process extract() call produces, asserted independently.
    cache_dir = Path(summary["cache_dir"])
    assert cache_dir.parent == out_dir
    ds = load_dataset(cache_dir)
    assert list(ds.item_ids) == ["P1", "P2", "P3", "P4"]
    assert list(ds.user_ids) == ["U1", "U2"]
    assert np.array_equal(np.load(cache_dir / "n_train.npy"), np.array([2, 1], dtype=np.int32))
    assert np.load(cache_dir / "n_train.npy").dtype == np.int32
    assert np.array_equal(np.load(cache_dir / "n_val.npy"), np.array([1, 0], dtype=np.int32))
    assert np.array_equal(np.load(cache_dir / "n_test.npy"), np.array([0, 2], dtype=np.int32))

    assert ds.train_csr.nnz == 3
    assert ds.train_csr.shape == (2, 4)
    assert list(ds.gt["val"].user_idx) == [0]
    assert list(ds.gt["val"].item_idx) == [2]
    test_gt = ds.gt["test"]
    assert list(test_gt.user_idx) == [1]
    assert set(test_gt.item_idx[test_gt.indptr[0]:test_gt.indptr[1]].tolist()) == {1, 3}

    pop_train_0 = ds.pop[("train_end", 0)]
    assert pop_train_0.dtype == np.float32
    assert list(pop_train_0) == [2.0, 1.0, 0.0, 0.0]
    assert list(ds.pop[("train_end", 365)]) == [2.0, 0.0, 0.0, 0.0]
    assert list(ds.pop[("val_end", 0)]) == [2.0, 0.0, 0.0, 0.0]
    assert list(ds.pop[("val_end", 365)]) == [0.0, 0.0, 1.0, 0.0]
    assert list(ds.pop[("train_end", 30)]) == [0.0, 0.0, 0.0, 0.0]  # no rows for this window

    assert ds.category_names[0] == "__unknown__"
    assert ds.item_category_codes.dtype == np.int32
    assert [ds.category_names[c] for c in ds.item_category_codes] == [
        "Cat A", "Cat B", "__unknown__", "Cat A",
    ]

    manifest = ds.manifest
    assert set(manifest["snapshot_ids"]) == set(TABLES)
    assert manifest["catalog_size"] == 4
    assert manifest["n_users"] == 2
    assert manifest["split_pair_counts"] == {"train": 3, "val": 1, "test": 2}


def test_extract_idempotent(spark, toy_gold, tmp_path, capsys):
    out_dir = tmp_path / "cache"
    extract(
        spark,
        out=out_dir,
        five_core_table=FIVE_CORE,
        user_stats_table=USER_STATS,
        item_features_table=ITEM_FEATURES,
        popularity_table=POPULARITY,
    )
    capsys.readouterr()
    summary2 = extract(
        spark,
        out=out_dir,
        five_core_table=FIVE_CORE,
        user_stats_table=USER_STATS,
        item_features_table=ITEM_FEATURES,
        popularity_table=POPULARITY,
    )
    assert summary2["status"] == "up_to_date"
    captured = capsys.readouterr()
    assert "cache up to date" in captured.out


# --- schema-aware item-feature projection (no main_category, e.g. ML-32M) ------

ITEM_FEATURES_NO_MAIN_CATEGORY_DDL = "parent_asin string, title string, genres string"
ITEM_FEATURES_NO_MAIN_CATEGORY = "local.gold.item_features_extract_no_category"


def test_build_item_categories_tolerates_missing_main_category(spark, toy_gold, tmp_path, capsys):
    """A table without ``main_category`` (the ML-32M ``item_features`` shape)
    must not raise an AnalysisException: the projection should select only the
    columns that exist, fall back every item to '__unknown__' (code 0), and
    report the missing column."""
    rows = [
        ("P1", "t1", "g1"),
        ("P2", "t2", "g2"),
        ("P3", "t3", "g3"),
        ("P4", "t4", "g4"),
    ]
    _write(spark, rows, ITEM_FEATURES_NO_MAIN_CATEGORY_DDL, ITEM_FEATURES_NO_MAIN_CATEGORY)
    _stamp_contract(spark, ITEM_FEATURES_NO_MAIN_CATEGORY, "gold_item_features_ml32m", "1")

    out_dir = tmp_path / "cache_direct"
    out_dir.mkdir()
    item_ids = ["P1", "P2", "P3", "P4"]
    capsys.readouterr()
    missing = extract_mod._build_item_categories(
        spark, ITEM_FEATURES_NO_MAIN_CATEGORY, item_ids, out_dir
    )
    captured = capsys.readouterr()

    assert missing == ["main_category"]
    assert "main_category" in captured.out

    codes = np.load(out_dir / "item_category_codes.npy")
    names = json.loads((out_dir / "item_category_names.json").read_text())
    assert names == ["__unknown__"]
    assert list(codes) == [0, 0, 0, 0]


def test_extract_end_to_end_with_missing_item_feature_column(spark, toy_gold, tmp_path):
    """Full ``extract()`` run against a table without ``main_category`` must
    succeed and stamp the missing-column list on the cache manifest, while
    leaving the manifest schema Amazon-compatible otherwise."""
    rows = [
        ("P1", "t1", "g1"),
        ("P2", "t2", "g2"),
        ("P3", "t3", "g3"),
        ("P4", "t4", "g4"),
    ]
    _write(spark, rows, ITEM_FEATURES_NO_MAIN_CATEGORY_DDL, ITEM_FEATURES_NO_MAIN_CATEGORY)
    _stamp_contract(spark, ITEM_FEATURES_NO_MAIN_CATEGORY, "gold_item_features_ml32m", "1")

    out_dir = tmp_path / "cache"
    summary = extract(
        spark,
        out=out_dir,
        five_core_table=FIVE_CORE,
        user_stats_table=USER_STATS,
        item_features_table=ITEM_FEATURES_NO_MAIN_CATEGORY,
        popularity_table=POPULARITY,
    )
    assert summary["status"] == "built"

    manifest = json.loads((Path(summary["cache_dir"]) / "cache_manifest.json").read_text())
    assert manifest["item_features_missing_columns"] == ["main_category"]

    codes = np.load(Path(summary["cache_dir"]) / "item_category_codes.npy")
    names = json.loads((Path(summary["cache_dir"]) / "item_category_names.json").read_text())
    assert names == ["__unknown__"]
    assert list(codes) == [0, 0, 0, 0]
