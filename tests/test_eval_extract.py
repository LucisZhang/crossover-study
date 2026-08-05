"""Eval cache extract + EvalDataset round-trip tests (Phase 2, T1).

Synthetic gold tables (distinct table names, passed into the extract functions —
never ``local.gold.*`` hardcoded) spanning TRAIN/VAL/TEST around the frozen
split boundaries in ``configs/splits.yaml``.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from batch_recsys_lab.eval.extract import extract
from batch_recsys_lab.eval.dataset import load_dataset
from batch_recsys_lab.features.splits import load_splits

pytestmark = pytest.mark.spark

UTC = timezone.utc
DAY = timedelta(days=1)

FIVE_CORE_DDL = "user_id string, parent_asin string, ts timestamp"
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
        ("U1", "P1", train_ts),
        ("U1", "P2", train_ts),
        ("U1", "P3", val_ts),   # U1's VAL GT
        ("U2", "P1", train_ts),
        ("U2", "P4", test_ts),  # U2's TEST GT
        ("U2", "P2", test_ts),  # U2's second TEST GT item
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
    assert manifest["schema_version"] == 1
    assert len(manifest["snapshot_ids"]) == 4
    assert manifest["catalog_size"] == 4
    assert manifest["n_users"] == 2
    assert "splits_file_sha256" in manifest
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
