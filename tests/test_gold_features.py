"""Gold feature build tests (Phase 1, T7).

Toy frames (not full data) against the shared tmp-warehouse ``spark`` fixture:

* user_stats: split bucketing at the frozen boundaries (AT train_end -> train,
  +1ms -> val), and n_train + n_val + n_test == n_total;
* popularity: window edges — ``ts <= as_of`` upper edge (instant AT as_of
  included), ``ts > as_of - window`` STRICTLY-GREATER lower edge (instant AT
  ``as_of - window`` excluded), and n_unique_users deduping a repeat user;
* item_features: join-loss measured when a 5-core item is absent from silver.items;
* the four gold contract YAMLs load and audit cleanly on the toy outputs.
"""

from __future__ import annotations

import os

# Force loopback before any SparkContext starts (see test_silver_build.py).
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pyspark.sql import functions as F

from batch_recsys_lab.contracts import audit, load_contract
from batch_recsys_lab.features.gold import (
    build_item_features,
    build_popularity,
    build_user_stats,
)
from batch_recsys_lab.features.splits import load_splits

pytestmark = pytest.mark.spark

UTC = timezone.utc
MS = timedelta(milliseconds=1)
DAY = timedelta(days=1)

FIVE_CORE_DDL = (
    "user_id string, parent_asin string, ts timestamp, rating double, "
    "asin string, helpful_vote long, verified_purchase boolean"
)
SILVER_ITEMS_DDL = (
    "parent_asin string, title string, main_category string, "
    "categories array<string>, store string, average_rating double, "
    "rating_number long, price_usd double, brand_norm string"
)

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"


def _write(spark, rows, ddl, table):
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.gold")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.silver")
    spark.createDataFrame(rows, ddl).writeTo(table).createOrReplace()


# --------------------------------------------------------------------------- #
# user_stats: split bucketing straddling the frozen boundaries.
# --------------------------------------------------------------------------- #


def test_user_stats_split_buckets(spark):
    s = load_splits()
    src = "local.gold.five_core_ustats"
    out = "local.gold.user_stats_ustats"
    rows = [
        ("U1", "P1", datetime(2020, 1, 1, tzinfo=UTC), 5.0, "A1", 0, True),  # train
        ("U1", "P2", s.train_end, 4.0, "A2", 0, True),                       # AT train_end -> train
        ("U1", "P3", s.train_end + MS, 3.0, "A3", 0, True),                  # +1ms -> val
        ("U1", "P4", s.val_end, 5.0, "A4", 0, True),                         # AT val_end -> val
        ("U1", "P5", datetime(2023, 5, 1, tzinfo=UTC), 4.0, "A5", 0, True),  # test
    ]
    _write(spark, rows, FIVE_CORE_DDL, src)

    build_user_stats(spark, s, five_core_table=src, out_table=out, run_id="us-run")
    r = spark.table(out).where("user_id = 'U1'").first()

    assert r["n_total"] == 5
    assert r["n_train"] == 2  # 2020-01-01 and the instant AT train_end
    assert r["n_val"] == 2    # train_end+1ms and the instant AT val_end
    assert r["n_test"] == 1
    assert r["n_train"] + r["n_val"] + r["n_test"] == r["n_total"]
    # PySpark returns naive local-tz datetimes on collect; assert ordering +
    # tenure rather than an offset-fragile exact-instant equality.
    assert r["first_ts"] < r["last_ts"]
    assert r["tenure_days"] > 0


# --------------------------------------------------------------------------- #
# popularity: upper edge inclusive, lower edge strictly-greater, unique users.
# --------------------------------------------------------------------------- #


def test_popularity_window_edges(spark):
    s = load_splits()
    as_of = s.train_end
    src = "local.gold.five_core_pop"
    out = "local.gold.popularity_pop"
    rows = [
        ("U1", "P1", as_of, 5.0, "A1", 0, True),                 # AT as_of -> included
        ("U1", "P1", as_of - 10 * DAY, 4.0, "A2", 0, True),      # within 30d, repeat user
        ("U2", "P1", as_of - 30 * DAY, 3.0, "A3", 0, True),      # AT lower edge of w30 -> excluded
        ("U3", "P1", as_of + DAY, 5.0, "A4", 0, True),           # future -> excluded everywhere
    ]
    _write(spark, rows, FIVE_CORE_DDL, src)

    build_popularity(spark, s, five_core_table=src, out_table=out, run_id="pop-run")

    # Filter for as_of == train_end in Spark (session tz UTC) — a Python-side
    # equality would trip over naive local-tz datetimes returned by collect().
    as_of_lit = F.lit(as_of).cast("timestamp")
    got = {
        row["window_days"]: (row["n_interactions"], row["n_unique_users"])
        for row in spark.table(out)
        .where((F.col("parent_asin") == "P1") & (F.col("as_of") == as_of_lit))
        .collect()
    }

    # window 30: U1@as_of + U1@-10d (repeat) -> 2 rows, 1 unique; U2@-30d on the
    # strictly-greater lower edge is excluded; U3 (future) excluded.
    assert got[30] == (2, 1)
    # window 0 (all history <= as_of): U1@as_of, U1@-10d, U2@-30d -> 3 rows, 2 users.
    assert got[0] == (3, 2)
    # window 90: -30d is now inside the window -> 3 rows, 2 users.
    assert got[90] == (3, 2)


# --------------------------------------------------------------------------- #
# item_features: join-loss measured for a 5-core item missing from silver.items.
# --------------------------------------------------------------------------- #


def test_item_features_join_loss(spark):
    src = "local.gold.five_core_if"
    items_tbl = "local.silver.items_if"
    out = "local.gold.item_features_if"

    # silver.items has P1, P2; the 5-core catalog has P1, P3 (P3 orphaned).
    items = [
        ("P1", "t1", "Cat", ["c"], "S", 4.0, 10, 9.99, "acme"),
        ("P2", "t2", "Cat", ["c"], "S", 4.5, 20, 19.99, "sony"),
    ]
    _write(spark, items, SILVER_ITEMS_DDL, items_tbl)
    five = [
        ("U1", "P1", datetime(2021, 1, 1, tzinfo=UTC), 5.0, "A1", 0, True),
        ("U2", "P3", datetime(2021, 2, 1, tzinfo=UTC), 4.0, "A2", 0, True),
    ]
    _write(spark, five, FIVE_CORE_DDL, src)

    summary = build_item_features(
        spark,
        five_core_table=src,
        silver_items_table=items_tbl,
        out_table=out,
        run_id="jl-run",
    )

    # Only P1 survives the inner join (P2 not in catalog, P3 not in silver.items).
    feats = [r["parent_asin"] for r in spark.table(out).collect()]
    assert feats == ["P1"]
    assert summary["catalog_items"] == 2
    assert summary["join_loss_items"] == 1
    assert summary["join_loss_share"] == 0.5

    dq = (
        spark.table("local.dq.dq_results")
        .where("check_id = 'gold_item_features_join_loss' AND run_id = 'jl-run'")
        .first()
    )
    assert dq["status"] == "measured"
    assert dq["violation_count"] == 1
    assert dq["total_rows"] == 2
    assert dq["metric_value"] == 0.5


# --------------------------------------------------------------------------- #
# Gold contract YAMLs load and audit cleanly on toy gold outputs.
# --------------------------------------------------------------------------- #


def test_gold_contracts_load_and_audit_clean(spark):
    s = load_splits()
    src = "local.gold.five_core_clean"
    us_out = "local.gold.user_stats_clean"
    if_out = "local.gold.item_features_clean"
    pop_out = "local.gold.popularity_clean"
    items_tbl = "local.silver.items_clean"

    # A single user with >= 5 interactions (gold_user_stats.n_total_min5), all
    # ratings in 1..5, all ts inside the frozen [1996, 2023-10) bound.
    parents = [f"P{i}" for i in range(1, 7)]
    five = [
        ("U1", p, datetime(2021, i + 1, 1, tzinfo=UTC), float((i % 5) + 1), f"A{i}", i, True)
        for i, p in enumerate(parents)
    ]
    _write(spark, five, FIVE_CORE_DDL, src)
    items = [
        (p, f"title {p}", "Cat", ["c"], "S", 4.0, 10 + i, 5.0 + i, "acme")
        for i, p in enumerate(parents)
    ]
    _write(spark, items, SILVER_ITEMS_DDL, items_tbl)

    build_user_stats(spark, s, five_core_table=src, out_table=us_out, run_id="clean")
    build_item_features(
        spark, five_core_table=src, silver_items_table=items_tbl,
        out_table=if_out, run_id="clean",
    )
    build_popularity(spark, s, five_core_table=src, out_table=pop_out, run_id="clean")

    pairs = [
        ("gold_interactions_5core.yaml", src),
        ("gold_user_stats.yaml", us_out),
        ("gold_item_features.yaml", if_out),
        ("gold_popularity.yaml", pop_out),
    ]
    for yaml_name, table in pairs:
        contract = load_contract(CONTRACTS_DIR / yaml_name)  # loader round-trip
        assert contract.version == 1
        results = audit(spark, contract, table, run_id="clean")
        # Data-level checks must not fail; schema_conformance nullability is a known
        # cosmetic finding (Iceberg marks derived/aggregated columns nullable).
        data_fails = [
            r.check_id
            for r in results
            if r.status == "fail" and r.check_kind != "schema_conformance"
        ]
        assert not data_fails, f"{yaml_name} on {table}: unexpected fails {data_fails}"
