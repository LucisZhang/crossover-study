"""Gold feature builds (Phase 1, T7; docs/engineering-log/UPGRADE_PLAN.md §8).

Three projections off the frozen 5-core interaction table
(``local.gold.interactions_5core``, produced by ``features.kcore``):

* :func:`build_user_stats` — one row per user: interaction counts bucketed by the
  frozen temporal splits (``features.splits``), first/last timestamp, tenure.
* :func:`build_item_features` — ``silver.items`` inner-joined to the *distinct*
  ``parent_asin`` catalog of the 5-core table. The join-loss (5-core items with no
  ``silver.items`` row) is *measured* into ``dq.dq_results`` — items are not
  dropped from the interaction table, only absent from the feature table.
* :func:`build_popularity` — the ``as_of × window_days`` popularity grid, computed
  with a **leak-free** upper edge (``ts <= as_of``) and a **strictly-greater**
  lower edge (``ts > as_of - window_days``): the instant *at* ``as_of`` is included,
  the instant *at* ``as_of - window_days`` is excluded. ``window_days == 0`` means
  all history ``ts <= as_of``. Written partitioned by ``(as_of, window_days)``.

These are feature projections, not row-accounting stages: only
``interactions_5core`` participates in the raw→gold waterfall (handled by
``kcore.py``). Hence *no* ``build_summary.jsonl`` line is written here — the CLI
just prints one JSON summary as its last stdout line, per repo convention.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from batch_recsys_lab.contracts import load_contract, write_dq_results
from batch_recsys_lab.contracts.engine import DqResult, _resolve_run_id
from batch_recsys_lab.features.splits import SplitConfig, load_splits
from batch_recsys_lab.features.splits import TEST, TRAIN, VAL
from batch_recsys_lab.spark_session import get_spark

# --- Locations ---------------------------------------------------------------
CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts"
ITEM_FEATURES_CONTRACT = CONTRACTS_DIR / "gold_item_features.yaml"

FIVE_CORE = "local.gold.interactions_5core"
SILVER_ITEMS = "local.silver.items"
USER_STATS = "local.gold.user_stats"
ITEM_FEATURES = "local.gold.item_features"
POPULARITY = "local.gold.popularity"

# as_of ranges over the frozen train/val boundaries (never the test boundary — a
# popularity feature computed at test end would leak the test period into ranking).
POPULARITY_WINDOWS = [30, 90, 365, 0]  # 0 = all history <= as_of

ITEM_FEATURE_COLS = [
    "parent_asin",
    "title",
    "brand_norm",
    "price_usd",
    "main_category",
    "categories",
    "average_rating",
    "rating_number",
]


def _ensure_namespace(spark: SparkSession, table: str) -> None:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {table.rsplit('.', 1)[0]}")


# --- user_stats --------------------------------------------------------------


def build_user_stats(
    spark: SparkSession,
    splits: SplitConfig,
    five_core_table: str = FIVE_CORE,
    out_table: str = USER_STATS,
    run_id: str | None = None,
) -> dict:
    """Build ``local.gold.user_stats`` from the 5-core interaction table."""
    start = time.perf_counter()
    rid, _ = _resolve_run_id(run_id)
    df = spark.table(five_core_table)
    label = splits.split_label(F.col("ts"))

    agg = (
        df.groupBy("user_id")
        .agg(
            F.count(F.lit(1)).cast("long").alias("n_total"),
            F.coalesce(F.sum((label == F.lit(TRAIN)).cast("long")), F.lit(0))
            .cast("long")
            .alias("n_train"),
            F.coalesce(F.sum((label == F.lit(VAL)).cast("long")), F.lit(0))
            .cast("long")
            .alias("n_val"),
            F.coalesce(F.sum((label == F.lit(TEST)).cast("long")), F.lit(0))
            .cast("long")
            .alias("n_test"),
            F.min("ts").alias("first_ts"),
            F.max("ts").alias("last_ts"),
        )
        .withColumn("tenure_days", F.datediff(F.col("last_ts"), F.col("first_ts")))
        .select(
            "user_id",
            "n_total",
            "n_train",
            "n_val",
            "n_test",
            "first_ts",
            "last_ts",
            "tenure_days",
        )
    )

    _ensure_namespace(spark, out_table)
    agg.writeTo(out_table).createOrReplace()
    return {
        "table": "user_stats",
        "run_id": rid,
        "rows": spark.table(out_table).count(),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }


# --- item_features -----------------------------------------------------------


def build_item_features(
    spark: SparkSession,
    five_core_table: str = FIVE_CORE,
    silver_items_table: str = SILVER_ITEMS,
    out_table: str = ITEM_FEATURES,
    run_id: str | None = None,
    contract_path: str | Path = ITEM_FEATURES_CONTRACT,
) -> dict:
    """Build ``local.gold.item_features`` (silver.items ∩ 5-core catalog).

    Measures the join-loss (distinct 5-core items with no ``silver.items`` row) into
    ``dq.dq_results`` as ``gold_item_features_join_loss`` (status ``measured``,
    metric = orphan share of the 5-core catalog).
    """
    start = time.perf_counter()
    rid, rts = _resolve_run_id(run_id)
    contract = load_contract(contract_path)

    items = spark.table(silver_items_table)
    catalog = spark.table(five_core_table).select("parent_asin").distinct()
    catalog_count = catalog.count()

    joined = items.join(catalog, "parent_asin", "inner").select(*ITEM_FEATURE_COLS)

    orphans = (
        catalog.join(
            items.select("parent_asin").distinct(), "parent_asin", "left_anti"
        ).count()
    )
    share = (orphans / catalog_count) if catalog_count else 0.0

    _ensure_namespace(spark, out_table)
    joined.writeTo(out_table).createOrReplace()

    write_dq_results(
        spark,
        [
            DqResult(
                run_id=rid,
                run_ts=rts,
                table_name=out_table,
                contract_name=contract.name,
                contract_version=contract.version,
                check_id="gold_item_features_join_loss",
                check_kind="measure",
                column="parent_asin",
                status="measured",
                violation_count=int(orphans),
                total_rows=int(catalog_count),
                metric_value=float(share),
                details=json.dumps(
                    {"orphan_items": int(orphans), "catalog_items": int(catalog_count)}
                ),
            )
        ],
    )
    return {
        "table": "item_features",
        "run_id": rid,
        "rows": spark.table(out_table).count(),
        "catalog_items": int(catalog_count),
        "join_loss_items": int(orphans),
        "join_loss_share": float(share),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }


# --- popularity --------------------------------------------------------------


def build_popularity(
    spark: SparkSession,
    splits: SplitConfig,
    five_core_table: str = FIVE_CORE,
    out_table: str = POPULARITY,
    run_id: str | None = None,
    windows: list[int] | None = None,
) -> dict:
    """Build ``local.gold.popularity`` over ``{train_end, val_end} × windows``.

    Window semantics (pinned): a row is in window ``w`` at ``as_of`` iff
    ``ts <= as_of`` AND (``w == 0`` OR ``ts > as_of - w days``). The lower edge is
    STRICTLY greater — the instant exactly ``w`` days before ``as_of`` is excluded;
    the instant exactly at ``as_of`` is included.
    """
    start = time.perf_counter()
    rid, _ = _resolve_run_id(run_id)
    windows = windows if windows is not None else POPULARITY_WINDOWS

    df = spark.table(five_core_table).select("user_id", "parent_asin", "ts")
    parts: list[DataFrame] = []
    for as_of in (splits.train_end, splits.val_end):
        as_of_lit = F.lit(as_of).cast("timestamp")
        base = df.where(F.col("ts") <= as_of_lit)  # leak-free upper edge
        for w in windows:
            sub = base
            if w != 0:
                lower = F.lit(as_of - timedelta(days=w)).cast("timestamp")
                sub = base.where(F.col("ts") > lower)  # strictly-greater lower edge
            parts.append(
                sub.groupBy("parent_asin")
                .agg(
                    F.count(F.lit(1)).cast("long").alias("n_interactions"),
                    F.countDistinct("user_id").cast("long").alias("n_unique_users"),
                )
                .select(
                    as_of_lit.alias("as_of"),
                    F.lit(w).cast("int").alias("window_days"),
                    "parent_asin",
                    "n_interactions",
                    "n_unique_users",
                )
            )

    out = reduce(DataFrame.unionByName, parts)
    _ensure_namespace(spark, out_table)
    # Iceberg identity partitioning on the 2-value as_of and 4-value window_days
    # grid — straightforward and keeps per-(as_of, window) reads pruned.
    out.writeTo(out_table).partitionedBy(
        F.col("as_of"), F.col("window_days")
    ).createOrReplace()
    return {
        "table": "popularity",
        "run_id": rid,
        "rows": spark.table(out_table).count(),
        "windows": windows,
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }


# --- orchestration -----------------------------------------------------------


def build_gold(
    spark: SparkSession,
    splits: SplitConfig | None = None,
    run_id: str | None = None,
    five_core_table: str = FIVE_CORE,
    silver_items_table: str = SILVER_ITEMS,
    user_stats_table: str = USER_STATS,
    item_features_table: str = ITEM_FEATURES,
    popularity_table: str = POPULARITY,
    splits_path: str | Path = None,
) -> dict:
    """Build all three gold feature tables; return a combined summary dict."""
    if splits is None:
        splits = load_splits(splits_path) if splits_path else load_splits()
    rid, _ = _resolve_run_id(run_id)
    user_stats = build_user_stats(spark, splits, five_core_table, user_stats_table, rid)
    item_features = build_item_features(
        spark, five_core_table, silver_items_table, item_features_table, rid
    )
    popularity = build_popularity(spark, splits, five_core_table, popularity_table, rid)
    return {
        "run_id": rid,
        "user_stats": user_stats,
        "item_features": item_features,
        "popularity": popularity,
    }


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.features.gold")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--run-id", default=None)
    # Table / config overrides (tests point these at toy tables).
    parser.add_argument("--five-core-table", default=FIVE_CORE)
    parser.add_argument("--silver-items-table", default=SILVER_ITEMS)
    parser.add_argument("--user-stats-table", default=USER_STATS)
    parser.add_argument("--item-features-table", default=ITEM_FEATURES)
    parser.add_argument("--popularity-table", default=POPULARITY)
    parser.add_argument("--splits-path", default=None)
    args = parser.parse_args(argv)

    spark = get_spark(
        app_name="gold-features",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        summary = build_gold(
            spark,
            run_id=args.run_id,
            five_core_table=args.five_core_table,
            silver_items_table=args.silver_items_table,
            user_stats_table=args.user_stats_table,
            item_features_table=args.item_features_table,
            popularity_table=args.popularity_table,
            splits_path=args.splits_path,
        )
    finally:
        spark.stop()

    # Summary JSON MUST be the last stdout line (repo convention).
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
