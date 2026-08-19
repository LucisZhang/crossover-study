"""ML-32M gold builds (Phase 9, T9-3a; UPGRADE_PLAN.md §8c).

Mirror of the Amazon gold stage, reusing the shared code wherever the shared code
is already generic over ``(user_id, parent_asin, ts)``:

* ``interactions_5core`` — the SAME iterative k-core prune
  (``features/kcore.py:run_kcore``, k=5, mirror design per §8b T8-4), with the
  ML-32M column projection passed in; the per-iteration funnel goes to
  ``local.dq_ml32m.kcore_funnel``.
* ``user_stats`` — the SAME ``features/gold.py:build_user_stats``, pointed at the
  ML-32M 5-core table and ``configs/splits_ml32m.yaml``.
* ``popularity`` — the SAME ``features/gold.py:build_popularity`` (same
  ``{train_end, val_end} × {30, 90, 365, 0}`` grid, same leak-free edges).
* ``item_features`` — built HERE, not reused: the Amazon projection selects
  brand/price/category columns that movies.csv does not have. The join-loss
  measure (5-core items with no ``silver_ml32m.items`` row) is reproduced verbatim
  in spirit, because that loss is exactly the edge the churn statistic's
  catalog join lives on.

**Not built (deliberate, disclosed):** ``gold.item_text``. That builder
(``features/item_text.py``) reads ``bronze.items`` description/features/store
fields that have no ML-32M counterpart, so reusing it would mean rewriting it, not
reusing it. The T9-3a churn statistic needs only ``interactions_5core`` +
``item_train_stats``; a title+genres text table is T9-3b's problem (the content
arm), not the data stage's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from batch_recsys_lab.contracts import load_contract, write_dq_results
from batch_recsys_lab.contracts.engine import DqResult, _resolve_run_id
from batch_recsys_lab.features.gold import (
    _ensure_namespace,
    build_popularity,
    build_user_stats,
)
from batch_recsys_lab.features.kcore import DEFAULT_K, build_summary, run_kcore
from batch_recsys_lab.features.silver_ml32m import DQ_TABLE, SILVER_INTERACTIONS
from batch_recsys_lab.features.splits import SplitConfig, load_splits
from batch_recsys_lab.spark_session import get_spark

# --- Locations ---------------------------------------------------------------
CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts" / "ml32m"
ITEM_FEATURES_CONTRACT = CONTRACTS_DIR / "gold_ml32m_item_features.yaml"
SPLITS_PATH = Path(__file__).resolve().parents[3] / "configs" / "splits_ml32m.yaml"

FIVE_CORE = "local.gold_ml32m.interactions_5core"
FUNNEL_TABLE = "local.dq_ml32m.kcore_funnel"
SILVER_ITEMS = "local.silver_ml32m.items"
USER_STATS = "local.gold_ml32m.user_stats"
ITEM_FEATURES = "local.gold_ml32m.item_features"
POPULARITY = "local.gold_ml32m.popularity"

# The k-core edge plus the ML-32M payload (no asin / helpful_vote /
# verified_purchase — those are Amazon review fields with no counterpart).
PROJECTION = ("user_id", "parent_asin", "ts", "rating")

ITEM_FEATURE_COLS = ["parent_asin", "title", "genres"]


# --- interactions_5core -------------------------------------------------------


def build_five_core(
    spark: SparkSession,
    source_table: str = SILVER_INTERACTIONS,
    target_table: str = FIVE_CORE,
    funnel_table: str = FUNNEL_TABLE,
    k: int = DEFAULT_K,
    run_id: str | None = None,
) -> dict:
    """Iterative k-core prune of the ML-32M silver interactions (shared engine)."""
    funnel = run_kcore(
        spark,
        source_table=source_table,
        target_table=target_table,
        funnel_table=funnel_table,
        k=k,
        run_id=run_id,
        projection=PROJECTION,
    )
    return build_summary(funnel, source_table, target_table, k)


# --- item_features ------------------------------------------------------------


def build_item_features(
    spark: SparkSession,
    five_core_table: str = FIVE_CORE,
    silver_items_table: str = SILVER_ITEMS,
    out_table: str = ITEM_FEATURES,
    run_id: str | None = None,
    contract_path: str | Path = ITEM_FEATURES_CONTRACT,
    dq_table: str = DQ_TABLE,
) -> dict:
    """Build ``local.gold_ml32m.item_features`` (silver items ∩ 5-core catalog).

    Measures the join-loss (distinct 5-core items with no ``silver_ml32m.items``
    row) into ``dq_ml32m.dq_results`` as ``gold_ml32m_item_features_join_loss``.
    Items are never dropped from the interaction table, only absent from the
    feature table — and this catalog IS the universe the churn statistic's TEST
    ground truth is joined against, so the loss is published, not assumed zero.
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
                check_id="gold_ml32m_item_features_join_loss",
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
        dq_table,
    )
    return {
        "table": "ml32m_item_features",
        "run_id": rid,
        "rows": spark.table(out_table).count(),
        "catalog_items": int(catalog_count),
        "join_loss_items": int(orphans),
        "join_loss_share": float(share),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }


# --- orchestration -----------------------------------------------------------


def build_gold_features(
    spark: SparkSession,
    splits: SplitConfig | None = None,
    run_id: str | None = None,
    five_core_table: str = FIVE_CORE,
    silver_items_table: str = SILVER_ITEMS,
    user_stats_table: str = USER_STATS,
    item_features_table: str = ITEM_FEATURES,
    popularity_table: str = POPULARITY,
    splits_path: str | Path = SPLITS_PATH,
) -> dict:
    """Build the three ML-32M gold feature tables; return a combined summary."""
    if splits is None:
        splits = load_splits(splits_path)
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

_STAGES = ("core", "features")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.features.gold_ml32m")
    parser.add_argument("--stage", required=True, choices=_STAGES)
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--run-id", default=None)
    # Table / config overrides (tests point these at toy tables).
    parser.add_argument("--source-table", default=SILVER_INTERACTIONS)
    parser.add_argument("--five-core-table", default=FIVE_CORE)
    parser.add_argument("--funnel-table", default=FUNNEL_TABLE)
    parser.add_argument("--silver-items-table", default=SILVER_ITEMS)
    parser.add_argument("--user-stats-table", default=USER_STATS)
    parser.add_argument("--item-features-table", default=ITEM_FEATURES)
    parser.add_argument("--popularity-table", default=POPULARITY)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--splits-path", default=str(SPLITS_PATH))
    args = parser.parse_args(argv)

    spark = get_spark(
        app_name=f"gold-ml32m-{args.stage}",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        if args.stage == "core":
            summary = build_five_core(
                spark,
                source_table=args.source_table,
                target_table=args.five_core_table,
                funnel_table=args.funnel_table,
                k=args.k,
                run_id=args.run_id,
            )
        else:
            summary = build_gold_features(
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
