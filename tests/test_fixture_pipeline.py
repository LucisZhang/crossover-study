"""End-to-end fixture pipeline acceptance test (Phase 1, T8) — the CI gate.

Drives the WHOLE Phase-1 engine on the bundled bronze fixtures, twice, against the
shared tmp-warehouse ``spark`` fixture (``tests/conftest.py``):

    fixtures → bronze tables
      → silver (items, then interactions)               [features.silver]
      → 5-core prune + funnel                            [features.kcore, k=5]
      → gold user_stats / item_features / popularity     [features.gold]
      → gold item_text                                   [features.item_text]
      → contract audit over every contracts/*.yaml       [contracts.run_audit]
      → reconciliation waterfall (exact)                 [features.waterfall]
      → per-table content hashes                         [features.verify_determinism]

and asserts the four Phase-1 acceptance clauses on the fixture substrate:
    (1) the contract audit reports ZERO ``status == 'fail'`` (§8 accept #1);
    (2) the raw→bronze→silver→gold waterfall sums exactly and matches live counts
        (§8 accept #2);
    (3) intra-run gold consistency: Σ user_stats.n_total == count(5-core);
    (4) silver+gold content hashes are IDENTICAL across the two independent builds
        (§8 accept #4, determinism).

k and sparsity notes
--------------------
Runs at the PRODUCTION k=5 (not the k=2/3 fallback the T8 brief floats): the frozen
``gold_user_stats`` contract asserts ``n_total >= 5`` for every surviving user,
which is exactly the k=5 guarantee — running a smaller k would manufacture a
contract violation the audit would (correctly) fail on. At 50k-row fixture sparsity
the 5-core may be tiny or empty; that is fine and is what the audit's empty-table
handling (``no_all_null`` on 0 rows → pass) exists for. We assert *consistency*,
never a specific 5-core size.
"""

from __future__ import annotations

import os

# This host cannot bind Spark to its own hostname; force loopback before any
# SparkContext starts (mirrors test_waterfall.py / test_silver_build.py).
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import json
from pathlib import Path

import pytest
from pyspark.sql import functions as F

from batch_recsys_lab.contracts.run_audit import run_audit
from batch_recsys_lab.features.gold import build_gold
from batch_recsys_lab.features.item_text import build_item_text
from batch_recsys_lab.features.kcore import run_kcore
from batch_recsys_lab.features.silver import build_interactions, build_items
from batch_recsys_lab.features.verify_determinism import compute_hashes
from batch_recsys_lab.features.waterfall import run_waterfall

pytestmark = pytest.mark.spark

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ITEMS_PARQUET = FIXTURES / "bronze_items_fixture.parquet"
REVIEWS_PARQUET = FIXTURES / "bronze_reviews_50k.parquet"

BRONZE_REVIEWS = "local.bronze.reviews"
BRONZE_ITEMS = "local.bronze.items"
SILVER_INTERACTIONS = "local.silver.interactions"
SILVER_ITEMS = "local.silver.items"
GOLD_5CORE = "local.gold.interactions_5core"
USER_STATS = "local.gold.user_stats"
ITEM_FEATURES = "local.gold.item_features"
POPULARITY = "local.gold.popularity"
ITEM_TEXT = "local.gold.item_text"
FUNNEL_TABLE = "local.dq.kcore_funnel"
WATERFALL_TABLE = "local.dq.waterfall"
DQ_RESULTS = "local.dq.dq_results"

# The tables whose content must be reproducible across the two builds (bronze is
# reloaded identically each build, so it would match too, but the acceptance
# clause is about the silver+gold *derivations*).
SILVER_GOLD_TABLES = [
    SILVER_INTERACTIONS,
    SILVER_ITEMS,
    GOLD_5CORE,
    USER_STATS,
    ITEM_FEATURES,
    POPULARITY,
]


# --------------------------------------------------------------------------- #
# Bronze fixture loaders (adapted from test_waterfall.py: production bronze
# schema; kept local so this acceptance test is self-contained).
# --------------------------------------------------------------------------- #


def _load_bronze_items(spark):
    spark.conf.set("spark.sql.caseSensitive", "true")
    try:
        df = spark.read.parquet(str(ITEMS_PARQUET))
        proj = df.select(
            "parent_asin", "main_category", "title", "average_rating",
            "rating_number", "price", "store", "categories",
            "features", "description",
            F.map_from_arrays(
                F.array(F.lit("Brand"), F.lit("Manufacturer")),
                F.array(F.col("details")["Brand"], F.col("details")["Manufacturer"]),
            ).alias("details"),
        )
        spark.sql("CREATE NAMESPACE IF NOT EXISTS local.bronze")
        proj.writeTo(BRONZE_ITEMS).createOrReplace()
    finally:
        spark.conf.set("spark.sql.caseSensitive", "false")


def _load_bronze_reviews(spark):
    df = spark.read.parquet(str(REVIEWS_PARQUET))
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.bronze")
    df.writeTo(BRONZE_REVIEWS).createOrReplace()


def _write_ingest_summary(path: Path, reviews_n: int, items_n: int) -> None:
    """Synthesize the raw→bronze feed (no ingest job runs here): written == the
    loaded bronze count, corrupt == 0 (the invariant a real ingest guarantees)."""
    lines = [
        {"table": BRONZE_REVIEWS, "table_name": "reviews", "total_parsed": reviews_n,
         "corrupt": 0, "written": reviews_n, "wall_clock_s": 1.0,
         "ingested_at": "2026-08-05T00:00:00+00:00"},
        {"table": BRONZE_ITEMS, "table_name": "items", "total_parsed": items_n,
         "corrupt": 0, "written": items_n, "wall_clock_s": 1.0,
         "ingested_at": "2026-08-05T00:00:00+00:00"},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")


# --------------------------------------------------------------------------- #
# One full build: bronze → silver → gold. Returns the tmp feed paths the
# waterfall consumes. Every table is createOrReplace, so a second call is an
# independent rebuild from the (re-loaded) bronze fixtures.
# --------------------------------------------------------------------------- #


def _build(spark, tmp: Path, run_id: str) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    build_summary = tmp / "build_summary.jsonl"
    ingest_summary = tmp / "ingest_summary.jsonl"

    _load_bronze_items(spark)
    _load_bronze_reviews(spark)

    build_items(spark, run_id=run_id, summary_path=str(build_summary), write_summary=True)
    build_interactions(
        spark, run_id=run_id, summary_path=str(build_summary), write_summary=True
    )

    reviews_n = spark.table(BRONZE_REVIEWS).count()
    items_n = spark.table(BRONZE_ITEMS).count()
    _write_ingest_summary(ingest_summary, reviews_n, items_n)

    # Production k=5 (see module docstring).
    run_kcore(
        spark,
        source_table=SILVER_INTERACTIONS,
        target_table=GOLD_5CORE,
        funnel_table=FUNNEL_TABLE,
        k=5,
        run_id=run_id,
    )
    build_gold(spark, run_id=run_id)
    build_item_text(spark, run_id=run_id)

    return {"build_summary": build_summary, "ingest_summary": ingest_summary}


def _assert_waterfall_exact(result: dict) -> None:
    for dataset, entry in result["datasets"].items():
        for e in entry["edges"]:
            assert e["sum_ok"], f"{dataset} edge {e['stage_from']}->{e['stage_to']} sum mismatch"
            assert e["count_ok"], f"{dataset} edge {e['stage_from']}->{e['stage_to']} live-count mismatch"
            assert e["reason_sum"] == e["source_rows"]
            assert e["kept_rows"] == e["target_count"]


def _run_and_assert(spark, tmp: Path, run_id: str) -> dict[str, dict]:
    feeds = _build(spark, tmp, run_id)

    # (1) audit: every contract passes on the fixture (zero hard fails).
    audit_summary = run_audit(spark, run_id=run_id, dq_table=DQ_RESULTS)
    assert not audit_summary["any_fail"], (
        "contract audit reported a hard fail on the fixture: "
        + json.dumps(audit_summary["tables"], indent=2)
    )
    # All seven contracts were audited (nothing silently skipped).
    assert len(audit_summary["tables"]) == 7
    assert audit_summary["skipped_tables"] == []

    # (2) waterfall reconciles exactly against live Iceberg counts.
    result = run_waterfall(
        spark,
        run_id=run_id,
        ingest_summary_path=str(feeds["ingest_summary"]),
        build_summary_path=str(feeds["build_summary"]),
        manifest_path=str(tmp / "MANIFEST.md"),
        json_out=str(tmp / "waterfall.json"),
    )
    _assert_waterfall_exact(result)

    # (3) gold consistency: every 5-core row is bucketed into exactly one user.
    n_5core = spark.table(GOLD_5CORE).count()
    n_bucketed = spark.table(USER_STATS).agg(F.sum("n_total")).first()[0] or 0
    assert int(n_bucketed) == int(n_5core)

    return compute_hashes(spark, SILVER_GOLD_TABLES)


def test_fixture_pipeline_end_to_end(spark, tmp_path):
    hashes1 = _run_and_assert(spark, tmp_path / "run1", "fixture-run-1")
    hashes2 = _run_and_assert(spark, tmp_path / "run2", "fixture-run-2")

    # (4) determinism: content-identical silver+gold across two independent builds.
    assert set(hashes1) == set(SILVER_GOLD_TABLES)
    assert hashes1 == hashes2, (
        "non-deterministic rebuild:\n"
        + "\n".join(
            f"  {t}: run1={hashes1[t]} run2={hashes2[t]}"
            for t in SILVER_GOLD_TABLES
            if hashes1.get(t) != hashes2.get(t)
        )
    )
