"""Reconciliation waterfall tests (Phase 1, T4b).

Drives the REAL pipeline on the bundled bronze fixtures against the shared
tmp-warehouse ``spark`` fixture (``tests/conftest.py``):

    fixtures → tmp bronze tables
      → build_items / build_interactions (features.silver, summaries → tmp jsonl)
      → run_kcore (features.kcore, gold + dq.kcore_funnel)
      → run_waterfall (features.waterfall)

and asserts the four things the task requires:
    (a) every edge sums exactly AND its `kept` matches the live table count;
    (b) the MANIFEST section is idempotent (run twice → one section);
    (c) waterfall.json has the documented structure;
    (d) a tampered build_summary count makes the waterfall raise, naming the edge.

k=5 at fixture sparsity may yield a small/empty 5-core — that is fine; we assert
*consistency* (sums + live counts), never a specific size.
"""

from __future__ import annotations

import os

# This host cannot bind Spark to its own hostname; force loopback before any
# SparkContext starts (see test_silver_build.py).
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import json
from pathlib import Path

import pytest
from pyspark.sql import functions as F

from batch_recsys_lab.features.kcore import run_kcore
from batch_recsys_lab.features.silver import build_interactions, build_items
from batch_recsys_lab.features.waterfall import (
    SECTION_HEADER,
    WaterfallError,
    run_waterfall,
)

pytestmark = pytest.mark.spark

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ITEMS_PARQUET = FIXTURES / "bronze_items_fixture.parquet"
REVIEWS_PARQUET = FIXTURES / "bronze_reviews_50k.parquet"

RUN_ID = "wf-test-run"

BRONZE_REVIEWS = "local.bronze.reviews"
BRONZE_ITEMS = "local.bronze.items"
SILVER_INTERACTIONS = "local.silver.interactions"
SILVER_ITEMS = "local.silver.items"
GOLD_5CORE = "local.gold.interactions_5core"
FUNNEL_TABLE = "local.dq.kcore_funnel"
WATERFALL_TABLE = "local.dq.waterfall"


# --------------------------------------------------------------------------- #
# Fixture loaders (mirror test_silver_build.py: production bronze schema).
# --------------------------------------------------------------------------- #


def _load_bronze_items(spark):
    spark.conf.set("spark.sql.caseSensitive", "true")
    try:
        df = spark.read.parquet(str(ITEMS_PARQUET))
        proj = df.select(
            "parent_asin", "main_category", "title", "average_rating",
            "rating_number", "price", "store", "categories",
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
    """Synthesize the raw→bronze feed. No ingest job runs in the fixture pipeline,
    so we record written == the loaded bronze count (corrupt = 0), which is the
    invariant a real bronze ingest guarantees (raw = written + corrupt)."""
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
# Module-scoped pipeline: build silver + gold once, expose the tmp feeds.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pipeline(spark, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("waterfall_feeds")
    build_summary = tmp / "build_summary.jsonl"
    ingest_summary = tmp / "ingest_summary.jsonl"

    _load_bronze_items(spark)
    _load_bronze_reviews(spark)

    build_items(spark, run_id=RUN_ID, summary_path=str(build_summary), write_summary=True)
    build_interactions(
        spark, run_id=RUN_ID, summary_path=str(build_summary), write_summary=True
    )

    reviews_n = spark.table(BRONZE_REVIEWS).count()
    items_n = spark.table(BRONZE_ITEMS).count()
    _write_ingest_summary(ingest_summary, reviews_n, items_n)

    run_kcore(
        spark,
        source_table=SILVER_INTERACTIONS,
        target_table=GOLD_5CORE,
        funnel_table=FUNNEL_TABLE,
        k=5,
        run_id=RUN_ID,
    )

    return {
        "build_summary": build_summary,
        "ingest_summary": ingest_summary,
        "reviews_n": reviews_n,
        "items_n": items_n,
    }


def _run(spark, pipeline, tmp_path, **overrides):
    kwargs = dict(
        run_id=RUN_ID,
        ingest_summary_path=str(pipeline["ingest_summary"]),
        build_summary_path=str(pipeline["build_summary"]),
        manifest_path=str(tmp_path / "MANIFEST.md"),
        json_out=str(tmp_path / "waterfall.json"),
    )
    kwargs.update(overrides)
    return run_waterfall(spark, **kwargs)


# --------------------------------------------------------------------------- #
# (a) Exact sums + live-count agreement on every edge.
# --------------------------------------------------------------------------- #


def test_every_edge_sums_and_matches_live_counts(spark, pipeline, tmp_path):
    result = _run(spark, pipeline, tmp_path)

    live = {
        BRONZE_REVIEWS: spark.table(BRONZE_REVIEWS).count(),
        BRONZE_ITEMS: spark.table(BRONZE_ITEMS).count(),
        SILVER_INTERACTIONS: spark.table(SILVER_INTERACTIONS).count(),
        SILVER_ITEMS: spark.table(SILVER_ITEMS).count(),
        GOLD_5CORE: spark.table(GOLD_5CORE).count(),
    }

    for dataset, entry in result["datasets"].items():
        for e in entry["edges"]:
            assert e["sum_ok"], f"{dataset} {e} sum mismatch"
            assert e["count_ok"], f"{dataset} {e} live-count mismatch"
            assert e["reason_sum"] == e["source_rows"]
            assert e["target_count"] == live[e["target_table"]]
            assert e["kept_rows"] == e["target_count"]

    # Chaining reviews: raw.written == bronze.count == silver.input;
    # silver.count == kcore input; kcore output == gold.count.
    rev = result["datasets"]["reviews"]["edges"]
    assert rev[0]["kept_rows"] == rev[1]["source_rows"]
    assert rev[1]["kept_rows"] == rev[2]["source_rows"]

    # The appended dq.waterfall rows reconstruct each edge's source exactly.
    wf = (
        spark.table(WATERFALL_TABLE)
        .where(F.col("run_id") == RUN_ID)
        .groupBy("dataset", "stage_from", "stage_to")
        .agg(F.sum("rows").alias("total"))
        .collect()
    )
    edge_source = {
        (d, e["stage_from"], e["stage_to"]): e["source_rows"]
        for d, entry in result["datasets"].items()
        for e in entry["edges"]
    }
    assert len(wf) == len(edge_source)
    for r in wf:
        assert r["total"] == edge_source[(r["dataset"], r["stage_from"], r["stage_to"])]


# --------------------------------------------------------------------------- #
# (b) Idempotent MANIFEST section.
# --------------------------------------------------------------------------- #


def test_manifest_section_idempotent(spark, pipeline, tmp_path):
    manifest = tmp_path / "MANIFEST.md"
    manifest.write_text("# Manifest\n\n## Existing section\n\nkeep me\n")

    _run(spark, pipeline, tmp_path, manifest_path=str(manifest))
    _run(spark, pipeline, tmp_path, manifest_path=str(manifest))

    text = manifest.read_text()
    assert text.count(SECTION_HEADER) == 1
    # Pre-existing content is preserved.
    assert "## Existing section" in text
    assert "keep me" in text


# --------------------------------------------------------------------------- #
# (c) waterfall.json structure.
# --------------------------------------------------------------------------- #


def test_waterfall_json_structure(spark, pipeline, tmp_path):
    _run(spark, pipeline, tmp_path)
    payload = json.loads((tmp_path / "waterfall.json").read_text())

    assert set(payload) == {"run_id", "generated_at", "datasets"}
    assert payload["run_id"] == RUN_ID
    assert set(payload["datasets"]) == {"reviews", "items"}

    reviews = payload["datasets"]["reviews"]
    assert reviews["chain"] == ["raw", "bronze", "silver", "gold"]
    assert [(e["stage_from"], e["stage_to"]) for e in reviews["edges"]] == [
        ("raw", "bronze"),
        ("bronze", "silver"),
        ("silver", "gold"),
    ]
    # k-core funnel published for reviews only; iteration 0 present.
    assert "kcore_funnel" in reviews
    assert reviews["kcore_funnel"][0]["iteration"] == 0
    assert reviews["kcore_run_id"] == RUN_ID

    items = payload["datasets"]["items"]
    assert items["chain"] == ["raw", "bronze", "silver"]
    assert "kcore_funnel" not in items

    # Edge shape.
    edge = reviews["edges"][0]
    assert set(edge) == {
        "stage_from", "stage_to", "source_rows", "kept_rows", "reason_sum",
        "target_table", "target_count", "sum_ok", "count_ok", "reasons",
    }
    assert any(r["reason"] == "kept" for r in edge["reasons"])


# --------------------------------------------------------------------------- #
# (d) Corruption → non-zero exit naming the bad edge.
# --------------------------------------------------------------------------- #


def test_tampered_build_summary_raises_naming_edge(spark, pipeline, tmp_path):
    lines = pipeline["build_summary"].read_text().splitlines()
    tampered = []
    for line in lines:
        rec = json.loads(line)
        if rec["table"] == "interactions":
            rec["kept"] = int(rec["kept"]) + 1  # break Σ == input_rows
        tampered.append(json.dumps(rec))
    bad = tmp_path / "build_summary_bad.jsonl"
    bad.write_text("\n".join(tampered) + "\n")

    with pytest.raises(WaterfallError) as exc:
        _run(
            spark, pipeline, tmp_path,
            build_summary_path=str(bad),
            write_manifest=False,
            write_json=False,
            append_table=False,
        )
    msg = str(exc.value)
    assert "reviews" in msg
    assert "bronze -> silver" in msg
