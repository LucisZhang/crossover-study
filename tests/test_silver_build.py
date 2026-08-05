"""Silver build tests (Phase 1, T3).

Covers, against the shared tmp-warehouse ``spark`` fixture (``tests/conftest.py``)
and the bundled bronze fixtures:

* waterfall conservation — ``kept + Σquarantined + exact_duplicate + superseded``
  equals the fixture row count exactly (50,000 reviews / 5,000 items);
* price parsing and brand normalization at the transform level;
* quarantine rows carry ``violation_reasons`` + ``primary_reason`` + ``run_id``.

The items fixture stores ``details`` as a struct with case-colliding field names
(``Assembly Required`` vs ``Assembly required``), which Spark's default
case-insensitive reader rejects; the loader reads it case-sensitively and
reshapes ``details`` into the ``map<string,string>`` that production
``bronze.items`` actually carries, so silver's map access is what gets tested.
"""

from __future__ import annotations

import os

# This host cannot bind Spark to its own hostname; force loopback before any
# SparkContext starts (the session-scoped fixture builds it lazily on first use).
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from pathlib import Path

import pytest
from pyspark.sql import functions as F

from batch_recsys_lab.contracts.engine import PRIMARY_COL, REASONS_COL
from batch_recsys_lab.features.silver import (
    build_interactions,
    build_items,
    transform_items,
)

pytestmark = pytest.mark.spark

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ITEMS_PARQUET = FIXTURES / "bronze_items_fixture.parquet"
REVIEWS_PARQUET = FIXTURES / "bronze_reviews_50k.parquet"

_ITEMS_BRONZE_DDL = (
    "parent_asin string, main_category string, title string, "
    "average_rating double, rating_number long, price string, store string, "
    "categories array<string>, details map<string,string>"
)
_REVIEWS_BRONZE_DDL = (
    "user_id string, parent_asin string, asin string, rating double, "
    "timestamp long, helpful_vote long, verified_purchase boolean"
)


# --------------------------------------------------------------------------- #
# Fixture loaders (bronze fixtures → tmp-warehouse bronze tables).
# --------------------------------------------------------------------------- #


def _load_bronze_items(spark, table="local.bronze.items"):
    """Reshape the fixture struct-``details`` into map<string,string> (production
    bronze schema) and publish as a bronze table."""
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
        proj.writeTo(table).createOrReplace()
    finally:
        spark.conf.set("spark.sql.caseSensitive", "false")


def _load_bronze_reviews(spark, table="local.bronze.reviews"):
    df = spark.read.parquet(str(REVIEWS_PARQUET))
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.bronze")
    df.writeTo(table).createOrReplace()


# --------------------------------------------------------------------------- #
# Conservation.
# --------------------------------------------------------------------------- #


def test_items_conservation(spark):
    _load_bronze_items(spark)
    s = build_items(spark, run_id="t-items", write_summary=False)

    assert s["input_rows"] == 5000
    assert s["kept"] > 0
    total = (
        s["kept"]
        + sum(s["quarantined"].values())
        + s["exact_duplicate"]
        + s["superseded_by_later_review"]
    )
    assert total == 5000
    # Published silver row count matches the reported kept count.
    assert spark.table("local.silver.items").count() == s["kept"]


def test_interactions_conservation(spark):
    _load_bronze_items(spark)  # items first — interactions FK measure needs it.
    build_items(spark, run_id="t-items-fk", write_summary=False)
    _load_bronze_reviews(spark)

    s = build_interactions(spark, run_id="t-int", write_summary=False)

    assert s["input_rows"] == 50000
    assert s["kept"] > 0
    total = (
        s["kept"]
        + sum(s["quarantined"].values())
        + s["exact_duplicate"]
        + s["superseded_by_later_review"]
    )
    assert total == 50000
    assert spark.table("local.silver.interactions").count() == s["kept"]


# --------------------------------------------------------------------------- #
# Price parsing + brand normalization (transform level).
# --------------------------------------------------------------------------- #


def test_price_and_brand_parsing(spark):
    rows = [
        ("P1", "Cat", "t", 4.0, 1, "39.99", "S", ["c"], {"Brand": "  SONY "}),
        ("P2", "Cat", "t", 4.0, 1, "$5", "S", ["c"], {"Brand": "", "Manufacturer": "Acme"}),
        ("P3", "Cat", "t", 4.0, 1, "12.99 - 19.99", "S", ["c"], {}),
        ("P4", "Cat", "t", 4.0, 1, "see price in cart", "S", ["c"], {"Manufacturer": None}),
    ]
    df = spark.createDataFrame(rows, _ITEMS_BRONZE_DDL)
    out = {
        r["parent_asin"]: r
        for r in transform_items(df)
        .select("parent_asin", "price_usd", "brand_norm", "_price_unparseable", "_brand_source")
        .collect()
    }

    assert out["P1"]["price_usd"] == 39.99
    assert out["P2"]["price_usd"] == 5.0
    assert out["P3"]["price_usd"] is None
    assert out["P4"]["price_usd"] is None
    assert out["P3"]["_price_unparseable"] is True
    assert out["P4"]["_price_unparseable"] is True
    assert out["P1"]["_price_unparseable"] is False

    assert out["P1"]["brand_norm"] == "sony"      # '  SONY ' → trim/lower
    assert out["P1"]["_brand_source"] == "Brand"
    assert out["P2"]["brand_norm"] == "acme"       # blank Brand → Manufacturer fallback
    assert out["P2"]["_brand_source"] == "Manufacturer"
    assert out["P3"]["brand_norm"] == "unknown"    # no brand → unknown
    assert out["P4"]["brand_norm"] == "unknown"


# --------------------------------------------------------------------------- #
# Quarantine metadata.
# --------------------------------------------------------------------------- #


def test_quarantine_rows_carry_reasons_and_run_id(spark):
    # Minimal item catalog so the interactions FK measure has a reference table.
    items_ddl = (
        "parent_asin string, title string, main_category string, "
        "categories array<string>, store string, average_rating double, "
        "rating_number long, price_usd double, brand_norm string"
    )
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.silver")
    spark.createDataFrame(
        [("P1", "t", "Cat", ["c"], "S", 4.0, 10, 9.99, "acme")], items_ddl
    ).writeTo("local.silver.items").createOrReplace()

    ts_2022 = 1_640_995_200_000  # 2022-01-01Z
    ts_1992 = 700_000_000_000     # < 1996-01-01Z lower bound
    reviews = [
        ("U1", "P1", "A1", 5.0, ts_2022, 3, True),       # good
        ("U2", "P1", "A2", 4.0, ts_2022, 1, False),      # good
        (None, "P1", "A3", 5.0, ts_2022, 0, True),       # keys_non_null
        ("U3", "P1", "A4", 7.0, ts_2022, 0, True),       # rating_domain
        ("U4", "P1", "A5", 3.0, ts_1992, 0, True),       # ts_range
    ]
    spark.createDataFrame(reviews, _REVIEWS_BRONZE_DDL).writeTo(
        "local.bronze.reviews"
    ).createOrReplace()

    run_id = "quarantine-run-xyz"
    s = build_interactions(spark, run_id=run_id, write_summary=False)

    assert s["input_rows"] == 5
    assert s["kept"] == 2
    assert sum(s["quarantined"].values()) == 3
    assert set(s["quarantined"]) == {"keys_non_null", "rating_domain", "ts_range"}

    q = spark.table("local.quarantine.interactions")
    assert q.count() == 3
    collected = q.select(REASONS_COL, PRIMARY_COL, "run_id").collect()
    for r in collected:
        assert r["run_id"] == run_id
        assert r[PRIMARY_COL] is not None
        assert len(r[REASONS_COL]) >= 1
        assert r[PRIMARY_COL] == r[REASONS_COL][0]  # primary = first declared-order reason
