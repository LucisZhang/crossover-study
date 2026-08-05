"""Bronze ingestion tests on tiny synthetic gz-jsonl fixtures.

Never touches the real (incomplete) data/raw downloads — builds small gzipped
jsonl files in a tmp dir and ingests them against a tmp Iceberg warehouse.

Marked ``spark`` (starts a local Spark session). Note: ``get_spark`` returns the
first session started in the process, so master/driver args on later calls are
ignored — harmless here since every test only needs a working local catalog.
"""

from __future__ import annotations

import gzip
import json

import pytest

from batch_recsys_lab.ingest.bronze import ingest_table
from batch_recsys_lab.spark_session import get_spark

pytestmark = pytest.mark.spark


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    warehouse = tmp_path_factory.mktemp("bronze_warehouse")
    session = get_spark(
        app_name="bronze-ingest-test",
        warehouse=warehouse,
        master="local[2]",
        driver_memory="2g",
    )
    yield session
    # Do not stop(): a session may be shared with other spark-marked tests in
    # the same process; pytest teardown of the JVM is fine at process exit.


def _write_gz_jsonl(path, records_and_raw):
    """Write mixed valid-dict / raw-string lines to a gzip jsonl file."""
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for item in records_and_raw:
            if isinstance(item, str):
                fh.write(item + "\n")  # raw (possibly corrupt) line verbatim
            else:
                fh.write(json.dumps(item) + "\n")


def _review(asin, rating=5.0):
    return {
        "rating": rating,
        "title": "Nice",
        "text": "a long review body the lab never uses",
        "images": [
            {
                "small_image_url": "s",
                "medium_image_url": "m",
                "large_image_url": "l",
                "attachment_type": "IMAGE",
            }
        ],
        "asin": asin,
        "parent_asin": asin + "P",
        "user_id": "U-" + asin,
        "timestamp": 1650000000000,
        "helpful_vote": 2,
        "verified_purchase": True,
    }


def test_ingest_reviews_projects_out_text_images_and_counts_corrupt(spark, tmp_path):
    src = tmp_path / "reviews.jsonl.gz"
    records = [_review(f"B{i:03d}") for i in range(5)]  # 5 valid
    records.append('{"rating": 4.0, "asin": "BROKEN"')  # 1 corrupt (truncated JSON)
    _write_gz_jsonl(src, records)

    summary = ingest_table(spark, "reviews", str(src), repartition=2)

    assert summary["table"] == "local.bronze.reviews"
    assert summary["total_parsed"] == 6
    assert summary["corrupt"] == 1
    assert summary["written"] == 5

    df = spark.table("local.bronze.reviews")
    assert df.count() == 5

    cols = set(df.columns)
    # §5: reviews drop text + images; corrupt marker never leaks into bronze.
    assert "text" not in cols
    assert "images" not in cols
    assert "_corrupt_record" not in cols
    # timestamp kept as raw long (bronze does not convert).
    assert dict(df.dtypes)["timestamp"] == "bigint"


def test_ingest_items_keeps_string_price_and_no_sentinel_nulls(spark, tmp_path):
    src = tmp_path / "meta.jsonl.gz"
    item_priced = {
        "main_category": "Electronics",
        "title": "Widget",
        "average_rating": 4.5,
        "rating_number": 100,
        "features": ["f1", "f2"],
        "description": ["desc"],
        "price": "24.99",  # string price the course DoubleType would have NULLed
        "images": [{"thumb": "t", "large": "l", "variant": "v", "hi_res": "h"}],
        "videos": [{"title": "vt", "url": "u", "user_id": "vu"}],
        "store": "Store",
        "categories": ["c1", "c2"],
        "details": {"Brand": "Acme"},
        "parent_asin": "P-PRICED",
        "bought_together": ["P-OTHER"],
    }
    item_null_price = {
        "main_category": "Electronics",
        "title": "No Price Widget",
        "average_rating": None,
        "rating_number": None,
        "features": [],
        "description": [],
        "price": None,  # stays NULL — no -1.0 sentinel
        "images": [],
        "videos": [],
        "store": None,
        "categories": [],
        "details": {},
        "parent_asin": "P-NULL",
        "bought_together": [],
    }
    item_range_price = {
        "main_category": "Electronics",
        "title": "Range Widget",
        "average_rating": 3.0,
        "rating_number": 5,
        "features": [],
        "description": [],
        "price": "10.00 - 20.00",  # range text: another reason price is STRING
        "parent_asin": "P-RANGE",
    }
    records = [item_priced, item_null_price, item_range_price]  # 3 valid
    records.append('{"parent_asin": "BROKEN", "price":')  # 1 corrupt
    _write_gz_jsonl(src, records)

    summary = ingest_table(spark, "items", str(src), repartition=2)

    assert summary["table"] == "local.bronze.items"
    assert summary["total_parsed"] == 4
    assert summary["corrupt"] == 1
    assert summary["written"] == 3

    df = spark.table("local.bronze.items")
    assert df.count() == 3

    # price is STRING (correction vs. seed DoubleType) and preserved verbatim.
    assert dict(df.dtypes)["price"] == "string"
    # items keep images/videos (only reviews project those out).
    assert "images" in df.columns
    assert "videos" in df.columns
    assert "_corrupt_record" not in df.columns

    by_asin = {r["parent_asin"]: r for r in df.collect()}
    assert by_asin["P-PRICED"]["price"] == "24.99"
    assert by_asin["P-RANGE"]["price"] == "10.00 - 20.00"

    # NULL stays NULL — no sentinel fills anywhere on the null-price row.
    null_row = by_asin["P-NULL"]
    assert null_row["price"] is None
    assert null_row["average_rating"] is None
    assert null_row["rating_number"] is None
    assert null_row["store"] is None
