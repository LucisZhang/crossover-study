"""Fast, Spark-free checks on the bundled CI fixture parquet files.

Skips entirely if the fixtures haven't been built yet (bronze.reviews must
exist first — see batch_recsys_lab.ingest.make_fixture).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEWS_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "bronze_reviews_50k.parquet"
ITEMS_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "bronze_items_fixture.parquet"

pytestmark = pytest.mark.skipif(
    not (REVIEWS_FIXTURE_PATH.exists() and ITEMS_FIXTURE_PATH.exists()),
    reason="fixtures not built yet",
)


@pytest.fixture(scope="module")
def reviews_table():
    return pq.read_table(REVIEWS_FIXTURE_PATH)


@pytest.fixture(scope="module")
def items_table():
    return pq.read_table(ITEMS_FIXTURE_PATH)


def test_reviews_fixture_row_count(reviews_table):
    assert reviews_table.num_rows == 50_000


def test_reviews_fixture_drops_text_and_images(reviews_table):
    names = set(reviews_table.column_names)
    assert "text" not in names
    assert "images" not in names


def test_items_fixture_row_count(items_table):
    assert items_table.num_rows <= 5_000


def test_items_fixture_price_is_string(items_table):
    schema = items_table.schema
    price_field = schema.field("price")
    # pyarrow may represent Spark/Parquet string columns as "string" or the
    # 64-bit-offset variant "large_string" depending on writer version; both
    # are logically UTF-8 strings, which is what this test guards.
    assert str(price_field.type) in ("string", "large_string")


def test_items_parent_asins_appear_in_reviews_fixture(reviews_table, items_table):
    reviews_parent_asins = set(reviews_table.column("parent_asin").to_pylist())
    items_parent_asins = set(items_table.column("parent_asin").to_pylist())
    assert items_parent_asins.issubset(reviews_parent_asins)
