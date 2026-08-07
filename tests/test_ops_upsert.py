"""Late-data MERGE upsert (Phase 5, T20). Fixture scale, tmp warehouse only."""

from __future__ import annotations

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import pytest

from batch_recsys_lab.ops import monthly, upsert

pytestmark = pytest.mark.spark

BACKFILL_END = "2023-06-30"
LATE_START = "2023-05-01"
HOLDOUT_PERMILLE = 200
UPDATE_PERMILLE = 200

TABLE = "local.ops.upsert_under_test"


def _content_checksum(spark, table) -> list:
    """Order-independent fingerprint of every column of every row."""
    rows = spark.sql(
        f"SELECT xxhash64(user_id, parent_asin, asin, rating, ts, helpful_vote, "
        f"verified_purchase) AS h FROM {table}"
    ).collect()
    return sorted(int(r["h"]) for r in rows)


@pytest.fixture(scope="module")
def merged(spark, ops_source):
    backfill = monthly.create_backfill(
        spark,
        source=ops_source,
        table=TABLE,
        backfill_end=BACKFILL_END,
        late_window_start=LATE_START,
        holdout_permille=HOLDOUT_PERMILLE,
    )
    result = upsert.late_data_merge(
        spark,
        table=TABLE,
        source=ops_source,
        holdout_permille=HOLDOUT_PERMILLE,
        update_permille=UPDATE_PERMILLE,
        late_window_start=LATE_START,
        backfill_end=BACKFILL_END,
    )
    return backfill, result


def test_merge_matched_and_inserted_counts_are_exact(spark, ops_source, merged):
    backfill, result = merged
    src = spark.table(ops_source)
    slice_rows = src.where(monthly.backfill_predicate(BACKFILL_END)).count()

    # Inserts are exactly what the backfill withheld — measured independently of
    # the holdout predicate, as (full slice - what the backfill actually wrote).
    assert result["inserted"] == slice_rows - backfill["rows_written"]
    assert result["inserted"] == backfill["holdout_rows"]
    assert result["inserted"] > 0
    assert result["insert_count_reconciles"] is True

    # Updates: the fixture contains no rating 5.0, so every 5.0 in the table is
    # an update this MERGE applied.
    assert result["matched_updated"] > 0
    assert spark.table(TABLE).where("rating = 5.0").count() == result["matched_updated"]

    # ...and every updated row sits in the late window, outside the holdout.
    assert (
        spark.table(TABLE)
        .where("rating = 5.0")
        .where(f"NOT ({monthly.late_window_predicate(LATE_START, BACKFILL_END)})")
        .count()
        == 0
    )


def test_post_merge_total_equals_full_source_slice(spark, ops_source, merged):
    _, result = merged
    slice_rows = spark.table(ops_source).where(
        monthly.backfill_predicate(BACKFILL_END)
    ).count()
    assert result["post_merge_total"] == slice_rows
    assert spark.table(TABLE).count() == slice_rows
    assert result["reconciles_with_source"] is True


def test_merge_is_idempotent(spark, ops_source, merged):
    """Re-running the identical MERGE changes neither row count nor values."""
    _, first = merged
    before_rows = spark.table(TABLE).count()
    before_checksum = _content_checksum(spark, TABLE)

    second = upsert.late_data_merge(
        spark,
        table=TABLE,
        source=ops_source,
        holdout_permille=HOLDOUT_PERMILLE,
        update_permille=UPDATE_PERMILLE,
        late_window_start=LATE_START,
        backfill_end=BACKFILL_END,
    )

    assert second["inserted"] == first["inserted"]        # same batch built...
    assert second["matched_updated"] == first["matched_updated"]
    assert second["post_merge_total"] == before_rows      # ...but nothing new landed
    assert spark.table(TABLE).count() == before_rows
    assert _content_checksum(spark, TABLE) == before_checksum
    assert (
        spark.table(TABLE).where("rating = 5.0").count() == first["matched_updated"]
    )


def test_insert_and_update_predicates_are_disjoint(spark, ops_source):
    hold = monthly.holdout_predicate(HOLDOUT_PERMILLE, LATE_START, BACKFILL_END)
    upd = upsert.update_predicate(
        HOLDOUT_PERMILLE, UPDATE_PERMILLE, LATE_START, BACKFILL_END
    )
    overlap = spark.table(ops_source).where(f"({hold}) AND ({upd})").count()
    assert overlap == 0


def test_upsert_refuses_non_ops_tables():
    for table in ("local.gold.interactions_5core", "local.silver.interactions"):
        with pytest.raises(ValueError, match="may only write tables under"):
            upsert.late_data_merge(None, table=table)
