"""Ops monthly backfill + incremental append (Phase 5, T20).

Fixture scale only: every table lives in the session's tmp warehouse under
``local.ops.*``; the real ``data/warehouse`` is never opened here.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import pytest

from batch_recsys_lab.ops import monthly
from batch_recsys_lab.ops.snapshot_metrics import table_metadata_for
from conftest import warehouse_of

pytestmark = pytest.mark.spark

BACKFILL_END = "2023-06-30"
LATE_START = "2023-05-01"
HOLDOUT_PERMILLE = 200  # fixture-scale: 200/1000 of ~122 late-window rows

TABLE = "local.ops.monthly_under_test"
TABLE_REPEAT = "local.ops.monthly_repeat"


def _identity_set(df):
    return {
        (r["user_id"], r["parent_asin"], r["ts"])
        for r in df.select("user_id", "parent_asin", "ts").collect()
    }


@pytest.fixture(scope="module")
def backfilled(spark, ops_source):
    return monthly.create_backfill(
        spark,
        source=ops_source,
        table=TABLE,
        backfill_end=BACKFILL_END,
        late_window_start=LATE_START,
        holdout_permille=HOLDOUT_PERMILLE,
    )


def test_partition_spec_is_months_ts(spark, backfilled):
    assert monthly.partition_transforms(spark, TABLE) == ["months(ts)"]
    # ...and the metadata JSON agrees (JVM-free view of the same fact).
    spec = table_metadata_for(warehouse_of(spark), TABLE)["partition_spec"]
    assert spec == [{"name": "ts_month", "transform": "month"}]


def test_backfill_and_holdout_partition_the_source_slice(spark, ops_source, backfilled):
    src = spark.table(ops_source)
    slice_df = src.where(monthly.backfill_predicate(BACKFILL_END))
    hold_pred = monthly.holdout_predicate(HOLDOUT_PERMILLE, LATE_START, BACKFILL_END)

    source_ids = _identity_set(slice_df)
    holdout_ids = _identity_set(slice_df.where(hold_pred))
    table_ids = _identity_set(spark.table(TABLE))

    assert holdout_ids, "fixture must produce a non-empty holdout"
    assert table_ids | holdout_ids == source_ids  # union covers the slice
    assert table_ids & holdout_ids == set()       # and the halves are disjoint

    assert backfilled["source_rows"] == len(source_ids)
    assert backfilled["holdout_rows"] == len(holdout_ids)
    assert backfilled["rows_written"] == len(table_ids)
    assert backfilled["reconciles_with_source"] is True

    # The holdout is confined to the last two months of the backfill window.
    assert holdout_ids <= _identity_set(
        slice_df.where(monthly.late_window_predicate(LATE_START, BACKFILL_END))
    )

    # Nothing after the backfill window leaked in.
    assert spark.table(TABLE).where("ts >= TIMESTAMP '2023-07-01 00:00:00'").count() == 0


def test_holdout_predicate_is_deterministic(spark, ops_source, backfilled):
    """A second, independent invocation selects exactly the same rows."""
    assert monthly.holdout_predicate(
        HOLDOUT_PERMILLE, LATE_START, BACKFILL_END
    ) == monthly.holdout_predicate(HOLDOUT_PERMILLE, LATE_START, BACKFILL_END)

    again = monthly.create_backfill(
        spark,
        source=ops_source,
        table=TABLE_REPEAT,
        backfill_end=BACKFILL_END,
        late_window_start=LATE_START,
        holdout_permille=HOLDOUT_PERMILLE,
    )
    assert again["holdout_rows"] == backfilled["holdout_rows"]
    assert _identity_set(spark.table(TABLE_REPEAT)) == _identity_set(spark.table(TABLE))


def test_append_month_snapshot_summary_matches_source_count(spark, ops_source, backfilled):
    warehouse = warehouse_of(spark)
    for month in ("2023-07", "2023-08"):
        rows_before = spark.table(TABLE).count()
        res = monthly.append_month(
            spark, table=TABLE, month=month, source=ops_source, backfill_end=BACKFILL_END
        )
        expected = spark.table(ops_source).where(monthly.month_predicate(month)).count()
        assert expected > 0
        assert res["month_source_rows"] == expected
        assert res["rows_written"] == expected
        assert res["reconciles_with_source"] is True
        assert spark.table(TABLE).count() == rows_before + expected

        meta = table_metadata_for(warehouse, TABLE)
        current = [
            s for s in meta["snapshots"] if s["snapshot_id"] == meta["current_snapshot_id"]
        ][0]
        assert current["operation"] == "append"
        assert current["added_records"] == expected
        assert current["total_records"] == rows_before + expected


def test_append_month_inside_backfill_window_is_refused(spark, ops_source, backfilled):
    with pytest.raises(ValueError, match="inside the backfill window"):
        monthly.append_month(
            spark, table=TABLE, month="2023-06", source=ops_source, backfill_end=BACKFILL_END
        )


def test_monthly_writers_refuse_non_ops_tables(ops_source):
    """Guard fires before Spark is touched (spark=None would explode otherwise)."""
    for table in ("local.gold.interactions_5core", "local.silver.interactions"):
        with pytest.raises(ValueError, match="may only write tables under"):
            monthly.create_backfill(None, source=ops_source, table=table)
        with pytest.raises(ValueError, match="may only write tables under"):
            monthly.append_month(None, table=table, month="2023-07")
