"""Ops monthly backfill + incremental append (Phase 5, T20).

Fixture scale only: every table lives in the session's tmp warehouse under
``local.ops.*``; the real ``data/warehouse`` is never opened here.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import pytest

from batch_recsys_lab.ops import maintenance, monthly
from batch_recsys_lab.ops.snapshot_metrics import files_stats, table_metadata_for
from conftest import warehouse_of

pytestmark = pytest.mark.spark

BACKFILL_END = "2023-06-30"
LATE_START = "2023-05-01"
HOLDOUT_PERMILLE = 200  # fixture-scale: 200/1000 of ~122 late-window rows

TABLE = "local.ops.monthly_under_test"
TABLE_REPEAT = "local.ops.monthly_repeat"

# Fragmentation (T23b) runs on its own backfilled copies so it cannot perturb
# the tables the backfill/append tests above assert on.
TABLE_FRAG_A = "local.ops.monthly_frag_a"
TABLE_FRAG_B = "local.ops.monthly_frag_b"
FRAG_MONTH = "2023-03"  # inside the backfill window, outside the holdout window


def _identity_set(df):
    return {
        (r["user_id"], r["parent_asin"], r["ts"])
        for r in df.select("user_id", "parent_asin", "ts").collect()
    }


def _day_counts(spark, table, month):
    """``{date: rows}`` for one month — the fragmentation slice profile."""
    rows = (
        spark.table(table)
        .where(monthly.month_predicate(month))
        .selectExpr("to_date(ts) AS d")
        .groupBy("d")
        .count()
        .collect()
    )
    return {r["d"]: int(r["count"]) for r in rows}


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
        # fragment_month deletes rows, so it carries the same guard.
        with pytest.raises(ValueError, match="may only write tables under"):
            monthly.fragment_month(None, table=table, month=FRAG_MONTH)
        with pytest.raises(ValueError, match="may only write tables under"):
            monthly.scratch_table_name(table)
        # ...and it cannot be talked into writing its scratch copy elsewhere.
        with pytest.raises(ValueError, match="may only write tables under"):
            monthly.fragment_month(
                None, table=TABLE_FRAG_A, month=FRAG_MONTH, scratch_table=table
            )


# --- fragmentation (T23b) -----------------------------------------------------


@pytest.fixture(scope="module")
def frag_backfilled(spark, ops_source):
    monthly.create_backfill(
        spark,
        source=ops_source,
        table=TABLE_FRAG_A,
        backfill_end=BACKFILL_END,
        late_window_start=LATE_START,
        holdout_permille=HOLDOUT_PERMILLE,
    )
    return TABLE_FRAG_A


@pytest.fixture(scope="module")
def fragmented(spark, ops_source, frag_backfilled):
    """Before-state + the fragment result, so the assertions can compare."""
    before = {
        "total": spark.table(TABLE_FRAG_A).count(),
        "month": spark.table(TABLE_FRAG_A)
        .where(monthly.month_predicate(FRAG_MONTH))
        .count(),
        "files": files_stats(spark, TABLE_FRAG_A)["file_count"],
        "ids": _identity_set(spark.table(TABLE_FRAG_A)),
        "day_counts": _day_counts(spark, TABLE_FRAG_A, FRAG_MONTH),
    }
    res = monthly.fragment_month(
        spark, table=TABLE_FRAG_A, source=ops_source, month=FRAG_MONTH
    )
    return before, res


def test_fragment_preserves_every_row(spark, fragmented):
    before, res = fragmented
    assert before["month"] > 0
    assert res["rows_before_total"] == before["total"]
    assert res["rows_after_total"] == before["total"]
    assert res["rows_month"] == before["month"]
    assert res["rows_preserved"] is True

    # ...and the table itself agrees, row-for-row, not just in aggregate.
    assert spark.table(TABLE_FRAG_A).count() == before["total"]
    assert (
        spark.table(TABLE_FRAG_A).where(monthly.month_predicate(FRAG_MONTH)).count()
        == before["month"]
    )
    assert _identity_set(spark.table(TABLE_FRAG_A)) == before["ids"]
    assert _day_counts(spark, TABLE_FRAG_A, FRAG_MONTH) == before["day_counts"]


def test_fragment_adds_one_file_per_non_empty_day(spark, fragmented):
    before, res = fragmented
    non_empty_days = len(before["day_counts"])
    assert non_empty_days > 1, "fixture month must span several days"
    assert res["n_slices"] == non_empty_days
    assert res["n_slices"] <= res["days_in_month"]
    assert res["files_added"] == res["n_slices"]
    assert res["one_file_per_slice"] is True

    # Fragmentation is the point: strictly more files than before.
    assert res["files_after"] > res["files_before"]
    assert files_stats(spark, TABLE_FRAG_A)["file_count"] == res["files_after"]

    # The scratch copy is dropped once the row counts have been verified.
    assert not maintenance.table_exists(spark, res["scratch_table"])
    assert res["scratch_table"] == TABLE_FRAG_A + "__fragment_scratch"


def test_fragment_is_deterministic(spark, ops_source, fragmented):
    """Identical input fragmented twice -> identical rows and identical slices."""
    _, res_a = fragmented
    monthly.create_backfill(
        spark,
        source=ops_source,
        table=TABLE_FRAG_B,
        backfill_end=BACKFILL_END,
        late_window_start=LATE_START,
        holdout_permille=HOLDOUT_PERMILLE,
    )
    res_b = monthly.fragment_month(
        spark, table=TABLE_FRAG_B, source=ops_source, month=FRAG_MONTH
    )

    assert _identity_set(spark.table(TABLE_FRAG_B)) == _identity_set(
        spark.table(TABLE_FRAG_A)
    )
    assert _day_counts(spark, TABLE_FRAG_B, FRAG_MONTH) == _day_counts(
        spark, TABLE_FRAG_A, FRAG_MONTH
    )
    for key in ("rows_month", "n_slices", "files_added", "slice_rows_min", "slice_rows_max"):
        assert res_b[key] == res_a[key], key


def test_compact_after_fragment_puts_the_files_back(spark, fragmented):
    """The exhibit's payoff: bin-packing has real work only post-fragmentation."""
    _, res = fragmented
    rows_before = spark.table(TABLE_FRAG_A).count()
    ids_before = _identity_set(spark.table(TABLE_FRAG_A))
    files_before = files_stats(spark, TABLE_FRAG_A)["file_count"]

    out = maintenance.compact(
        spark, TABLE_FRAG_A, options={"min-input-files": "2"}
    )
    files_after = files_stats(spark, TABLE_FRAG_A)["file_count"]

    assert out["rewritten_files"] >= res["n_slices"]
    assert out["added_files"] < out["rewritten_files"]
    assert out["failed_files"] == 0
    assert files_after < files_before
    assert spark.table(TABLE_FRAG_A).count() == rows_before
    assert _identity_set(spark.table(TABLE_FRAG_A)) == ids_before
