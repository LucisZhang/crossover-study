"""Ops scenario runner: record shape, append-once, epilogue guards (Phase 5, T20).

Everything here runs against the session's tmp warehouse and a tmp results file
— never ``data/warehouse`` and never the real ``results/runs.jsonl``.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import json

import pytest

from batch_recsys_lab.ops import run_scenario
from batch_recsys_lab.ops.run_scenario import ProtectedTableMoved, check_protected
from conftest import warehouse_of

TABLE = "local.ops.scenario_under_test"
BACKFILL_END = "2023-06-30"
LATE_START = "2023-05-01"
HOLDOUT_PERMILLE = 200

REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "run_id",
    "run_ts",
    "git_sha",
    "git_dirty",
    "scenario",
    "table",
    "params",
    "snapshot_before",
    "snapshot_after",
    "rows_before",
    "rows_after",
    "files_before",
    "files_after",
    "bytes_before",
    "bytes_after",
    "wall_clock_s",
    "disk_avail_gb",
}


def _kwargs(spark, ops_source):
    return {
        "warehouse": warehouse_of(spark),
        "table": TABLE,
        "source": ops_source,
        "backfill_end": BACKFILL_END,
        "late_window_start": LATE_START,
        "holdout_permille": HOLDOUT_PERMILLE,
    }


# --- pure-python guard --------------------------------------------------------


def test_protected_guard_raises_when_a_snapshot_moved(tmp_path):
    """No Spark: the guard must be trustworthy even when Spark misbehaved."""
    # The table does not exist in this warehouse -> current id is None, so a
    # non-None "before" is a move.
    with pytest.raises(ProtectedTableMoved, match="PROTECTED TABLE MOVED"):
        check_protected(tmp_path, {"local.gold.interactions_5core": 123456789})
    # Absent before and after: unchanged, no raise.
    assert check_protected(tmp_path, {"local.gold.interactions_5core": None}) == {
        "local.gold.interactions_5core": None
    }


def test_unknown_step_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown step"):
        run_scenario.run_step(None, "vacuum", warehouse=tmp_path, table=TABLE)


def test_run_step_refuses_non_ops_tables(tmp_path):
    with pytest.raises(ValueError, match="may only write tables under"):
        run_scenario.run_step(None, "compact", warehouse=tmp_path, table="local.gold.x")


def test_namespace_dir_removal_is_narrowly_scoped(tmp_path):
    warehouse = tmp_path / "warehouse"
    for ns in ("ops", "gold"):
        (warehouse / ns / "t" / "data").mkdir(parents=True)

    with pytest.raises(ValueError, match="only the 'ops' namespace"):
        run_scenario.remove_ops_namespace_dir(warehouse, "gold")
    with pytest.raises(ValueError, match="only the 'ops' namespace"):
        run_scenario.remove_ops_namespace_dir(warehouse, "../ops")
    assert (warehouse / "gold").is_dir()

    target, removed = run_scenario.remove_ops_namespace_dir(warehouse, "ops")
    assert removed is True
    assert target == (warehouse / "ops").resolve()
    assert not (warehouse / "ops").exists()
    assert (warehouse / "gold").is_dir()  # untouched

    # Idempotent: a second call is a no-op, not an error.
    assert run_scenario.remove_ops_namespace_dir(warehouse, "ops")[1] is False


# --- end-to-end record shape --------------------------------------------------


@pytest.mark.spark
def test_backfill_record_shape_and_single_append(spark, ops_source, tmp_path):
    results = tmp_path / "runs.jsonl"
    out = run_scenario.run_and_record(
        spark, "backfill", results=results, **_kwargs(spark, ops_source)
    )
    record = out["record"]

    assert out["exit_code"] == 0
    assert REQUIRED_KEYS <= set(record)
    assert record["kind"] == "ops"
    assert record["scenario"] == "backfill"
    assert record["table"] == TABLE
    assert record["snapshot_before"] is None      # table did not exist yet
    assert record["snapshot_after"] is not None
    assert record["rows_before"] == 0
    assert record["rows_after"] == record["rows_written"] > 0
    assert record["files_after"] >= 1
    assert record["bytes_after"] > 0
    assert record["reconciles_with_source"] is True
    assert record["holdout_rows"] > 0
    assert record["source_rows"] == record["rows_after"] + record["holdout_rows"]

    # The exact predicates that ran are recorded verbatim.
    params = record["params"]
    assert params["partition_transform"] == "months(ts)"
    assert params["holdout_permille"] == HOLDOUT_PERMILLE
    assert "pmod(xxhash64(user_id, parent_asin, ts), 1000) < 200" in params["holdout_predicate"]
    assert params["identity_cols"] == ["user_id", "parent_asin", "ts"]

    # Exactly one line was appended, and it is the record.
    lines = results.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record


@pytest.mark.spark
def test_each_step_appends_exactly_one_record(spark, ops_source, tmp_path):
    """append -> upsert -> compact -> expire on top of the backfill above."""
    results = tmp_path / "runs.jsonl"
    kwargs = _kwargs(spark, ops_source)

    run_scenario.run_and_record(spark, "backfill", results=results, **kwargs)
    run_scenario.run_and_record(
        spark, "append", results=results, month="2023-07", **kwargs
    )
    run_scenario.run_and_record(spark, "upsert", results=results, **kwargs)
    run_scenario.run_and_record(
        spark,
        "compact",
        results=results,
        compact_options={"min-input-files": "2"},
        **kwargs,
    )
    run_scenario.run_and_record(spark, "expire", results=results, retain_last=2, **kwargs)

    records = [json.loads(ln) for ln in results.read_text().splitlines()]
    assert len(records) == 5
    assert [r["scenario"] for r in records] == [
        "backfill",
        "append",
        "upsert",
        "compact",
        "expire",
    ]
    assert all(REQUIRED_KEYS <= set(r) for r in records)
    assert all(r["kind"] == "ops" for r in records)

    _, appended, upserted, compacted, expired = records

    assert appended["month"] == "2023-07"
    assert appended["month_source_rows"] == appended["rows_after"] - appended["rows_before"]

    assert upserted["inserted"] > 0
    assert upserted["matched_updated"] > 0
    assert upserted["post_merge_total"] == upserted["rows_after"]
    assert upserted["reconciles_with_source"] is True
    assert "MERGE INTO" in upserted["params"]["merge_sql"]

    # Bin-packing on this fixture may find nothing to do (one file per month
    # partition already); the strict "file count decreases" claim is proved on a
    # deliberately fragmented table in test_ops_maintenance.py. What must hold
    # here is that compaction never adds files and never changes content.
    assert compacted["files_after"] <= compacted["files_before"]
    assert compacted["rows_after"] == compacted["rows_before"]
    assert compacted["params"]["procedure"] == "rewrite_data_files"

    assert expired["params"]["procedure"] == "expire_snapshots"
    assert expired["params"]["retain_last"] == 2
    assert expired["snapshot_count_after"] == 2
    assert expired["rows_after"] == expired["rows_before"]

    # Snapshot chaining across the whole scenario.
    assert appended["snapshot_before"] == records[0]["snapshot_after"]
    assert upserted["snapshot_before"] == appended["snapshot_after"]


@pytest.mark.spark
def test_disk_floor_is_a_hard_stop(spark, ops_source, tmp_path):
    """A floor above any plausible free space forces the non-zero exit path —
    and the record is still appended first (it is durable evidence)."""
    results = tmp_path / "runs.jsonl"
    out = run_scenario.run_and_record(
        spark,
        "backfill",
        results=results,
        min_disk_gb=10**9,
        **_kwargs(spark, ops_source),
    )
    assert out["exit_code"] == 2
    assert len(results.read_text().splitlines()) == 1


@pytest.mark.spark
def test_clean_drops_the_ops_table(spark, ops_source, tmp_path):
    """``remove_dir=False``: the shared session warehouse holds other tests'
    ops tables, so only the DROP ... PURGE half is exercised here (the rmtree
    half is covered by test_namespace_dir_removal_is_narrowly_scoped)."""
    table = "local.ops.clean_under_test"
    run_scenario.run_and_record(
        spark,
        "backfill",
        results=tmp_path / "runs.jsonl",
        **{**_kwargs(spark, ops_source), "table": table},
    )
    assert spark.catalog.tableExists(table)

    summary = run_scenario.clean(
        spark, table=table, warehouse=warehouse_of(spark), remove_dir=False
    )
    assert summary["dropped"] is True
    assert summary["dir_removed"] is False
    assert not spark.catalog.tableExists(table)
    assert summary["namespace_dir"].endswith("/ops")


def test_clean_refuses_non_ops_tables():
    with pytest.raises(ValueError, match="may only write tables under"):
        run_scenario.clean(None, table="local.gold.interactions_5core")
