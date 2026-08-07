"""Ops scenario runner: one step, one ``kind="ops"`` record (Phase 5, T20).

    python -m batch_recsys_lab.ops.run_scenario --step backfill|append|upsert|compact|expire

Every step follows the same shape: capture the before-state (snapshot id, file
count/bytes, row count), run exactly one operation against ``local.ops.*``,
capture the after-state, and APPEND one record to ``results/runs.jsonl``
(append-only, CLAUDE.md invariant #3 — via ``eval.runlog.append_record``).

Two epilogue checks run after the record is appended, on every step:

1. **Protected-table guard.** The current snapshot IDs of ``local.gold.*`` (all
   four) and ``local.silver.interactions``, read JVM-free at step start, must be
   unchanged. If any moved, the step aborts loudly — a moved snapshot means the
   published lakehouse was mutated and every recorded eval record that cites it
   is now unverifiable.
2. **Disk floor.** ``df -g`` Avail below 8 GB prints a CRITICAL warning and exits
   non-zero. This is a hard stop for the phase: compaction and expiry both need
   headroom to write new files before deleting old ones.

``--step clean`` is the teardown behind ``make clean-ops``: it drops the ops
table with PURGE and appends NO record (nothing was measured).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from batch_recsys_lab.contracts.engine import _resolve_run_id
from batch_recsys_lab.eval import runlog
from batch_recsys_lab.ops import maintenance, monthly, upsert
from batch_recsys_lab.ops.maintenance import PROTECTED_TABLES, require_ops_table
from batch_recsys_lab.ops.snapshot_metrics import (
    current_snapshot_ids,
    disk_avail_gb,
    files_stats,
    snapshot_summary,
    table_metadata_for,
)
from batch_recsys_lab.spark_session import get_spark

DEFAULT_RESULTS = "results/runs.jsonl"
DEFAULT_WAREHOUSE = "data/warehouse"
MIN_DISK_GB = 8.0

STEPS = ("backfill", "append", "upsert", "compact", "expire")


class ProtectedTableMoved(RuntimeError):
    """A protected (published) table's snapshot moved during an ops step."""


# --- state capture ------------------------------------------------------------


def _table_state(spark, warehouse: str | Path, table: str) -> dict:
    meta = table_metadata_for(warehouse, table)
    files = files_stats(spark, table)
    rows = spark.table(table).count() if maintenance.table_exists(spark, table) else 0
    return {
        "snapshot": meta["current_snapshot_id"],
        "snapshot_count": meta["snapshot_count"],
        "rows": int(rows),
        "files": files["file_count"],
        "bytes": files["total_bytes"],
        "avg_file_mb": files["avg_file_mb"],
    }


def check_protected(warehouse: str | Path, before: dict[str, int | None]) -> dict:
    """Re-read the protected tables' snapshot IDs and compare (JVM-free)."""
    after = current_snapshot_ids(warehouse, before.keys())
    moved = {t: (before[t], after[t]) for t in before if before[t] != after[t]}
    if moved:
        raise ProtectedTableMoved(
            "PROTECTED TABLE MOVED during an ops step — the published lakehouse "
            "must never be written by batch_recsys_lab.ops: "
            + "; ".join(f"{t}: {b} -> {a}" for t, (b, a) in sorted(moved.items()))
        )
    return after


# --- the step -----------------------------------------------------------------


def run_step(
    spark,
    step: str,
    *,
    warehouse: str | Path = DEFAULT_WAREHOUSE,
    table: str = monthly.OPS_MONTHLY,
    source: str = monthly.SILVER_INTERACTIONS,
    month: str | None = None,
    backfill_end: str = monthly.BACKFILL_END,
    late_window_start: str = monthly.LATE_WINDOW_START,
    holdout_permille: int = monthly.HOLDOUT_PERMILLE,
    update_permille: int = upsert.UPDATE_PERMILLE,
    retain_last: int = 2,
    compact_options: dict[str, str] | None = None,
    run_id: str | None = None,
    protected_tables=PROTECTED_TABLES,
    disk_path: str | Path = ".",
) -> dict:
    """Run one scenario step and return ``{record, protected, disk_avail_gb}``.

    Does NOT append the record and does NOT exit — see :func:`run_and_record`.
    """
    if step not in STEPS:
        raise ValueError(f"unknown step {step!r}; expected one of {STEPS}")
    require_ops_table(table)

    rid, rts = _resolve_run_id(run_id)
    protected_before = current_snapshot_ids(warehouse, protected_tables)
    before = _table_state(spark, warehouse, table)
    start = time.perf_counter()

    params: dict = {
        "source": source,
        "warehouse": str(warehouse),
        "identity_cols": list(monthly.IDENTITY_COLS),
        "hash_modulus": monthly.HASH_MODULUS,
    }
    extras: dict = {}

    if step == "backfill":
        m = monthly.create_backfill(
            spark,
            source=source,
            table=table,
            backfill_end=backfill_end,
            late_window_start=late_window_start,
            holdout_permille=holdout_permille,
        )
        params |= {
            "partition_transform": m["partition_transform"],
            "backfill_end": m["backfill_end"],
            "late_window_start": m["late_window_start"],
            "holdout_permille": m["holdout_permille"],
            "backfill_predicate": m["backfill_predicate"],
            "late_window_predicate": m["late_window_predicate"],
            "holdout_predicate": m["holdout_predicate"],
        }
        extras = {
            "source_rows": m["source_rows"],
            "late_window_source_rows": m["late_window_source_rows"],
            "holdout_rows": m["holdout_rows"],
            "rows_written": m["rows_written"],
            "reconciles_with_source": m["reconciles_with_source"],
            "holdout_is_subset_of_late_window": m["holdout_is_subset_of_late_window"],
        }

    elif step == "append":
        if not month:
            raise ValueError("--month YYYY-MM is required for the append step")
        m = monthly.append_month(
            spark, table=table, month=month, source=source, backfill_end=backfill_end
        )
        params |= {"month": m["month"], "month_predicate": m["month_predicate"]}
        extras = {
            "month": m["month"],
            "month_source_rows": m["month_source_rows"],
            "rows_written": m["rows_written"],
            "reconciles_with_source": m["reconciles_with_source"],
        }

    elif step == "upsert":
        m = upsert.late_data_merge(
            spark,
            table=table,
            source=source,
            holdout_permille=holdout_permille,
            update_permille=update_permille,
            late_window_start=late_window_start,
            backfill_end=backfill_end,
        )
        params |= {
            "holdout_permille": m["holdout_permille"],
            "update_permille": m["update_permille"],
            "insert_predicate": m["insert_predicate"],
            "update_predicate": m["update_predicate"],
            "update_mutation": m["update_mutation"],
            "merge_condition": m["merge_condition"],
            "merge_sql": m["merge_sql"],
        }
        extras = {
            "matched_updated": m["matched_updated"],
            "inserted": m["inserted"],
            "post_merge_total": m["post_merge_total"],
            "insert_count_reconciles": m["insert_count_reconciles"],
            "backfill_window_rows": m["backfill_window_rows"],
            "source_backfill_window_rows": m["source_backfill_window_rows"],
            "reconciles_with_source": m["reconciles_with_source"],
        }

    elif step == "compact":
        m = maintenance.compact(spark, table, options=compact_options)
        params |= {"procedure": m["procedure"], "call": m["call"], "options": m["options"]}
        extras = {
            "rewritten_files": m["rewritten_files"],
            "added_files": m["added_files"],
            "rewritten_bytes": m["rewritten_bytes"],
            "failed_files": m["failed_files"],
        }

    else:  # expire
        m = maintenance.expire(spark, table, retain_last=retain_last)
        params |= {
            "procedure": m["procedure"],
            "call": m["call"],
            "retain_last": m["retain_last"],
            "older_than": m["older_than"],
        }
        extras = {
            "deleted_data_files": m["deleted_data_files"],
            "deleted_manifest_files": m["deleted_manifest_files"],
            "deleted_manifest_lists": m["deleted_manifest_lists"],
            "deleted_position_delete_files": m["deleted_position_delete_files"],
            "deleted_equality_delete_files": m["deleted_equality_delete_files"],
            "deleted_statistics_files": m["deleted_statistics_files"],
        }

    wall = round(time.perf_counter() - start, 3)
    after = _table_state(spark, warehouse, table)
    git = runlog.git_info()

    record = {
        "schema_version": runlog.record_schema_version,
        "kind": "ops",
        "run_id": rid,
        "run_ts": rts,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "scenario": step,
        "table": table,
        "params": params,
        "snapshot_before": before["snapshot"],
        "snapshot_after": after["snapshot"],
        "snapshot_count_before": before["snapshot_count"],
        "snapshot_count_after": after["snapshot_count"],
        "snapshot_summary_after": (
            snapshot_summary(warehouse, table, after["snapshot"])
            if after["snapshot"] is not None
            else {}
        ),
        "rows_before": before["rows"],
        "rows_after": after["rows"],
        "files_before": before["files"],
        "files_after": after["files"],
        "bytes_before": before["bytes"],
        "bytes_after": after["bytes"],
        "avg_file_mb_before": before["avg_file_mb"],
        "avg_file_mb_after": after["avg_file_mb"],
        "protected_snapshots": protected_before,
        "wall_clock_s": wall,
        "disk_avail_gb": disk_avail_gb(disk_path),
        "hardware": runlog.hardware_string(),
    }
    record.update(extras)

    return {
        "record": record,
        "protected_before": protected_before,
        "warehouse": str(warehouse),
    }


def run_and_record(
    spark,
    step: str,
    *,
    results: str | Path = DEFAULT_RESULTS,
    min_disk_gb: float = MIN_DISK_GB,
    **kwargs,
) -> dict:
    """:func:`run_step` + append the record + run both epilogue checks.

    Returns ``{record, exit_code}``. The disk floor is reported as a non-zero
    exit code rather than an exception so the record is always durable first.
    """
    out = run_step(spark, step, **kwargs)
    record = out["record"]
    runlog.append_record(record, results)

    # Epilogue 1: published tables must not have moved (raises if they did).
    check_protected(out["warehouse"], out["protected_before"])

    # Epilogue 2: disk floor.
    exit_code = 0
    avail = record["disk_avail_gb"]
    if avail < min_disk_gb:
        print(
            f"CRITICAL: only {avail:.2f} GB available (floor {min_disk_gb:.0f} GB). "
            "Stop the ops phase and free space before running another step — "
            "compaction writes new files before the old ones are deleted.",
            file=sys.stderr,
        )
        exit_code = 2
    return {"record": record, "exit_code": exit_code}


# --- teardown -----------------------------------------------------------------


def remove_ops_namespace_dir(
    warehouse: str | Path, namespace: str = "ops"
) -> tuple[Path, bool]:
    """``rmtree`` of ``<warehouse>/ops`` — and of nothing else, ever.

    Three independent conditions must hold: the namespace is literally ``ops``,
    the resolved target's parent is exactly the resolved warehouse root (so
    ``..`` or an absolute path cannot escape), and the target is a directory.
    """
    if namespace != "ops":
        raise ValueError(
            f"Refusing to remove {namespace!r}: only the 'ops' namespace directory "
            "may be deleted."
        )
    warehouse_path = Path(warehouse).resolve()
    target = (warehouse_path / namespace).resolve()
    if target.parent != warehouse_path:
        raise ValueError(f"Refusing to remove {target}: not directly under {warehouse_path}")
    if not target.is_dir():
        return target, False
    shutil.rmtree(target)
    return target, True


def clean(
    spark,
    table: str = monthly.OPS_MONTHLY,
    warehouse: str | Path = DEFAULT_WAREHOUSE,
    remove_dir: bool = True,
) -> dict:
    """Drop the ops table (PURGE) and remove ``<warehouse>/ops``. No record."""
    require_ops_table(table)
    dropped = maintenance.drop_ops_table(spark, table)
    namespace = table.split(".")[1]
    if remove_dir:
        namespace_dir, removed = remove_ops_namespace_dir(warehouse, namespace)
    else:
        namespace_dir, removed = Path(warehouse).resolve() / namespace, False
    return {**dropped, "namespace_dir": str(namespace_dir), "dir_removed": removed}


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.ops.run_scenario")
    parser.add_argument("--step", required=True, choices=[*STEPS, "clean"])
    parser.add_argument("--month", default=None, help="YYYY-MM (append step)")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--warehouse", default=DEFAULT_WAREHOUSE)
    parser.add_argument("--table", default=monthly.OPS_MONTHLY)
    parser.add_argument("--source", default=monthly.SILVER_INTERACTIONS)
    parser.add_argument("--backfill-end", default=monthly.BACKFILL_END)
    parser.add_argument("--late-window-start", default=monthly.LATE_WINDOW_START)
    parser.add_argument("--holdout-permille", type=int, default=monthly.HOLDOUT_PERMILLE)
    parser.add_argument("--update-permille", type=int, default=upsert.UPDATE_PERMILLE)
    parser.add_argument("--retain-last", type=int, default=2)
    parser.add_argument("--min-input-files", default=None,
                        help="rewrite_data_files option (compact step)")
    parser.add_argument("--min-disk-gb", type=float, default=MIN_DISK_GB)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    args = parser.parse_args(argv)

    spark = get_spark(
        app_name=f"ops-{args.step}",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        if args.step == "clean":
            summary = clean(spark, table=args.table, warehouse=args.warehouse)
            print(json.dumps(summary))
            return 0

        compact_options = (
            {"min-input-files": str(args.min_input_files)}
            if args.min_input_files
            else None
        )
        out = run_and_record(
            spark,
            args.step,
            results=args.results,
            min_disk_gb=args.min_disk_gb,
            warehouse=args.warehouse,
            table=args.table,
            source=args.source,
            month=args.month,
            backfill_end=args.backfill_end,
            late_window_start=args.late_window_start,
            holdout_permille=args.holdout_permille,
            update_permille=args.update_permille,
            retain_last=args.retain_last,
            compact_options=compact_options,
            run_id=args.run_id,
        )
    finally:
        spark.stop()

    # Summary JSON MUST be the last stdout line (repo convention).
    print(json.dumps(out["record"]))
    return out["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
