"""Monthly-partitioned ops table: backfill + incremental append (Phase 5, T20).

``local.ops.interactions_monthly`` is a **disposable copy** of
``local.silver.interactions``, partitioned by Iceberg hidden partitioning
``months(ts)``. It exists so the lakehouse-ops exhibits (incremental append,
late-data MERGE, compaction, snapshot expiry) can be demonstrated on a real
43M-row table without ever touching the published tables that recorded eval
records cite.

Source columns (verified against ``features/silver.py``'s
``SILVER_INTERACTION_COLS`` and the live table): ``user_id``, ``parent_asin``,
``asin``, ``rating``, ``ts``, ``helpful_vote``, ``verified_purchase``. The
identity of an interaction is ``(user_id, parent_asin, ts)`` — silver's D2 dedup
already collapses to one row per ``(user_id, parent_asin)``, so this triple is
unique and is used as the MERGE key in :mod:`~batch_recsys_lab.ops.upsert`.

Late-arrival holdout
--------------------
The backfill deliberately omits a deterministic slice of the last two months of
its window so the T22 upsert scenario has genuine *inserts* to land, not just
synthetic rows. The predicate is a pure function of the row's identity, so it is
reproducible without a seed file:

    (ts >= TIMESTAMP '2023-05-01 00:00:00' AND ts < TIMESTAMP '2023-07-01 00:00:00')
    AND pmod(xxhash64(user_id, parent_asin, ts), 1000) < <holdout_permille>

Sizing (``docs/volume_by_month.md``, silver.interactions): 2023-05 = 144,889 and
2023-06 = 93,827 rows, i.e. 238,716 rows in the late window. At the default
50‰ the holdout is ≈11.9k rows — large enough to be a visible batch, small
enough to remain a plausible late-arrival share. 20‰ would give only ≈4.8k.

Date semantics: ``backfill_end`` is an INCLUSIVE calendar date, applied as the
exclusive timestamp bound ``ts < <backfill_end + 1 day> 00:00:00`` so the whole
of the final day is included (``ts <= TIMESTAMP '2023-06-30'`` would silently
drop all but the first instant of 2023-06-30).

Fragmentation (T23b)
--------------------
:func:`fragment_month` exists because the compaction exhibit had nothing to
compact: the backfill writes each ``months(ts)`` partition in one shot, so the
43.4M-row table carried exactly ONE data file per partition and
``rewrite_data_files`` was a measured no-op. Fragmentation simulates what a real
micro-batch ingestion pipeline produces — one small file per arrival batch — by
deleting one month and re-appending exactly the same rows one calendar day at a
time. It is content-preserving by construction: the row set is a copy of the
month's own pre-delete content, not a re-derivation from ``source``, so it
survives the T22 MERGE's rating updates and the holdout inserts without having
to replay their predicates.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from batch_recsys_lab.ops.maintenance import (
    ensure_ops_namespace,
    require_ops_table,
    table_exists,
)
from batch_recsys_lab.ops.snapshot_metrics import files_stats

SILVER_INTERACTIONS = "local.silver.interactions"
OPS_MONTHLY = "local.ops.interactions_monthly"

# Verified silver.interactions column names (see module docstring).
USER_COL = "user_id"
ITEM_COL = "parent_asin"
TS_COL = "ts"
IDENTITY_COLS = (USER_COL, ITEM_COL, TS_COL)

PARTITION_TRANSFORM = f"months({TS_COL})"

# Backfill window end (INCLUSIVE date) and the 2-month late-arrival window.
BACKFILL_END = "2023-06-30"
LATE_WINDOW_START = "2023-05-01"

# Deterministic holdout share, in parts per thousand (see module docstring).
HOLDOUT_PERMILLE = 50
HASH_MODULUS = 1000

# Fragmentation (T23b): the design string recorded verbatim in the ops record,
# and the suffix of the durable scratch copy the day slices are read back from.
FRAGMENT_DESIGN = (
    "delete month, re-append in daily slices from a materialized scratch copy "
    "— simulated micro-batch ingestion for the compaction exhibit"
)
FRAGMENT_SCRATCH_SUFFIX = "__fragment_scratch"
FRAGMENT_SLICE_GRANULARITY = "day"


# --- predicate builders (the strings recorded verbatim in the ops record) -----


def _ts_literal(day: str | date) -> str:
    d = day if isinstance(day, date) else date.fromisoformat(str(day))
    return f"TIMESTAMP '{d.isoformat()} 00:00:00'"


def _exclusive_end(inclusive_day: str | date) -> date:
    d = (
        inclusive_day
        if isinstance(inclusive_day, date)
        else date.fromisoformat(str(inclusive_day))
    )
    return d + timedelta(days=1)


def backfill_predicate(backfill_end: str = BACKFILL_END) -> str:
    """``ts`` on or before ``backfill_end`` (whole day inclusive)."""
    return f"{TS_COL} < {_ts_literal(_exclusive_end(backfill_end))}"


def late_window_predicate(
    late_window_start: str = LATE_WINDOW_START, backfill_end: str = BACKFILL_END
) -> str:
    """The 2-month window the holdout is drawn from."""
    return (
        f"{TS_COL} >= {_ts_literal(late_window_start)} "
        f"AND {TS_COL} < {_ts_literal(_exclusive_end(backfill_end))}"
    )


def holdout_predicate(
    holdout_permille: int = HOLDOUT_PERMILLE,
    late_window_start: str = LATE_WINDOW_START,
    backfill_end: str = BACKFILL_END,
) -> str:
    """Rows withheld from the backfill (they arrive later, via MERGE)."""
    return (
        f"({late_window_predicate(late_window_start, backfill_end)}) "
        f"AND pmod(xxhash64({', '.join(IDENTITY_COLS)}), {HASH_MODULUS}) "
        f"< {int(holdout_permille)}"
    )


def _month_bounds(month: str) -> tuple[date, date]:
    """``("2023-06")`` -> ``(2023-06-01, 2023-07-01)`` (end exclusive)."""
    start = datetime.strptime(month, "%Y-%m").date()
    end = date(start.year + (start.month // 12), (start.month % 12) + 1, 1)
    return start, end


def month_predicate(month: str) -> str:
    """``ts`` inside calendar month ``YYYY-MM`` — aligned to the ``months(ts)``
    partition boundary, so a DELETE with it is a whole-partition drop."""
    start, end = _month_bounds(month)
    return f"{TS_COL} >= {_ts_literal(start)} AND {TS_COL} < {_ts_literal(end)}"


def day_predicate(day: date) -> str:
    """``ts`` inside one calendar day (the fragmentation slice predicate)."""
    return (
        f"{TS_COL} >= {_ts_literal(day)} "
        f"AND {TS_COL} < {_ts_literal(day + timedelta(days=1))}"
    )


def month_days(month: str) -> list[date]:
    """Every calendar day of ``YYYY-MM``, ascending."""
    start, end = _month_bounds(month)
    days, cur = [], start
    while cur < end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


# --- helpers ------------------------------------------------------------------


def _check_source_columns(df: DataFrame) -> None:
    missing = [c for c in IDENTITY_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"source table is missing expected column(s) {missing}; "
            f"batch_recsys_lab.ops assumes silver.interactions columns "
            f"{IDENTITY_COLS} (see module docstring)"
        )


def _create_partitioned(spark: SparkSession, table: str, schema_ddl: str) -> None:
    """Create the ops table with an explicit ``months(ts)`` partition spec."""
    require_ops_table(table)
    ensure_ops_namespace(spark, table.rsplit(".", 1)[0])
    spark.sql(f"DROP TABLE IF EXISTS {table} PURGE")
    spark.sql(
        f"CREATE TABLE {table} ({schema_ddl}) USING iceberg "
        f"PARTITIONED BY ({PARTITION_TRANSFORM})"
    )


def partition_transforms(spark: SparkSession, table: str) -> list[str]:
    """Partition-spec transforms of ``table`` as ``SHOW CREATE TABLE`` renders
    them, e.g. ``["months(ts)"]``. Splits on TOP-LEVEL commas only, so
    multi-argument transforms (``bucket(16, user_id)``) stay intact."""
    ddl = spark.sql(f"SHOW CREATE TABLE {table}").collect()[0][0]
    marker = "PARTITIONED BY ("
    if marker not in ddl:
        return []
    rest = ddl.split(marker, 1)[1]
    depth, end = 1, len(rest)
    for i, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    body, out, depth, buf = rest[:end], [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf.strip())
    return out


# --- backfill -----------------------------------------------------------------


def create_backfill(
    spark: SparkSession,
    source: str = SILVER_INTERACTIONS,
    table: str = OPS_MONTHLY,
    backfill_end: str = BACKFILL_END,
    late_window_start: str = LATE_WINDOW_START,
    holdout_permille: int = HOLDOUT_PERMILLE,
) -> dict:
    """Create ``table`` partitioned by ``months(ts)`` and load the backfill.

    Content = every ``source`` row with ``ts`` on or before ``backfill_end``,
    MINUS the deterministic late-arrival holdout over the final two months.
    Reconciliation (asserted in the returned dict, not raised): the backfill and
    the holdout exactly partition the source slice.
    """
    require_ops_table(table)
    start = time.perf_counter()

    src = spark.table(source)
    _check_source_columns(src)
    cols = src.columns

    window_pred = backfill_predicate(backfill_end)
    late_pred = late_window_predicate(late_window_start, backfill_end)
    hold_pred = holdout_predicate(holdout_permille, late_window_start, backfill_end)

    slice_df = src.where(window_pred)
    source_rows = slice_df.count()
    late_window_source_rows = slice_df.where(late_pred).count()
    holdout_rows = slice_df.where(hold_pred).count()

    landed = slice_df.where(f"NOT ({hold_pred})").select(*cols)

    _create_partitioned(spark, table, landed.schema.toDDL())
    landed.writeTo(table).append()
    rows_written = spark.table(table).count()

    return {
        "source": source,
        "table": table,
        "partition_transform": PARTITION_TRANSFORM,
        "backfill_end": backfill_end,
        "late_window_start": late_window_start,
        "holdout_permille": int(holdout_permille),
        "hash_modulus": HASH_MODULUS,
        "identity_cols": list(IDENTITY_COLS),
        "backfill_predicate": window_pred,
        "late_window_predicate": late_pred,
        "holdout_predicate": hold_pred,
        "source_rows": int(source_rows),
        "late_window_source_rows": int(late_window_source_rows),
        "holdout_rows": int(holdout_rows),
        "rows_written": int(rows_written),
        "reconciles_with_source": bool(source_rows == rows_written + holdout_rows),
        "holdout_is_subset_of_late_window": bool(
            holdout_rows <= late_window_source_rows
        ),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }


# --- incremental append -------------------------------------------------------


def append_month(
    spark: SparkSession,
    table: str = OPS_MONTHLY,
    month: str = "2023-07",
    source: str = SILVER_INTERACTIONS,
    backfill_end: str = BACKFILL_END,
) -> dict:
    """Append one calendar month of ``source`` rows to the ops table.

    Months strictly after ``backfill_end`` carry no holdout: the whole month is
    appended, so ``rows_written == month_source_rows`` exactly.
    """
    require_ops_table(table)
    start = time.perf_counter()
    if not table_exists(spark, table):
        raise RuntimeError(
            f"{table} does not exist — run the backfill step before appending months."
        )

    month_start = datetime.strptime(month, "%Y-%m").date()
    if month_start < _exclusive_end(backfill_end):
        raise ValueError(
            f"month {month} is inside the backfill window (ends {backfill_end}); "
            "appending it would duplicate already-landed rows."
        )

    src = spark.table(source)
    _check_source_columns(src)
    pred = month_predicate(month)
    batch = src.where(pred).select(*spark.table(table).columns)
    month_source_rows = batch.count()

    rows_before = spark.table(table).count()
    batch.writeTo(table).append()
    rows_after = spark.table(table).count()

    return {
        "source": source,
        "table": table,
        "month": month,
        "month_predicate": pred,
        "month_source_rows": int(month_source_rows),
        "rows_written": int(rows_after - rows_before),
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "reconciles_with_source": bool(rows_after - rows_before == month_source_rows),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }


# --- fragmentation (T23b) -----------------------------------------------------


def scratch_table_name(table: str) -> str:
    """Name of the durable scratch copy used to fragment ``table``.

    Derived from the target so two ops tables can be fragmented independently,
    and guarded like every other name this package writes.
    """
    return require_ops_table(require_ops_table(table) + FRAGMENT_SCRATCH_SUFFIX)


def fragment_month(
    spark: SparkSession,
    table: str = OPS_MONTHLY,
    source: str = SILVER_INTERACTIONS,
    month: str = "2023-06",
    scratch_table: str | None = None,
) -> dict:
    """Rewrite one month of ``table`` as one small data file per calendar day.

    Procedure (each step is content-preserving, and the two row-count identities
    are re-checked before anything is dropped):

    1. **Materialize** the month's CURRENT rows into a durable Iceberg scratch
       table (``<table>__fragment_scratch``), and verify the copy is complete.
       Durable, not ``DataFrame.cache()``: a cached DataFrame is a best-effort
       accelerator, and its lineage — a scan of the very rows step 2 deletes —
       is what Spark recomputes if a block is evicted. Losing that race would
       silently drop the month.
    2. **Delete** the month with the partition-aligned
       :func:`month_predicate`, i.e. a whole-``months(ts)``-partition drop.
    3. **Re-append** the scratch rows one calendar day at a time, each slice
       ``repartition(1)``-ed so it lands as exactly one data file. Days with no
       rows are skipped (an empty append would still be a snapshot, and could
       write an empty file).
    4. **Verify** the total row count and the month's row count are both back to
       their pre-delete values, then drop the scratch table. On failure the
       scratch table is deliberately LEFT IN PLACE — it is the only remaining
       copy of the month.

    ``source`` is accepted for call-signature symmetry with the other steps and
    is recorded for provenance; it is NOT read. Re-deriving the month from
    ``source`` would have to replay the backfill holdout AND the T22 MERGE's
    ``rating := 5.0`` updates to reproduce post-merge state — error-prone in a
    way that copying the table's own rows is not.
    """
    require_ops_table(table)
    scratch = scratch_table_name(table) if scratch_table is None else scratch_table
    require_ops_table(scratch)
    if scratch == table:
        raise ValueError(f"scratch table must differ from the target table ({table!r})")

    start = time.perf_counter()
    if not table_exists(spark, table):
        raise RuntimeError(
            f"{table} does not exist — run the backfill step before fragmenting."
        )

    pred = month_predicate(month)
    cols = spark.table(table).columns
    rows_before_total = spark.table(table).count()
    rows_month = spark.table(table).where(pred).count()
    if rows_month == 0:
        raise ValueError(
            f"{table} has no rows in {month} ({pred}) — nothing to fragment."
        )
    files_before = files_stats(spark, table)["file_count"]

    # 1. durable scratch copy, verified complete BEFORE the delete.
    ensure_ops_namespace(spark, scratch.rsplit(".", 1)[0])
    spark.sql(f"DROP TABLE IF EXISTS {scratch} PURGE")
    spark.table(table).where(pred).select(*cols).writeTo(scratch).create()
    scratch_rows = spark.table(scratch).count()
    if scratch_rows != rows_month:
        raise RuntimeError(
            f"fragment aborted BEFORE deleting anything: scratch copy of {month} has "
            f"{scratch_rows} rows, expected {rows_month}."
        )

    # Slice sizes come from one pass over the scratch copy; empty days are skipped.
    day_counts = {
        r["d"]: int(r["n"])
        for r in spark.table(scratch)
        .groupBy(F.to_date(F.col(TS_COL)).alias("d"))
        .count()
        .withColumnRenamed("count", "n")
        .collect()
    }
    days = [d for d in month_days(month) if day_counts.get(d, 0) > 0]
    slice_rows = [day_counts[d] for d in days]
    if sum(slice_rows) != rows_month:
        raise RuntimeError(
            f"fragment aborted BEFORE deleting anything: day slices cover "
            f"{sum(slice_rows)} rows, expected {rows_month} — a row's ts falls "
            f"outside the calendar days of {month}."
        )

    # 2. partition-aligned delete.
    delete_sql = f"DELETE FROM {table} WHERE {pred}"
    spark.sql(delete_sql)
    rows_after_delete = spark.table(table).count()
    if rows_after_delete != rows_before_total - rows_month:
        raise RuntimeError(
            f"DELETE removed {rows_before_total - rows_after_delete} rows, expected "
            f"{rows_month}. The month's rows are preserved in {scratch}."
        )
    files_after_delete = files_stats(spark, table)["file_count"]

    # 3. one append (= one data file) per non-empty calendar day.
    scratch_df = spark.table(scratch)
    for day in days:
        (
            scratch_df.where(day_predicate(day))
            .select(*cols)
            .repartition(1)
            .writeTo(table)
            .append()
        )

    # 4. verify, then drop the scratch copy.
    rows_after_total = spark.table(table).count()
    rows_month_after = spark.table(table).where(pred).count()
    if rows_after_total != rows_before_total:
        raise RuntimeError(
            f"fragment changed the table row count: {rows_before_total} -> "
            f"{rows_after_total}. The month's rows are preserved in {scratch}."
        )
    if rows_month_after != rows_month:
        raise RuntimeError(
            f"fragment changed {month}'s row count: {rows_month} -> {rows_month_after}. "
            f"The month's rows are preserved in {scratch}."
        )
    files_after = files_stats(spark, table)["file_count"]
    spark.sql(f"DROP TABLE IF EXISTS {scratch} PURGE")

    files_added = files_after - files_after_delete
    return {
        "source": source,
        "table": table,
        "month": month,
        "month_predicate": pred,
        "delete_sql": delete_sql,
        "scratch_table": scratch,
        "fragmentation_design": FRAGMENT_DESIGN,
        "slice_granularity": FRAGMENT_SLICE_GRANULARITY,
        "rows_month": int(rows_month),
        "n_slices": len(days),
        "slice_rows_min": int(min(slice_rows)),
        "slice_rows_max": int(max(slice_rows)),
        "days_in_month": len(month_days(month)),
        "files_added": int(files_added),
        "files_before": int(files_before),
        "files_after_delete": int(files_after_delete),
        "files_after": int(files_after),
        "one_file_per_slice": bool(files_added == len(days)),
        "rows_before_total": int(rows_before_total),
        "rows_after_total": int(rows_after_total),
        "rows_preserved": bool(rows_after_total == rows_before_total),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }
