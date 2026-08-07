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
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

from pyspark.sql import DataFrame, SparkSession

from batch_recsys_lab.ops.maintenance import (
    ensure_ops_namespace,
    require_ops_table,
    table_exists,
)

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


def month_predicate(month: str) -> str:
    """``ts`` inside calendar month ``YYYY-MM``."""
    start = datetime.strptime(month, "%Y-%m").date()
    end = date(start.year + (start.month // 12), (start.month % 12) + 1, 1)
    return f"{TS_COL} >= {_ts_literal(start)} AND {TS_COL} < {_ts_literal(end)}"


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
