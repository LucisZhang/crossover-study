"""Contract engine: gate + audit + dq_results sink (Phase 1, T1; UPGRADE_PLAN.md §8).

Two entry points:

* :func:`gate` — row-level split of an *unpublished* DataFrame into a kept frame
  and a quarantine frame, in a single pass. Every row with any quarantine-action
  violation goes to quarantine exactly once, annotated with the ordered list of
  reasons it violated and its ``primary_reason`` (the first violated check in the
  contract's declared order — that order IS the fixed priority, D5).
* :func:`audit` — table-level aggregates against a *published* table, returning
  :class:`DqResult` rows: schema/type/nullability conformance, ``no_all_null``,
  ``no_control_chars``, ``orphan_rate``, ``unknown_share``, and a zero-violation
  re-assertion of the row-level quarantine rules (they must hold post-publish).

No collects on big data: gate computes all its counts in one aggregation; audit
computes all table-local aggregates in one aggregation (orphan_rate needs a join).
Only the single one-row aggregate results are collected to the driver.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import reduce

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from batch_recsys_lab.contracts.checks import (
    ROW_LEVEL_KINDS,
    no_all_null_nonnull_exprs,
    orphan_stats,
    row_violation_column,
    unknown_share_agg_expr,
)
from batch_recsys_lab.contracts.loader import Contract

REASONS_COL = "violation_reasons"
PRIMARY_COL = "primary_reason"
DQ_RESULTS_TABLE = "local.dq.dq_results"

# Contract dtype spelling → Spark simpleString (only where they differ).
_DTYPE_TO_SPARK = {"long": "bigint"}


@dataclass(frozen=True)
class GateResult:
    """Outcome of :func:`gate`.

    ``kept_df`` carries only the input columns; ``quarantine_df`` carries the
    input columns plus ``violation_reasons`` (array<string>, declared order) and
    ``primary_reason`` (string, the first). ``violation_counts`` is per
    quarantine check_id. Row counts are materialized (one aggregation).
    """

    kept_df: DataFrame
    quarantine_df: DataFrame
    violation_counts: dict[str, int]
    total_rows: int
    quarantined_rows: int


@dataclass(frozen=True)
class DqResult:
    """One row of the ``local.dq.dq_results`` ledger."""

    run_id: str
    run_ts: str
    table_name: str
    contract_name: str
    contract_version: int
    check_id: str
    check_kind: str
    column: str | None
    status: str  # pass | fail | measured
    violation_count: int
    total_rows: int
    metric_value: float | None
    details: str  # JSON string


# --- gate --------------------------------------------------------------------


def gate(df: DataFrame, contract: Contract) -> GateResult:
    """Split ``df`` into kept / quarantine per the contract's quarantine checks."""
    column_types = {cs.name: cs.dtype for cs in contract.columns}
    q_checks = [
        c for c in contract.checks if c.action == "quarantine" and c.kind in ROW_LEVEL_KINDS
    ]

    if not q_checks:
        empty_q = (
            df.where(F.lit(False))
            .withColumn(REASONS_COL, F.array().cast("array<string>"))
            .withColumn(PRIMARY_COL, F.lit(None).cast("string"))
        )
        return GateResult(
            kept_df=df,
            quarantine_df=empty_q,
            violation_counts={},
            total_rows=df.count(),
            quarantined_rows=0,
        )

    # One coalesced boolean per check, in declared order.
    viol = [
        (c.check_id, F.coalesce(row_violation_column(c, column_types), F.lit(False)))
        for c in q_checks
    ]
    any_viol = reduce(lambda a, b: a | b, (v for _, v in viol))

    # Reasons array in declared order, nulls (non-violations) filtered out.
    reason_elems = [F.when(v, F.lit(cid)) for cid, v in viol]
    reasons = F.filter(F.array(*reason_elems), lambda x: x.isNotNull())
    primary = F.element_at(reasons, 1)

    quarantine_df = (
        df.where(any_viol).withColumn(REASONS_COL, reasons).withColumn(PRIMARY_COL, primary)
    )
    kept_df = df.where(~any_viol)

    # All counts in a single aggregation pass.
    agg_exprs = [
        F.count(F.lit(1)).alias("__total"),
        F.sum(F.when(any_viol, F.lit(1)).otherwise(F.lit(0))).alias("__quarantined"),
    ]
    agg_exprs += [F.sum(F.when(v, F.lit(1)).otherwise(F.lit(0))).alias(cid) for cid, v in viol]
    row = df.agg(*agg_exprs).collect()[0]

    counts = {cid: int(row[cid] or 0) for cid, _ in viol}
    return GateResult(
        kept_df=kept_df,
        quarantine_df=quarantine_df,
        violation_counts=counts,
        total_rows=int(row["__total"] or 0),
        quarantined_rows=int(row["__quarantined"] or 0),
    )


# --- audit -------------------------------------------------------------------


def audit(
    spark: SparkSession,
    contract: Contract,
    table_name: str | None = None,
    run_id: str | None = None,
) -> list[DqResult]:
    """Run table-level checks against a published table → list of DqResult."""
    table = table_name or contract.table
    df = spark.table(table)
    column_types = {cs.name: cs.dtype for cs in contract.columns}
    rid, rts = _resolve_run_id(run_id)
    total = df.count()

    def make(check_id, kind, column, status, count, metric, details) -> DqResult:
        return DqResult(
            run_id=rid,
            run_ts=rts,
            table_name=table,
            contract_name=contract.name,
            contract_version=contract.version,
            check_id=check_id,
            check_kind=kind,
            column=column,
            status=status,
            violation_count=int(count),
            total_rows=total,
            metric_value=(None if metric is None else float(metric)),
            details=json.dumps(details, default=str),
        )

    results: list[DqResult] = list(_schema_conformance(df, contract, make))

    # Single aggregation for every table-local statistic.
    agg_exprs: list[Column] = []
    has_no_all_null = any(c.kind == "no_all_null" for c in contract.checks)
    if has_no_all_null:
        agg_exprs += no_all_null_nonnull_exprs([cs.name for cs in contract.columns])
    for chk in contract.checks:
        if chk.kind in ROW_LEVEL_KINDS:
            agg_exprs.append(
                F.sum(row_violation_column(chk, column_types).cast("long")).alias(f"v__{chk.check_id}")
            )
        elif chk.kind == "unknown_share":
            agg_exprs.append(unknown_share_agg_expr(chk).alias(f"u__{chk.check_id}"))
    row = df.agg(*agg_exprs).collect()[0] if agg_exprs else None

    for chk in contract.checks:
        column = chk.columns[0] if len(chk.columns) == 1 else None

        if chk.kind == "no_all_null":
            dead = [cs.name for cs in contract.columns if int(row[f"nonnull__{cs.name}"] or 0) == 0]
            results.append(
                make(
                    chk.check_id,
                    chk.kind,
                    None,
                    "fail" if dead else "pass",
                    len(dead),
                    None,
                    {"dead_columns": dead},
                )
            )

        elif chk.kind == "orphan_rate":
            orphans, denom = orphan_stats(spark, df, chk)
            rate = (orphans / denom) if denom else 0.0
            results.append(
                make(
                    chk.check_id,
                    chk.kind,
                    column,
                    "measured",
                    orphans,
                    rate,
                    {"ref_table": chk.ref_table, "ref_column": chk.ref_column, "denominator": denom},
                )
            )

        elif chk.kind == "unknown_share":
            count = int(row[f"u__{chk.check_id}"] or 0)
            share = (count / total) if total else 0.0
            results.append(
                make(
                    chk.check_id,
                    chk.kind,
                    column,
                    "measured",
                    count,
                    share,
                    {"sentinel": chk.value},
                )
            )

        elif chk.kind == "no_control_chars":
            count = int(row[f"v__{chk.check_id}"] or 0)
            status = _row_status(chk.action, count)
            results.append(
                make(chk.check_id, chk.kind, column, status, count, None, {"columns": list(chk.columns)})
            )

        elif chk.kind in ROW_LEVEL_KINDS:
            # Zero-violation re-assertion: these quarantine rules must hold on the
            # published table (the offenders were already routed away).
            count = int(row[f"v__{chk.check_id}"] or 0)
            status = _row_status(chk.action, count)
            results.append(
                make(chk.check_id, chk.kind, column, status, count, None, {"columns": list(chk.columns)})
            )

    return results


def _row_status(action: str, count: int) -> str:
    if action == "measure":
        return "measured"
    return "pass" if count == 0 else "fail"


def _schema_conformance(df: DataFrame, contract: Contract, make) -> list[DqResult]:
    """One DqResult per expected column: presence / dtype / nullability match."""
    actual = {f.name: f for f in df.schema.fields}
    out: list[DqResult] = []
    for cs in contract.columns:
        field = actual.get(cs.name)
        if field is None:
            status, details = "fail", {"reason": "missing_column"}
        else:
            expected_dtype = _DTYPE_TO_SPARK.get(cs.dtype, cs.dtype)
            actual_dtype = field.dataType.simpleString()
            dtype_ok = actual_dtype == expected_dtype
            null_ok = field.nullable == cs.nullable
            status = "pass" if (dtype_ok and null_ok) else "fail"
            details = {
                "expected_dtype": expected_dtype,
                "actual_dtype": actual_dtype,
                "expected_nullable": cs.nullable,
                "actual_nullable": field.nullable,
            }
        out.append(
            make(f"schema__{cs.name}", "schema_conformance", cs.name, status, 0, None, details)
        )
    return out


# --- dq_results sink ---------------------------------------------------------

DQ_RESULTS_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), False),
        StructField("run_ts", StringType(), False),
        StructField("table_name", StringType(), False),
        StructField("contract_name", StringType(), False),
        StructField("contract_version", LongType(), False),
        StructField("check_id", StringType(), False),
        StructField("check_kind", StringType(), False),
        StructField("column", StringType(), True),
        StructField("status", StringType(), False),
        StructField("violation_count", LongType(), False),
        StructField("total_rows", LongType(), False),
        StructField("metric_value", DoubleType(), True),
        StructField("details", StringType(), True),
    ]
)


def write_dq_results(
    spark: SparkSession,
    results: list[DqResult],
    table: str = DQ_RESULTS_TABLE,
) -> None:
    """Append ``results`` to the append-only dq_results table (created if absent)."""
    if not results:
        return
    namespace = table.rsplit(".", 1)[0]
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")

    rows = [
        (
            r.run_id,
            r.run_ts,
            r.table_name,
            r.contract_name,
            int(r.contract_version),
            r.check_id,
            r.check_kind,
            r.column,
            r.status,
            int(r.violation_count),
            int(r.total_rows),
            r.metric_value,
            r.details,
        )
        for r in results
    ]
    sdf = spark.createDataFrame(rows, DQ_RESULTS_SCHEMA)
    if spark.catalog.tableExists(table):
        sdf.writeTo(table).append()
    else:
        sdf.writeTo(table).create()


# --- run id ------------------------------------------------------------------


def _resolve_run_id(run_id: str | None) -> tuple[str, str]:
    """Return ``(run_id, run_ts_iso)``. Precedence: arg > $RECSYS_RUN_ID > generated.

    Generated form is ``<UTC compact ts>-<git short sha>`` (D3). run_ts is always
    a fresh ISO-8601 UTC instant.
    """
    now = datetime.now(timezone.utc)
    run_ts = now.isoformat()
    if run_id:
        return run_id, run_ts
    env = os.environ.get("RECSYS_RUN_ID")
    if env:
        return env, run_ts
    sha = _git_short_sha()
    generated = now.strftime("%Y%m%dT%H%M%SZ") + (f"-{sha}" if sha else "")
    return generated, run_ts


def _git_short_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover - git absent / not a repo
        pass
    return None
