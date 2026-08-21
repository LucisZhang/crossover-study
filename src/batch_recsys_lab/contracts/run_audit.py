"""Contract audit CLI (Phase 1, T8; docs/engineering-log/UPGRADE_PLAN.md §8, acceptance #1).

Runs every ``contracts/*.yaml`` contract against its published Iceberg table via
:func:`batch_recsys_lab.contracts.engine.audit`, appends all :class:`DqResult`
rows to the append-only ``local.dq.dq_results`` ledger, stamps each audited table
with the contract identity as Iceberg ``TBLPROPERTIES`` (the §7 hook Phase 2's
eval manifest reads back), prints a human-readable table×check pass/fail/measured
matrix, and exits non-zero if ANY check ends ``status == 'fail'``.

Two audit-level normalizations, applied to the raw engine results *before* they
are written / matrixed / graded. Both are documented here because they change how
a ``fail`` from the engine is graded, and neither weakens a real data defect:

* **Iceberg nullability caveat (T3).** Spark/Iceberg ``createOrReplace`` writes
  every column as physically ``nullable = true``, so ``schema_conformance`` reports
  ``fail`` for every ``nullable: false`` contract column even when the data is
  clean. Where the ONLY discrepancy is *declared-non-null-but-physically-nullable*
  (dtype matches; expected nullable ``False``; actual nullable ``True``) we downgrade
  the status to ``measured`` and annotate the details with the caveat. This is
  purely a physical-schema artifact — actual NULL *values* in a declared-non-null
  column are still caught as hard ``fail`` by that column's ``not_null`` check and
  by the zero-violation re-assertion of the row-level quarantine rules, neither of
  which is touched here. A genuine dtype mismatch (or a *missing* column) stays a
  hard ``fail``.

* **Empty-table ``no_all_null``.** On a table with zero rows the engine reports
  every column as "dead" (non-null count 0) and ``no_all_null`` fails. A 0-row
  table has no all-NULL *values*, only vacuous emptiness (the fixture 5-core, hence
  ``gold.popularity`` / ``gold.user_stats``, can legitimately be empty at fixture
  sparsity). We downgrade ``no_all_null`` to ``pass`` when ``total_rows == 0`` and
  note the 0-row reason in the details. Every other check on an empty table already
  passes vacuously (0 violations), so this is the only empty-table special case.

Nothing else is downgraded: every other ``fail`` (including a real ``no_all_null``
on a *non-empty* table, or any ``fail``-action row check with surviving violators)
remains a hard failure and drives a non-zero exit.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from pyspark.sql import SparkSession

from batch_recsys_lab.contracts.engine import (
    DQ_RESULTS_TABLE,
    DqResult,
    audit,
    write_dq_results,
)
from batch_recsys_lab.contracts.engine import _resolve_run_id
from batch_recsys_lab.contracts.loader import Contract, load_contract
from batch_recsys_lab.spark_session import get_spark

# contracts/ lives at the repo root (…/batch-recsys-lab/contracts).
CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts"
DEFAULT_DQ_TABLE = DQ_RESULTS_TABLE  # local.dq.dq_results
DEFAULT_CATALOG = "local"

NULLABILITY_NOTE = (
    "Spark/Iceberg createOrReplace marks all columns physically nullable; the "
    "declared-non-null column is otherwise conformant (dtype matches). NULL values "
    "are still hard-failed by the column's not_null check / zero-violation "
    "re-assertion. Downgraded fail->measured (T8)."
)
EMPTY_TABLE_NOTE = (
    "Table has 0 rows; no_all_null is vacuous on an empty table (no NULL values "
    "exist). Downgraded fail->pass (T8)."
)


class AuditError(RuntimeError):
    """A referenced table is missing and skipping was not permitted."""


# --- result normalization ----------------------------------------------------


def _normalize(results: list[DqResult]) -> list[DqResult]:
    """Apply the two documented audit-level downgrades; return a new list."""
    out: list[DqResult] = []
    for r in results:
        if r.status == "fail" and r.check_kind == "schema_conformance":
            d = _details(r)
            only_nullability = (
                d.get("actual_dtype") == d.get("expected_dtype")
                and d.get("expected_nullable") is False
                and d.get("actual_nullable") is True
            )
            if only_nullability:
                out.append(
                    replace(
                        r,
                        status="measured",
                        details=json.dumps({**d, "note": NULLABILITY_NOTE}),
                    )
                )
                continue
        if r.status == "fail" and r.check_kind == "no_all_null" and r.total_rows == 0:
            d = _details(r)
            out.append(
                replace(
                    r,
                    status="pass",
                    details=json.dumps({**d, "note": EMPTY_TABLE_NOTE}),
                )
            )
            continue
        out.append(r)
    return out


def _details(r: DqResult) -> dict:
    try:
        d = json.loads(r.details)
        return d if isinstance(d, dict) else {"_raw": r.details}
    except (TypeError, ValueError):
        return {"_raw": r.details}


# --- table resolution --------------------------------------------------------


def _resolve_table(contract: Contract, table_prefix: str | None) -> str:
    """The physical table to audit for ``contract``.

    ``table_prefix`` swaps the leading catalog token of the contract table (e.g.
    ``local.silver.items`` -> ``<prefix>.silver.items``) so a test harness can point
    the audit at an alternately-catalogued warehouse. Default (None) audits the
    contract's declared table verbatim.
    """
    if not table_prefix:
        return contract.table
    parts = contract.table.split(".", 1)
    rest = parts[1] if len(parts) == 2 else contract.table
    return f"{table_prefix}.{rest}"


# --- orchestration -----------------------------------------------------------


def run_audit(
    spark: SparkSession,
    *,
    contracts_dir: str | Path = CONTRACTS_DIR,
    dq_table: str = DEFAULT_DQ_TABLE,
    run_id: str | None = None,
    skip_missing: bool = False,
    table_prefix: str | None = None,
) -> dict:
    """Audit every contract, ledger the results, stamp tables, return a summary.

    Raises :class:`AuditError` if a contract's table is absent and
    ``skip_missing`` is False. Never raises on a data-quality ``fail`` — that is
    reported via the returned summary (``any_fail``) and the CLI exit code.
    """
    rid, _ = _resolve_run_id(run_id)
    contract_paths = sorted(Path(contracts_dir).glob("*.yaml"))
    if not contract_paths:
        raise AuditError(f"no contract YAMLs found under {contracts_dir}")

    all_results: list[DqResult] = []
    tables: dict[str, dict] = {}
    skipped: list[str] = []

    for path in contract_paths:
        contract = load_contract(path)
        table = _resolve_table(contract, table_prefix)

        if not spark.catalog.tableExists(table):
            if skip_missing:
                print(f"[warn] table {table} does not exist yet; skipping (--skip-missing)")
                skipped.append(table)
                continue
            raise AuditError(
                f"table {table} (contract {contract.name!r}) does not exist; build it "
                "first or pass --skip-missing (only with --table-prefix)"
            )

        results = _normalize(audit(spark, contract, table_name=table, run_id=rid))
        all_results.extend(results)

        # §7 hook: stamp the contract identity so Phase 2's eval manifest can read
        # which contract/version this table was audited against.
        spark.sql(
            f"ALTER TABLE {table} SET TBLPROPERTIES "
            f"('contracts.name'='{contract.name}', "
            f"'contracts.version'='{contract.version}')"
        )

        checks = {}
        for r in results:
            checks[r.check_id] = {
                "kind": r.check_kind,
                "status": r.status,
                "violation_count": r.violation_count,
                "metric_value": r.metric_value,
            }
        counts = {"pass": 0, "fail": 0, "measured": 0}
        for r in results:
            counts[r.status] = counts.get(r.status, 0) + 1
        tables[table] = {
            "contract": contract.name,
            "version": contract.version,
            "total_rows": results[0].total_rows if results else 0,
            "counts": counts,
            "checks": checks,
        }

    write_dq_results(spark, all_results, dq_table)

    any_fail = any(r.status == "fail" for r in all_results)
    return {
        "run_id": rid,
        "dq_table": dq_table,
        "any_fail": any_fail,
        "n_results": len(all_results),
        "skipped_tables": skipped,
        "tables": tables,
    }


# --- rendering ---------------------------------------------------------------


def _mark(status: str) -> str:
    return {"pass": "PASS", "fail": "FAIL", "measured": "MEAS"}.get(status, status.upper())


def _print_matrix(summary: dict) -> None:
    """Human-readable table × check pass/fail/measured matrix on stdout."""
    print("=" * 72)
    print(f"contract audit  ·  run_id={summary['run_id']}  ·  dq_table={summary['dq_table']}")
    print("=" * 72)
    for table, info in summary["tables"].items():
        c = info["counts"]
        print(
            f"\n{table}  [contract={info['contract']} v{info['version']}, "
            f"rows={info['total_rows']:,}]  "
            f"pass={c['pass']} fail={c['fail']} measured={c['measured']}"
        )
        for check_id, chk in info["checks"].items():
            extra = ""
            if chk["status"] == "measured" and chk["metric_value"] is not None:
                extra = f"  (metric={chk['metric_value']:.6g}, n={chk['violation_count']})"
            elif chk["status"] == "fail":
                extra = f"  (violations={chk['violation_count']})"
            print(f"  [{_mark(chk['status'])}] {check_id} · {chk['kind']}{extra}")
    if summary["skipped_tables"]:
        print(f"\nskipped (missing): {', '.join(summary['skipped_tables'])}")
    verdict = "FAIL" if summary["any_fail"] else "PASS"
    print("\n" + "-" * 72)
    print(f"overall: {verdict}  ({summary['n_results']} checks recorded)")
    print("-" * 72)


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.contracts.run_audit")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--contracts-dir", default=str(CONTRACTS_DIR))
    parser.add_argument("--dq-table", default=DEFAULT_DQ_TABLE)
    # Test/override knobs.
    parser.add_argument(
        "--table-prefix",
        default=None,
        help="swap the leading catalog token of every contract table (non-default)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="warn-and-skip contracts whose table is absent "
        "(only allowed together with --table-prefix, to avoid masking a real "
        "missing table in a default production audit)",
    )
    args = parser.parse_args(argv)

    if args.skip_missing and not args.table_prefix:
        parser.error("--skip-missing is only allowed together with --table-prefix")

    spark = get_spark(
        app_name="contracts-audit",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        summary = run_audit(
            spark,
            contracts_dir=args.contracts_dir,
            dq_table=args.dq_table,
            run_id=args.run_id,
            skip_missing=args.skip_missing,
            table_prefix=args.table_prefix,
        )
    except AuditError as exc:
        print(f"[error] contract audit failed: {exc}", file=sys.stderr)
        return 2
    finally:
        spark.stop()

    _print_matrix(summary)
    # Summary JSON MUST be the last stdout line (repo convention).
    print(json.dumps(summary))
    return 1 if summary["any_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
