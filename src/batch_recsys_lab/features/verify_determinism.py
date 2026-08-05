"""Per-table content hashing + determinism verification (Phase 1, T8; §8 accept #4).

Determinism (D9) is defined as *content-identical across two builds*: for every
published table we compute, in a SINGLE aggregation pass, ``(row_count,
content_hash)`` where ``content_hash = Σ xxhash64(canonicalized columns)`` over all
rows. The sum is used deliberately: it is commutative, so the hash is independent
of row and partition order (there is no ``ORDER BY`` in the aggregation, which
would otherwise be a determinism hazard of its own). The per-row hash is summed as
``decimal(38,0)`` so the accumulation is exact and cannot overflow (or, under
Spark ANSI mode, error) — up to ~1e26 for tens of millions of ~1e19 hashes.

Type canonicalization (so the hash is stable and type-robust):

* ``timestamp`` -> ``unix_micros`` (a plain ``long``): microsecond-precision, no
  timezone/rendering ambiguity.
* ``date`` -> days since epoch (a ``long``).
* ``array`` / ``map`` / ``struct`` -> ``to_json`` (a canonical string). Two builds
  of the same deterministic code emit identical element/entry order, so their JSON
  is byte-identical; this is a *cross-build* determinism check, not a cross-engine
  canonical form.
* everything else (string / numeric / boolean / binary) -> the column as-is;
  ``xxhash64`` hashes these deterministically, NULL included.

Two modes:

* **write** (default): hash the default (or ``--tables``) table set and write
  ``{table: {rows, content_hash}}`` plus ``run_id`` / ``generated_at`` to
  ``data/table_hashes.json`` (``--out``).
* **compare** (``--compare <path>``): recompute the current warehouse hashes for
  the tables named in the reference file and diff them; print per-table
  ``identical`` / ``different`` / ``missing``; exit non-zero on ANY difference or
  missing table. This is the ``make data-verify`` engine (build, hash, rebuild,
  verify).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import Column, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DateType, MapType, StructType, TimestampType

from batch_recsys_lab.features.kcore import _resolve_run_id
from batch_recsys_lab.spark_session import get_spark

DEFAULT_OUT = "data/table_hashes.json"

# The published tables whose content must be reproducible across rebuilds. Bronze
# is included because it is the deterministic substrate silver/gold are hashed
# against; a real `make data` rebuild leaves bronze untouched, so it must match.
DEFAULT_TABLES = [
    "local.bronze.reviews",
    "local.bronze.items",
    "local.silver.interactions",
    "local.silver.items",
    "local.gold.interactions_5core",
    "local.gold.user_stats",
    "local.gold.item_features",
    "local.gold.popularity",
]


class DeterminismError(RuntimeError):
    """A referenced table is missing and skipping was not permitted."""


# --- hashing -----------------------------------------------------------------


def _canon_column(field) -> Column:
    """Canonical, hash-stable expression for one column (see module docstring)."""
    name = field.name
    dt = field.dataType
    col = F.col(f"`{name}`")
    if isinstance(dt, TimestampType):
        return F.expr(f"unix_micros(`{name}`)")
    if isinstance(dt, DateType):
        return F.datediff(col, F.lit("1970-01-01").cast("date")).cast("long")
    if isinstance(dt, (ArrayType, MapType, StructType)):
        return F.to_json(col)
    return col


def table_hash(spark: SparkSession, table: str) -> dict:
    """Return ``{"rows": int, "content_hash": str}`` for ``table`` in one pass."""
    df = spark.table(table)
    canon = [_canon_column(f) for f in df.schema.fields]
    row_hash = F.xxhash64(*canon).cast("decimal(38,0)")
    agg = df.agg(
        F.count(F.lit(1)).alias("rows"),
        # Empty table -> sum is NULL -> coalesce to 0 so both builds agree.
        F.coalesce(F.sum(row_hash), F.lit(0).cast("decimal(38,0)")).alias("content_hash"),
    ).first()
    return {"rows": int(agg["rows"]), "content_hash": str(agg["content_hash"])}


def compute_hashes(
    spark: SparkSession, tables: list[str], skip_missing: bool = False
) -> dict[str, dict]:
    """Hash each table; skip (or raise on) any that do not exist."""
    out: dict[str, dict] = {}
    for t in tables:
        if not spark.catalog.tableExists(t):
            if skip_missing:
                print(f"[warn] table {t} does not exist; skipping (--skip-missing)")
                continue
            raise DeterminismError(f"table {t} does not exist; build it first")
        out[t] = table_hash(spark, t)
    return out


# --- compare -----------------------------------------------------------------


def compare_hashes(
    spark: SparkSession, reference: dict, tables: list[str] | None
) -> tuple[bool, list[dict]]:
    """Diff current warehouse hashes against a reference ``{table: {...}}`` map.

    Returns ``(all_identical, rows)`` where each row is a per-table verdict dict.
    A table listed in the reference but absent from the warehouse is ``missing``
    (counts as a difference).
    """
    ref_tables = reference.get("tables", reference)
    check_tables = tables if tables is not None else sorted(ref_tables)
    rows: list[dict] = []
    all_identical = True
    for t in check_tables:
        ref = ref_tables.get(t)
        if ref is None:
            rows.append({"table": t, "verdict": "unreferenced", "detail": "not in reference file"})
            all_identical = False
            continue
        if not spark.catalog.tableExists(t):
            rows.append({"table": t, "verdict": "missing", "detail": "table absent from warehouse"})
            all_identical = False
            continue
        cur = table_hash(spark, t)
        identical = (
            int(cur["rows"]) == int(ref["rows"])
            and str(cur["content_hash"]) == str(ref["content_hash"])
        )
        rows.append(
            {
                "table": t,
                "verdict": "identical" if identical else "different",
                "reference": ref,
                "current": cur,
            }
        )
        if not identical:
            all_identical = False
    return all_identical, rows


# --- io ----------------------------------------------------------------------


def _write_out(out_path: str, run_id: str, hashes: dict[str, dict]) -> dict:
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": hashes,
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.features.verify_determinism")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--compare", default=None, help="reference hashes JSON to diff against")
    parser.add_argument(
        "--tables",
        default=None,
        help="comma-separated table override (default: the published silver+gold+bronze set)",
    )
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args(argv)

    tables = [t.strip() for t in args.tables.split(",")] if args.tables else None

    spark = get_spark(
        app_name="verify-determinism",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        if args.compare:
            reference = json.loads(Path(args.compare).read_text())
            all_identical, rows = compare_hashes(spark, reference, tables)
            for r in rows:
                if r["verdict"] == "identical":
                    print(f"[ok]   {r['table']}: identical")
                elif r["verdict"] in ("missing", "unreferenced"):
                    print(f"[FAIL] {r['table']}: {r['verdict']} ({r['detail']})")
                else:
                    print(
                        f"[FAIL] {r['table']}: DIFFERENT "
                        f"reference={r['reference']} current={r['current']}"
                    )
            result = {
                "mode": "compare",
                "reference": args.compare,
                "all_identical": all_identical,
                "tables": rows,
            }
            print(json.dumps(result))
            return 0 if all_identical else 1

        rid = _resolve_run_id(args.run_id)
        hashes = compute_hashes(spark, tables or DEFAULT_TABLES, skip_missing=args.skip_missing)
        payload = _write_out(args.out, rid, hashes)
    except DeterminismError as exc:
        print(f"[error] determinism hashing failed: {exc}", file=sys.stderr)
        return 2
    finally:
        spark.stop()

    print(f"wrote {len(payload['tables'])} table hashes to {args.out}")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
