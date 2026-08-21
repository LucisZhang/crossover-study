"""Bronze reconciliation for ML-32M: live ``local.bronze_ml32m.*`` counts vs the
row counts recorded in ``data/MANIFEST_ML32M.md`` (Phase 9, T9-3a; docs/engineering-log/UPGRADE_PLAN.md
§8c).

    python -m batch_recsys_lab.ingest.reconcile_ml32m [--manifest PATH]

Why this exists (it is not ceremony). The first real ingest landed 87,584 of
87,585 movies: Spark's CSV default escape is ``\\`` while MovieLens uses RFC-4180
doubled quotes, so one title blew its field apart and the row was dropped as
corrupt. Nothing failed — the number simply came out one short, and only a manual
count caught it. This job is the check that would have caught it: an *exact*
equality gate between the bytes we hashed and the rows we landed.

Two deliberate differences from the Amazon ``ingest/reconcile.py``:

* **Exact, not indicative.** The Amazon published counts are rounded marketing
  figures ("43.9M"), so that job reports a delta and writes it into the manifest
  as documentation. Here the manifest carries an exactly counted number of data
  rows per CSV, so any delta is a defect: mismatch exits non-zero.
* **Read-only.** It never edits the manifest. ``data/MANIFEST_ML32M.md`` is
  hashed into every ML-32M run record's ``dataset_manifest_hash``; a verification
  step that mutates the artifact it verifies would move that hash under recorded
  runs.

The comparison is only valid because the bronze reader does NOT enable
``multiLine``: one physical line == one CSV record, which is what makes the
manifest's byte-counted ``\\n`` totals comparable to Spark's row count. If a
future release embeds newlines in quoted fields, this job fails loudly — the
right outcome, not a silent re-interpretation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from batch_recsys_lab.ingest.bronze_ml32m import BRONZE_ML32M_NAMESPACE
from batch_recsys_lab.ingest.download_ml32m import MANIFEST_ML32M_PATH, parse_manifest

# bronze table -> the manifest filename whose data-row count it must equal.
TABLE_SOURCES: dict[str, str] = {
    "ratings": "ratings.csv",
    "movies": "movies.csv",
    "tags": "tags.csv",
}


def _expected_rows(manifest_path: Path) -> dict[str, int]:
    """Manifest data-row counts for every bronze table, or raise with the gap."""
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} does not exist. Run `make manifest-ml32m` first "
            "(it writes the ML-32M manifest; data/MANIFEST.md is a different file "
            "and must not be used for this dataset)."
        )
    entries = parse_manifest(manifest_path)
    expected: dict[str, int] = {}
    missing: list[str] = []
    for table, filename in TABLE_SOURCES.items():
        entry = entries.get(filename)
        if entry is None or entry.get("data_rows") is None:
            missing.append(filename)
            continue
        expected[table] = int(entry["data_rows"])
    if missing:
        raise RuntimeError(
            f"{manifest_path} has no '- Data rows (excl. header): N' line for "
            f"{missing}. Regenerate it with `make manifest-ml32m` after `fetch` has "
            "extracted every CSV; a table with no recorded row count cannot be "
            "reconciled, and an unreconciled bronze count is exactly how the "
            "87,584-of-87,585 movies bug survived."
        )
    return expected


def compare(
    expected: dict[str, int], observed: dict[str, int | None]
) -> tuple[list[str], list[str]]:
    """Pure core: ``(report lines, failure messages)``. Equality, not tolerance."""
    lines = [f"{'table':<10}{'manifest rows':>16}{'bronze rows':>14}{'delta':>10}  verdict"]
    failures: list[str] = []
    for table, filename in TABLE_SOURCES.items():
        want = expected[table]
        got = observed.get(table)
        if got is None:
            lines.append(
                f"{table:<10}{want:>16}{'MISSING':>14}{'-':>10}  FAIL (table not ingested)"
            )
            failures.append(f"{BRONZE_ML32M_NAMESPACE}.{table} does not exist")
            continue
        delta = got - want
        lines.append(
            f"{table:<10}{want:>16}{got:>14}{delta:>+10}  "
            f"{'ok' if delta == 0 else 'FAIL'} (vs {filename})"
        )
        if delta:
            failures.append(
                f"{BRONZE_ML32M_NAMESPACE}.{table}: {got} rows != {want} in {filename} "
                f"(delta {delta:+d})"
            )
    return lines, failures


def reconcile(manifest_path: str | Path | None = None) -> int:
    """Exit code: 0 if every bronze count equals its manifest row count."""
    path = Path(manifest_path) if manifest_path else MANIFEST_ML32M_PATH
    try:
        expected = _expected_rows(path)
    except RuntimeError as exc:  # fast, table-free failure: never start Spark for it
        print(f"[error] {exc}")
        return 1

    # Imported lazily so a missing/short manifest never touches Spark.
    from batch_recsys_lab.spark_session import get_spark

    spark = get_spark(app_name="bronze-reconcile-ml32m")
    try:
        observed: dict[str, int | None] = {}
        for table in TABLE_SOURCES:
            full = f"{BRONZE_ML32M_NAMESPACE}.{table}"
            observed[table] = spark.table(full).count() if spark.catalog.tableExists(full) else None
    finally:
        spark.stop()

    lines, failures = compare(expected, observed)
    print(f"[ml32m] bronze reconciliation against {path}")
    for line in lines:
        print(line)

    if failures:
        print(
            "[error] bronze does not reconcile with the manifest:\n  - "
            + "\n  - ".join(failures)
            + "\n  A shortfall is usually a CSV parse defect (check the ingest "
            "summary's `corrupt` count and the reader's quote/escape options), not "
            "a manifest error. Do NOT proceed to silver."
        )
        return 1
    print("[ok] every bronze_ml32m table matches the manifest's recorded data-row count")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.ingest.reconcile_ml32m")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest path (default: data/MANIFEST_ML32M.md).",
    )
    args = parser.parse_args(argv)
    return reconcile(args.manifest)


if __name__ == "__main__":
    sys.exit(main())
