"""Bronze reconciliation: compare observed bronze table counts against the
published (rounded) Amazon Reviews 2023 Electronics counts, and idempotently
write the result into the ``## Bronze reconciliation`` section of
``data/MANIFEST.md`` (docs/engineering-log/UPGRADE_PLAN.md §8, Phase 0 T3+).

Usage:
    python -m batch_recsys_lab.ingest.reconcile [--wall-clock reviews=NNNs,items=NNNs]

The manifest itself (``data/MANIFEST.md``) is produced by ``make manifest``
(``batch_recsys_lab.ingest.download manifest``); this module only replaces the
"Bronze reconciliation" section in place. If the manifest does not exist yet,
this errors cleanly *before* touching any Spark/Iceberg tables — a missing
manifest is a fast, table-free check.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.md"

BRONZE_NAMESPACE = "local.bronze"

# Published (rounded) figures per the Amazon Reviews 2023 / Hou et al. 2024
# release site. The observed bronze count is canonical; these are only the
# rounded headline figures used for a sanity-check delta.
PUBLISHED_TABLES: dict[str, tuple[str, int]] = {
    "reviews": ("43.9M", 43_900_000),
    "items": ("1.61M (rounded)", 1_610_000),
}

SECTION_HEADER = "## Bronze reconciliation"
SECTION_RE = re.compile(
    rf"{re.escape(SECTION_HEADER)}\n(.*?)(\n## |\Z)", flags=re.DOTALL
)


def _parse_wall_clock(spec: str | None) -> dict[str, str]:
    """Parse ``--wall-clock reviews=NNNs,items=NNNs`` into a dict of table -> string."""
    if not spec:
        return {}
    out: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid --wall-clock segment {part!r}; expected table=NNNs")
        table, value = part.split("=", 1)
        out[table.strip()] = value.strip()
    return out


def _table_exists(spark, table: str) -> bool:
    full_table = f"{BRONZE_NAMESPACE}.{table}"
    try:
        return spark.catalog.tableExists(full_table)
    except Exception:
        return False


def _observed_counts(spark) -> dict[str, int | None]:
    """Return observed row counts for known bronze tables; None if a table doesn't exist."""
    counts: dict[str, int | None] = {}
    for table in PUBLISHED_TABLES:
        full_table = f"{BRONZE_NAMESPACE}.{table}"
        if not _table_exists(spark, table):
            print(f"[skip] {full_table}: table does not exist yet")
            counts[table] = None
            continue
        count = spark.table(full_table).count()
        print(f"[ok] {full_table}: observed count = {count}")
        counts[table] = count
    return counts


def _format_delta(observed: int | None, published_literal: int) -> str:
    if observed is None:
        return "n/a"
    delta = observed - published_literal
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:,}"


def _render_section(
    observed_counts: dict[str, int | None],
    wall_clock: dict[str, str],
) -> str:
    lines = []
    lines.append(SECTION_HEADER)
    lines.append("")
    lines.append("| Table | Published (rounded) | Observed | Delta vs rounded |")
    lines.append("|---|---|---|---|")
    for table, (published_str, published_literal) in PUBLISHED_TABLES.items():
        observed = observed_counts.get(table)
        observed_str = f"{observed:,}" if observed is not None else "not ingested yet"
        delta_str = _format_delta(observed, published_literal)
        lines.append(f"| {table} | {published_str} | {observed_str} | {delta_str} |")
    lines.append("")
    lines.append(
        "Note: published figures are rounded (per the Amazon Reviews 2023 / "
        "Hou et al. 2024 release site); the observed bronze count is canonical. "
        "Delta is observed minus the literal rounded published number, not a "
        "measure of ingestion correctness."
    )
    if wall_clock:
        lines.append("")
        wc_str = ", ".join(f"{table}={value}" for table, value in wall_clock.items())
        lines.append(f"Ingest wall-clock: {wc_str}")
    lines.append("")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"Last verified: {now}")
    return "\n".join(lines)


def _replace_section(manifest_text: str, new_section: str) -> str:
    if not SECTION_RE.search(manifest_text):
        raise ValueError(
            f"data/MANIFEST.md does not contain a {SECTION_HEADER!r} section; "
            "run `make manifest` first."
        )

    def _sub(match: re.Match) -> str:
        trailing = match.group(2)
        # Preserve whether the section was followed by another '## ' header or EOF.
        if trailing == "\n## ":
            return new_section + "\n\n## "
        return new_section + "\n"

    return SECTION_RE.sub(_sub, manifest_text, count=1)


def reconcile(wall_clock_spec: str | None = None) -> int:
    # Check manifest existence FIRST — a missing manifest is a fast, table-free
    # error; we do not want to spin up Spark just to fail on a missing file.
    if not MANIFEST_PATH.exists():
        print(
            f"[error] {MANIFEST_PATH} does not exist. Run `make manifest` first "
            "to generate data/MANIFEST.md before running bronze-verify."
        )
        return 1

    manifest_text = MANIFEST_PATH.read_text()
    if not SECTION_RE.search(manifest_text):
        print(
            f"[error] {MANIFEST_PATH} does not contain a {SECTION_HEADER!r} "
            "section. Run `make manifest` first to regenerate a manifest with "
            "the placeholder section."
        )
        return 1

    wall_clock = _parse_wall_clock(wall_clock_spec)

    # Import lazily so a missing-manifest failure never touches Spark.
    from batch_recsys_lab.spark_session import get_spark

    spark = get_spark(app_name="bronze-reconcile")
    try:
        observed_counts = _observed_counts(spark)
    finally:
        spark.stop()

    new_section = _render_section(observed_counts, wall_clock)
    updated_text = _replace_section(manifest_text, new_section)
    MANIFEST_PATH.write_text(updated_text)
    print(f"[done] updated {SECTION_HEADER!r} section in {MANIFEST_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.ingest.reconcile")
    parser.add_argument(
        "--wall-clock",
        default=None,
        help="Optional ingest wall-clock to record, e.g. 'reviews=1234s,items=210s'.",
    )
    args = parser.parse_args(argv)
    return reconcile(args.wall_clock)


if __name__ == "__main__":
    sys.exit(main())
