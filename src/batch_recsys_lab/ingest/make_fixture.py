"""Build the bundled CI fixture: a deterministic ~50k-row sample of
``local.bronze.reviews`` plus the matching items slice from
``local.bronze.items`` (docs/engineering-log/UPGRADE_PLAN.md §8, repo map: ``tests/fixtures/``).

Usage:
    python -m batch_recsys_lab.ingest.make_fixture

Sampling is deterministic content-hash sampling, NOT a `LIMIT` off an
arbitrary scan order: we keep rows where
``pmod(xxhash64(user_id, asin, timestamp), FIXTURE_HASH_MOD) == 0``, then take
the first 50,000 of that (still-large) subset ordered by
``xxhash64(user_id, asin, timestamp)`` for a stable, reproducible ordering.
Re-running against the same bronze snapshot reproduces byte-identical fixture
files.

Outputs:
    tests/fixtures/bronze_reviews_50k.parquet   -- single-file parquet, 50,000 rows
    tests/fixtures/bronze_items_fixture.parquet -- single-file parquet, <=5,000 rows
    tests/fixtures/README.md                    -- provenance

Both parquet files are written as a single file via pyarrow (not Spark's
directory-parquet writer) since 50k/5k rows collect comfortably into memory.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql.functions import col, pmod, xxhash64

from batch_recsys_lab.spark_session import get_spark

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
REVIEWS_FIXTURE_PATH = FIXTURES_DIR / "bronze_reviews_50k.parquet"
ITEMS_FIXTURE_PATH = FIXTURES_DIR / "bronze_items_fixture.parquet"
README_PATH = FIXTURES_DIR / "README.md"

REVIEWS_TABLE = "local.bronze.reviews"
ITEMS_TABLE = "local.bronze.items"

# 43.9M reviews / FIXTURE_HASH_MOD ~= 54.9k rows in the pre-filter subset,
# comfortably above the 50,000-row target we LIMIT down to.
FIXTURE_HASH_MOD = 800

REVIEWS_ROW_TARGET = 50_000
ITEMS_ROW_TARGET = 5_000

SIZE_WARNING_BYTES = 50 * 1024 * 1024  # 50MB
ITEMS_HEAVY_COLUMNS = ("images", "videos")


def _content_hash(*columns: str):
    return xxhash64(*[col(c) for c in columns])


def _build_reviews_fixture(spark) -> pa.Table:
    hash_expr = _content_hash("user_id", "asin", "timestamp")
    sampled = (
        spark.table(REVIEWS_TABLE)
        .withColumn("_fixture_hash", hash_expr)
        .filter(pmod(col("_fixture_hash"), FIXTURE_HASH_MOD) == 0)
        .orderBy(col("_fixture_hash"))
        .limit(REVIEWS_ROW_TARGET)
        .drop("_fixture_hash")
    )
    pdf = sampled.toPandas()
    if len(pdf) != REVIEWS_ROW_TARGET:
        print(
            f"[warn] reviews fixture has {len(pdf)} rows, expected "
            f"{REVIEWS_ROW_TARGET}; FIXTURE_HASH_MOD={FIXTURE_HASH_MOD} pre-filter "
            "subset may be too small for this bronze snapshot."
        )
    return pa.Table.from_pandas(pdf, preserve_index=False)


def _build_items_fixture(spark, reviews_table: pa.Table) -> pa.Table:
    parent_asins = reviews_table.column("parent_asin").to_pylist()
    distinct_parent_asins = sorted(set(parent_asins))

    spark_parent_asins = spark.createDataFrame(
        [(p,) for p in distinct_parent_asins], ["parent_asin"]
    )
    hash_expr = xxhash64(col("parent_asin"))
    sampled = (
        spark.table(ITEMS_TABLE)
        .join(spark_parent_asins, on="parent_asin", how="inner")
        .withColumn("_fixture_hash", hash_expr)
        .orderBy(col("_fixture_hash"))
        .limit(ITEMS_ROW_TARGET)
        .drop("_fixture_hash")
    )
    pdf = sampled.toPandas()
    return pa.Table.from_pandas(pdf, preserve_index=False)


def build() -> dict:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    spark = get_spark(app_name="make-fixture")
    try:
        reviews_table = _build_reviews_fixture(spark)
        items_table = _build_items_fixture(spark, reviews_table)
    finally:
        spark.stop()

    pq.write_table(reviews_table, REVIEWS_FIXTURE_PATH)

    dropped_items_columns: list[str] = []
    total_bytes = REVIEWS_FIXTURE_PATH.stat().st_size + _table_nbytes(items_table)
    if total_bytes > SIZE_WARNING_BYTES:
        to_drop = [c for c in ITEMS_HEAVY_COLUMNS if c in items_table.column_names]
        if to_drop:
            print(
                f"[warn] combined fixture size ({total_bytes} bytes) exceeds "
                f"{SIZE_WARNING_BYTES} bytes; dropping {to_drop} from the items "
                "fixture only."
            )
            items_table = items_table.drop(to_drop)
            dropped_items_columns = to_drop

    pq.write_table(items_table, ITEMS_FIXTURE_PATH)

    summary = {
        "reviews_fixture": str(REVIEWS_FIXTURE_PATH),
        "reviews_rows": reviews_table.num_rows,
        "items_fixture": str(ITEMS_FIXTURE_PATH),
        "items_rows": items_table.num_rows,
        "dropped_items_columns": dropped_items_columns,
    }
    _write_readme(summary)
    return summary


def _table_nbytes(table: pa.Table) -> int:
    return table.nbytes


def _write_readme(summary: dict) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dropped = summary["dropped_items_columns"]
    dropped_note = (
        f"Dropped from items fixture (combined size > 50MB): {', '.join(dropped)}."
        if dropped
        else "No columns dropped (combined fixture size <= 50MB)."
    )
    lines = [
        "# Bundled bronze fixtures",
        "",
        "Deterministic ~50k-row sample of `local.bronze.reviews` plus the "
        "matching items slice from `local.bronze.items`, used as the CI "
        "substrate (docs/engineering-log/UPGRADE_PLAN.md repo map, `tests/fixtures/`).",
        "",
        "## Provenance",
        "",
        f"- Source tables: `{REVIEWS_TABLE}`, `{ITEMS_TABLE}`",
        "- Sampling rule (reviews): rows where "
        f"`pmod(xxhash64(user_id, asin, timestamp), {FIXTURE_HASH_MOD}) == 0`, "
        f"then `ORDER BY xxhash64(user_id, asin, timestamp) LIMIT {REVIEWS_ROW_TARGET}`.",
        "- Sampling rule (items): distinct `parent_asin` from the reviews "
        f"sample, joined to `{ITEMS_TABLE}`, `ORDER BY xxhash64(parent_asin) "
        f"LIMIT {ITEMS_ROW_TARGET}`.",
        f"- Generated: {generated} (filled at runtime by `make fixture`)",
        f"- Reviews fixture rows: {summary['reviews_rows']}",
        f"- Items fixture rows: {summary['items_rows']}",
        f"- {dropped_note}",
        "",
        "## Files",
        "",
        f"- `{REVIEWS_FIXTURE_PATH.name}` — single-file parquet, "
        f"{REVIEWS_ROW_TARGET} rows, bronze reviews schema.",
        f"- `{ITEMS_FIXTURE_PATH.name}` — single-file parquet, up to "
        f"{ITEMS_ROW_TARGET} rows, bronze items schema"
        + (" minus dropped columns above." if dropped else "."),
        "",
        "## Regeneration",
        "",
        "Regeneration is deterministic given the same bronze snapshot: "
        "`make fixture` (`python -m batch_recsys_lab.ingest.make_fixture`) "
        "re-derives byte-identical output because the sampling predicate is a "
        "pure content hash of `(user_id, asin, timestamp)` / `parent_asin`, not "
        "an arbitrary scan order.",
        "",
    ]
    README_PATH.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    summary = build()
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
