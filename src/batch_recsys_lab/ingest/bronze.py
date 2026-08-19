"""Bronze ingestion: gz-jsonl → Iceberg bronze tables (Phase 0, UPGRADE_PLAN.md §8).

Bronze is the *faithful* layer: explicit schemas, no schema inference, no type
coercion beyond what the declared schema forces, and — critically — **no
sentinel fills**. NULL stays NULL. Any parsing, unit normalization, or key
extraction is silver's job (Phase 1). The one deliberate projection at bronze is
documented below.

Review text / images projection (UPGRADE_PLAN.md §5)
---------------------------------------------------
``bronze.reviews`` PROJECTS OUT the ``text`` and ``images`` columns before
write: "the lab never uses review text; item text comes from metadata". This is
the §5 disk budget lever (keeps the lakehouse ≈ 5–8GB). ``bronze.items`` keeps
its ``images``/``videos`` — only *reviews* drop text/images.

The ``timestamp`` column is kept as the raw epoch-ms ``long`` from the source;
bronze does not convert it (silver, Phase 1, owns that).

Corrupt-record accounting (never silently dropped)
--------------------------------------------------
The ancestor course project used a bare ``except: continue`` that silently
discarded unparseable rows. We correct that: rows are read in PERMISSIVE mode
with an explicit ``_corrupt_record`` column, the corrupt count is *reported*
(printed as a JSON summary line), and only the valid rows are written to bronze.

Spark gotchas handled here:
  * With an *explicit* schema, PERMISSIVE mode only captures corrupt rows if the
    ``_corrupt_record`` column is itself part of the read schema — so we append
    it to the declared schema before reading.
  * Referencing only ``_corrupt_record`` in a query against a raw JSON source is
    disallowed by Spark unless the parsed frame is cached/persisted first
    (otherwise Spark re-parses and the column analysis fails). We therefore
    persist the parsed frame, materialize it with a ``count()``, and only then
    filter on the corrupt marker.

Memory design (single 40M+ row non-splittable gz):
  We ``persist(DISK_ONLY)`` the parsed frame. DISK_ONLY spills blocks to local
  disk rather than pinning ~8–10GB in the driver heap, which is within the §5
  budget and keeps the driver bounded. We then take two cheap counts off the
  cached frame (total and corrupt), write the valid rows, and unpersist. This is
  a single logical parse of the gz (the persisted result is reused for both
  counts and the write) — not a decompress-twice design.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from batch_recsys_lab.spark_session import get_spark

CORRUPT_COL = "_corrupt_record"
BRONZE_NAMESPACE = "local.bronze"
INGEST_SUMMARY_LOG = "data/ingest_summary.jsonl"

# --- Reviews schema (Amazon Reviews 2023 review record, per dataset docs). ----
# Explicit schema, NO inference. `text` and `images` are read (so PERMISSIVE
# parsing sees the real record shape) but projected out before write per §5.
REVIEWS_SCHEMA = StructType(
    [
        StructField("rating", DoubleType()),
        StructField("title", StringType()),
        StructField("text", StringType()),
        StructField(
            "images",
            ArrayType(
                StructType(
                    [
                        StructField("small_image_url", StringType()),
                        StructField("medium_image_url", StringType()),
                        StructField("large_image_url", StringType()),
                        StructField("attachment_type", StringType()),
                    ]
                )
            ),
        ),
        StructField("asin", StringType()),
        StructField("parent_asin", StringType()),
        StructField("user_id", StringType()),
        StructField("timestamp", LongType()),  # epoch ms, raw — silver converts.
        StructField("helpful_vote", LongType()),
        StructField("verified_purchase", BooleanType()),
    ]
)

# --- Items schema (Amazon Reviews 2023 item-metadata record). -----------------
# Seeded from the course StructType in docs/seed-archive/jsonl_to_parquet.ipynb
# (read-only reference), with the following CORRECTIONS, each cited:
#   * price: course used DoubleType, which silently NULLs string prices like
#     "24.99" and range/"See price in cart" text. We keep price as STRING at
#     bronze and defer parsing to silver (Phase 1). No information is lost.
#   * videos: course struct had only {title, url}; the 2023 records also carry
#     a `user_id` field on video entries, added here.
#   * NO sentinel fills: the course did `na.fill({price:-1.0, rating_number:0,
#     average_rating:0.0, ...})`. Bronze does none of this — NULL stays NULL.
#   * `details` is kept as raw map<string,string>. NO per-key extraction at
#     bronze (course pulled Brand/Manufacturer/... into columns) — that is
#     Phase 1 silver. NOTE: map<string,string> stringifies non-string JSON
#     values inside `details`; acceptable at bronze (faithful-enough), silver
#     re-derives typed keys from this raw map.
ITEMS_SCHEMA = StructType(
    [
        StructField("main_category", StringType()),
        StructField("title", StringType()),
        StructField("average_rating", DoubleType()),
        StructField("rating_number", LongType()),
        StructField("features", ArrayType(StringType())),
        StructField("description", ArrayType(StringType())),
        StructField("price", StringType()),  # correction: was DoubleType (seed).
        StructField(
            "images",
            ArrayType(
                StructType(
                    [
                        StructField("thumb", StringType()),
                        StructField("large", StringType()),
                        StructField("variant", StringType()),
                        StructField("hi_res", StringType()),
                    ]
                )
            ),
        ),
        StructField(
            "videos",
            ArrayType(
                StructType(
                    [
                        StructField("title", StringType()),
                        StructField("url", StringType()),
                        StructField("user_id", StringType()),  # correction: added.
                    ]
                )
            ),
        ),
        StructField("store", StringType()),
        StructField("categories", ArrayType(StringType())),
        StructField("details", MapType(StringType(), StringType())),  # raw, no extraction.
        StructField("parent_asin", StringType()),
        StructField("bought_together", ArrayType(StringType())),
    ]
)


@dataclass(frozen=True)
class TableSpec:
    schema: StructType
    default_input: str
    default_repartition: int
    # Columns dropped before write (reviews only, per §5).
    project_out: tuple[str, ...] = field(default_factory=tuple)
    # --- source-format / destination knobs (Phase 9, T9-3a) -------------------
    # Defaults reproduce the Amazon gz-jsonl path byte-for-byte: fmt "json" takes
    # the same `spark.read...json(path)` branch it always did, no extra reader
    # options, and the same `local.bronze` namespace. A second dataset (ML-32M,
    # csv) sets these; nothing about the Amazon lane changes.
    fmt: str = "json"
    read_options: dict[str, str] = field(default_factory=dict)
    namespace: str = BRONZE_NAMESPACE
    # Positional-schema guard for headered CSV: the exact header line the file
    # must start with. None (Amazon) skips the check entirely.
    expected_header: tuple[str, ...] | None = None


TABLE_SPECS: dict[str, TableSpec] = {
    "reviews": TableSpec(
        schema=REVIEWS_SCHEMA,
        default_input="data/raw/Electronics.jsonl.gz",
        default_repartition=64,
        project_out=("text", "images"),
    ),
    "items": TableSpec(
        schema=ITEMS_SCHEMA,
        default_input="data/raw/meta_Electronics.jsonl.gz",
        default_repartition=16,
        project_out=(),
    ),
}


def _schema_with_corrupt(schema: StructType) -> StructType:
    """Append the corrupt-record marker so PERMISSIVE mode can populate it."""
    return StructType(schema.fields + [StructField(CORRUPT_COL, StringType())])


def _check_header(input_path: str, expected: tuple[str, ...]) -> list[str]:
    """Assert a headered CSV starts with exactly ``expected`` (Phase 9, T9-3a).

    A CSV is read POSITIONALLY against the declared schema (``enforceSchema``),
    so an upstream column reorder or rename would be ingested silently under the
    old names. The header is therefore verified as data, before Spark is asked to
    read anything, and a mismatch is a hard error. Returns the observed header.
    """
    with open(input_path, encoding="utf-8-sig") as fh:
        first_line = fh.readline()
    observed = [name.strip().strip('"') for name in first_line.rstrip("\r\n").split(",")]
    if tuple(observed) != tuple(expected):
        raise ValueError(
            f"{input_path}: CSV header {observed} != expected {list(expected)}. The "
            "declared schema is applied positionally, so an unverified header would "
            "mislabel every column; refusing to ingest."
        )
    return observed


def ingest_table(
    spark: SparkSession,
    table: str,
    input_path: str,
    repartition: int,
    specs: dict[str, TableSpec] | None = None,
) -> dict:
    """Ingest one raw file into ``<namespace>.<table>`` (createOrReplace).

    Returns a summary dict: ``{table, total_parsed, corrupt, written,
    wall_clock_s}``. Corrupt rows are excluded from the written table but their
    count is reported — never silently dropped.

    ``specs`` defaults to this module's Amazon :data:`TABLE_SPECS`; a sibling
    dataset module (``ingest/bronze_ml32m.py``) passes its own spec table so the
    corrupt-record accounting below is shared, not copied.
    """
    specs = TABLE_SPECS if specs is None else specs
    if table not in specs:
        raise ValueError(f"unknown table {table!r}; expected one of {sorted(specs)}")
    spec = specs[table]
    full_table = f"{spec.namespace}.{table}"

    start = time.perf_counter()

    if spec.expected_header is not None:
        _check_header(input_path, spec.expected_header)

    read_schema = _schema_with_corrupt(spec.schema)
    reader = (
        spark.read.schema(read_schema)
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT_COL)
    )
    for key, value in spec.read_options.items():
        reader = reader.option(key, value)
    # Explicit branch (not `.format(fmt).load(...)`) so the Amazon gz-jsonl call
    # is the exact expression it has always been.
    parsed = reader.csv(input_path) if spec.fmt == "csv" else reader.json(input_path)

    # Persist to DISK so we parse the gz once and reuse it for both counts and
    # the write, without pinning the full frame in the driver heap. The count()
    # below materializes the cache; only after that can we safely filter on the
    # corrupt marker (Spark forbids referencing _corrupt_record straight off a
    # raw JSON scan).
    parsed = parsed.persist(StorageLevel.DISK_ONLY)
    try:
        total_parsed = parsed.count()
        corrupt = parsed.filter(col(CORRUPT_COL).isNotNull()).count()
        # Every row is either corrupt (marker non-null) or valid (marker null),
        # a clean partition of the parse — so written == total - corrupt exactly,
        # no extra scan needed.
        written = total_parsed - corrupt

        valid: DataFrame = parsed.filter(col(CORRUPT_COL).isNull()).drop(CORRUPT_COL)
        if spec.project_out:
            valid = valid.drop(*spec.project_out)

        # gz is non-splittable: the read is a single task by physics. Repartition
        # AFTER read so the write and downstream reads are parallelized.
        valid = valid.repartition(repartition)

        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {BRONZE_NAMESPACE}")
        valid.writeTo(full_table).createOrReplace()
    finally:
        parsed.unpersist()

    wall_clock_s = round(time.perf_counter() - start, 3)
    return {
        "table": full_table,
        "total_parsed": total_parsed,
        "corrupt": corrupt,
        "written": written,
        "wall_clock_s": wall_clock_s,
    }


def _append_summary_log(summary: dict, table: str) -> None:
    """Enrich ``summary`` in place with table name / timestamp and append it as
    one line to data/ingest_summary.jsonl (same content as what gets printed).
    """
    summary.setdefault("table_name", table)
    summary["ingested_at"] = datetime.now(timezone.utc).isoformat()

    log_path = Path(INGEST_SUMMARY_LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(summary) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.ingest.bronze")
    parser.add_argument("--table", required=True, choices=sorted(TABLE_SPECS))
    parser.add_argument("--input", default=None, help="Override the default gz-jsonl input path.")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument(
        "--repartition",
        type=int,
        default=None,
        help="Post-read partition count (default: 64 reviews / 16 items).",
    )
    args = parser.parse_args(argv)

    spec = TABLE_SPECS[args.table]
    input_path = args.input or spec.default_input
    repartition = args.repartition if args.repartition is not None else spec.default_repartition

    spark = get_spark(
        app_name=f"bronze-ingest-{args.table}",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        summary = ingest_table(spark, args.table, input_path, repartition)
    finally:
        spark.stop()

    _append_summary_log(summary, args.table)

    # Summary JSON MUST be the last stdout line so the orchestrator can parse it.
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
