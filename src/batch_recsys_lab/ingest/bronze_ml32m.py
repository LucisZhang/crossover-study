"""ML-32M bronze ingestion: csv → Iceberg ``local.bronze_ml32m.*`` (Phase 9, T9-3a;
UPGRADE_PLAN.md §8c).

The regime-contrast dataset (MovieLens 32M; Harper & Konstan 2015) enters the
lakehouse through the SAME bronze discipline as the Amazon lane — explicit
schemas, no inference, no sentinel fills, PERMISSIVE parsing with the corrupt
count *reported* rather than silently dropped — by reusing
``ingest.bronze.ingest_table`` with its own spec table. The only differences are
declared in :data:`TABLE_SPECS`:

* **format** — ``csv`` with ``header=true`` instead of gz-jsonl. CSV is
  splittable, so unlike the Amazon 6.5GB non-splittable gz the read is already
  parallel; the DISK_ONLY persist in ``ingest_table`` still buys the
  parse-once-count-twice-write-once shape.
* **``escape='"'`` (RFC 4180)** — Spark's CSV default escape is backslash, but the
  MovieLens files escape a quote inside a quoted field by DOUBLING it. With the
  default, movieId 284105 (``... Presents: "We're Newbridge, ..."``) blew the
  field apart and was silently lost: the real run landed 87,584 of 87,585
  movies. The option is set on EVERY spec, not just ``movies``: ratings/tags do
  not change behaviour under it (ratings is purely numeric, tags is already
  RFC-4180), and one reader dialect for the dataset is one fact to verify rather
  than three. ``multiLine`` is deliberately NOT enabled — keeping one record per
  physical line is what makes the manifest's byte-counted data-row totals a
  valid cross-check in ``ingest/reconcile_ml32m.py``.
* **positional-schema guard** — a headered CSV read against a declared schema is
  positional (``enforceSchema``), so an upstream rename/reorder would be ingested
  silently under the old names. ``expected_header`` makes the header itself a
  checked fact: ``ingest_table`` reads the first line and refuses to ingest on
  any mismatch.
* **namespace** — ``local.bronze_ml32m``, a physically separate warehouse
  directory. The Amazon namespaces are never opened by this lane.

Raw types are kept exactly as published: ``userId``/``movieId`` stay integral
keys, ``timestamp`` stays the raw epoch-**seconds** long (the Amazon lane's is
epoch-millis; silver owns the conversion in both cases), ``rating`` stays the
0.5..5.0 half-star double. Column names are the source's own — bronze is the
faithful layer, silver renames.

``tags.csv`` IS ingested (§8c T9-3b's content arm is "title+genres+tags", so the
data stage must land it). ``links.csv`` is not: it holds only IMDb/TMDb foreign
keys, which nothing in this lab reads. It stays inside the downloaded zip and is
called out as deliberately unused in ``data/MANIFEST_ML32M.md``.

    uv run python -m batch_recsys_lab.ingest.bronze_ml32m --table ratings
    uv run python -m batch_recsys_lab.ingest.bronze_ml32m --table movies
    uv run python -m batch_recsys_lab.ingest.bronze_ml32m --table tags
"""

from __future__ import annotations

import argparse
import json
import sys

from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from batch_recsys_lab.ingest.bronze import TableSpec, ingest_table, _append_summary_log
from batch_recsys_lab.spark_session import get_spark

BRONZE_ML32M_NAMESPACE = "local.bronze_ml32m"
RAW_DIR = "data/raw/ml32m"

# --- ratings.csv (userId,movieId,rating,timestamp) -----------------------------
# timestamp is epoch SECONDS UTC (ML-32M README); kept raw, silver converts.
RATINGS_SCHEMA = StructType(
    [
        StructField("userId", LongType()),
        StructField("movieId", LongType()),
        StructField("rating", DoubleType()),
        StructField("timestamp", LongType()),
    ]
)

# --- movies.csv (movieId,title,genres) -----------------------------------------
# `genres` is the raw pipe-separated string ("Action|Adventure|Sci-Fi", or the
# literal "(no genres listed)"). Bronze does NOT split it — silver does.
MOVIES_SCHEMA = StructType(
    [
        StructField("movieId", LongType()),
        StructField("title", StringType()),
        StructField("genres", StringType()),
    ]
)

# --- tags.csv (userId,movieId,tag,timestamp) -----------------------------------
# Free text: tags carry commas and doubled quotes, which is exactly why the
# RFC-4180 escape below is not optional. timestamp is epoch SECONDS, as ratings.
TAGS_SCHEMA = StructType(
    [
        StructField("userId", LongType()),
        StructField("movieId", LongType()),
        StructField("tag", StringType()),
        StructField("timestamp", LongType()),
    ]
)

# escape='"': RFC-4180 doubled quotes, NOT Spark's default backslash. See the
# module docstring — the default silently dropped one movies.csv row on the real
# 87,585-row file.
_CSV_OPTIONS = {"header": "true", "escape": '"'}

TABLE_SPECS: dict[str, TableSpec] = {
    "ratings": TableSpec(
        schema=RATINGS_SCHEMA,
        default_input=f"{RAW_DIR}/ratings.csv",
        default_repartition=32,
        fmt="csv",
        read_options=_CSV_OPTIONS,
        namespace=BRONZE_ML32M_NAMESPACE,
        expected_header=("userId", "movieId", "rating", "timestamp"),
    ),
    "movies": TableSpec(
        schema=MOVIES_SCHEMA,
        default_input=f"{RAW_DIR}/movies.csv",
        default_repartition=4,
        fmt="csv",
        read_options=_CSV_OPTIONS,
        namespace=BRONZE_ML32M_NAMESPACE,
        expected_header=("movieId", "title", "genres"),
    ),
    "tags": TableSpec(
        schema=TAGS_SCHEMA,
        default_input=f"{RAW_DIR}/tags.csv",
        default_repartition=4,
        fmt="csv",
        read_options=_CSV_OPTIONS,
        namespace=BRONZE_ML32M_NAMESPACE,
        expected_header=("userId", "movieId", "tag", "timestamp"),
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.ingest.bronze_ml32m")
    parser.add_argument("--table", required=True, choices=sorted(TABLE_SPECS))
    parser.add_argument("--input", default=None, help="Override the default csv input path.")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument(
        "--repartition",
        type=int,
        default=None,
        help="Post-read partition count (default: 32 ratings / 4 movies / 4 tags).",
    )
    args = parser.parse_args(argv)

    spec = TABLE_SPECS[args.table]
    input_path = args.input or spec.default_input
    repartition = args.repartition if args.repartition is not None else spec.default_repartition

    spark = get_spark(
        app_name=f"bronze-ingest-ml32m-{args.table}",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        summary = ingest_table(spark, args.table, input_path, repartition, specs=TABLE_SPECS)
    finally:
        spark.stop()

    _append_summary_log(summary, f"ml32m_{args.table}")

    # Summary JSON MUST be the last stdout line (repo convention).
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
