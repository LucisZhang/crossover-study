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

``tags.csv`` and ``links.csv`` are not ingested: nothing downstream reads them
(the content arm, if it runs in T9-3b, is title+genres). They stay inside the
downloaded zip.

    uv run python -m batch_recsys_lab.ingest.bronze_ml32m --table ratings
    uv run python -m batch_recsys_lab.ingest.bronze_ml32m --table movies
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

_CSV_OPTIONS = {"header": "true"}

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
        help="Post-read partition count (default: 32 ratings / 4 movies).",
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
