"""Per-item TRAIN-support / recency / first-seen stats (Phase 8, T8-1;
UPGRADE_PLAN.md §8b).

ONE Spark aggregation over ``local.gold.interactions_5core`` producing, per
``parent_asin``:

* ``n_train_support`` — number of interactions with ``ts <= train_end``. Silver
  dedup guarantees one row per ``(user, item, ts)`` identity, so this is also the
  count of distinct TRAIN users who touched the item.
* ``last_train_ts`` — ``max(ts)`` over TRAIN rows only; NULL when the item has no
  TRAIN interaction at all.
* ``first_seen_ts`` — ``min(ts)`` over **all** rows of the table (any split).

**Leak discipline.** The two TRAIN aggregates use ``ts <= train_end``
exclusively, so nothing a TRAIN-frozen model could not have known enters them.
``first_seen_ts`` deliberately spans all splits because its only job is to *date*
an item (an interaction-based proxy for release date, disclosed as such in the
preregistration); it is used to describe the catalog, never to score, rank, tune
or select. The regime map's own axis labels a first-seen after ``train_end`` as
``post-cutoff``, which is exactly the fact being measured.

**Timestamp transport.** The comparison against ``train_end``
(2022-06-30T23:59:59.999Z — millisecond precision) is done SERVER-SIDE with the
frozen ``features.splits`` literal, and the aggregated instants leave Spark as
``unix_millis`` int64 rather than as pandas datetimes: a naive-local round trip
would put the exact millisecond boundary at the mercy of the driver's timezone.
The parquet stores them back as ``timestamp[ms, tz=UTC]``, so downstream readers
get an unambiguous instant with no rounding.

Output (snapshot-keyed, idempotent — a matching manifest short-circuits)::

    data/eval/item_train_stats/<interactions_5core_snapshot_id>/
        item_train_stats.parquet   parent_asin (sorted asc = catalog order),
                                   n_train_support, last_train_ts, first_seen_ts
        manifest.json              schema_version, snapshot id, train_end,
                                   row count, parquet sha256, splits file hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from batch_recsys_lab.features.splits import SplitConfig, load_splits

FIVE_CORE = "local.gold.interactions_5core"
DEFAULT_OUT = "data/eval/item_train_stats"
STATS_FILENAME = "item_train_stats.parquet"
MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1

# Arrow schema of the published parquet. Instants are tz-aware millisecond
# timestamps (see module docstring); last_train_ts is nullable by construction.
STATS_SCHEMA = pa.schema(
    [
        pa.field("parent_asin", pa.string(), nullable=False),
        pa.field("n_train_support", pa.int64(), nullable=False),
        pa.field("last_train_ts", pa.timestamp("ms", tz="UTC"), nullable=True),
        pa.field("first_seen_ts", pa.timestamp("ms", tz="UTC"), nullable=False),
    ]
)


def _snapshot_id(spark: SparkSession, table: str) -> int:
    row = spark.sql(f"SELECT snapshot_id FROM {table}.refs WHERE name = 'main'").first()
    if row is None:
        raise RuntimeError(f"table {table} has no 'main' ref snapshot")
    return int(row["snapshot_id"])


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _up_to_date(out_dir: Path, snapshot_id: int) -> bool:
    manifest_path = out_dir / MANIFEST_FILENAME
    if not manifest_path.exists() or not (out_dir / STATS_FILENAME).exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    return (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("interactions_5core_snapshot_id") == snapshot_id
    )


def compute_item_train_stats(
    spark: SparkSession,
    splits: SplitConfig,
    five_core_table: str = FIVE_CORE,
):
    """The single aggregation. Returns a Spark DataFrame with the four columns
    (instants as ``*_ms`` int64 epoch milliseconds), one row per ``parent_asin``.

    ``F.count(col)`` skips NULLs, so counting the TRAIN-masked timestamp column
    counts exactly the rows with ``ts <= train_end`` — and ``F.max`` of the same
    column is NULL precisely for items with no TRAIN interaction, which is the
    "absent-in-TRAIN" bucket the regime map needs.
    """
    df = spark.table(five_core_table).select("parent_asin", "ts")
    train_ts = F.when(F.col("ts") <= splits._lit(splits.train_end), F.col("ts"))
    return (
        df.withColumn("_train_ts", train_ts)
        .groupBy("parent_asin")
        .agg(
            F.count(F.col("_train_ts")).cast("long").alias("n_train_support"),
            F.max(F.col("_train_ts")).alias("_last_train_ts"),
            F.min(F.col("ts")).alias("_first_seen_ts"),
        )
        .select(
            F.col("parent_asin"),
            F.col("n_train_support"),
            F.expr("unix_millis(_last_train_ts)").alias("last_train_ms"),
            F.expr("unix_millis(_first_seen_ts)").alias("first_seen_ms"),
        )
        .orderBy("parent_asin")
    )


def build(
    spark: SparkSession,
    out: str | Path = DEFAULT_OUT,
    five_core_table: str = FIVE_CORE,
    splits_path: str | Path | None = None,
    force: bool = False,
) -> dict:
    """Build (or skip, if up to date) the snapshot-keyed item-stats parquet."""
    start = time.perf_counter()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    splits = load_splits(splits_path) if splits_path else load_splits()

    snapshot_id = _snapshot_id(spark, five_core_table)
    out_dir = Path(out) / str(snapshot_id)
    if not force and _up_to_date(out_dir, snapshot_id):
        print(f"item_train_stats up to date: {out_dir}")
        return {"status": "up_to_date", "out_dir": str(out_dir), "snapshot_id": snapshot_id}

    # 368k rows at the 5-core catalog size — an Arrow transfer, not a collect().
    pdf = compute_item_train_stats(spark, splits, five_core_table).toPandas()

    parent_asin = pdf["parent_asin"].tolist()
    n_train_support = pdf["n_train_support"].to_numpy(dtype=np.int64)
    # Nullable int64 epoch-ms: pandas gives object/float for a column with NULLs,
    # so build the Arrow array from an explicit None-carrying list.
    last_ms = [None if v is None or v != v else int(v) for v in pdf["last_train_ms"].tolist()]
    first_ms = [int(v) for v in pdf["first_seen_ms"].tolist()]

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "parent_asin": pa.array(parent_asin, type=pa.string()),
            "n_train_support": pa.array(n_train_support, type=pa.int64()),
            "last_train_ts": pa.array(last_ms, type=pa.int64()).cast(
                pa.timestamp("ms", tz="UTC")
            ),
            "first_seen_ts": pa.array(first_ms, type=pa.int64()).cast(
                pa.timestamp("ms", tz="UTC")
            ),
        },
        schema=STATS_SCHEMA,
    )
    stats_path = out_dir / STATS_FILENAME
    pq.write_table(table, stats_path)

    n_zero = int((n_train_support == 0).sum())
    n_low = int(((n_train_support >= 1) & (n_train_support <= 4)).sum())
    n_high = int((n_train_support >= 5).sum())

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "source_table": five_core_table,
        "interactions_5core_snapshot_id": snapshot_id,
        "train_end": splits.train_end.isoformat(),
        "splits_version": splits.version,
        "splits_file_sha256": hashlib.sha256(
            Path(splits_path).read_bytes()
            if splits_path
            else (Path(__file__).resolve().parents[3] / "configs" / "splits.yaml").read_bytes()
        ).hexdigest(),
        "n_items": int(table.num_rows),
        "support_bucket_counts": {"zero": n_zero, "low": n_low, "high": n_high},
        "stats_parquet": STATS_FILENAME,
        "stats_parquet_sha256": _sha256_file(stats_path),
    }
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

    return {
        "status": "built",
        "out_dir": str(out_dir),
        "snapshot_id": snapshot_id,
        "n_items": int(table.num_rows),
        "support_bucket_counts": manifest["support_bucket_counts"],
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.features.item_train_stats")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--five-core-table", default=FIVE_CORE)
    parser.add_argument("--splits-path", default=None)
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    args = parser.parse_args(argv)

    from batch_recsys_lab.spark_session import get_spark

    spark = get_spark(
        app_name="item-train-stats",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        summary = build(
            spark,
            out=args.out,
            five_core_table=args.five_core_table,
            splits_path=args.splits_path,
            force=args.force,
        )
    finally:
        spark.stop()

    # Summary JSON MUST be the last stdout line (repo convention).
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
