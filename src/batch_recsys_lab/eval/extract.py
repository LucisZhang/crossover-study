"""Spark -> numpy/scipy cache extract (Phase 2, T1; UPGRADE_PLAN.md §8 "Architecture").

Two-process design: this module is Step A, run once via Spark to compact the four
``gold`` Iceberg tables (``interactions_5core``, ``user_stats``, ``item_features``,
``popularity``) into a snapshot-keyed numpy/parquet cache directory that Step B
(``eval/dataset.py``, ``eval/harness.py``) loads without ever starting a JVM.

Cache layout, ``<out>/<interactions_5core_snapshot_id>/``::

    cache_manifest.json     schema_version, snapshot IDs, contract identities,
                             catalog/user counts, splits file hash
    item_ids.parquet        single string column "parent_asin", sorted ascending
                             (catalog order = the GLOBAL deterministic tie-break
                             used everywhere item_idx appears: rank ties, top-K
                             ties, and any "== 0 -> item index" comparison)
    user_ids.parquet        single string column "user_id", sorted ascending
    n_train.npy             int32, len U, aligned to user_ids
    n_val.npy                int32, len U, aligned to user_ids
    n_test.npy                int32, len U, aligned to user_ids
    train_user_idx.npy      int32, TRAIN pairs, user index
    train_item_idx.npy      int32, TRAIN pairs, item index (aligned to train_user_idx)
    val_user_idx.npy        int32, VAL GT pairs
    val_item_idx.npy        int32, VAL GT pairs
    test_user_idx.npy       int32, TEST GT pairs
    test_item_idx.npy       int32, TEST GT pairs
    pop_<asof>_<window>.npy float32, len I, dense n_interactions vector
                             (asof in {"train_end","val_end"}, window in
                             {0,30,90,365})
    item_category_codes.npy int32, len I, aligned to item_ids
    item_category_names.json JSON list[str]; index == code; "__unknown__" is the
                             code for a NULL/missing main_category

The cache is idempotent and keyed by the ``interactions_5core`` snapshot ID: if
the directory already exists and its manifest's snapshot IDs all match the live
tables, extraction is skipped (exit 0, "cache up to date").
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

from batch_recsys_lab.features.splits import SplitConfig, load_splits

# --- default table names (real warehouse) -------------------------------------
FIVE_CORE = "local.gold.interactions_5core"
USER_STATS = "local.gold.user_stats"
ITEM_FEATURES = "local.gold.item_features"
POPULARITY = "local.gold.popularity"

WINDOWS = (0, 30, 90, 365)
AS_OF_LABELS = ("train_end", "val_end")
SPLITS_PATH = Path(__file__).resolve().parents[3] / "configs" / "splits.yaml"

CACHE_SCHEMA_VERSION = 1


# --- small helpers -------------------------------------------------------------


def _snapshot_id(spark: SparkSession, table: str) -> int:
    row = spark.sql(f"SELECT snapshot_id FROM {table}.refs WHERE name = 'main'").first()
    if row is None:
        raise RuntimeError(f"table {table} has no 'main' ref snapshot")
    return int(row["snapshot_id"])


def _contract_identity(spark: SparkSession, table: str) -> dict:
    """Read back the ``contracts.name``/``contracts.version`` TBLPROPERTIES stamped
    by ``contracts/run_audit.py``."""
    props = {row["key"]: row["value"] for row in spark.sql(f"SHOW TBLPROPERTIES {table}").collect()}
    return {
        "name": props.get("contracts.name"),
        "version": props.get("contracts.version"),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_string_column(values: list[str], name: str, out_dir: Path) -> None:
    table = pa.table({name: pa.array(values, type=pa.string())})
    pq.write_table(table, out_dir / f"{name}s.parquet")


def _save_npy(arr: np.ndarray, out_dir: Path, name: str) -> None:
    np.save(out_dir / f"{name}.npy", arr, allow_pickle=False)


# --- extraction steps ------------------------------------------------------


def _fetch_snapshot_ids(
    spark: SparkSession,
    five_core_table: str,
    user_stats_table: str,
    item_features_table: str,
    popularity_table: str,
) -> dict[str, int]:
    return {
        five_core_table: _snapshot_id(spark, five_core_table),
        user_stats_table: _snapshot_id(spark, user_stats_table),
        item_features_table: _snapshot_id(spark, item_features_table),
        popularity_table: _snapshot_id(spark, popularity_table),
    }


def _cache_up_to_date(cache_dir: Path, live_snapshot_ids: dict[str, int]) -> bool:
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    cached = manifest.get("snapshot_ids", {})
    return all(cached.get(t) == sid for t, sid in live_snapshot_ids.items())


def _build_item_index(spark: SparkSession, item_features_table: str, five_core_table: str, out_dir: Path) -> tuple[list[str], int, int]:
    # Catalog order = sorted ascending parent_asin from item_features. This
    # ordering IS the global deterministic tie-break used downstream (rank ties,
    # top-K argpartition ties) — every consumer of item_idx must derive it from
    # this exact sort.
    rows = (
        spark.table(item_features_table)
        .select("parent_asin")
        .distinct()
        .orderBy("parent_asin")
        .collect()
    )
    item_ids = [r["parent_asin"] for r in rows]
    n_5core_distinct = spark.table(five_core_table).select("parent_asin").distinct().count()
    _save_string_column(item_ids, "item_id", out_dir)
    return item_ids, len(item_ids), n_5core_distinct


def _build_user_index(spark: SparkSession, user_stats_table: str, out_dir: Path) -> list[str]:
    df = spark.table(user_stats_table).orderBy("user_id")
    rows = df.select("user_id", "n_train", "n_val", "n_test").collect()
    user_ids = [r["user_id"] for r in rows]
    n_train = np.array([r["n_train"] for r in rows], dtype=np.int32)
    n_val = np.array([r["n_val"] for r in rows], dtype=np.int32)
    n_test = np.array([r["n_test"] for r in rows], dtype=np.int32)
    _save_string_column(user_ids, "user_id", out_dir)
    _save_npy(n_train, out_dir, "n_train")
    _save_npy(n_val, out_dir, "n_val")
    _save_npy(n_test, out_dir, "n_test")
    return user_ids


def _build_pairs(
    spark: SparkSession,
    five_core_table: str,
    splits: SplitConfig,
    user_ids: list[str],
    item_ids: list[str],
    out_dir: Path,
) -> dict[str, int]:
    """Join 5-core rows to the user/item indexes; save (user_idx, item_idx) pairs
    per split via Arrow-backed toPandas (never a Python row-by-row collect)."""
    user_idx_df = spark.createDataFrame(
        [(uid, i) for i, uid in enumerate(user_ids)], "user_id string, user_idx int"
    )
    item_idx_df = spark.createDataFrame(
        [(iid, i) for i, iid in enumerate(item_ids)], "parent_asin string, item_idx int"
    )

    base = spark.table(five_core_table).select("user_id", "parent_asin", "ts")
    label = splits.split_label("ts")
    labeled = base.withColumn("split", label)

    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        pairs = (
            labeled.where(labeled["split"] == split_name)
            .join(user_idx_df, "user_id", "inner")
            .join(item_idx_df, "parent_asin", "inner")
            .select("user_idx", "item_idx")
        )
        pdf = pairs.toPandas()
        user_arr = pdf["user_idx"].to_numpy(dtype=np.int32)
        item_arr = pdf["item_idx"].to_numpy(dtype=np.int32)
        _save_npy(user_arr, out_dir, f"{split_name}_user_idx")
        _save_npy(item_arr, out_dir, f"{split_name}_item_idx")
        counts[split_name] = int(len(pdf))
    return counts


def _build_popularity_vectors(
    spark: SparkSession,
    popularity_table: str,
    splits: SplitConfig,
    item_ids: list[str],
    out_dir: Path,
) -> None:
    from pyspark.sql import functions as F

    item_pos = {iid: i for i, iid in enumerate(item_ids)}
    n_items = len(item_ids)
    as_of_to_label = {
        splits.train_end: "train_end",
        splits.val_end: "val_end",
    }

    # Resolve the as_of -> label mapping SERVER-SIDE (Spark session tz is pinned
    # to UTC), never by comparing collected Python datetimes: ``collect()`` can
    # hand back naive local-tz instants (see features/gold.py tests), which would
    # make an instant-equality check offset-fragile.
    df = spark.table(popularity_table).select("as_of", "window_days", "parent_asin", "n_interactions")
    label_col = F.lit(None).cast("string")
    for boundary, label in as_of_to_label.items():
        label_col = F.when(F.col("as_of") == F.lit(boundary).cast("timestamp"), F.lit(label)).otherwise(
            label_col
        )
    labeled = df.withColumn("as_of_label", label_col).where(F.col("as_of_label").isNotNull())
    rows = labeled.collect()

    buckets: dict[tuple[str, int], np.ndarray] = {}
    for label in as_of_to_label.values():
        for w in WINDOWS:
            buckets[(label, w)] = np.zeros(n_items, dtype=np.float32)

    for r in rows:
        w = int(r["window_days"])
        if w not in WINDOWS:
            continue
        pos = item_pos.get(r["parent_asin"])
        if pos is None:
            continue
        buckets[(r["as_of_label"], w)][pos] = float(r["n_interactions"])

    for (label, w), vec in buckets.items():
        _save_npy(vec, out_dir, f"pop_{label}_{w}")


def _build_item_categories(spark: SparkSession, item_features_table: str, item_ids: list[str], out_dir: Path) -> None:
    item_pos = {iid: i for i, iid in enumerate(item_ids)}
    n_items = len(item_ids)
    rows = spark.table(item_features_table).select("parent_asin", "main_category").collect()

    # "__unknown__" is always code 0, deterministic regardless of row order, so a
    # NULL/missing main_category never depends on collect() ordering.
    names: list[str] = ["__unknown__"]
    name_to_code: dict[str, int] = {"__unknown__": 0}
    codes = np.zeros(n_items, dtype=np.int32)

    def _code_for(name: str | None) -> int:
        key = name if name is not None else "__unknown__"
        if key not in name_to_code:
            name_to_code[key] = len(names)
            names.append(key)
        return name_to_code[key]

    # Deterministic: process rows in a stable (sorted-by-parent_asin) order so
    # new-category code assignment does not depend on Spark's collect() ordering.
    for r in sorted(rows, key=lambda r: r["parent_asin"]):
        pos = item_pos.get(r["parent_asin"])
        if pos is None:
            continue
        codes[pos] = _code_for(r["main_category"])

    _save_npy(codes, out_dir, "item_category_codes")
    (out_dir / "item_category_names.json").write_text(json.dumps(names))


# --- orchestration -----------------------------------------------------------


def extract(
    spark: SparkSession,
    out: str | Path = "data/eval/cache",
    five_core_table: str = FIVE_CORE,
    user_stats_table: str = USER_STATS,
    item_features_table: str = ITEM_FEATURES,
    popularity_table: str = POPULARITY,
    splits_path: str | Path = SPLITS_PATH,
) -> dict:
    """Build (or skip, if up to date) the snapshot-keyed eval cache. Returns a
    summary dict; the CLI prints it as the last stdout line."""
    start = time.perf_counter()
    splits = load_splits(splits_path)

    live_snapshot_ids = _fetch_snapshot_ids(
        spark, five_core_table, user_stats_table, item_features_table, popularity_table
    )
    cache_root = Path(out)
    cache_dir = cache_root / str(live_snapshot_ids[five_core_table])

    if _cache_up_to_date(cache_dir, live_snapshot_ids):
        print(f"cache up to date: {cache_dir}")
        return {"status": "up_to_date", "cache_dir": str(cache_dir)}

    cache_dir.mkdir(parents=True, exist_ok=True)

    item_ids, catalog_size, n_5core_distinct = _build_item_index(
        spark, item_features_table, five_core_table, cache_dir
    )
    user_ids = _build_user_index(spark, user_stats_table, cache_dir)
    split_counts = _build_pairs(spark, five_core_table, splits, user_ids, item_ids, cache_dir)
    _build_popularity_vectors(spark, popularity_table, splits, item_ids, cache_dir)
    _build_item_categories(spark, item_features_table, item_ids, cache_dir)

    contract_identities = {
        five_core_table: _contract_identity(spark, five_core_table),
        user_stats_table: _contract_identity(spark, user_stats_table),
        item_features_table: _contract_identity(spark, item_features_table),
        popularity_table: _contract_identity(spark, popularity_table),
    }
    splits_bytes = Path(splits_path).read_bytes()

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "snapshot_ids": live_snapshot_ids,
        "contract_identities": contract_identities,
        "catalog_size": catalog_size,
        "n_5core_distinct_items": n_5core_distinct,
        "n_users": len(user_ids),
        "split_pair_counts": split_counts,
        "splits_file_sha256": hashlib.sha256(splits_bytes).hexdigest(),
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2))

    summary = {
        "status": "built",
        "cache_dir": str(cache_dir),
        "catalog_size": catalog_size,
        "n_users": len(user_ids),
        "split_pair_counts": split_counts,
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }
    return summary


# --- CLI -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.eval.extract")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--out", default="data/eval/cache")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    args = parser.parse_args(argv)

    from batch_recsys_lab.spark_session import get_spark

    spark = get_spark(
        app_name="eval-extract",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        summary = extract(spark, out=args.out)
    finally:
        spark.stop()

    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
