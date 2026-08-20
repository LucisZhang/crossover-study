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
    train_rating.npy        float32, TRAIN pairs, rating (aligned to train_user_idx)
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

**Pinned (time-travel) mode** — Phase 5, T18. Passing ``pinned_snapshot_ids``
(plus ``pinned_contracts``) switches every table read to Iceberg time travel
(``spark.read.option("snapshot-id", sid).table(name)``) so the cache is rebuilt
from the exact snapshots a recorded run used, regardless of what the live tables
now hold. In pinned mode the live snapshot IDs are never fetched, the cache dir
is keyed by the PINNED ``interactions_5core`` snapshot, and the contract
name/version identities come from the caller (the recorded record's
``contracts``) rather than live ``TBLPROPERTIES`` — because ``SHOW
TBLPROPERTIES`` has no time-travel form, so a live read would silently stamp
today's contract version onto a historical extract. Live mode is untouched.

**Driver-side scale** — Phase 7, T-A2. Every bulk table -> driver transfer goes
through Arrow (``toPandas``), never ``collect()`` into Python ``Row`` objects:
the un-cored silver universe is 18.3M users / ~36M pairs / 1.6M items, where
Row materialization alone is several GB and minutes of interpreter time. The
driver-side result budget is therefore a real constraint —
``spark.driver.maxResultSize`` is settable from the CLI (``--max-result-size``,
default 4g) because it must be fixed before the JVM starts. Determinism is
unchanged by the Arrow rewrite: ``toPandas`` collects partitions in the same
order ``collect()`` does, so every ORDER BY-driven artifact keeps the exact
ordering (and dtype) it had before — see the per-function notes.
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
import pandas as pd
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

CACHE_SCHEMA_VERSION = 2


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


def _read_table(spark: SparkSession, table: str, snapshot_ids: dict[str, int] | None = None):
    """Read ``table`` live, or time-travelled to a pinned snapshot.

    ``snapshot_ids`` is ``None``/empty in live mode -> plain ``spark.table``
    (byte-for-byte the previous behavior). Otherwise the table MUST have an entry
    in the mapping: a silent fallback to the live table is exactly the failure
    this mode exists to prevent, so a missing key raises.
    """
    if not snapshot_ids:
        return spark.table(table)
    sid = snapshot_ids.get(table)
    if sid is None:
        raise KeyError(
            f"pinned snapshot id missing for table {table!r}; pinned tables are "
            f"{sorted(snapshot_ids)}"
        )
    return spark.read.option("snapshot-id", str(int(sid))).table(table)


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
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False
    cached = manifest.get("snapshot_ids", {})
    return all(cached.get(t) == sid for t, sid in live_snapshot_ids.items())


def _build_item_index(
    spark: SparkSession,
    item_features_table: str,
    five_core_table: str,
    out_dir: Path,
    snapshot_ids: dict[str, int] | None = None,
) -> tuple[list[str], int, int]:
    # Catalog order = sorted ascending parent_asin from item_features. This
    # ordering IS the global deterministic tie-break used downstream (rank ties,
    # top-K argpartition ties) — every consumer of item_idx must derive it from
    # this exact sort. The ORDER BY does the sorting server-side and toPandas
    # preserves partition order, so item_ids is byte-identical to the previous
    # collect()-based build.
    pdf = (
        _read_table(spark, item_features_table, snapshot_ids)
        .select("parent_asin")
        .distinct()
        .orderBy("parent_asin")
        .toPandas()
    )
    item_ids = pdf["parent_asin"].tolist()
    n_5core_distinct = (
        _read_table(spark, five_core_table, snapshot_ids).select("parent_asin").distinct().count()
    )
    _save_string_column(item_ids, "item_id", out_dir)
    return item_ids, len(item_ids), n_5core_distinct


def _build_user_index(
    spark: SparkSession,
    user_stats_table: str,
    out_dir: Path,
    snapshot_ids: dict[str, int] | None = None,
) -> list[str]:
    # User order = sorted ascending user_id (the alignment contract for
    # n_train/n_val/n_test and for every user_idx downstream). ORDER BY is
    # server-side and toPandas preserves partition order, so user_ids and the
    # three count vectors are byte-identical to the previous collect() build —
    # the int32 casts below are the same narrowing the np.array(..., int32)
    # constructor did, applied to the Arrow int64 columns.
    df = _read_table(spark, user_stats_table, snapshot_ids).orderBy("user_id")
    pdf = df.select("user_id", "n_train", "n_val", "n_test").toPandas()
    user_ids = pdf["user_id"].tolist()
    n_train = pdf["n_train"].to_numpy(dtype=np.int32)
    n_val = pdf["n_val"].to_numpy(dtype=np.int32)
    n_test = pdf["n_test"].to_numpy(dtype=np.int32)
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
    snapshot_ids: dict[str, int] | None = None,
) -> dict[str, int]:
    """Join 5-core rows to the user/item indexes; save (user_idx, item_idx) pairs
    per split via Arrow-backed toPandas (never a Python row-by-row collect).

    NOTE (determinism, T18): the ROW ORDER of the saved pair arrays is a Spark
    shuffle artifact, not semantics. Downstream it is order-invariant for TRAIN
    (COO->CSR sorts) and for GT membership (``dataset._build_gt`` sorts by user);
    only the within-user GT item order can vary, which reproduce.py checks for
    explicitly (order-normalized cache digest) rather than assuming."""
    # The idx maps ship to Spark through Arrow (pandas -> createDataFrame with an
    # explicit schema), not as a Python list of tuples: at 18.3M users the list
    # form is ~2GB of driver tuples plus a per-row pickle. Contents are identical
    # — position i of the (sorted) id list is index i, declared int32/string by
    # the same DDL as before — so the joins below are unchanged.
    user_idx_df = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": pd.Series(user_ids, dtype=object),
                "user_idx": np.arange(len(user_ids), dtype=np.int32),
            }
        ),
        schema="user_id string, user_idx int",
    )
    item_idx_df = spark.createDataFrame(
        pd.DataFrame(
            {
                "parent_asin": pd.Series(item_ids, dtype=object),
                "item_idx": np.arange(len(item_ids), dtype=np.int32),
            }
        ),
        schema="parent_asin string, item_idx int",
    )

    base = _read_table(spark, five_core_table, snapshot_ids).select(
        "user_id", "parent_asin", "ts", "rating"
    )
    label = splits.split_label("ts")
    labeled = base.withColumn("split", label)

    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        cols = ["user_idx", "item_idx", "rating"] if split_name == "train" else ["user_idx", "item_idx"]
        pairs = (
            labeled.where(labeled["split"] == split_name)
            .join(user_idx_df, "user_id", "inner")
            .join(item_idx_df, "parent_asin", "inner")
            .select(*cols)
        )
        pdf = pairs.toPandas()
        user_arr = pdf["user_idx"].to_numpy(dtype=np.int32)
        item_arr = pdf["item_idx"].to_numpy(dtype=np.int32)
        _save_npy(user_arr, out_dir, f"{split_name}_user_idx")
        _save_npy(item_arr, out_dir, f"{split_name}_item_idx")
        if split_name == "train":
            rating_arr = pdf["rating"].to_numpy(dtype=np.float32)
            _save_npy(rating_arr, out_dir, "train_rating")
        counts[split_name] = int(len(pdf))
    return counts


def _build_popularity_vectors(
    spark: SparkSession,
    popularity_table: str,
    splits: SplitConfig,
    item_ids: list[str],
    out_dir: Path,
    snapshot_ids: dict[str, int] | None = None,
) -> None:
    from pyspark.sql import functions as F

    n_items = len(item_ids)
    as_of_to_label = {
        splits.train_end: "train_end",
        splits.val_end: "val_end",
    }

    # Resolve the as_of -> label mapping SERVER-SIDE (Spark session tz is pinned
    # to UTC), never by comparing collected Python datetimes: ``collect()`` can
    # hand back naive local-tz instants (see features/gold.py tests), which would
    # make an instant-equality check offset-fragile.
    df = _read_table(spark, popularity_table, snapshot_ids).select(
        "as_of", "window_days", "parent_asin", "n_interactions"
    )
    label_col = F.lit(None).cast("string")
    for boundary, label in as_of_to_label.items():
        label_col = F.when(F.col("as_of") == F.lit(boundary).cast("timestamp"), F.lit(label)).otherwise(
            label_col
        )
    labeled = df.withColumn("as_of_label", label_col).where(F.col("as_of_label").isNotNull())
    pdf = labeled.select("as_of_label", "window_days", "parent_asin", "n_interactions").toPandas()

    buckets: dict[tuple[str, int], np.ndarray] = {}
    for label in as_of_to_label.values():
        for w in WINDOWS:
            buckets[(label, w)] = np.zeros(n_items, dtype=np.float32)

    # Vectorized equivalent of the previous per-Row loop, with the same two skip
    # rules (window not in WINDOWS; parent_asin not in the catalog -> get_indexer
    # returns -1) and the same last-row-wins behavior for a repeated
    # (as_of, window, item) key: pandas keeps the collect() row order and numpy
    # fancy-index assignment writes duplicates in order. int64 -> float32 is one
    # round-to-nearest, exactly as float(int) -> float32 store was.
    if len(pdf):
        pos = pd.Index(item_ids).get_indexer(pdf["parent_asin"].to_numpy())
        windows = pdf["window_days"].to_numpy(dtype=np.int64)
        keep = (pos >= 0) & np.isin(windows, np.asarray(WINDOWS, dtype=np.int64))
        pos = pos[keep]
        windows = windows[keep]
        labels = pdf["as_of_label"].to_numpy()[keep]
        values = pdf["n_interactions"].to_numpy(dtype=np.float32)[keep]
        for (label, w), vec in buckets.items():
            sel = (labels == label) & (windows == w)
            if sel.any():
                vec[pos[sel]] = values[sel]

    for (label, w), vec in buckets.items():
        _save_npy(vec, out_dir, f"pop_{label}_{w}")


def _build_item_categories(
    spark: SparkSession,
    item_features_table: str,
    item_ids: list[str],
    out_dir: Path,
    snapshot_ids: dict[str, int] | None = None,
) -> None:
    n_items = len(item_ids)
    pdf = (
        _read_table(spark, item_features_table, snapshot_ids)
        .select("parent_asin", "main_category")
        .toPandas()
    )

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
    # new-category code assignment does not depend on Spark's row ordering. The
    # sort is the same Python string comparison the previous ``sorted(rows,
    # key=...)`` used, and it is stable, so the emitted code order is unchanged.
    pdf = pdf.sort_values("parent_asin", kind="stable")
    positions = pd.Index(item_ids).get_indexer(pdf["parent_asin"].to_numpy())
    # Arrow hands NULL strings back as None; guard NaN too so a missing
    # main_category can never become a category NAME (it is always code 0).
    categories = [None if pd.isna(v) else v for v in pdf["main_category"].tolist()]
    for pos, category in zip(positions, categories):
        if pos < 0:
            continue
        codes[pos] = _code_for(category)

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
    pinned_snapshot_ids: dict[str, int] | None = None,
    pinned_contracts: dict[str, dict] | None = None,
) -> dict:
    """Build (or skip, if up to date) the snapshot-keyed eval cache. Returns a
    summary dict; the CLI prints it as the last stdout line.

    ``pinned_snapshot_ids`` (with ``pinned_contracts``) switches to time-travel
    mode — see the module docstring. Both must cover all four tables.
    """
    start = time.perf_counter()
    # Runtime-settable and performance-only: it selects the Arrow transport for
    # toPandas()/createDataFrame(pandas) below. Output bytes do not depend on it
    # (the fallback path produces the same rows in the same order); at un-cored
    # scale the difference is minutes-to-hours of driver time.
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    splits = load_splits(splits_path)
    tables = (five_core_table, user_stats_table, item_features_table, popularity_table)

    pinned = pinned_snapshot_ids is not None
    if pinned:
        snapshot_ids = {}
        for table in tables:
            if table not in pinned_snapshot_ids:
                raise ValueError(
                    f"pinned extract: no snapshot id for {table!r} (have "
                    f"{sorted(pinned_snapshot_ids)})"
                )
            snapshot_ids[table] = int(pinned_snapshot_ids[table])
        if pinned_contracts is None:
            raise ValueError(
                "pinned extract requires pinned_contracts: SHOW TBLPROPERTIES has no "
                "time-travel form, so contract identities must come from the recorded "
                "run, not from today's live table properties."
            )
        for table in tables:
            if table not in pinned_contracts:
                raise ValueError(
                    f"pinned extract: no contract identity for {table!r} (have "
                    f"{sorted(pinned_contracts)})"
                )
        read_snapshots: dict[str, int] | None = snapshot_ids
    else:
        snapshot_ids = _fetch_snapshot_ids(
            spark, five_core_table, user_stats_table, item_features_table, popularity_table
        )
        read_snapshots = None

    cache_root = Path(out)
    cache_dir = cache_root / str(snapshot_ids[five_core_table])

    if _cache_up_to_date(cache_dir, snapshot_ids):
        print(f"cache up to date: {cache_dir}")
        return {"status": "up_to_date", "cache_dir": str(cache_dir), "pinned": pinned}

    cache_dir.mkdir(parents=True, exist_ok=True)

    item_ids, catalog_size, n_5core_distinct = _build_item_index(
        spark, item_features_table, five_core_table, cache_dir, read_snapshots
    )
    user_ids = _build_user_index(spark, user_stats_table, cache_dir, read_snapshots)
    split_counts = _build_pairs(
        spark, five_core_table, splits, user_ids, item_ids, cache_dir, read_snapshots
    )
    _build_popularity_vectors(
        spark, popularity_table, splits, item_ids, cache_dir, read_snapshots
    )
    _build_item_categories(spark, item_features_table, item_ids, cache_dir, read_snapshots)

    if pinned:
        contract_identities = {t: dict(pinned_contracts[t]) for t in tables}
    else:
        contract_identities = {t: _contract_identity(spark, t) for t in tables}
    splits_bytes = Path(splits_path).read_bytes()

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "snapshot_ids": snapshot_ids,
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
        "pinned": pinned,
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
    parser.add_argument(
        "--max-result-size",
        default="4g",
        help=(
            "spark.driver.maxResultSize. Every artifact here is collected to the "
            "driver, and the un-cored universe (18.3M users / ~36M pairs) blows "
            "past Spark's 1g default. Pre-JVM conf, so it is set at session build."
        ),
    )
    parser.add_argument(
        "--five-core-table",
        default=FIVE_CORE,
        help="interactions table; its snapshot id keys the cache directory",
    )
    parser.add_argument("--user-stats-table", default=USER_STATS)
    parser.add_argument("--item-features-table", default=ITEM_FEATURES)
    parser.add_argument("--popularity-table", default=POPULARITY)
    parser.add_argument(
        "--splits-path",
        default=str(SPLITS_PATH),
        help=(
            "Path to the frozen splits YAML used to label TRAIN/VAL/TEST while "
            "building the pair cache (default: configs/splits.yaml)."
        ),
    )
    parser.add_argument(
        "--pinned-record",
        default=None,
        help=(
            "path to a JSON file carrying 'iceberg_snapshots' and 'contracts' "
            "(i.e. a results/runs.jsonl record). Switches every table read to "
            "Iceberg time travel at those snapshot IDs."
        ),
    )
    args = parser.parse_args(argv)

    pinned_snapshot_ids = pinned_contracts = None
    if args.pinned_record:
        rec = json.loads(Path(args.pinned_record).read_text())
        pinned_snapshot_ids = rec["iceberg_snapshots"]
        pinned_contracts = rec["contracts"]

    from batch_recsys_lab.spark_session import get_spark

    spark = get_spark(
        app_name="eval-extract",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
        extra_conf={"spark.driver.maxResultSize": args.max_result_size},
    )
    try:
        summary = extract(
            spark,
            out=args.out,
            five_core_table=args.five_core_table,
            user_stats_table=args.user_stats_table,
            item_features_table=args.item_features_table,
            popularity_table=args.popularity_table,
            splits_path=args.splits_path,
            pinned_snapshot_ids=pinned_snapshot_ids,
            pinned_contracts=pinned_contracts,
        )
    finally:
        spark.stop()

    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
