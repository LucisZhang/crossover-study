"""Per-TRAIN-pair age extension for the eval cache (Phase 8, T8-2; UPGRADE_PLAN.md §8b).

``eval/extract.py`` deliberately drops timestamps: the eval cache keeps only
``(train_user_idx, train_item_idx, train_rating)``. The recency-matched arms
preregistered for T8-2 (EXPERIMENT_LOG.md 2026-08-17T11:44Z) need one more
column — how stale each TRAIN pair is at the train cutoff — so this module is a
ONE-SHOT, additive extension of an existing cache directory::

    <cache>/<snapshot>/train_age_days.npy        float32, len == n TRAIN pairs,
                                                 aligned to train_user_idx
    <cache>/<snapshot>/train_age_manifest.json   provenance for the above

``age_days = (train_end − latest TRAIN ts of that (user, item) pair) / 1 day``,
fractional, with ``train_end`` read from ``configs/splits.yaml`` (never
hardcoded). Because silver's D2 dedup already keeps exactly one row per
``(user_id, parent_asin)`` (``features/silver.py: keep_latest``), "latest ts" is
that row's ``ts``; the ``groupBy(...).max(ts)`` below is therefore an identity on
real data and doubles as the assertion that the dedup invariant still holds (a
duplicated pair aborts the job).

Nothing else in the cache is touched, no Iceberg table is written, and
``results/runs.jsonl`` is never opened.

Alignment (the load-bearing guarantee)
--------------------------------------
The saved TRAIN pair arrays' ROW ORDER is a Spark shuffle artifact, not
semantics (see ``extract._build_pairs``): re-running the same query is NOT
guaranteed to re-emit rows in the same order. Positional alignment is therefore
established by KEY, not by luck:

1. every TRAIN pair is read at the SAME pinned Iceberg snapshot the cache
   directory is keyed by (Iceberg time travel, exactly as ``extract.py``'s
   pinned mode), through the SAME user/item index space — the index maps are
   rebuilt from the cache's own ``user_ids.parquet`` / ``item_ids.parquet``, so
   the ``user_idx``/``item_idx`` values cannot drift from the cached arrays;
2. the recomputed ``(user_idx, item_idx)`` multiset is asserted EXACTLY equal to
   the cached one (same length, both duplicate-free, every cached pair present);
3. ages are then gathered into cached-array order.

Any violation is a hard abort BEFORE anything is written. The manifest also
records ``spark_row_order_matched`` — whether the collected row order happened
to coincide with the cached order — as an observation, never as a requirement.

Determinism: the written array is a pure function of (pinned snapshot, cached
pair arrays, ``configs/splits.yaml``); the Spark row order it was collected in
cannot change a single byte.
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
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.dataset import TRAIN_AGE_FILE, TRAIN_AGE_STEM
from batch_recsys_lab.eval.extract import FIVE_CORE, SPLITS_PATH, _read_table, _save_npy
from batch_recsys_lab.features.splits import SplitConfig, load_splits

AGE_MANIFEST = "train_age_manifest.json"
AGE_SCHEMA_VERSION = 1
MICROS_PER_DAY = 86_400_000_000


# --- pure numpy core (unit-tested without a JVM) ------------------------------


def _composite_keys(user_idx: np.ndarray, item_idx: np.ndarray, n_items: int) -> np.ndarray:
    """``user_idx * n_items + item_idx`` as int64 — a bijection on the pair space.

    Overflow is impossible at lab scale (1.6M users × 0.37M items ≈ 6e11 << 2^63)
    but is asserted rather than assumed.
    """
    n_items = int(n_items)
    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got {n_items}")
    if len(user_idx):
        hi = int(user_idx.max()) * n_items + int(item_idx.max())
        if hi >= 2**62:
            raise OverflowError(
                f"composite key {hi} too large for int64 keying (n_items={n_items})"
            )
    return user_idx.astype(np.int64) * np.int64(n_items) + item_idx.astype(np.int64)


def align_ages(
    cached_user_idx: np.ndarray,
    cached_item_idx: np.ndarray,
    pair_user_idx: np.ndarray,
    pair_item_idx: np.ndarray,
    pair_age_days: np.ndarray,
    n_items: int,
) -> np.ndarray:
    """Gather recomputed per-pair ages into the CACHED pair arrays' row order.

    Raises (hard abort, caller writes nothing) if the recomputed pair multiset is
    not exactly the cached one: different length, a duplicate ``(user, item)``
    pair on either side, or any cached pair missing from the recomputation. Also
    rejects non-finite or negative ages — TRAIN is ``ts <= train_end``, so a
    negative age means the split boundary or the snapshot is wrong.

    Returns
    -------
    numpy.ndarray
        float32, ``len(cached_user_idx)``, ``ages[k]`` is the age of the pair at
        row ``k`` of the cached TRAIN arrays.
    """
    if len(cached_user_idx) != len(cached_item_idx):
        raise ValueError(
            f"cached TRAIN arrays disagree in length: user {len(cached_user_idx)} "
            f"!= item {len(cached_item_idx)}"
        )
    if not (len(pair_user_idx) == len(pair_item_idx) == len(pair_age_days)):
        raise ValueError(
            f"recomputed columns disagree in length: user {len(pair_user_idx)}, "
            f"item {len(pair_item_idx)}, age {len(pair_age_days)}"
        )
    if len(pair_user_idx) != len(cached_user_idx):
        raise ValueError(
            f"TRAIN pair count mismatch: cache has {len(cached_user_idx)} pairs, "
            f"recomputation produced {len(pair_user_idx)}. The cache and the pinned "
            f"snapshot are not the same TRAIN universe — aborting without writing."
        )

    cached_keys = _composite_keys(cached_user_idx, cached_item_idx, n_items)
    pair_keys = _composite_keys(pair_user_idx, pair_item_idx, n_items)

    cached_sorted_keys = np.sort(cached_keys)
    if len(cached_sorted_keys) > 1 and (np.diff(cached_sorted_keys) == 0).any():
        n_dup = int((np.diff(cached_sorted_keys) == 0).sum())
        raise ValueError(
            f"cached TRAIN pairs contain {n_dup} duplicate (user, item) keys; "
            f"the cache is corrupt — aborting without writing."
        )

    order = np.argsort(pair_keys, kind="stable")
    sorted_keys = pair_keys[order]
    if len(sorted_keys) > 1 and (np.diff(sorted_keys) == 0).any():
        n_dup = int((np.diff(sorted_keys) == 0).sum())
        raise ValueError(
            f"recomputed TRAIN pairs contain {n_dup} duplicate (user, item) keys; "
            f"silver's keep-latest dedup guarantees uniqueness — aborting."
        )

    if len(sorted_keys) == 0:
        return np.zeros(0, dtype=np.float32)

    pos = np.searchsorted(sorted_keys, cached_keys)
    np.clip(pos, 0, len(sorted_keys) - 1, out=pos)
    found = sorted_keys[pos] == cached_keys
    if not found.all():
        n_missing = int((~found).sum())
        first = int(np.flatnonzero(~found)[0])
        raise ValueError(
            f"{n_missing} cached TRAIN pairs are absent from the recomputation "
            f"(first at cached row {first}: user_idx={int(cached_user_idx[first])}, "
            f"item_idx={int(cached_item_idx[first])}) — aborting without writing."
        )
    # Equal lengths + duplicate-free + every cached key found  =>  the two pair
    # multisets are identical, so the cached side is duplicate-free too.

    ages = np.asarray(pair_age_days, dtype=np.float64)[order][pos]
    if not np.isfinite(ages).all():
        raise ValueError("recomputed age_days contains non-finite values — aborting.")
    if (ages < 0).any():
        worst = float(ages.min())
        raise ValueError(
            f"recomputed age_days has negative entries (min {worst}); TRAIN is "
            f"ts <= train_end, so this means the split boundary or the pinned "
            f"snapshot is wrong — aborting."
        )
    return ages.astype(np.float32)


# --- Spark side ---------------------------------------------------------------


def _read_string_column(path: Path) -> list[str]:
    return pq.read_table(path).column(0).to_pylist()


def _index_frame(spark: SparkSession, ids: list[str], id_col: str, idx_col: str):
    """Ship an id -> index map to Spark through Arrow (never a list of tuples).

    Identical construction to ``extract._build_pairs``: position ``i`` of the
    (sorted) id list is index ``i``, declared ``string``/``int``.
    """
    return spark.createDataFrame(
        pd.DataFrame(
            {
                id_col: pd.Series(ids, dtype=object),
                idx_col: np.arange(len(ids), dtype=np.int32),
            }
        ),
        schema=f"{id_col} string, {idx_col} int",
    )


def _pair_ages_pdf(
    spark: SparkSession,
    cache_dir: Path,
    five_core_table: str,
    snapshot_id: int,
    splits: SplitConfig,
) -> pd.DataFrame:
    """One TRAIN row per ``(user_idx, item_idx)`` with fractional ``age_days``.

    Read at the pinned ``snapshot_id`` (Iceberg time travel), labelled by the
    frozen ``SplitConfig.split_label`` — the exact same TRAIN predicate
    ``extract._build_pairs`` used — and aggregated ``max(ts)`` per pair (an
    identity given silver's keep-latest dedup; see the module docstring).

    ``age_days`` is computed from ``unix_micros`` deltas, never from a
    ``timestamp -> double`` cast: at epoch scale the double cast loses
    microsecond resolution, and the boundary ``train_end`` carries millis.
    """
    user_ids = _read_string_column(cache_dir / "user_ids.parquet")
    item_ids = _read_string_column(cache_dir / "item_ids.parquet")

    base = _read_table(spark, five_core_table, {five_core_table: snapshot_id}).select(
        "user_id", "parent_asin", "ts"
    )
    train = base.where(splits.split_label("ts") == F.lit("train"))
    joined = (
        train.join(_index_frame(spark, user_ids, "user_id", "user_idx"), "user_id", "inner")
        .join(_index_frame(spark, item_ids, "parent_asin", "item_idx"), "parent_asin", "inner")
        .select("user_idx", "item_idx", "ts")
    )
    per_pair = joined.groupBy("user_idx", "item_idx").agg(F.max("ts").alias("ts_latest"))
    train_end_micros = F.unix_micros(F.lit(splits.train_end).cast("timestamp"))
    with_age = per_pair.withColumn(
        "age_days",
        (train_end_micros - F.unix_micros(F.col("ts_latest"))) / F.lit(float(MICROS_PER_DAY)),
    ).select("user_idx", "item_idx", "age_days")
    return with_age.toPandas()


def _up_to_date(cache_dir: Path, snapshot_id: int, splits_sha: str, n_pairs: int) -> bool:
    man_path = cache_dir / AGE_MANIFEST
    if not man_path.exists() or not (cache_dir / TRAIN_AGE_FILE).exists():
        return False
    try:
        man = json.loads(man_path.read_text())
    except (OSError, ValueError):
        return False
    return (
        man.get("schema_version") == AGE_SCHEMA_VERSION
        and int(man.get("five_core_snapshot_id", -1)) == int(snapshot_id)
        and man.get("splits_file_sha256") == splits_sha
        and int(man.get("n_train_pairs", -1)) == int(n_pairs)
    )


def extract_age(
    spark: SparkSession,
    cache_dir: str | Path,
    five_core_table: str = FIVE_CORE,
    splits_path: str | Path = SPLITS_PATH,
    force: bool = False,
) -> dict:
    """Write ``train_age_days.npy`` next to an existing cache's TRAIN arrays.

    ``cache_dir`` must be a resolved snapshot subdir (holding
    ``cache_manifest.json``). The five-core table is read by TIME TRAVEL at the
    snapshot id that manifest records, so the ages always describe the same
    TRAIN universe the cached pair arrays came from, whatever the live table now
    holds. Idempotent: an existing, matching ``train_age_manifest.json`` short-
    circuits (``force=True`` rebuilds).

    Returns a summary dict; the CLI prints it as the last stdout line.
    """
    start = time.perf_counter()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
    snapshot_id = int(manifest["snapshot_ids"][five_core_table])
    if cache_dir.name != str(snapshot_id):
        raise ValueError(
            f"cache dir {cache_dir} is not keyed by its own five-core snapshot "
            f"{snapshot_id}; refusing to extend an ambiguous cache."
        )

    splits = load_splits(splits_path)
    splits_sha = hashlib.sha256(Path(splits_path).read_bytes()).hexdigest()
    if splits_sha != manifest.get("splits_file_sha256"):
        raise ValueError(
            "configs/splits.yaml has changed since this cache was built "
            f"(cache {manifest.get('splits_file_sha256')!r} != now {splits_sha!r}); "
            "the frozen split boundaries are an invariant — aborting."
        )

    cached_user_idx = np.load(cache_dir / "train_user_idx.npy", allow_pickle=False)
    cached_item_idx = np.load(cache_dir / "train_item_idx.npy", allow_pickle=False)
    n_pairs = int(len(cached_user_idx))
    recorded = int((manifest.get("split_pair_counts") or {}).get("train", n_pairs))
    if recorded != n_pairs:
        raise ValueError(
            f"cache manifest records {recorded} TRAIN pairs but train_user_idx.npy "
            f"holds {n_pairs} — aborting."
        )

    if not force and _up_to_date(cache_dir, snapshot_id, splits_sha, n_pairs):
        print(f"train_age_days up to date: {cache_dir / TRAIN_AGE_FILE}")
        return {
            "status": "up_to_date",
            "cache_dir": str(cache_dir),
            "n_train_pairs": n_pairs,
        }

    n_items = int(manifest["catalog_size"])
    pdf = _pair_ages_pdf(spark, cache_dir, five_core_table, snapshot_id, splits)

    pair_user_idx = pdf["user_idx"].to_numpy(dtype=np.int32)
    pair_item_idx = pdf["item_idx"].to_numpy(dtype=np.int32)
    pair_age_days = pdf["age_days"].to_numpy(dtype=np.float64)

    ages = align_ages(
        cached_user_idx,
        cached_item_idx,
        pair_user_idx,
        pair_item_idx,
        pair_age_days,
        n_items,
    )
    row_order_matched = bool(
        len(pair_user_idx) == len(cached_user_idx)
        and np.array_equal(pair_user_idx, cached_user_idx)
        and np.array_equal(pair_item_idx, cached_item_idx)
    )

    _save_npy(ages, cache_dir, TRAIN_AGE_STEM)

    age_manifest = {
        "schema_version": AGE_SCHEMA_VERSION,
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "five_core_table": five_core_table,
        "five_core_snapshot_id": snapshot_id,
        "splits_file_sha256": splits_sha,
        "train_end": splits.train_end.isoformat(),
        "n_train_pairs": n_pairs,
        "age_days_min": float(ages.min()) if n_pairs else None,
        "age_days_max": float(ages.max()) if n_pairs else None,
        "age_days_mean": float(ages.mean()) if n_pairs else None,
        "spark_row_order_matched": row_order_matched,
        "train_age_days_sha256": hashlib.sha256(
            (cache_dir / TRAIN_AGE_FILE).read_bytes()
        ).hexdigest(),
        "git_sha": runlog.git_info()["git_sha"],
        "spark_version": spark.version,
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }
    (cache_dir / AGE_MANIFEST).write_text(json.dumps(age_manifest, indent=2))

    return {
        "status": "built",
        "cache_dir": str(cache_dir),
        "five_core_snapshot_id": snapshot_id,
        "n_train_pairs": n_pairs,
        "age_days_min": age_manifest["age_days_min"],
        "age_days_max": age_manifest["age_days_max"],
        "spark_row_order_matched": row_order_matched,
        "wall_clock_s": age_manifest["wall_clock_s"],
    }


# --- CLI -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.eval.extract_age")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument(
        "--cache-dir",
        default="data/eval/cache",
        help="cache root or an explicit snapshot subdir (resolved like the harness)",
    )
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--max-result-size", default="4g")
    parser.add_argument("--five-core-table", default=FIVE_CORE)
    parser.add_argument("--splits", default=str(SPLITS_PATH))
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if the age manifest matches"
    )
    args = parser.parse_args(argv)

    # Imported here (not at module top) purely to keep the CLI's failure on a
    # bad --cache-dir ahead of the JVM start.
    from batch_recsys_lab.eval.harness import _resolve_cache_dir

    cache_dir = _resolve_cache_dir(args.cache_dir)

    from batch_recsys_lab.spark_session import get_spark

    spark = get_spark(
        app_name="eval-extract-age",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
        extra_conf={"spark.driver.maxResultSize": args.max_result_size},
    )
    try:
        summary = extract_age(
            spark,
            cache_dir,
            five_core_table=args.five_core_table,
            splits_path=args.splits,
            force=args.force,
        )
    finally:
        spark.stop()

    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
