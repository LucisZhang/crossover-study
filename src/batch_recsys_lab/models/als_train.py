"""Implicit-ALS training — Step A: train Spark MLlib ALS once, persist factors
(Phase 2, T2/T3; UPGRADE_PLAN.md §8 "Architecture", two-process eval design).

This is the ONLY ALS module that imports ``pyspark``. It reads the SAME eval
config yaml the harness consumes (``model.name == "als"``,
``model.params`` = rank/reg_param/alpha/max_iter/weighting, ``seeds.model`` =
seed), trains ``pyspark.ml.recommendation.ALS`` with ``implicitPrefs=True``, and
writes dense float32 factor matrices ``U`` (n_users × rank) / ``V`` (n_items ×
rank) plus an ``als_manifest.json`` under the snapshot-and-param-keyed artifact
directory that ``models.als`` (Step B) loads. The harness never starts a JVM.

Determinism stance: Spark ALS retraining is NOT bit-stable — implicit-ALS solves
a least-squares system whose float reduction order depends on partitioning and
parallelism, so two runs with identical seed/params can differ in the low bits.
Reproducibility is therefore claimed not at the training step but via:
  * the persisted artifact — both factor npys' sha256s are recorded in the
    manifest and the run record, and Step B's ``U @ Vᵀ`` rescoring is fully
    bit-deterministic given fixed factors;
  * the recorded seed + params, with headline configs reporting a 3-seed mean±sd
    that bounds the residual stochastic variance.

Driver-memory trap: at rank 128 the concatenated factors are ≈840MB; a
``collect()``/``toPandas()`` on ``model.userFactors`` would materialize that in
the driver heap and OOM. Instead the factors are written to parquet, read back
with pyarrow, and scattered by id into pre-zeroed dense arrays.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.harness import _resolve_cache_dir
from batch_recsys_lab.models.als import (
    als_param_hash,
    artifact_dir,
    canonical_params,
    five_core_snapshot_id,
    sha256_file,
)


def _load_cache_manifest(cache_dir: Path) -> dict:
    return json.loads((cache_dir / "cache_manifest.json").read_text())


def _artifact_up_to_date(adir: Path, param_hash: str, params: dict, snap: int) -> bool:
    """True if an artifact manifest already matches this param_hash/params/snapshot."""
    man_path = adir / "als_manifest.json"
    if not man_path.exists():
        return False
    try:
        am = json.loads(man_path.read_text())
    except (OSError, ValueError):
        return False
    return (
        am.get("param_hash") == param_hash
        and am.get("params") == params
        and int(am.get("five_core_snapshot_id", -1)) == snap
    )


def _scatter_factors(parquet_dir: Path, n_rows: int, rank: int) -> np.ndarray:
    """Read a Spark factor parquet dir (columns ``id`` int, ``features``
    list<float>) and scatter into a pre-zeroed dense ``(n_rows, rank)`` float32
    array by id. Rows for ids ALS never emitted (cold users / untrained items)
    stay zero — the intentional segment-0 collapse (see ``models.als``)."""
    table = pq.read_table(parquet_dir)
    ids = np.asarray(table.column("id").to_pylist(), dtype=np.int64)
    feats = np.array(table.column("features").to_pylist(), dtype=np.float32)
    out = np.zeros((n_rows, rank), dtype=np.float32)
    if len(ids) > 0:
        if feats.shape[1] != rank:
            raise ValueError(
                f"factor width {feats.shape[1]} != rank {rank} in {parquet_dir}"
            )
        out[ids] = feats
    return out


def train_als(
    spark,
    cache_dir: Path,
    *,
    rank: int,
    reg_param: float,
    alpha: float,
    max_iter: int,
    weighting: str,
    seed: int,
    factors_root: str | Path = "data/eval/als",
    git_sha: str | None = None,
    checkpoint_interval: int = 5,
) -> Path:
    """Train ALS on one snapshot-keyed cache dir and persist the artifact.

    ``cache_dir`` is an already-resolved snapshot subdir (contains
    ``cache_manifest.json`` + ``train_*_idx.npy``). Returns the artifact dir.
    Idempotent: if a matching artifact already exists, prints a skip line and
    returns without touching Spark output.
    """
    from pyspark.ml.recommendation import ALS

    cache_dir = Path(cache_dir)
    manifest = _load_cache_manifest(cache_dir)
    snap = five_core_snapshot_id(manifest)
    params = canonical_params(
        rank=rank,
        reg_param=reg_param,
        alpha=alpha,
        max_iter=max_iter,
        weighting=weighting,
        seed=seed,
    )
    param_hash = als_param_hash(params)
    adir = artifact_dir(factors_root, snap, param_hash)

    if _artifact_up_to_date(adir, param_hash, params, snap):
        print(f"artifact exists, skipping: {adir}")
        return adir

    if weighting not in ("binary", "rating"):
        raise ValueError(f"weighting must be 'binary' or 'rating', got {weighting!r}")

    n_users = int(manifest["n_users"])
    n_items = int(manifest["catalog_size"])

    t0 = time.perf_counter()

    train_user_idx = np.load(cache_dir / "train_user_idx.npy", allow_pickle=False)
    train_item_idx = np.load(cache_dir / "train_item_idx.npy", allow_pickle=False)
    n_train_pairs = int(len(train_user_idx))

    rating_path = cache_dir / "train_rating.npy"
    if weighting == "rating":
        if not rating_path.exists():
            raise FileNotFoundError(
                f"weighting='rating' needs {rating_path}, which is missing. Rebuild "
                f"the eval cache (make eval-extract) so train_rating.npy is present, "
                f"or use weighting='binary'."
            )
        rating = np.load(rating_path, allow_pickle=False).astype(np.float32, copy=False)
    else:
        # Binary implicit feedback: every observed pair is a 1.0 positive. A
        # missing train_rating.npy is tolerated here (ratings are unused).
        rating = np.ones(n_train_pairs, dtype=np.float32)

    # Build the training DataFrame with a fixed layout (repartition by user_idx).
    # Via pandas so createDataFrame takes the Arrow path (never a per-row Python
    # collect). Columns: user_idx (int), item_idx (int), rating (float).
    import pandas as pd

    pdf = pd.DataFrame(
        {
            "user_idx": train_user_idx.astype(np.int32),
            "item_idx": train_item_idx.astype(np.int32),
            "rating": rating.astype(np.float32),
        }
    )
    df = spark.createDataFrame(pdf).repartition(64, "user_idx")

    # Checkpointing bounds ALS's iterative lineage growth; only checkpointing
    # truncates lineage and reclaims shuffle files, so a lower interval bounds
    # retained shuffle at the cost of more checkpoint writes (train.checkpoint_interval
    # in the eval config; training-infra knob only, excluded from the param hash).
    adir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = adir / "_checkpoint"
    spark.sparkContext.setCheckpointDir(str(ckpt_dir))

    als = ALS(
        rank=rank,
        regParam=reg_param,
        alpha=alpha,
        maxIter=max_iter,
        seed=seed,
        implicitPrefs=True,
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="rating",
        checkpointInterval=checkpoint_interval,
        intermediateStorageLevel="MEMORY_AND_DISK",
        finalStorageLevel="MEMORY_AND_DISK",
        coldStartStrategy="nan",
    )
    model = als.fit(df)

    # Extract factors WITHOUT collect()/toPandas(): write to parquet, read back
    # with pyarrow, scatter by id into pre-zeroed dense float32 arrays.
    tmp = adir / "_factors_tmp"
    user_pq = tmp / "user"
    item_pq = tmp / "item"
    if tmp.exists():
        shutil.rmtree(tmp)
    model.userFactors.write.mode("overwrite").parquet(str(user_pq))
    model.itemFactors.write.mode("overwrite").parquet(str(item_pq))

    U = _scatter_factors(user_pq, n_users, rank)
    V = _scatter_factors(item_pq, n_items, rank)

    np.save(adir / "user_factors.npy", U, allow_pickle=False)
    np.save(adir / "item_factors.npy", V, allow_pickle=False)

    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    wall = round(time.perf_counter() - t0, 3)
    if git_sha is None:
        git_sha = runlog.git_info()["git_sha"]

    als_manifest = {
        "schema_version": 1,
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "seed": int(seed),
        "weighting": weighting,
        "param_hash": param_hash,
        "snapshot_ids": manifest["snapshot_ids"],
        "five_core_snapshot_id": snap,
        "n_users": n_users,
        "n_items": n_items,
        "n_train_pairs": n_train_pairs,
        "user_factors_shape": list(U.shape),
        "item_factors_shape": list(V.shape),
        "user_factors_sha256": sha256_file(adir / "user_factors.npy"),
        "item_factors_sha256": sha256_file(adir / "item_factors.npy"),
        "git_sha": git_sha,
        "spark_version": spark.version,
        "wall_clock_s": wall,
        "checkpoint_interval": int(checkpoint_interval),
    }
    (adir / "als_manifest.json").write_text(json.dumps(als_manifest, indent=2))
    print(
        f"trained ALS: {adir}  n_users={n_users} n_items={n_items} "
        f"n_train_pairs={n_train_pairs} rank={rank} weighting={weighting} wall={wall}s"
    )
    return adir


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.models.als_train")
    parser.add_argument("--config", required=True)
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    args = parser.parse_args(argv)

    import yaml

    config = yaml.safe_load(Path(args.config).read_text())
    model_cfg = config["model"]
    if model_cfg.get("name") != "als":
        raise ValueError(f"als_train expects model.name == 'als', got {model_cfg.get('name')!r}")
    params = dict(model_cfg.get("params") or {})
    seed = (config.get("seeds", {}) or {}).get("model")
    if seed is None:
        raise ValueError("als model requires seeds.model in the config")

    factors_root = params.get("factors_root", "data/eval/als")
    checkpoint_interval = int(config.get("train", {}).get("checkpoint_interval", 5))
    cache_dir = _resolve_cache_dir(config["cache_dir"])

    # Idempotency pre-check BEFORE starting Spark: the artifact identity is fully
    # determined by the cache manifest + params, no JVM required.
    manifest = _load_cache_manifest(cache_dir)
    snap = five_core_snapshot_id(manifest)
    canon = canonical_params(
        rank=int(params["rank"]),
        reg_param=float(params["reg_param"]),
        alpha=float(params["alpha"]),
        max_iter=int(params["max_iter"]),
        weighting=str(params["weighting"]),
        seed=int(seed),
    )
    param_hash = als_param_hash(canon)
    adir = artifact_dir(factors_root, snap, param_hash)
    if _artifact_up_to_date(adir, param_hash, canon, snap):
        print(f"artifact exists, skipping: {adir}")
        return 0

    from batch_recsys_lab.spark_session import get_spark

    spark = get_spark(
        app_name="als-train",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        train_als(
            spark,
            cache_dir,
            rank=int(params["rank"]),
            reg_param=float(params["reg_param"]),
            alpha=float(params["alpha"]),
            max_iter=int(params["max_iter"]),
            weighting=str(params["weighting"]),
            seed=int(seed),
            factors_root=factors_root,
            checkpoint_interval=checkpoint_interval,
        )
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
