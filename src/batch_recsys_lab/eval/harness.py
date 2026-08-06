"""Eval harness — config -> dataset -> model -> batched scoring loop -> metrics
-> bootstrap -> append-only record (Phase 2, T5; UPGRADE_PLAN.md §8 "Architecture",
Step B).

Pure numpy/scipy: no Spark. Loads the snapshot-keyed cache built by
``eval/extract.py`` into an ``EvalDataset``, scores eval users in batches, masks
TRAIN-seen items to ``-inf`` ONCE (model-agnostic exclusion — CLAUDE.md invariant
#4, full-catalog ranking), derives all accuracy metrics from exact GT ranks, and
writes one ``kind="eval"`` record to ``results/runs.jsonl`` with per-user artifact
parquet, bootstrap CIs, per-segment blocks, and beyond-accuracy metrics.

Integrity guards run before the append: TEST-split runs refuse a dirty git tree,
and the cache's Iceberg snapshot IDs are re-verified against the live tables
(stale-cache guard) unless ``allow_stale``.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from batch_recsys_lab.contracts.engine import _resolve_run_id
from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.bootstrap import ci_mean, segment_cis
from batch_recsys_lab.eval.dataset import EvalDataset, load_dataset
from batch_recsys_lab.eval.metrics import (
    accuracy_metrics,
    coverage_at_k,
    gini,
    gt_ranks,
    novelty_per_user,
    pop_share_at_k,
    topk_indices,
)
from batch_recsys_lab.eval.protocol import segment_of
from batch_recsys_lab.models.als import ALSRecommender
from batch_recsys_lab.models.content import ContentRecommender
from batch_recsys_lab.models.content_blend import ContentPopBlendRecommender
from batch_recsys_lab.models.item_knn import ItemKNNRecommender
from batch_recsys_lab.models.popularity import PopularityRecommender
from batch_recsys_lab.models.popularity_category import PopularityCategoryRecommender
from batch_recsys_lab.models.random_rec import RandomRecommender

DEFAULT_BATCH_SIZE = 1024
DEFAULT_K_LIST = (10, 20, 50)
TOPK_STORE = 50


# --- config helpers ----------------------------------------------------------


def _resolve_cache_dir(cache_dir: str | Path) -> Path:
    """Resolve to the single snapshot subdir under ``cache_dir``.

    If ``cache_dir`` itself holds a ``cache_manifest.json`` it is used directly
    (explicit full path). Otherwise it must contain exactly one snapshot subdir
    with a manifest; 0 or >1 is an error (ambiguous which snapshot to score).
    """
    p = Path(cache_dir)
    if (p / "cache_manifest.json").exists():
        return p
    subdirs = [
        d for d in sorted(p.iterdir()) if d.is_dir() and (d / "cache_manifest.json").exists()
    ] if p.exists() else []
    if len(subdirs) == 1:
        return subdirs[0]
    raise RuntimeError(
        f"cache_dir {p} must contain exactly one snapshot subdir (found "
        f"{len(subdirs)}); give an explicit full cache path if there are several."
    )


def _build_model(model_cfg: dict, seeds: dict):
    """Instantiate a model from ``config['model']`` via a small registry."""
    name = model_cfg["name"]
    params = dict(model_cfg.get("params") or {})
    if name == "random":
        seed = seeds.get("model")
        if seed is None:
            raise ValueError("random model requires seeds.model in the config")
        return RandomRecommender(seed=int(seed))
    if name == "popularity":
        return PopularityRecommender(
            as_of=params["as_of"], window_days=int(params["window_days"])
        )
    if name == "popularity_category":
        return PopularityCategoryRecommender(
            as_of=params["as_of"], window_days=int(params["window_days"])
        )
    if name == "item_knn":
        return ItemKNNRecommender(
            top_n=int(params["top_n"]),
            shrinkage=float(params.get("shrinkage", 0.0)),
            block_size=int(params.get("block_size", 8192)),
        )
    if name == "als":
        seed = seeds.get("model")
        if seed is None:
            raise ValueError("als model requires seeds.model in the config")
        return ALSRecommender(
            rank=int(params["rank"]),
            reg_param=float(params["reg_param"]),
            alpha=float(params["alpha"]),
            max_iter=int(params["max_iter"]),
            weighting=str(params["weighting"]),
            seed=int(seed),
            factors_root=params.get("factors_root", "data/eval/als"),
        )
    if name == "content":
        return ContentRecommender(
            recipe_hash=params["recipe_hash"],
            artifact_root=params.get("artifact_root", "data/eval/minilm"),
        )
    if name == "content_pop_blend":
        return ContentPopBlendRecommender(
            alpha=float(params["alpha"]),
            as_of=params["as_of"],
            window_days=int(params["window_days"]),
            recipe_hash=params["recipe_hash"],
            artifact_root=params.get("artifact_root", "data/eval/minilm"),
        )
    raise ValueError(f"unknown model name: {name!r}")


# --- the scoring loop --------------------------------------------------------


def _score_all(
    ds: EvalDataset, model, eval_user_idx: np.ndarray, gt, batch_size: int, k_list
):
    """Batched score -> mask -> ranks -> metrics + top-50. Returns per-user metric
    vectors (aligned to ``eval_user_idx``) and the top-50 index matrix."""
    n_users = len(eval_user_idx)
    n_items = len(ds.item_ids)
    kstore = min(TOPK_STORE, n_items)

    metric_chunks: dict[str, list[np.ndarray]] = {}
    topk_rows = np.empty((n_users, kstore), dtype=np.int32)

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        batch_users = eval_user_idx[start:end]

        scores = model.score_batch(batch_users)  # (B, I) fresh writable float32

        # Exclusion happens here, once, model-agnostically: mask TRAIN-seen to -inf.
        sub = ds.train_csr[batch_users]
        row_ids = np.repeat(np.arange(sub.shape[0]), np.diff(sub.indptr))
        scores[row_ids, sub.indices] = -np.inf

        # Batch-local CSR-ragged GT (eval_user_idx == gt.user_idx, same order).
        base = int(gt.indptr[start])
        batch_indptr = gt.indptr[start : end + 1] - base
        batch_items = gt.item_idx[base : int(gt.indptr[end])]

        ranks = gt_ranks(scores, batch_indptr, batch_items)
        batch_metrics = accuracy_metrics(batch_indptr, ranks, k_list)
        for name, vec in batch_metrics.items():
            metric_chunks.setdefault(name, []).append(vec)

        topk_rows[start:end] = topk_indices(scores, TOPK_STORE)

    metric_vecs = {name: np.concatenate(chunks) for name, chunks in metric_chunks.items()}
    return metric_vecs, topk_rows


# --- per-user artifact -------------------------------------------------------


def _write_artifact(
    path: Path,
    user_ids: np.ndarray,
    eval_user_idx: np.ndarray,
    segment_labels: np.ndarray,
    metric_vecs: dict[str, np.ndarray],
    novelty_vec: np.ndarray,
    topk_rows: np.ndarray,
) -> None:
    """Per-user parquet: user_id, user_idx, segment, each metric column,
    novelty@10, and ``top50`` as a variable-length int32 list column (documented
    choice: a list column keeps the schema stable regardless of catalog size <50)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: dict[str, pa.Array] = {
        "user_id": pa.array([str(u) for u in user_ids[eval_user_idx]], type=pa.string()),
        "user_idx": pa.array(np.asarray(eval_user_idx, dtype=np.int32), type=pa.int32()),
        "segment": pa.array([str(s) for s in segment_labels], type=pa.string()),
    }
    for name, vec in metric_vecs.items():
        cols[name] = pa.array(np.asarray(vec, dtype=np.float64), type=pa.float64())
    cols["novelty@10"] = pa.array(np.asarray(novelty_vec, dtype=np.float64), type=pa.float64())
    cols["top50"] = pa.array(
        [row.tolist() for row in np.asarray(topk_rows, dtype=np.int32)],
        type=pa.list_(pa.int32()),
    )
    pq.write_table(pa.table(cols), path)


# --- public entry point ------------------------------------------------------


def run_eval(
    config: dict,
    config_path: str | Path,
    results_path: str | Path,
    allow_stale: bool = False,
    run_id: str | None = None,
    manifest_path: str | Path | None = None,
    splits_path: str | Path | None = None,
) -> dict:
    """Run one eval config end-to-end and append its record. Returns the record.

    ``manifest_path`` / ``splits_path`` default to the repo's committed
    ``data/MANIFEST.md`` / ``configs/splits.yaml`` (overridable for tests).
    """
    t0 = time.monotonic()

    protocol_cfg = config["protocol"]
    eval_split = protocol_cfg["eval_split"]
    knowledge_cutoff = protocol_cfg.get("knowledge_cutoff", "train_end")
    k_list = tuple(int(k) for k in protocol_cfg.get("k_list", DEFAULT_K_LIST))
    batch_size = int(protocol_cfg.get("batch_size", DEFAULT_BATCH_SIZE))

    boot = config.get("bootstrap", {})
    n_resamples = int(boot.get("n_resamples", 1000))
    boot_seed = int(boot.get("seed", 20260805))
    seeds_cfg = config.get("seeds", {}) or {}

    manifest_path = Path(manifest_path) if manifest_path is not None else runlog.DEFAULT_MANIFEST_PATH
    splits_path = Path(splits_path) if splits_path is not None else runlog.DEFAULT_SPLITS_PATH
    warehouse = config.get("warehouse", "data/warehouse")

    # --- load cache + fit model ---
    cache_dir = _resolve_cache_dir(config["cache_dir"])
    ds = load_dataset(cache_dir)
    manifest = ds.manifest

    git = runlog.git_info()

    # --- integrity guards (before any scoring) ---
    runlog.check_stale_cache(manifest["snapshot_ids"], warehouse, allow_stale)
    runlog.check_test_dirty(eval_split, git["git_dirty"])

    model = _build_model(config["model"], seeds_cfg)
    model.fit(ds)

    gt = ds.gt[eval_split]
    eval_user_idx = np.asarray(gt.user_idx, dtype=np.int64)
    n_users = len(eval_user_idx)
    n_items = len(ds.item_ids)
    if n_users == 0:
        raise RuntimeError(f"no eval users for split {eval_split!r}")

    item_train_counts = np.asarray(ds.train_csr.sum(axis=0)).ravel()

    metric_vecs, topk_rows = _score_all(ds, model, eval_user_idx, gt, batch_size, k_list)
    topk10 = topk_rows[:, : min(10, topk_rows.shape[1])]

    segment_labels = np.asarray([str(s) for s in segment_of(ds.n_train[eval_user_idx])])
    novelty_vec = novelty_per_user(topk10, item_train_counts)

    # --- bootstrap CIs (global + per segment) ---
    global_metrics: dict[str, dict] = {}
    per_segment: dict[str, dict] = {}
    for name, vec in metric_vecs.items():
        global_metrics[name] = ci_mean(vec, n_resamples=n_resamples, seed=boot_seed)
        seg = segment_cis(vec, segment_labels, n_resamples=n_resamples, seed=boot_seed)
        for label, d in seg.items():
            blk = per_segment.setdefault(label, {"n_users": d["n_users"]})
            blk[name] = {"value": d["value"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]}

    metrics_block = {"global": global_metrics, "per_segment": per_segment}

    # --- beyond-accuracy ---
    novelty_ci = ci_mean(novelty_vec, n_resamples=n_resamples, seed=boot_seed)
    beyond_accuracy = {
        "coverage@10": coverage_at_k(topk10, n_items),
        "pop_share@10": pop_share_at_k(topk10, item_train_counts),
        "catalog_gini": gini(item_train_counts),
        "novelty@10": novelty_ci,
    }

    # --- run identity + artifact ---
    if run_id is None:
        run_id, run_ts = _resolve_run_id(None)
    else:
        run_ts = datetime.now(timezone.utc).isoformat()

    artifact_dir = Path(config.get("per_user_dir", "data/eval/per_user"))
    artifact_path = artifact_dir / f"{run_id}_{model.name}.parquet"
    _write_artifact(
        artifact_path,
        ds.user_ids,
        eval_user_idx,
        segment_labels,
        metric_vecs,
        novelty_vec,
        topk_rows,
    )

    protocol_block = {
        "eval_split": eval_split,
        "knowledge_cutoff": knowledge_cutoff,
        "exclusion": "train_seen",
        "catalog_size": n_items,
        "n_users": n_users,
        "k_list": list(k_list),
        "batch_size": batch_size,
    }

    record = runlog.build_record(
        kind="eval",
        run_id=run_id,
        run_ts=run_ts,
        git_sha=git["git_sha"],
        git_dirty=git["git_dirty"],
        config_path=config_path,
        config_hash=runlog.config_hash(config_path),
        splits=runlog.splits_block(splits_path),
        dataset_manifest_hash=runlog.dataset_manifest_hash(manifest_path),
        iceberg_snapshots=manifest["snapshot_ids"],
        contracts=manifest["contract_identities"],
        protocol=protocol_block,
        model={"name": model.name, "params": model.params},
        seeds={"bootstrap": boot_seed, "model": seeds_cfg.get("model")},
        metrics=metrics_block,
        beyond_accuracy=beyond_accuracy,
        per_user_artifact=str(artifact_path),
        wall_clock_s=round(time.monotonic() - t0, 3),
        hardware=runlog.hardware_string(),
    )

    runlog.append_record(record, results_path)
    _print_summary(record)
    return record


def _print_summary(record: dict) -> None:
    g = record["metrics"]["global"]
    def fmt(m: str) -> str:
        d = g.get(m)
        if d is None:
            return f"{m}=n/a"
        return f"{m}={d['value']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]"

    print(
        f"[{record['model']['name']}] split={record['protocol']['eval_split']} "
        f"n_users={record['protocol']['n_users']}  {fmt('recall@20')}  {fmt('ndcg@10')}"
    )
    print("  per-segment NDCG@10:")
    for label in ("0", "1-4", "5-9", "10-19", "20+"):
        blk = record["metrics"]["per_segment"].get(label)
        if blk is None or "ndcg@10" not in blk:
            continue
        d = blk["ndcg@10"]
        print(
            f"    {label:>6}  n={blk['n_users']:>7}  "
            f"ndcg@10={d['value']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]"
        )
