"""ANN index artifact + latency/overlap receipt (Phase 4, T16; UPGRADE_PLAN.md §8).

Demo-facing artifact ONLY. hnswlib is an approximate-nearest-neighbor index
built over the T10 MiniLM item embeddings for the pick-a-shopper demo's
snappy neighbor lookups. It is NEVER used to compute anything that appears in
``results/runs.jsonl`` as a ``kind="eval"`` / ``kind="paired_delta"`` record —
those all score the full catalog by exact chunked matmul (CLAUDE.md invariant
#4). This module appends exactly one ``kind="ann_receipt"`` record, carrying
an explicit ``used_in_eval_metrics: false`` field, so that boundary is
provable from the log itself.

JVM-free; runs under the torch-carrying ``embed`` dependency group (hnswlib is
declared there) — see Makefile target ``ann-index``.

Artifact layout: ``<embeddings artifact dir>/ann_index.bin`` +
``ann_manifest.json``, i.e. sibling to the T10
``data/eval/minilm/<snapshot>/<recipe_hash_short>/{embeddings.npy,minilm_manifest.json}``.

Idempotent: if ``ann_index.bin`` + ``ann_manifest.json`` already exist and the
manifest's source-embeddings hash / build parameters match the requested
build, the build step is skipped (exit without re-building). ``--measure``
runs the receipt measurement regardless (against whatever index is on disk).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.dataset import load_dataset
from batch_recsys_lab.models.content import l2_normalize_rows

DEFAULT_ARTIFACT_ROOT = Path("data/eval/minilm")
DEFAULT_CACHE_ROOT = Path("data/eval/cache")
DEFAULT_RESULTS_PATH = Path("results/runs.jsonl")

DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 200  # default ef_search recorded in the manifest for the built index
EF_SEARCH_GRID = (50, 100, 200)

RECEIPT_N_USERS = 10_000
RECEIPT_SEED = 20260805
TOPK = 10
MATMUL_CHUNK = 2048


# --- provenance helpers (mirrors minilm_embed.py conventions) ----------------


def _git_short_sha() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _hardware_description() -> str:
    chip = None
    try:
        proc = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if proc.returncode == 0:
            chip = proc.stdout.strip()
    except (OSError, FileNotFoundError):
        chip = None
    plat = platform.platform()
    return f"{plat} · {chip}" if chip else plat


# --- index build ---------------------------------------------------------


def _index_up_to_date(
    adir: Path,
    *,
    embeddings_sha256: str,
    m: int,
    ef_construction: int,
) -> bool:
    man_path = adir / "ann_manifest.json"
    bin_path = adir / "ann_index.bin"
    if not man_path.exists() or not bin_path.exists():
        return False
    try:
        man = json.loads(man_path.read_text())
    except (OSError, ValueError):
        return False
    return (
        man.get("source_embeddings_sha256") == embeddings_sha256
        and man.get("M") == m
        and man.get("ef_construction") == ef_construction
    )


def build_index(
    adir: Path,
    *,
    m: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF_CONSTRUCTION,
    ef_search: int = DEFAULT_EF_SEARCH,
) -> tuple[Path, dict]:
    """Build (or skip, if up to date) the hnswlib cosine index over the L2-
    normalized fp32 embeddings at ``adir/embeddings.npy``. Returns
    ``(adir, manifest_dict)``."""
    import hnswlib
    from importlib.metadata import version as _pkg_version

    man_path = adir / "minilm_manifest.json"
    emb_path = adir / "embeddings.npy"
    if not man_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"MiniLM embedding artifact not found at {adir} (expected "
            "minilm_manifest.json + embeddings.npy). Run `make embed-items` first."
        )
    minilm_manifest = json.loads(man_path.read_text())
    embeddings_sha256 = minilm_manifest["embeddings_sha256"]

    ann_man_path = adir / "ann_manifest.json"
    ann_bin_path = adir / "ann_index.bin"

    if _index_up_to_date(
        adir, embeddings_sha256=embeddings_sha256, m=m, ef_construction=ef_construction
    ):
        print(f"up to date: {ann_bin_path}")
        return adir, json.loads(ann_man_path.read_text())

    E = np.load(emb_path, allow_pickle=False).astype(np.float32, copy=False)
    E_norm = l2_normalize_rows(E)
    n_items, dim = E_norm.shape

    started_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n_items, ef_construction=ef_construction, M=m)
    index.add_items(E_norm, np.arange(n_items))
    index.set_ef(ef_search)

    wall = round(time.perf_counter() - t0, 3)
    finished_ts = datetime.now(timezone.utc).isoformat()

    index.save_index(str(ann_bin_path))

    manifest = {
        "schema_version": 1,
        "hnswlib_version": _pkg_version("hnswlib"),
        "space": "cosine",
        "M": m,
        "ef_construction": ef_construction,
        "ef_search": ef_search,
        "element_count": int(n_items),
        "dim": int(dim),
        "source_embeddings_path": str(emb_path),
        "source_embeddings_sha256": embeddings_sha256,
        "source_minilm_recipe_hash": minilm_manifest.get("recipe_hash"),
        "five_core_snapshot_id": minilm_manifest.get("five_core_snapshot_id"),
        "build_wall_clock_s": wall,
        "started_ts": started_ts,
        "finished_ts": finished_ts,
        "hardware": _hardware_description(),
        "git_sha": _git_short_sha(),
    }
    ann_man_path.write_text(json.dumps(manifest, indent=2))
    print(f"built ANN index: {ann_bin_path}  n_items={n_items} dim={dim} wall={wall}s")
    return adir, manifest


# --- receipt measurement --------------------------------------------------


def _content_profiles(train_csr, E_norm: np.ndarray, user_idx: np.ndarray) -> np.ndarray:
    """Same profile computation as ``ContentRecommender.score_batch`` (mean-
    pooled, L2-normalized TRAIN-item embeddings; cold users -> zero vector)."""
    sub = train_csr[user_idx]
    counts = np.asarray(sub.sum(axis=1)).ravel()
    safe_counts = np.where(counts == 0, 1.0, counts)
    summed = sub @ E_norm
    profiles = np.asarray(summed, dtype=np.float32) / safe_counts[:, None]
    profiles[counts == 0] = 0.0
    return l2_normalize_rows(profiles)


def _exact_topk_batch(profiles: np.ndarray, E_norm: np.ndarray, k: int, chunk: int = MATMUL_CHUNK):
    """Chunked exact brute-force cosine top-k over all queries at once (batched
    matmul — used for both the exact top-k answer and the amortized exact
    per-query latency estimate). Returns (topk_idx (Q,k), wall_clock_s)."""
    n_q = profiles.shape[0]
    topk_idx = np.zeros((n_q, k), dtype=np.int64)
    t0 = time.perf_counter()
    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        sims = profiles[start:end] @ E_norm.T  # (b, n_items)
        # argpartition for top-k, then order by similarity descending.
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(end - start)[:, None]
        order = np.argsort(-sims[row_idx, part], axis=1)
        topk_idx[start:end] = part[row_idx, order]
    wall = time.perf_counter() - t0
    return topk_idx, wall


def measure_receipt(
    adir: Path,
    cache_dir: Path,
    *,
    n_users: int = RECEIPT_N_USERS,
    seed: int = RECEIPT_SEED,
    k: int = TOPK,
    ef_grid: tuple[int, ...] = EF_SEARCH_GRID,
) -> dict:
    import hnswlib

    ds = load_dataset(cache_dir)
    E = np.load(adir / "embeddings.npy", allow_pickle=False).astype(np.float32, copy=False)
    E_norm = l2_normalize_rows(E)
    n_items, dim = E_norm.shape

    rng = np.random.default_rng(seed)
    eligible = np.nonzero(ds.n_train > 0)[0]
    n_sample = min(n_users, len(eligible))
    sampled_users = rng.choice(eligible, size=n_sample, replace=False)
    sampled_users.sort()

    profiles = _content_profiles(ds.train_csr, E_norm, sampled_users)

    # Exact top-k, batched (amortized latency regime).
    exact_topk, exact_batched_wall = _exact_topk_batch(profiles, E_norm, k)
    exact_per_query_amortized_s = exact_batched_wall / n_sample

    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(adir / "ann_index.bin"), max_elements=n_items)

    ef_results: dict[str, dict] = {}
    for ef in ef_grid:
        index.set_ef(max(ef, k))
        overlaps = np.zeros(n_sample, dtype=np.float64)
        latencies = np.zeros(n_sample, dtype=np.float64)
        for i in range(n_sample):
            q = profiles[i : i + 1]
            t0 = time.perf_counter()
            labels, _ = index.knn_query(q, k=k)
            latencies[i] = time.perf_counter() - t0
            ann_set = set(labels[0].tolist())
            exact_set = set(exact_topk[i].tolist())
            overlaps[i] = len(ann_set & exact_set) / k
        ef_results[str(ef)] = {
            "mean_overlap_at_10": round(float(overlaps.mean()), 6),
            "ann_latency_median_s": round(float(np.median(latencies)), 8),
            "ann_latency_p95_s": round(float(np.percentile(latencies, 95)), 8),
        }

    return {
        "n_users_sampled": int(n_sample),
        "seed": seed,
        "k": k,
        "ef_grid": list(ef_grid),
        "per_ef": ef_results,
        "exact_batched_wall_clock_s": round(exact_batched_wall, 6),
        "exact_per_query_amortized_s": round(exact_per_query_amortized_s, 8),
        "exact_amortized_note": (
            "Exact latency is amortized from a batched chunked matmul over all "
            f"{n_sample} sampled users at once (throughput regime), not a "
            "single-query call; ANN latency above is measured single-threaded, "
            "one query at a time (serving regime). These are different serving "
            "regimes and are not directly comparable as a speedup ratio without "
            "that caveat."
        ),
        "matmul_chunk": MATMUL_CHUNK,
    }


# --- runs.jsonl record -----------------------------------------------------


def append_receipt_record(
    adir: Path,
    ann_manifest: dict,
    minilm_manifest: dict,
    receipt: dict,
    *,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
) -> dict:
    git = runlog.git_info()
    from batch_recsys_lab.contracts.engine import _resolve_run_id

    run_id, run_ts = _resolve_run_id(None)

    record = {
        "schema_version": runlog.record_schema_version,
        "kind": "ann_receipt",
        "run_id": run_id,
        "run_ts": run_ts,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "artifact": {
            "ann_index_bin": str(adir / "ann_index.bin"),
            "ann_manifest_json": str(adir / "ann_manifest.json"),
            "source_embeddings_npy": str(adir / "embeddings.npy"),
            "source_embeddings_sha256": ann_manifest.get("source_embeddings_sha256"),
            "source_minilm_recipe_hash": minilm_manifest.get("recipe_hash"),
            "five_core_snapshot_id": ann_manifest.get("five_core_snapshot_id"),
        },
        "parameters": {
            "hnswlib_version": ann_manifest.get("hnswlib_version"),
            "space": ann_manifest.get("space"),
            "M": ann_manifest.get("M"),
            "ef_construction": ann_manifest.get("ef_construction"),
            "ef_search_built": ann_manifest.get("ef_search"),
            "element_count": ann_manifest.get("element_count"),
            "dim": ann_manifest.get("dim"),
            "build_wall_clock_s": ann_manifest.get("build_wall_clock_s"),
            "hardware": ann_manifest.get("hardware"),
        },
        "receipt": receipt,
        "used_in_eval_metrics": False,
        "note": (
            "Demo-facing artifact only. Every kind='eval' / kind='paired_delta' "
            "record in this log uses exact full-catalog ranking via chunked "
            "matmul (CLAUDE.md invariant #4); this ANN index is never in that "
            "code path and does not affect any reported metric."
        ),
    }
    runlog.append_record(record, results_path)
    return record


# --- CLI -------------------------------------------------------------------


def _resolve_artifact_dir(artifact_root: Path, snapshot: str | None) -> Path:
    if snapshot is not None:
        snap = snapshot
    else:
        subdirs = [p for p in artifact_root.iterdir() if p.is_dir()]
        if len(subdirs) != 1:
            raise ValueError(
                f"expected exactly one snapshot subdir under {artifact_root}, found "
                f"{[p.name for p in subdirs]}; pass --five-core-snapshot"
            )
        snap = subdirs[0].name
    snap_dir = artifact_root / snap
    subdirs = [p for p in snap_dir.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise ValueError(
            f"expected exactly one recipe-hash subdir under {snap_dir}, found "
            f"{[p.name for p in subdirs]}"
        )
    return subdirs[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.models.ann_index")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--five-core-snapshot", default=None)
    parser.add_argument("--M", type=int, default=DEFAULT_M)
    parser.add_argument("--ef-construction", type=int, default=DEFAULT_EF_CONSTRUCTION)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument(
        "--measure", action="store_true", help="run the overlap/latency receipt and append to runs.jsonl"
    )
    parser.add_argument("--n-users", type=int, default=RECEIPT_N_USERS)
    parser.add_argument("--seed", type=int, default=RECEIPT_SEED)
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH))
    args = parser.parse_args(argv)

    artifact_root = Path(args.artifact_root)
    adir = _resolve_artifact_dir(artifact_root, args.five_core_snapshot)
    cache_dir = Path(args.cache_root) / adir.parent.name

    adir, ann_manifest = build_index(
        adir, m=args.M, ef_construction=args.ef_construction, ef_search=args.ef_search
    )

    if args.measure:
        minilm_manifest = json.loads((adir / "minilm_manifest.json").read_text())
        receipt = measure_receipt(adir, cache_dir, n_users=args.n_users, seed=args.seed)
        print(json.dumps(receipt, indent=2))
        record = append_receipt_record(
            adir, ann_manifest, minilm_manifest, receipt, results_path=args.results_path
        )
        print(f"appended ann_receipt record run_id={record['run_id']} to {args.results_path}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
