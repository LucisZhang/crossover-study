"""MiniLM item-embedding — Step A (Phase 4, T10; UPGRADE_PLAN.md §8).

JVM-free job (no pyspark import) that reads the T9 export
(``data/eval/text/<five_core_snapshot_id>/item_text.parquet`` +
``export_manifest.json``), re-verifies alignment to the eval cache's
``item_ids`` order, assembles a fixed text recipe per item, embeds with
sentence-transformers ``all-MiniLM-L6-v2`` (MPS if available else CPU), and
writes a snapshot + recipe-hash-keyed artifact:
``data/eval/minilm/<five_core_snapshot_id>/<recipe_hash_short>/``:

* ``embeddings.npy`` — fp16, shape (n_items, 384), row-aligned to item_ids.
* ``minilm_manifest.json`` — full provenance (model id/revision, recipe,
  source hashes, embedding hash, library/hardware versions, timings).

Idempotent: if the artifact dir already has a manifest with matching
``recipe_hash`` and source hashes, skip (exit 0) without loading the model.

Recipe v1 (``v1_title_brand_cat_features``): ``title + " " + brand_norm + " "
+ main_category + " " + " ".join(features)``, skipping null/empty parts.
``description``/``categories`` are NOT part of this recipe. ``brand_norm``
"unknown" is kept as-is (not treated as missing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

RECIPE_ID = "v1_title_brand_cat_features"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 256
PROGRESS_EVERY = 50_000

DEFAULT_TEXT_ROOT = Path("data/eval/text")
DEFAULT_ARTIFACT_ROOT = Path("data/eval/minilm")
DEFAULT_CACHE_ROOT = Path("data/eval/cache")


# --- hashing / recipe --------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def item_ids_sha256(item_ids: list[str]) -> str:
    """Same convention as item_text export_manifest.json's item_ids_sha256."""
    return hashlib.sha256("\n".join(item_ids).encode("utf-8")).hexdigest()


def recipe_hash(recipe_id: str, fields: list[str], joiner: str, model_id: str) -> str:
    spec = {
        "recipe_id": recipe_id,
        "fields": fields,
        "joiner": joiner,
        "model_id": model_id,
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(canonical.encode("utf-8"))


def build_recipe_text(row: dict) -> str:
    """Recipe v1: title, brand_norm, main_category, features (joined), skipping
    null/empty parts. ``brand_norm == "unknown"`` is kept as-is."""
    parts: list[str] = []
    title = row.get("title")
    if title:
        parts.append(str(title))
    brand = row.get("brand_norm")
    if brand:
        parts.append(str(brand))
    main_cat = row.get("main_category")
    if main_cat:
        parts.append(str(main_cat))
    features = row.get("features")
    if features:
        parts.extend(str(f) for f in features if f)
    return " ".join(parts)


# --- provenance helpers -------------------------------------------------------


def _git_short_sha() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _hardware_description() -> str:
    chip = None
    try:
        proc = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            chip = proc.stdout.strip()
    except (OSError, FileNotFoundError):
        chip = None
    plat = platform.platform()
    return f"{plat} · {chip}" if chip else plat


def _resolve_device() -> tuple[str, str | None]:
    """Returns (device, failure_detail). failure_detail is set only if MPS
    was attempted and failed, in which case device falls back to 'cpu'."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps", None
    except Exception as exc:  # pragma: no cover - defensive
        return "cpu", f"MPS probe failed: {exc!r}"
    return "cpu", None


# --- manifest / idempotency ---------------------------------------------------


def _artifact_up_to_date(
    adir: Path,
    *,
    recipe_hash_val: str,
    export_parquet_sha256: str,
    item_ids_sha256_val: str,
) -> bool:
    man_path = adir / "minilm_manifest.json"
    if not man_path.exists() or not (adir / "embeddings.npy").exists():
        return False
    try:
        man = json.loads(man_path.read_text())
    except (OSError, ValueError):
        return False
    return (
        man.get("recipe_hash") == recipe_hash_val
        and man.get("source_export_parquet_sha256") == export_parquet_sha256
        and man.get("item_ids_sha256") == item_ids_sha256_val
    )


# --- main embed job -------------------------------------------------------


def embed_items(
    text_dir: Path,
    cache_dir: Path,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    text_dir = Path(text_dir)
    cache_dir = Path(cache_dir)

    export_manifest_path = text_dir / "export_manifest.json"
    export_parquet_path = text_dir / "item_text.parquet"
    if not export_manifest_path.exists() or not export_parquet_path.exists():
        raise FileNotFoundError(
            f"T9 export not found at {text_dir}; run `make item-text-export` first"
        )
    export_manifest = json.loads(export_manifest_path.read_text())
    five_core_snapshot = int(export_manifest["five_core_snapshot_id"])
    if not export_manifest.get("aligned_to_cache"):
        raise AssertionError(f"{export_manifest_path} aligned_to_cache is not True")

    # Re-verify the export parquet's own sha256 (don't trust the manifest blindly).
    export_parquet_sha256 = sha256_file(export_parquet_path)
    if export_parquet_sha256 != export_manifest.get("parquet_sha256"):
        raise AssertionError(
            f"item_text.parquet sha256 mismatch: recomputed {export_parquet_sha256} "
            f"!= manifest {export_manifest.get('parquet_sha256')}"
        )

    table = pq.read_table(export_parquet_path)
    parent_asins = table.column("parent_asin").to_pylist()

    # Re-verify item_ids_sha256 against the recomputed parent_asin sequence.
    recomputed_item_ids_sha256 = item_ids_sha256(parent_asins)
    if recomputed_item_ids_sha256 != export_manifest.get("item_ids_sha256"):
        raise AssertionError(
            "item_text.parquet parent_asin sequence sha256 mismatch vs "
            f"export_manifest.json: recomputed {recomputed_item_ids_sha256} != "
            f"manifest {export_manifest.get('item_ids_sha256')}"
        )

    # Re-verify against the eval cache's item_ids.parquet directly.
    cache_item_ids_path = cache_dir / "item_ids.parquet"
    if not cache_item_ids_path.exists():
        raise FileNotFoundError(f"eval cache item_ids not found at {cache_item_ids_path}")
    cache_item_ids = pq.read_table(cache_item_ids_path).column("item_id").to_pylist()
    cache_item_ids_sha256 = item_ids_sha256(cache_item_ids)
    if cache_item_ids_sha256 != recomputed_item_ids_sha256:
        raise AssertionError(
            "item_text.parquet parent_asin sequence does not match eval cache "
            f"item_ids order: {recomputed_item_ids_sha256} != {cache_item_ids_sha256}"
        )

    fields = ["title", "brand_norm", "main_category", "features"]
    rhash = recipe_hash(RECIPE_ID, fields, " ", MODEL_ID)
    rhash_short = rhash[:12]

    adir = Path(artifact_root) / str(five_core_snapshot) / rhash_short

    if _artifact_up_to_date(
        adir,
        recipe_hash_val=rhash,
        export_parquet_sha256=export_parquet_sha256,
        item_ids_sha256_val=recomputed_item_ids_sha256,
    ):
        print(f"up to date: {adir}")
        return adir

    n_items = table.num_rows
    titles = table.column("title").to_pylist()
    brands = table.column("brand_norm").to_pylist()
    main_cats = table.column("main_category").to_pylist()
    features_col = table.column("features").to_pylist()

    texts: list[str] = []
    for i in range(n_items):
        row = {
            "title": titles[i],
            "brand_norm": brands[i],
            "main_category": main_cats[i],
            "features": features_col[i],
        }
        texts.append(build_recipe_text(row))

    device, mps_failure_detail = _resolve_device()

    started_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(MODEL_ID, device=device)
    except Exception as exc:
        if device == "mps":
            mps_failure_detail = f"model load/encode on mps failed: {exc!r}"
            device = "cpu"
            model = SentenceTransformer(MODEL_ID, device=device)
        else:
            raise

    # Model revision: sentence-transformers caches under the HF hub cache; pull
    # the snapshot commit hash via huggingface_hub's local cache scan.
    model_revision = None
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == MODEL_ID and repo.repo_type == "model":
                revisions = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                if revisions:
                    model_revision = revisions[0].commit_hash
                break
    except Exception:
        model_revision = None

    embeddings = np.zeros((n_items, EMBED_DIM), dtype=np.float32)
    for start in range(0, n_items, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_items)
        batch = texts[start:end]
        try:
            out = model.encode(
                batch,
                batch_size=BATCH_SIZE,
                normalize_embeddings=False,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            if device == "mps":
                mps_failure_detail = f"encode on mps failed at batch starting {start}: {exc!r}"
                device = "cpu"
                model = SentenceTransformer(MODEL_ID, device=device)
                out = model.encode(
                    batch,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=False,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            else:
                raise
        embeddings[start:end] = out.astype(np.float32)
        if end % PROGRESS_EVERY < BATCH_SIZE or end == n_items:
            print(f"embedded {end}/{n_items} items")

    embeddings_fp16 = embeddings.astype(np.float16)

    wall = round(time.perf_counter() - t0, 3)
    finished_ts = datetime.now(timezone.utc).isoformat()

    adir.mkdir(parents=True, exist_ok=True)
    emb_path = adir / "embeddings.npy"
    np.save(emb_path, embeddings_fp16, allow_pickle=False)
    embeddings_sha256 = sha256_file(emb_path)

    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "recipe_id": RECIPE_ID,
        "recipe_fields": fields,
        "recipe_joiner": " ",
        "recipe_hash": rhash,
        "recipe_hash_short": rhash_short,
        "five_core_snapshot_id": five_core_snapshot,
        "source_export_parquet_sha256": export_parquet_sha256,
        "item_ids_sha256": recomputed_item_ids_sha256,
        "embeddings_sha256": embeddings_sha256,
        "embedding_dim": EMBED_DIM,
        "embedding_dtype": "float16",
        "row_count": n_items,
        "torch_version": torch.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "transformers_version": transformers.__version__,
        "device": device,
        "mps_failure_detail": mps_failure_detail,
        "hardware": _hardware_description(),
        "batch_size": BATCH_SIZE,
        "wall_clock_s": wall,
        "started_ts": started_ts,
        "finished_ts": finished_ts,
        "git_sha": _git_short_sha(),
    }
    (adir / "minilm_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"embedded MiniLM: {adir}  n_items={n_items} dim={EMBED_DIM} "
        f"device={device} wall={wall}s"
    )
    return adir


# --- spot-check ----------------------------------------------------------


def spot_check(adir: Path, text_dir: Path, query: str, k: int = 5) -> None:
    table = pq.read_table(text_dir / "item_text.parquet")
    parent_asins = table.column("parent_asin").to_pylist()
    titles = table.column("title").to_pylist()

    embeddings = np.load(adir / "embeddings.npy", allow_pickle=False).astype(np.float32)

    query_idx = None
    for i, (asin, title) in enumerate(zip(parent_asins, titles)):
        if asin == query or (title and query.lower() in title.lower()):
            query_idx = i
            break
    if query_idx is None:
        raise ValueError(f"no item matched query {query!r} (parent_asin or title substring)")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms
    sims = normed @ normed[query_idx]
    order = np.argsort(-sims)

    print(f"query: parent_asin={parent_asins[query_idx]!r} title={titles[query_idx]!r}")
    shown = 0
    for idx in order:
        if idx == query_idx:
            continue
        print(f"  {shown + 1}. sim={sims[idx]:.4f} asin={parent_asins[idx]!r} title={titles[idx]!r}")
        shown += 1
        if shown >= k:
            break


# --- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.models.minilm_embed")
    parser.add_argument("--text-root", default=str(DEFAULT_TEXT_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument(
        "--five-core-snapshot",
        default=None,
        help="explicit five_core_snapshot_id; defaults to the single subdir under --text-root",
    )
    parser.add_argument(
        "--neighbors",
        default=None,
        help="spot-check mode: parent_asin or title substring to query neighbors for",
    )
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    text_root = Path(args.text_root)
    if args.five_core_snapshot is not None:
        snap = args.five_core_snapshot
    else:
        subdirs = [p for p in text_root.iterdir() if p.is_dir()]
        if len(subdirs) != 1:
            raise ValueError(
                f"expected exactly one snapshot subdir under {text_root}, found "
                f"{[p.name for p in subdirs]}; pass --five-core-snapshot"
            )
        snap = subdirs[0].name

    text_dir = text_root / snap
    cache_dir = Path(args.cache_root) / snap

    if args.neighbors is not None:
        # Locate the artifact dir for this snapshot (same recipe as embed_items).
        artifact_snap_dir = Path(args.artifact_root) / snap
        subdirs = [p for p in artifact_snap_dir.iterdir() if p.is_dir()] if artifact_snap_dir.exists() else []
        if len(subdirs) != 1:
            raise ValueError(
                f"expected exactly one recipe-hash subdir under {artifact_snap_dir}, found "
                f"{[p.name for p in subdirs]}"
            )
        spot_check(subdirs[0], text_dir, args.neighbors, k=args.k)
        return 0

    embed_items(text_dir, cache_dir, artifact_root=args.artifact_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
