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

Recipe ``v1_ml32m_title_genres_tags`` (Phase 9, T9-3b; EXPERIMENT_LOG "Phase 9
T9-3b preregistration" §3) mirrors it on the ML-32M lane: ``title + " " +
" ".join(genres) + " " + " ".join(tags_top10)``, same joiner, same
skip-empty rule, same model. Its tag-aggregation rule is a new degree of
freedom, so it is bound into the artifact identity through
:func:`recipe_hash`'s ``extra`` mapping — which enters the canonical JSON
**only when non-None**, mirroring the ``half_life_days``-enters-the-param-hash-
only-under-``time_decay`` pattern, so the Amazon recipe hash
(``1f7878ff82bf…``) is provably unchanged (pinned by
``tests/test_ml32m_recipe.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

RECIPE_ID = "v1_title_brand_cat_features"
RECIPE_ID_ML32M = "v1_ml32m_title_genres_tags"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 256
PROGRESS_EVERY = 50_000

DEFAULT_TEXT_ROOT = Path("data/eval/text")
DEFAULT_ARTIFACT_ROOT = Path("data/eval/minilm")
DEFAULT_CACHE_ROOT = Path("data/eval/cache")

DEFAULT_TEXT_ROOT_ML32M = Path("data/eval/text_ml32m")
DEFAULT_ARTIFACT_ROOT_ML32M = Path("data/eval/minilm_ml32m")
DEFAULT_CACHE_ROOT_ML32M = Path("data/eval/cache_ml32m")


# --- hashing / recipe --------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def item_ids_sha256(item_ids: list[str]) -> str:
    """Same convention as item_text export_manifest.json's item_ids_sha256."""
    return hashlib.sha256("\n".join(item_ids).encode("utf-8")).hexdigest()


def recipe_hash(
    recipe_id: str,
    fields: list[str],
    joiner: str,
    model_id: str,
    extra: Mapping[str, object] | None = None,
) -> str:
    """SHA-256 of the canonical recipe spec.

    ``extra`` binds recipe-specific degrees of freedom that are not expressible
    as a field list (T9-3b §3h: the ML-32M tag source, cutoff, normalization,
    weight, ordering and cap). It enters the canonical JSON **only when
    non-None** — the same "optional key only under the option that needs it"
    pattern as ``half_life_days`` in the ALS param hash — so every recipe hash
    recorded before this parameter existed is byte-identical afterwards
    (asserted by ``tests/test_ml32m_recipe.py`` against the recorded Amazon
    prefix ``1f7878ff82bf``).
    """
    spec: dict[str, object] = {
        "recipe_id": recipe_id,
        "fields": fields,
        "joiner": joiner,
        "model_id": model_id,
    }
    if extra is not None:
        spec["extra"] = dict(extra)
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


def build_recipe_text_ml32m(row: dict) -> str:
    """Recipe ``v1_ml32m_title_genres_tags`` (T9-3b §3g, exact).

    ``" ".join`` of ``[title] + genres + tags_top10``, null/empty parts skipped,
    no separator tokens and no field labels — the same shape as
    :func:`build_recipe_text`. Titles keep their MovieLens year suffix
    (``"Toy Story (1995)"``): kept, not stripped, and disclosed. Over-long
    assembled text is truncated by the model's own ``max_seq_length=256``,
    deterministically, exactly as the Amazon recipe relies on.
    """
    parts: list[str] = []
    title = row.get("title")
    if title:
        parts.append(str(title))
    genres = row.get("genres")
    if genres is not None:
        parts.extend(str(g) for g in genres if g)
    tags = row.get("tags_top10")
    if tags is not None:
        parts.extend(str(t) for t in tags if t)
    return " ".join(parts)


# --- recipe registry ----------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """One frozen text recipe: identity, source columns, and default roots.

    ``fields`` is the hashed field list (recipe identity); ``source_columns`` is
    what the export parquet must carry to assemble the text. They coincide today
    on both lanes and are kept separate so a future recipe can hash a field name
    that is not literally a column.
    """

    recipe_id: str
    fields: tuple[str, ...]
    joiner: str
    model_id: str
    build_text: Callable[[dict], str]
    source_columns: tuple[str, ...]
    text_root: Path
    cache_root: Path
    artifact_root: Path
    extra: Mapping[str, object] | None = None

    def hash(self) -> str:
        return recipe_hash(
            self.recipe_id, list(self.fields), self.joiner, self.model_id, self.extra
        )


AMAZON_RECIPE = Recipe(
    recipe_id=RECIPE_ID,
    fields=("title", "brand_norm", "main_category", "features"),
    joiner=" ",
    model_id=MODEL_ID,
    build_text=build_recipe_text,
    source_columns=("title", "brand_norm", "main_category", "features"),
    text_root=DEFAULT_TEXT_ROOT,
    cache_root=DEFAULT_CACHE_ROOT,
    artifact_root=DEFAULT_ARTIFACT_ROOT,
    extra=None,  # NEVER set: this is what keeps 1f7878ff82bf… byte-identical.
)

# T9-3b §3(h), verbatim. Every value here is normative preregistered text — a
# change to any string is a change to the recipe's identity (and therefore to
# the artifact path), which is exactly the point of hashing it.
ML32M_RECIPE_EXTRA: dict[str, object] = {
    "tag_source": "local.silver_ml32m.tags",
    "tag_cutoff": "2022-06-30T23:59:59.999Z",  # inclusive
    "tag_norm": "silver_sanitized|lower|trim",
    "tag_weight": "count_distinct_user_id",
    "tag_order": "weight_desc,tag_asc",
    "tag_top_k": 10,
    "genres_source": "local.gold_ml32m.item_features.genres",
    "genres_order": "as_stored",
    "empty_policy": "skip",
}

ML32M_RECIPE = Recipe(
    recipe_id=RECIPE_ID_ML32M,
    fields=("title", "genres", "tags_top10"),
    joiner=" ",
    model_id=MODEL_ID,  # the SAME locally cached artifact the Amazon recipe used
    build_text=build_recipe_text_ml32m,
    source_columns=("title", "genres", "tags_top10"),
    text_root=DEFAULT_TEXT_ROOT_ML32M,
    cache_root=DEFAULT_CACHE_ROOT_ML32M,
    artifact_root=DEFAULT_ARTIFACT_ROOT_ML32M,
    extra=ML32M_RECIPE_EXTRA,
)

RECIPES: dict[str, Recipe] = {"amazon": AMAZON_RECIPE, "ml32m": ML32M_RECIPE}


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
    recipe: Recipe = AMAZON_RECIPE,
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

    fields = list(recipe.fields)
    rhash = recipe.hash()
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
    missing_cols = [c for c in recipe.source_columns if c not in table.column_names]
    if missing_cols:
        raise AssertionError(
            f"item_text.parquet at {text_dir} is missing recipe "
            f"{recipe.recipe_id} source columns {missing_cols}"
        )
    columns = {name: table.column(name).to_pylist() for name in recipe.source_columns}

    texts: list[str] = []
    for i in range(n_items):
        row = {name: values[i] for name, values in columns.items()}
        texts.append(recipe.build_text(row))

    device, mps_failure_detail = _resolve_device()

    started_ts = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(recipe.model_id, device=device)
    except Exception as exc:
        if device == "mps":
            mps_failure_detail = f"model load/encode on mps failed: {exc!r}"
            device = "cpu"
            model = SentenceTransformer(recipe.model_id, device=device)
        else:
            raise

    # Model revision: sentence-transformers caches under the HF hub cache; pull
    # the snapshot commit hash via huggingface_hub's local cache scan.
    model_revision = None
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == recipe.model_id and repo.repo_type == "model":
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
                model = SentenceTransformer(recipe.model_id, device=device)
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
        "model_id": recipe.model_id,
        # §3(h): the resolved revision of the LOCALLY CACHED model artifact is
        # recorded as provenance — no revision hash is preregistered, because
        # none had been verified at registration time.
        "model_revision": model_revision,
        "recipe_id": recipe.recipe_id,
        "recipe_fields": fields,
        "recipe_joiner": recipe.joiner,
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
    # Only recipes that HAVE an `extra` carry the key, so an Amazon manifest
    # written today is shaped exactly like the recorded one.
    if recipe.extra is not None:
        manifest["recipe_extra"] = dict(recipe.extra)
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
    parser.add_argument(
        "--recipe",
        choices=sorted(RECIPES),
        default="amazon",
        help="text recipe / lane: 'amazon' (v1_title_brand_cat_features) or "
        "'ml32m' (v1_ml32m_title_genres_tags). Selects the recipe's default "
        "text/cache/artifact roots unless they are given explicitly.",
    )
    parser.add_argument("--text-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--artifact-root", default=None)
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

    recipe = RECIPES[args.recipe]
    text_root = Path(args.text_root) if args.text_root else recipe.text_root
    cache_root = Path(args.cache_root) if args.cache_root else recipe.cache_root
    artifact_root = Path(args.artifact_root) if args.artifact_root else recipe.artifact_root

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
    cache_dir = cache_root / snap

    if args.neighbors is not None:
        # Locate the artifact dir for this snapshot (same recipe as embed_items).
        artifact_snap_dir = artifact_root / snap
        subdirs = [p for p in artifact_snap_dir.iterdir() if p.is_dir()] if artifact_snap_dir.exists() else []
        if len(subdirs) != 1:
            raise ValueError(
                f"expected exactly one recipe-hash subdir under {artifact_snap_dir}, found "
                f"{[p.name for p in subdirs]}"
            )
        spot_check(subdirs[0], text_dir, args.neighbors, k=args.k)
        return 0

    embed_items(text_dir, cache_dir, artifact_root=artifact_root, recipe=recipe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
