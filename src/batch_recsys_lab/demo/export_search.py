"""Semantic-search demo payload — Phase 6, T35 (plan §9 exhibit 3).

Projects the pinned MiniLM item embeddings onto the 50k most-popular items and
writes the five files the in-browser exhibit needs, into ``demo/data/search/``
(**not committed** — cut-order item #2):

* ``embeddings_int8.bin`` — ``rows × dim`` int8, C-order, per-row symmetric
  quantization (``scale_i = max|v_i| / 127``); rows are the 50k items of
  ``data/demo_export/search_items_raw.parquet`` **in file order**.
* ``scales_f32.bin`` — ``rows`` little-endian float32, one per row.
* ``embeddings_meta.json`` — shape, ordering rule, quantization spec, the
  SHA-256 of both ``.bin`` files, and the provenance chain up to the
  ``kind="ann_receipt"`` record.
* ``items_meta.json`` — parallel arrays of descriptive item metadata.
* ``example_queries.json`` — ~12 canned queries embedded with the **real**
  Python model and recipe pooling, with exact top-k reference results, each hit
  carrying its descriptive metadata inline so this one small file is a
  self-contained FALLBACK MODE for the exhibit.

Evidence class (important). Similarity scores here are a **capability
demonstration**, not evaluation evidence: they are cosine similarities over item
text, not a ranking metric on held-out interactions, and no full-catalog
protocol applies. The exhibit renders them without the traced-number affordance.
What is anchored is the *provenance* of the embeddings: this exporter re-hashes
``embeddings.npy`` on disk and refuses to run unless it matches both the MiniLM
manifest and ``source_embeddings_sha256`` on the ann_receipt record.

Ordering rule (re-asserted, not assumed): ``pop_train_end_365`` descending,
ties broken by **ascending catalog index** — the ``np.lexsort`` in
``demo/shopper_history_job.py``. The exporter recomputes that comparison over
the parquet's own columns and aborts on the first violation, so a re-generated
slice with a different tie-break cannot silently ship.

Row alignment: ``embeddings.npy`` row *i* is catalog index *i* (the eval cache's
``item_ids.parquet`` order, asserted at embed time by ``minilm_embed.py``), so
the slice is ``embeddings[catalog_index]`` in parquet order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

INT8_MAX = 127

DEFAULT_CONFIG = "configs/search_export.yaml"
SCHEMA_VERSION = 1


# --- small helpers -------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(doc: Any) -> str:
    """Same convention as demo/export_core.py: stable separators, trailing NL."""
    return json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def _fail(msg: str) -> "NoReturn":  # noqa: F821
    raise SystemExit(f"export_search: {msg}")


# --- pure functions the tests pin ----------------------------------------------


def quantize_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8 quantization.

    ``scale_i = max_j |mat[i, j]| / 127``; ``q = round(v / scale)`` clipped to
    ``[-127, 127]`` (``-128`` is never emitted, which keeps the codebook
    symmetric and makes the dequantized range exactly ``±max|v_i|``).

    An all-zero row gets ``scale_i = 0`` and an all-zero code row, which
    dequantizes back to exactly zero.

    Returns ``(int8 codes, float32 scales)``.
    """
    if mat.ndim != 2:
        raise ValueError(f"quantize_rows expects a 2-D matrix, got shape {mat.shape}")
    m = np.asarray(mat, dtype=np.float32)
    peak = np.abs(m).max(axis=1)
    scales = (peak / INT8_MAX).astype(np.float32)
    safe = np.where(scales > 0, scales, np.float32(1.0))[:, None]
    codes = np.rint(m / safe)
    np.clip(codes, -INT8_MAX, INT8_MAX, out=codes)
    return codes.astype(np.int8), scales


def dequantize_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_rows` (float32)."""
    return codes.astype(np.float32) * np.asarray(scales, dtype=np.float32)[:, None]


def verify_pop_ordering(pop: np.ndarray, catalog_index: np.ndarray) -> None:
    """Assert the slice is ``pop`` desc, ties broken by ascending catalog index.

    Raises ``ValueError`` naming the first offending adjacent pair. This is the
    ordering ``demo/shopper_history_job.py`` produces with
    ``np.lexsort((arange(n), -pop))``; re-deriving it here means a slice built
    with a different tie-break aborts the export instead of shipping.
    """
    pop = np.asarray(pop, dtype=np.float64)
    idx = np.asarray(catalog_index, dtype=np.int64)
    if pop.shape != idx.shape:
        raise ValueError(f"pop {pop.shape} and catalog_index {idx.shape} differ in length")
    if np.isnan(pop).any():
        raise ValueError("popularity column contains NaN")
    drop = np.nonzero(pop[1:] > pop[:-1])[0]
    if drop.size:
        i = int(drop[0])
        raise ValueError(
            f"popularity not descending at rows {i}->{i + 1}: {pop[i]} < {pop[i + 1]}"
        )
    tie = np.nonzero((pop[1:] == pop[:-1]) & (idx[1:] < idx[:-1]))[0]
    if tie.size:
        i = int(tie[0])
        raise ValueError(
            f"tie at popularity {pop[i]} broken the wrong way at rows {i}->{i + 1}: "
            f"catalog_index {idx[i]} then {idx[i + 1]} (expected ascending)"
        )


def top_k_rows(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, ties broken by ascending row index.

    Deterministic on every platform: ``argsort`` alone is not (kind-dependent
    tie order), so ties are resolved explicitly by the secondary key.
    """
    s = np.asarray(scores, dtype=np.float64)
    k = min(k, s.shape[0])
    return np.lexsort((np.arange(s.shape[0]), -s))[:k]


def overlap_at_k(a: list[Any], b: list[Any], k: int) -> int:
    """Set overlap of the first k elements of two ranked lists."""
    return len(set(a[:k]) & set(b[:k]))


# --- config --------------------------------------------------------------------


_REQUIRED = (
    "runs_log",
    "ann_receipt_run_id",
    "five_core_snapshot_id",
    "recipe_hash_short",
    "items_parquet",
    "popularity_column",
    "out_dir",
    "queries",
)


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        _fail(f"config {path}: missing required keys {missing}")
    if not cfg["queries"]:
        _fail(f"config {path}: queries[] is empty")
    if len(set(cfg["queries"])) != len(cfg["queries"]):
        _fail(f"config {path}: duplicate canned queries")
    return cfg


def find_record(runs_log: str | Path, run_id: str) -> dict:
    with open(runs_log, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("run_id") == run_id:
                return rec
    _fail(f"run_id {run_id!r} not found in {runs_log}")


# --- the export ------------------------------------------------------------------


def export(cfg: dict, *, skip_queries: bool = False) -> dict:
    out_dir = Path(cfg["out_dir"])
    snapshot_id = int(cfg["five_core_snapshot_id"])
    recipe_short = str(cfg["recipe_hash_short"])

    minilm_dir = Path(
        cfg["minilm_dir"]
        if cfg.get("minilm_dir")
        else Path("data/eval/minilm") / str(snapshot_id) / recipe_short
    )
    emb_path = minilm_dir / "embeddings.npy"
    man_path = minilm_dir / "minilm_manifest.json"
    for p in (emb_path, man_path):
        if not p.exists():
            _fail(f"missing MiniLM artifact {p}")
    manifest = json.loads(man_path.read_text())

    # --- provenance gate: the artifact on disk must be the recorded one --------
    record = find_record(cfg["runs_log"], cfg["ann_receipt_run_id"])
    if record.get("kind") != "ann_receipt":
        _fail(f"record {cfg['ann_receipt_run_id']} has kind {record.get('kind')!r}, expected ann_receipt")
    rec_art = record.get("artifact", {})
    print(f"hashing {emb_path} …", flush=True)
    emb_sha = sha256_file(emb_path)
    for label, expected in (
        ("minilm_manifest.embeddings_sha256", manifest.get("embeddings_sha256")),
        ("ann_receipt.artifact.source_embeddings_sha256", rec_art.get("source_embeddings_sha256")),
    ):
        if emb_sha != expected:
            _fail(
                f"{emb_path} sha256 {emb_sha} != {label} {expected} — the embeddings artifact "
                "is not the one the results log anchors; refusing to export"
            )
    if manifest.get("recipe_hash") != rec_art.get("source_minilm_recipe_hash"):
        _fail("minilm_manifest.recipe_hash disagrees with the ann_receipt record")
    if int(manifest.get("five_core_snapshot_id", -1)) != snapshot_id:
        _fail(f"minilm_manifest snapshot {manifest.get('five_core_snapshot_id')} != config {snapshot_id}")

    # --- the 50k slice ---------------------------------------------------------
    items_path = Path(cfg["items_parquet"])
    if not items_path.exists():
        _fail(f"missing item slice {items_path} — run `make demo-shoppers` (writes search_items_raw.parquet)")
    table = pq.read_table(items_path)
    pop_col = cfg["popularity_column"]
    for col in ("catalog_index", "item_id", pop_col, "title", "brand_norm", "price_usd", "main_category"):
        if col not in table.column_names:
            _fail(f"{items_path}: missing column {col!r}")

    catalog_index = np.asarray(table.column("catalog_index").to_pylist(), dtype=np.int64)
    item_ids = [str(x) for x in table.column("item_id").to_pylist()]
    pop = np.asarray(table.column(pop_col).to_pylist(), dtype=np.float64)
    rows = len(item_ids)
    expected_rows = cfg.get("expected_rows")
    if expected_rows is not None and rows != int(expected_rows):
        _fail(f"{items_path} has {rows} rows, config expects {expected_rows}")
    if len(set(item_ids)) != rows:
        _fail(f"{items_path}: item_id column is not unique")

    try:
        verify_pop_ordering(pop, catalog_index)
    except ValueError as exc:
        _fail(f"{items_path} ordering check failed: {exc}")
    print(f"ordering verified: {pop_col} desc, ties by ascending catalog index ({rows} rows)")

    # --- slice + normalize + quantize ------------------------------------------
    full = np.load(emb_path, mmap_mode="r", allow_pickle=False)
    dim = int(manifest["embedding_dim"])
    if full.shape != (int(manifest["row_count"]), dim):
        _fail(f"{emb_path} shape {full.shape} != manifest ({manifest['row_count']}, {dim})")
    if catalog_index.min() < 0 or catalog_index.max() >= full.shape[0]:
        _fail(f"catalog_index out of range for a catalog of {full.shape[0]} items")

    sliced = np.asarray(full[catalog_index], dtype=np.float32)
    norms = np.linalg.norm(sliced, axis=1)
    # The recipe's SentenceTransformer ends in a Normalize module, so the stored
    # fp16 rows are already unit-norm to fp16 precision; re-normalizing in
    # float32 removes that rounding so cosine == dot product exactly.
    max_norm_drift = float(np.abs(norms - 1.0).max())
    if max_norm_drift > 1e-2:
        _fail(
            f"stored embeddings are not unit-norm (max |‖v‖-1| = {max_norm_drift:.4g}); "
            "the recipe's Normalize assumption no longer holds"
        )
    safe_norms = np.where(norms > 0, norms, np.float32(1.0))[:, None]
    unit = (sliced / safe_norms).astype(np.float32)

    codes, scales = quantize_rows(unit)
    deq = dequantize_rows(codes, scales)
    err = np.abs(deq - unit)
    cos = (deq * unit).sum(axis=1) / np.maximum(np.linalg.norm(deq, axis=1), 1e-12)

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "embeddings_int8.bin"
    scales_path = out_dir / "scales_f32.bin"
    bin_path.write_bytes(np.ascontiguousarray(codes, dtype=np.int8).tobytes(order="C"))
    scales_path.write_bytes(np.ascontiguousarray(scales.astype("<f4")).tobytes(order="C"))
    bin_sha = sha256_file(bin_path)
    scales_sha = sha256_file(scales_path)

    quant = {
        "scheme": "per_row_symmetric_int8",
        "formula": "scale_i = max_j |v_ij| / 127 ; q_ij = clip(round(v_ij / scale_i), -127, 127)",
        "int8_max": INT8_MAX,
        "layout": "C-order, rows x dim, one int8 per component",
        "scales_dtype": "little-endian float32, one per row",
        "pre_quantization": "float32 L2-normalisation of the stored fp16 row",
        "measured_max_abs_component_error": float(err.max()),
        "measured_mean_abs_component_error": float(err.mean()),
        "measured_min_cosine_vs_fp16": float(cos.min()),
        "measured_mean_cosine_vs_fp16": float(cos.mean()),
    }

    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "batch_recsys_lab.demo.export_search",
        "evidence_class": "demonstration",
        "evidence_note": (
            "Similarity scores from this payload are a capability demonstration, not "
            "evaluation evidence: cosine over item text, no held-out interactions, no "
            "full-catalog ranking protocol. They are rendered without the traced-number "
            "affordance. Only the provenance chain below is anchored to the results log."
        ),
        "rows": rows,
        "n_items": rows,  # AGREED alias (docs/demo-data-schemas.md)
        "dim": dim,
        "ordering": {
            "rule": f"{pop_col} descending, ties broken by ascending catalog index",
            "popularity_column": pop_col,
            "source": str(items_path),
            "produced_by": "batch_recsys_lab.demo.shopper_history_job (configs/shoppers_export.yaml: search_slice)",
            "verified_at_export": True,
            "row_alignment": (
                "embeddings.npy row i is catalog index i (eval-cache item_ids order); "
                "payload row r is embeddings.npy row items_meta.catalog_index[r]"
            ),
        },
        "quantization": quant,
        "files": {
            "embeddings_int8.bin": {"bytes": bin_path.stat().st_size, "sha256": f"sha256:{bin_sha}"},
            "scales_f32.bin": {"bytes": scales_path.stat().st_size, "sha256": f"sha256:{scales_sha}"},
        },
        "source": {
            "embeddings_npy": str(emb_path),
            "source_embeddings_sha256": f"sha256:{emb_sha}",
            "catalog_rows": int(manifest["row_count"]),
            "embedding_dtype": manifest["embedding_dtype"],
            "recipe_id": manifest["recipe_id"],
            "recipe_hash": manifest["recipe_hash"],
            "recipe_fields": manifest["recipe_fields"],
            "model_id": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "five_core_snapshot_id": snapshot_id,
            "ann_receipt_run_id": record["run_id"],
            "ann_receipt_git_sha": record.get("git_sha"),
        },
        # AGREED aliases kept at the top level for docs/demo-data-schemas.md
        "source_embeddings_sha256": f"sha256:{emb_sha}",
        "five_core_snapshot_id": snapshot_id,
    }
    (out_dir / "embeddings_meta.json").write_text(dump_json(meta))

    items_meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "batch_recsys_lab.demo.export_search",
        "rows": rows,
        "note": (
            "Parallel arrays, index-aligned to the payload rows. All descriptive: "
            "gold.item_features metadata at the pinned snapshot, presence-checked, "
            "never value-matched against the results log."
        ),
        "columns": ["item_id", "title", "brand", "price_usd", "main_category", "catalog_index", pop_col],
        "item_id": item_ids,
        "title": table.column("title").to_pylist(),
        "brand": table.column("brand_norm").to_pylist(),
        "price_usd": table.column("price_usd").to_pylist(),
        "main_category": table.column("main_category").to_pylist(),
        "catalog_index": [int(i) for i in catalog_index],
        pop_col: [float(p) for p in pop],
    }
    (out_dir / "items_meta.json").write_text(dump_json(items_meta))

    result = {
        "rows": rows,
        "dim": dim,
        "embeddings_int8.bin": bin_path.stat().st_size,
        "scales_f32.bin": scales_path.stat().st_size,
        "embeddings_meta.json": (out_dir / "embeddings_meta.json").stat().st_size,
        "items_meta.json": (out_dir / "items_meta.json").stat().st_size,
        "quantization": quant,
    }

    if skip_queries:
        print("--skip-queries: example_queries.json NOT written (fallback mode would be unavailable)")
        return result

    q_doc, q_stats = compute_example_queries(
        cfg,
        manifest=manifest,
        unit=unit,
        deq=deq,
        item_ids=item_ids,
        descriptive={
            "title": items_meta["title"],
            "brand": items_meta["brand"],
            "price_usd": items_meta["price_usd"],
            "main_category": items_meta["main_category"],
        },
        meta_source=meta["source"],
    )
    (out_dir / "example_queries.json").write_text(dump_json(q_doc))
    result["example_queries.json"] = (out_dir / "example_queries.json").stat().st_size
    result.update(q_stats)
    return result


def compute_example_queries(
    cfg: dict,
    *,
    manifest: dict,
    unit: np.ndarray,
    deq: np.ndarray,
    item_ids: list[str],
    descriptive: dict[str, list],
    meta_source: dict,
) -> tuple[dict, dict]:
    """Embed the canned queries with the REAL model + recipe pooling.

    Same model id and the same pooling as ``models/minilm_embed.py``: the
    ``SentenceTransformer`` module stack (Transformer → mean Pooling →
    Normalize), ``encode(normalize_embeddings=False)``, then an explicit float32
    L2 normalisation — identical treatment to the item side, so the reference
    scores are exact cosines.

    Each reference hit carries its descriptive metadata inline. That is what makes
    this file a **self-contained fallback**: the exhibit renders all 12 canned
    queries from these ~70kB alone, without the 12.4MB ``items_meta.json`` and
    without either ``.bin`` — 120 rows of metadata, not 50,000.
    """
    from batch_recsys_lab.models import minilm_embed as recipe

    queries = list(cfg["queries"])
    k = int(cfg.get("top_k", 10))
    device, mps_detail = recipe._resolve_device()

    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(recipe.MODEL_ID, device=device)
    load_s = time.perf_counter() - t0

    detected_revision = None
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == recipe.MODEL_ID and repo.repo_type == "model":
                revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                if revs:
                    detected_revision = revs[0].commit_hash
                break
    except Exception:  # pragma: no cover - best effort, same as minilm_embed
        detected_revision = None
    recorded_revision = manifest.get("model_revision")
    if detected_revision and recorded_revision and detected_revision != recorded_revision:
        _fail(
            f"local {recipe.MODEL_ID} revision {detected_revision} != the revision the item "
            f"embeddings were computed with ({recorded_revision}); the canned queries would not "
            "be comparable to the payload"
        )

    t1 = time.perf_counter()
    raw = model.encode(
        queries,
        batch_size=len(queries),
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    encode_s = time.perf_counter() - t1
    qn = np.linalg.norm(raw, axis=1)
    qvecs = raw / np.where(qn > 0, qn, np.float32(1.0))[:, None]

    exact = unit @ qvecs.T  # (rows, n_queries) float32
    approx = deq @ qvecs.T

    entries = []
    overlaps = []
    for j, q in enumerate(queries):
        ref = top_k_rows(exact[:, j], k)
        alt = top_k_rows(approx[:, j], k)
        ref_ids = [item_ids[i] for i in ref]
        alt_ids = [item_ids[i] for i in alt]
        ov = overlap_at_k(ref_ids, alt_ids, k)
        overlaps.append(ov)
        entries.append(
            {
                "query_index": j,
                "query": q,
                "top_k": k,
                "results": [
                    {
                        "rank": r + 1,
                        "row": int(i),
                        "item_id": item_ids[i],
                        "score": float(exact[i, j]),
                        "title": descriptive["title"][i],
                        "brand": descriptive["brand"][i],
                        "price_usd": descriptive["price_usd"][i],
                        "main_category": descriptive["main_category"][i],
                    }
                    for r, i in enumerate(ref)
                ],
                "int8_top_k_item_ids": alt_ids,
                "int8_overlap_at_k": ov,
            }
        )

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "batch_recsys_lab.demo.export_search",
        "evidence_class": "demonstration",
        "evidence_note": (
            "Reference results for the fallback path and for the in-UI parity receipt. "
            "Scores are cosine similarities over item text — a capability demonstration, "
            "not an evaluation metric."
        ),
        "top_k": k,
        "n_queries": len(queries),
        "fallback": {
            "self_contained": True,
            "note": (
                "Each hit carries its own descriptive metadata, so demo/js/search.js can render "
                "FALLBACK MODE from this file alone — no items_meta.json, no .bin, no vendored "
                "model. This is the documented degradation path of cut-order item #2."
            ),
        },
        "reference": {
            "computed_against": "float32 L2-normalised view of the fp16 embeddings.npy slice (exact, not the int8 payload)",
            "model_id": recipe.MODEL_ID,
            "model_revision": recorded_revision,
            "detected_local_revision": detected_revision,
            "pooling": "mean (SentenceTransformer module stack) + float32 L2 normalisation",
            "recipe_id": manifest["recipe_id"],
            "recipe_hash": manifest["recipe_hash"],
            "sentence_transformers_version": manifest["sentence_transformers_version"],
            "device": device,
            "mps_failure_detail": mps_detail,
            "tie_break": "score descending, ties by ascending payload row",
        },
        "int8_parity": {
            "note": (
                "Export-side quantization parity: the int8 payload's own top-k against the "
                "exact fp16 reference. The browser adds the ONNX-model half of the gap; the "
                "exhibit's parity receipt reports the combined overlap."
            ),
            "mean_overlap_at_k": float(np.mean(overlaps)),
            "min_overlap_at_k": int(np.min(overlaps)),
            "per_query_overlap_at_k": [int(o) for o in overlaps],
        },
        "provenance": meta_source,
        "queries": entries,
    }
    stats = {
        "model_load_s": round(load_s, 3),
        "encode_s": round(encode_s, 4),
        "device": device,
        "int8_mean_overlap_at_k": float(np.mean(overlaps)),
        "int8_min_overlap_at_k": int(np.min(overlaps)),
    }
    return doc, stats


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="batch_recsys_lab.demo.export_search")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument(
        "--skip-queries",
        action="store_true",
        help="skip the sentence-transformers step (payload only; no fallback data)",
    )
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    res = export(cfg, skip_queries=args.skip_queries)
    wall = time.perf_counter() - t0

    q = res["quantization"]
    print("")
    print(f"wrote {cfg['out_dir']}/  ({res['rows']} rows x {res['dim']} dim)")
    for name in (
        "embeddings_int8.bin",
        "scales_f32.bin",
        "embeddings_meta.json",
        "items_meta.json",
        "example_queries.json",
    ):
        if name in res:
            print(f"  {name:24s} {res[name]:>12,} bytes  ({res[name] / 1e6:.2f} MB)")
    print(
        f"  int8 error: max |Δcomponent| {q['measured_max_abs_component_error']:.3e}, "
        f"min cosine vs fp16 {q['measured_min_cosine_vs_fp16']:.6f}"
    )
    if "int8_mean_overlap_at_k" in res:
        print(
            f"  int8 parity vs exact reference: mean overlap@k "
            f"{res['int8_mean_overlap_at_k']:.2f}, min {res['int8_min_overlap_at_k']}"
        )
        print(f"  queries embedded on device={res['device']} (model load {res['model_load_s']}s, encode {res['encode_s']}s)")
    print(f"  wall clock {wall:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
