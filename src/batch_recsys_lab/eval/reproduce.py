"""``make reproduce-headline`` — snapshot-pinned re-run of the recorded headline
eval (Phase 5, T18; UPGRADE_PLAN.md §8 Phase 5, §12 "never cut").

    uv run python -m batch_recsys_lab.eval.reproduce \
        [--headline configs/headline.yaml] [--skip-extract]

This module REPRODUCES an existing record; it never runs a new experiment. The
frozen-TEST invariant (CLAUDE.md #1) is not at risk because nothing here chooses
a config, a split, or a model — all three are read out of the recorded run.

What it does
------------
1. Reads ``configs/headline.yaml`` -> ``headline_run_id``; locates that record in
   ``results/runs.jsonl`` (exactly one match required).
2. Refuses to run from a dirty git tree: a reproduction claim is a claim about a
   commit.
3. Rebuilds the eval cache with a PINNED extract — Iceberg time travel at the
   recorded ``iceberg_snapshots`` — into ``<cache_repro_root>/<5core snapshot>/``.
   Idempotent: an existing manifest with matching pinned IDs is reused. This is
   the only step that starts Spark.
4. Verifies the reproduced cache against the recorded provenance (manifest IDs,
   per-file hashes vs the surviving live cache, MiniLM artifact hashes).
5. Re-runs the eval from the ORIGINAL config file (whose sha256 must still equal
   the recorded ``config_hash``), with ``append=False``, overriding only the cache
   and per-user-artifact locations IN MEMORY so ``config_hash`` is untouched.
6. Diffs the candidate record against the recorded one on the deterministic
   fields, compares the per-user parquet, and appends ONE ``kind="reproduce"``
   record. Exit 0 iff the verdict is ``byte_exact``.

Determinism: what is compared, and why
--------------------------------------
:data:`FIELDS_COMPARED` are the fields that MUST be identical given the same
cache bytes, config and seeds. :data:`FIELDS_EXCLUDED` are per-invocation
identity/timing/host fields (``run_id``, ``run_ts``, ``wall_clock_s``,
``hardware``, ``git_sha``, ``git_dirty``, ``config_path``,
``per_user_artifact``) plus the trivially-constant ``schema_version``/``kind``.

Two audited sources of legitimate variation, neither of which is papered over:

* **Pair-array row order.** ``extract._build_pairs`` writes TRAIN/VAL/TEST pair
  arrays in Spark's output order. That order is a shuffle artifact. It cannot
  change TRAIN masking (COO->CSR sorts) nor GT membership (``_build_gt`` sorts by
  user), but it CAN change the within-user order of GT items, and hence the
  float summation order of a user's DCG. So the cache comparison reports BOTH a
  strict per-file sha256 result (:data:`cache_files_match`) and an
  order-normalized one (``cache_files_canonical_match``): strict-False +
  canonical-True is precisely "same content, different row order", which is the
  explanation to reach for if a metric differs in its last bits.
* **Cache manifest ``created_ts``.** Wall-clock by construction, so
  ``cache_manifest.json`` is excluded from the file-hash comparison; its
  semantic content (snapshot IDs) is checked separately and exactly.

Everything else in the record is a pure function of (cache bytes, config bytes,
splits.yaml, data/MANIFEST.md, MiniLM artifact, seeds) — all of which are hashed
into the record itself, so a mismatch always points at a named input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from batch_recsys_lab.contracts.engine import _resolve_run_id
from batch_recsys_lab.eval import runlog

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HEADLINE = REPO_ROOT / "configs" / "headline.yaml"

FIELDS_COMPARED = (
    "config_hash",
    "splits",
    "dataset_manifest_hash",
    "iceberg_snapshots",
    "contracts",
    "protocol",
    "model",
    "seeds",
    "metrics",
    "beyond_accuracy",
)

FIELDS_EXCLUDED = (
    "schema_version",
    "kind",
    "run_id",
    "run_ts",
    "git_sha",
    "git_dirty",
    "config_path",
    "per_user_artifact",
    "wall_clock_s",
    "hardware",
)

# Cache files whose bytes are order-invariant by construction (sorted indexes,
# position-aligned vectors) -> compared by strict sha256.
_PAIR_GROUPS = {
    "train": ("train_user_idx.npy", "train_item_idx.npy", "train_rating.npy"),
    "val": ("val_user_idx.npy", "val_item_idx.npy"),
    "test": ("test_user_idx.npy", "test_item_idx.npy"),
}
# Excluded from the file-hash comparison: carries a wall-clock created_ts.
_MANIFEST_NAME = "cache_manifest.json"


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<missing>"


MISSING = _Missing()


# --- record location ---------------------------------------------------------


def load_headline(path: str | Path = DEFAULT_HEADLINE) -> dict:
    """Load the committed headline pin. ``headline_run_id`` is mandatory."""
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    if not cfg.get("headline_run_id"):
        raise ValueError(f"{path}: headline_run_id is required")
    cfg.setdefault("results_path", "results/runs.jsonl")
    cfg.setdefault("cache_repro_root", "data/eval/cache_repro")
    return cfg


def find_record(
    results_path: str | Path, run_id: str, kind: str = "eval", config_path: str | None = None
) -> dict:
    """The single ``kind`` record with ``run_id``.

    Unlike ``compare._resolve_arm`` (last match wins), reproduction demands an
    UNAMBIGUOUS target: 0 matches and >1 matches are both errors, because
    "reproduce the headline" must name exactly one thing.

    ``config_path``, if given, disambiguates a run_id collision (two records
    with the SAME run_id string, e.g. from overlapping campaigns that both
    derived their id from ``date -u ... -git_sha`` in the same second) by also
    requiring ``rec["config_path"] == config_path``. This only narrows an
    otherwise->1-match set; it never widens 0 matches, and callers that never
    pass it see the exact prior behavior (Amazon headline.yaml is unaffected).
    """
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"results log {results_path} not found")
    all_records = [
        json.loads(line) for line in results_path.read_text().splitlines() if line.strip()
    ]
    matches = [rec for rec in all_records if rec.get("kind") == kind and rec.get("run_id") == run_id]
    if not matches:
        raise ValueError(f"no kind={kind!r} record with run_id={run_id!r} in {results_path}")
    if len(matches) > 1 and config_path is not None:
        narrowed = [rec for rec in matches if rec.get("config_path") == config_path]
        if narrowed:
            matches = narrowed
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} kind={kind!r} records with run_id={run_id!r} in "
            f"{results_path}; the headline pin must identify exactly one run"
            + (f" (config_path={config_path!r} did not disambiguate)" if config_path else "")
        )
    return matches[0]


# --- record comparison -------------------------------------------------------


def json_roundtrip(obj):
    """Normalize through the exact serialization ``runs.jsonl`` was written with.

    ``append_record`` uses ``json.dumps(..., separators=(",", ":"))``; parsing that
    back puts the candidate in the same representation as the recorded line, so
    float comparison is comparison of what WOULD have been written (and numpy
    scalars collapse to plain floats/ints).
    """
    return json.loads(json.dumps(obj, separators=(",", ":"), sort_keys=False))


def _type_drift(a, b) -> bool:
    """True when two ``==``-equal values have different JSON types (``True`` vs
    ``1``, ``1`` vs ``1.0``) — schema drift is a mismatch even when the values
    compare equal in Python."""
    if isinstance(a, bool) != isinstance(b, bool):
        return True
    return isinstance(a, float) != isinstance(b, float)


def _leaf_diff(path: str, a, b) -> dict:
    return {
        "path": path,
        "recorded": None if a is MISSING else a,
        "candidate": None if b is MISSING else b,
        **({"recorded_missing": True} if a is MISSING else {}),
        **({"candidate_missing": True} if b is MISSING else {}),
    }


def _walk(path: str, a, b, out: list) -> None:
    if a is MISSING and b is MISSING:
        return
    if a is MISSING or b is MISSING:
        out.append(_leaf_diff(path, a, b))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a.keys()) + [k for k in b if k not in a]:
            _walk(f"{path}.{k}" if path else str(k), a.get(k, MISSING), b.get(k, MISSING), out)
        return
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(f"{path}[{i}]", x, y, out)
        return
    if a != b or _type_drift(a, b):
        out.append(_leaf_diff(path, a, b))


def diff_records(recorded: dict, candidate: dict, fields=FIELDS_COMPARED) -> list[dict]:
    """Field-level diff over ``fields``; empty list == deterministic-field equality.

    ``candidate`` is round-tripped through the runs.jsonl serialization first, so
    the comparison is against the bytes that WOULD be appended.
    """
    cand = json_roundtrip(candidate)
    out: list[dict] = []
    for field in fields:
        _walk(field, recorded.get(field, MISSING), cand.get(field, MISSING), out)
    return out


def verify_config_hash(config_path: str | Path, recorded_hash: str) -> None:
    """The recorded config file must still hash to ``recorded_hash``.

    If it does not, reproduction is impossible by definition: the run would be of
    a different config. Fail loudly rather than reporting a mismatch verdict that
    looks like a determinism bug.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"recorded config {config_path} no longer exists; the headline run "
            "cannot be reproduced from this tree"
        )
    current = runlog.config_hash(config_path)
    if current != recorded_hash:
        raise RuntimeError(
            f"config file changed since the recorded run: {config_path} hashes "
            f"{current} but the record carries {recorded_hash}. A snapshot-pinned "
            "reproduction requires the ORIGINAL config bytes — restore them from "
            "the recorded git SHA (invariant #3: supersede, never rewrite)."
        )


# --- cache / artifact comparison ---------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pair_digest(dir_path: Path, names) -> str:
    """Order-normalized digest of one pair group: lexsort the aligned columns by
    (user, item[, rating]) and hash the sorted bytes. Equal digests == same set of
    pairs, regardless of the Spark output order."""
    arrs = [np.load(dir_path / n, allow_pickle=False) for n in names]
    order = np.lexsort(tuple(reversed(arrs)))  # last key is primary -> arrs[0]
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a[order]).tobytes())
    return h.hexdigest()


def compare_cache_dirs(original: Path, repro: Path) -> dict:
    """Compare a reproduced cache dir against the original live one.

    Returns ``{"files_match", "canonical_match", "detail"}``. ``files_match`` is
    strict per-file sha256 over every file EXCEPT ``cache_manifest.json`` (which
    carries a wall-clock ``created_ts``). ``canonical_match`` relaxes only the
    pair arrays to an order-normalized digest — see the module docstring.
    """
    orig_names = {p.name for p in original.iterdir() if p.is_file()} - {_MANIFEST_NAME}
    repro_names = {p.name for p in repro.iterdir() if p.is_file()} - {_MANIFEST_NAME}
    detail: dict = {
        "compared_files": sorted(orig_names & repro_names),
        "only_in_original": sorted(orig_names - repro_names),
        "only_in_repro": sorted(repro_names - orig_names),
        "sha256_mismatches": [],
        "canonical_mismatches": [],
        "excluded_files": [_MANIFEST_NAME],
    }

    for name in detail["compared_files"]:
        if _sha256_file(original / name) != _sha256_file(repro / name):
            detail["sha256_mismatches"].append(name)

    pair_files = {n for names in _PAIR_GROUPS.values() for n in names}
    for group, names in _PAIR_GROUPS.items():
        if not all(n in orig_names and n in repro_names for n in names):
            continue
        if _pair_digest(original, names) != _pair_digest(repro, names):
            detail["canonical_mismatches"].append(group)

    same_set = not detail["only_in_original"] and not detail["only_in_repro"]
    files_match = same_set and not detail["sha256_mismatches"]
    non_pair_mismatch = [n for n in detail["sha256_mismatches"] if n not in pair_files]
    canonical_match = (
        same_set and not non_pair_mismatch and not detail["canonical_mismatches"]
    )
    return {"files_match": files_match, "canonical_match": canonical_match, "detail": detail}


def verify_artifact_hashes(model: dict, root: Path = REPO_ROOT) -> dict:
    """Re-verify the model-artifact hashes the record carries against disk.

    The record's ``model.params`` (possibly nested, e.g. ``content_params`` inside
    a blend) carries ``artifact_root`` + ``recipe_hash`` +
    ``five_core_snapshot_id`` + ``embeddings_sha256`` / ``item_ids_sha256``. Those
    hash fields ARE the artifact identity; this recomputes the file hash and
    re-reads the manifest. Returns ``{"match": bool|None, "detail": {...}}`` —
    ``None`` when the record carries no artifact hashes (nothing to verify) or the
    artifact directory is gone.
    """
    params = (model or {}).get("params") or {}
    candidates = [params] + [v for v in params.values() if isinstance(v, dict)]
    block = next((p for p in candidates if "embeddings_sha256" in p), None)
    if block is None:
        return {"match": None, "detail": {"reason": "record carries no artifact hashes"}}

    adir = (
        root
        / str(block.get("artifact_root", "data/eval/minilm"))
        / str(block["five_core_snapshot_id"])
        / str(block["recipe_hash"])
    )
    man_path = adir / "minilm_manifest.json"
    emb_path = adir / "embeddings.npy"
    if not man_path.exists() or not emb_path.exists():
        return {"match": None, "detail": {"reason": f"artifact not on disk at {adir}"}}

    man = json.loads(man_path.read_text())
    checks = {
        "manifest_embeddings_sha256": man.get("embeddings_sha256") == block["embeddings_sha256"],
        "manifest_item_ids_sha256": man.get("item_ids_sha256") == block.get("item_ids_sha256"),
        "recomputed_embeddings_sha256": _sha256_file(emb_path) == block["embeddings_sha256"],
    }
    if "embedding_dim" in block:
        checks["embedding_dim"] = int(man.get("embedding_dim", -1)) == int(block["embedding_dim"])
    return {
        "match": all(checks.values()),
        "detail": {"artifact_dir": str(adir), "checks": checks},
    }


def compare_per_user(original: Path, repro: Path) -> dict:
    """Exact array comparison of two per-user artifacts.

    Both are sorted by ``user_id`` (unique per row -> a total order), then the
    user-id vectors and every shared column are compared for exact equality. No
    tolerance: a reproduction that shifts a per-user metric is a mismatch.
    """
    import pyarrow.parquet as pq

    a = pq.read_table(original)
    b = pq.read_table(repro)
    detail: dict = {
        "n_rows_original": a.num_rows,
        "n_rows_repro": b.num_rows,
        "only_in_original_columns": sorted(set(a.column_names) - set(b.column_names)),
        "only_in_repro_columns": sorted(set(b.column_names) - set(a.column_names)),
        "mismatched_columns": [],
    }
    if a.num_rows != b.num_rows or detail["only_in_original_columns"] or detail[
        "only_in_repro_columns"
    ]:
        return {"match": False, "detail": detail}

    a_ids = np.array(a.column("user_id").to_pylist(), dtype=object)
    b_ids = np.array(b.column("user_id").to_pylist(), dtype=object)
    ao = np.argsort(a_ids, kind="stable")
    bo = np.argsort(b_ids, kind="stable")
    if not np.array_equal(a_ids[ao], b_ids[bo]):
        detail["mismatched_columns"].append("user_id")
        return {"match": False, "detail": detail}

    for name in a.column_names:
        if name == "user_id":
            continue
        acol = a.column(name).to_pylist()
        bcol = b.column(name).to_pylist()
        if [acol[i] for i in ao] != [bcol[i] for i in bo]:
            detail["mismatched_columns"].append(name)
    return {"match": not detail["mismatched_columns"], "detail": detail}


# --- orchestration -----------------------------------------------------------


def _pinned_cache_dir(cache_repro_root: Path, snapshots: dict, five_core_table: str) -> Path:
    return cache_repro_root / str(int(snapshots[five_core_table]))


def _tables_from_config(config: dict) -> dict:
    from batch_recsys_lab.eval import extract as extract_mod

    tables = config.get("tables") or {}
    return {
        "five_core": tables.get("five_core", extract_mod.FIVE_CORE),
        "user_stats": tables.get("user_stats", extract_mod.USER_STATS),
        "item_features": tables.get("item_features", extract_mod.ITEM_FEATURES),
        "popularity": tables.get("popularity", extract_mod.POPULARITY),
    }


def _resolve_splits_path(config: dict) -> str:
    """Same precedence as ``run_eval.py``'s CLI: the config's own ``splits_path``
    key, else the repo default (Amazon ``configs/splits.yaml``)."""
    return config.get("splits_path") or str(runlog.DEFAULT_SPLITS_PATH)


def _resolve_manifest_path(config: dict) -> str:
    """Same precedence as ``run_eval.py``'s CLI: the config's own ``manifest_path``
    key, else the repo default (``data/MANIFEST.md``)."""
    return config.get("manifest_path") or str(runlog.DEFAULT_MANIFEST_PATH)


def _run_pinned_extract(
    record: dict,
    config: dict,
    cache_dir: Path,
    warehouse: Path,
    master: str,
    driver_memory: str,
) -> dict:
    """Spark step: rebuild the cache by time travel at the recorded snapshots."""
    from batch_recsys_lab.eval.extract import extract
    from batch_recsys_lab.spark_session import get_spark

    tables = _tables_from_config(config)
    splits_path = _resolve_splits_path(config)
    spark = get_spark(
        app_name="reproduce-headline",
        warehouse=warehouse,
        master=master,
        driver_memory=driver_memory,
    )
    try:
        return extract(
            spark,
            out=cache_dir.parent,
            five_core_table=tables["five_core"],
            user_stats_table=tables["user_stats"],
            item_features_table=tables["item_features"],
            popularity_table=tables["popularity"],
            pinned_snapshot_ids=record["iceberg_snapshots"],
            pinned_contracts=record["contracts"],
            splits_path=splits_path,
        )
    finally:
        spark.stop()


def reproduce(
    headline_path: str | Path = DEFAULT_HEADLINE,
    skip_extract: bool = False,
    root: str | Path = REPO_ROOT,
    master: str = "local[10]",
    driver_memory: str = "8g",
    append: bool = True,
) -> dict:
    """Full reproduce-headline flow. Returns the ``kind="reproduce"`` record."""
    root = Path(root)
    headline = load_headline(headline_path)
    run_id_target = headline["headline_run_id"]
    results_path = root / headline["results_path"]
    cache_repro_root = root / headline["cache_repro_root"]

    record = find_record(
        results_path, run_id_target, config_path=headline.get("expected_config_path")
    )

    git = runlog.git_info()
    if git["git_dirty"]:
        raise RuntimeError(
            "Refusing to reproduce the headline from a dirty git working tree: a "
            "reproduction is a claim about a commit (CLAUDE.md invariant #1/#3). "
            "Commit or stash your changes first."
        )

    config_path = root / record["config_path"]
    verify_config_hash(config_path, record["config_hash"])
    config = yaml.safe_load(config_path.read_text())

    tables = _tables_from_config(config)
    snapshots = record["iceberg_snapshots"]
    cache_dir = _pinned_cache_dir(cache_repro_root, snapshots, tables["five_core"])

    # --- (c) pinned extract, idempotent -------------------------------------
    t0 = time.monotonic()
    manifest_path = cache_dir / "cache_manifest.json"
    already = False
    if manifest_path.exists():
        try:
            runlog.check_pinned_cache(
                json.loads(manifest_path.read_text()).get("snapshot_ids", {}), snapshots
            )
            already = True
        except RuntimeError:
            already = False
    if already:
        print(f"[reproduce] pinned cache reused: {cache_dir}")
    elif skip_extract:
        raise RuntimeError(
            f"--skip-extract given but no valid pinned cache at {cache_dir}; run "
            "without --skip-extract to build it."
        )
    else:
        warehouse = root / config.get("warehouse", "data/warehouse")
        print(f"[reproduce] pinned extract -> {cache_dir} (warehouse={warehouse})")
        _run_pinned_extract(record, config, cache_dir, warehouse, master, driver_memory)
    extract_wall_clock_s = round(time.monotonic() - t0, 3)

    # --- (d) verifications ---------------------------------------------------
    repro_manifest = json.loads(manifest_path.read_text())
    try:
        runlog.check_pinned_cache(repro_manifest.get("snapshot_ids", {}), snapshots)
        pinned_cache_manifest_match = True
    except RuntimeError as exc:
        print(f"[reproduce] WARNING: {exc}")
        pinned_cache_manifest_match = False

    # The original live cache: the config's cache root (snapshot-keyed), unless the
    # config already names a full cache path — mirrors harness._resolve_cache_dir.
    cfg_cache_root = root / config.get("cache_dir", "data/eval/cache")
    orig_cache_dir = (
        cfg_cache_root
        if (cfg_cache_root / "cache_manifest.json").exists()
        else cfg_cache_root / str(int(snapshots[tables["five_core"]]))
    )
    if orig_cache_dir.is_dir() and orig_cache_dir.resolve() != cache_dir.resolve():
        cache_cmp = compare_cache_dirs(orig_cache_dir, cache_dir)
        cache_files_match = cache_cmp["files_match"]
        cache_files_canonical_match = cache_cmp["canonical_match"]
        cache_compare_detail = cache_cmp["detail"]
    else:
        cache_files_match = None
        cache_files_canonical_match = None
        cache_compare_detail = {"reason": f"original cache dir not present at {orig_cache_dir}"}

    artifact_cmp = verify_artifact_hashes(record["model"], root)
    artifact_hashes_match = artifact_cmp["match"]

    # --- (e) re-run the eval from the ORIGINAL config ------------------------
    from batch_recsys_lab.eval.harness import run_eval

    # In-memory overrides ONLY: config_hash must stay the hash of the file bytes.
    config["cache_dir"] = str(cache_dir)
    config["per_user_dir"] = str(cache_repro_root / "per_user")

    t1 = time.monotonic()
    candidate = run_eval(
        config,
        config_path=config_path,
        results_path=results_path,
        append=False,
        expected_snapshot_ids=snapshots,
        splits_path=_resolve_splits_path(config),
        manifest_path=_resolve_manifest_path(config),
    )
    eval_wall_clock_s = round(time.monotonic() - t1, 3)

    # --- (f) deterministic-field diff ---------------------------------------
    diff = diff_records(record, candidate)

    # --- (g) per-user artifact -----------------------------------------------
    orig_artifact = root / record["per_user_artifact"]
    if orig_artifact.exists():
        pu = compare_per_user(orig_artifact, Path(candidate["per_user_artifact"]))
        per_user_artifact_match = pu["match"]
        per_user_detail = pu["detail"]
    else:
        per_user_artifact_match = None
        per_user_detail = {"reason": f"original artifact not present at {orig_artifact}"}

    # --- (h) the reproduce record -------------------------------------------
    byte_exact = (not diff) and pinned_cache_manifest_match
    run_id, run_ts = _resolve_run_id(None)
    repro_record = {
        "schema_version": runlog.record_schema_version,
        "kind": "reproduce",
        "run_id": run_id,
        "run_ts": run_ts,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "headline_config": str(Path(headline_path)),
        "reproduces_run_id": run_id_target,
        "recorded_git_sha": record.get("git_sha"),
        "verdict": "byte_exact" if byte_exact else "mismatch",
        "diff": diff,
        "fields_compared": list(FIELDS_COMPARED),
        "fields_excluded": list(FIELDS_EXCLUDED),
        "pinned_cache_manifest_match": pinned_cache_manifest_match,
        "repro_cache_dir": str(cache_dir),
        "cache_files_match": cache_files_match,
        "cache_files_canonical_match": cache_files_canonical_match,
        "cache_compare_detail": cache_compare_detail,
        "per_user_artifact_match": per_user_artifact_match,
        "per_user_compare_detail": per_user_detail,
        "artifact_hashes_match": artifact_hashes_match,
        "artifact_compare_detail": artifact_cmp["detail"],
        "extract_wall_clock_s": extract_wall_clock_s,
        "eval_wall_clock_s": eval_wall_clock_s,
        "hardware": runlog.hardware_string(),
    }
    if append:
        runlog.append_record(repro_record, results_path)
    print_summary(repro_record)
    return repro_record


# --- reporting ---------------------------------------------------------------


def _status(value) -> str:
    if value is None:
        return "SKIP"
    return "OK  " if value else "FAIL"


def print_summary(rec: dict) -> None:
    rows = [
        ("pinned cache manifest == recorded snapshots", rec["pinned_cache_manifest_match"]),
        ("cache files sha256 == original cache", rec["cache_files_match"]),
        ("cache pair arrays (order-normalized)", rec["cache_files_canonical_match"]),
        ("model artifact hashes == recorded", rec["artifact_hashes_match"]),
        ("deterministic record fields identical", not rec["diff"]),
        ("per-user artifact arrays identical", rec["per_user_artifact_match"]),
    ]
    width = max(len(label) for label, _ in rows)
    print()
    print(f"reproduce {rec['reproduces_run_id']}  (recorded @ {rec['recorded_git_sha']})")
    print("-" * (width + 8))
    for label, value in rows:
        print(f"  {_status(value)}  {label:<{width}}")
    print("-" * (width + 8))
    if rec["diff"]:
        print(f"  {len(rec['diff'])} differing field(s):")
        for d in rec["diff"][:20]:
            print(f"    {d['path']}: recorded={d['recorded']!r} candidate={d['candidate']!r}")
        if len(rec["diff"]) > 20:
            print(f"    … {len(rec['diff']) - 20} more")
    print(
        f"  verdict={rec['verdict']}  extract={rec['extract_wall_clock_s']}s  "
        f"eval={rec['eval_wall_clock_s']}s"
    )
    print()


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.eval.reproduce")
    parser.add_argument("--headline", default=str(DEFAULT_HEADLINE))
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="require an existing valid pinned cache instead of rebuilding it",
    )
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    args = parser.parse_args(argv)

    try:
        rec = reproduce(
            headline_path=args.headline,
            skip_extract=args.skip_extract,
            master=args.master,
            driver_memory=args.driver_memory,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report + non-zero exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0 if rec["verdict"] == "byte_exact" else 1


if __name__ == "__main__":
    sys.exit(main())
