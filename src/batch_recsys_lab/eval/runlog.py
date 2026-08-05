"""Append-only run log: record schema, provenance hashing, integrity guards
(Phase 2, T5; UPGRADE_PLAN.md §8 "runs.jsonl record").

``results/runs.jsonl`` is append-only and committed (CLAUDE.md invariant #3): a
wrong run is superseded by a new record, never rewritten. Every record carries
the full provenance manifest — config hash, git SHA + dirty flag, dataset
manifest hash, Iceberg snapshot IDs, contract identities, seeds — so any metric
that reaches the case study traces back to an exactly reproducible run.

This module never starts Spark. The stale-cache guard reads Iceberg metadata
files directly (:func:`iceberg_snapshot_id`) so a pure-numpy scoring process can
verify its snapshot-keyed cache against the live tables before appending.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import yaml

record_schema_version = 1

# Repo root (…/src/batch_recsys_lab/eval/runlog.py -> parents[3]).
_REPO_ROOT = Path(__file__).resolve().parents[3]


# --- hashing / provenance ----------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """``"sha256:<hex>"`` digest of a file's raw bytes."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


# Run-output paths excluded from the dirty determination: these are append-only
# artifacts produced BY the eval/logging process itself (results/runs.jsonl,
# EXPERIMENT_LOG.md), not code. The dirty guard exists to pin code reproducibility
# (CLAUDE.md invariant #1/#3: a TEST number must trace to a committed SHA) — it is
# not meant to pin the outputs the runs themselves generate. Without this
# exclusion, the very first eval run leaves results/runs.jsonl modified/untracked,
# which would make every subsequent TEST run in the same session refuse.
_DIRTY_EXCLUDE_PATHS = frozenset({"results/runs.jsonl", "EXPERIMENT_LOG.md"})


def _dirty_from_porcelain(lines: list[str]) -> bool:
    """True if any ``git status --porcelain`` line refers to a path outside
    :data:`_DIRTY_EXCLUDE_PATHS`.

    Handles both tracked-modified lines (``" M path"``, ``"MM path"``, etc.) and
    untracked lines (``"?? path"``). Porcelain lines are ``XY path`` (or
    ``XY orig -> path`` for renames); the path is taken as everything after the
    2-character status + space, with any rename arrow resolved to the new path.
    """
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        # Porcelain format: "XY path" (status is first 2 chars, then a space).
        path = line[3:] if len(line) > 3 else line.strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if path not in _DIRTY_EXCLUDE_PATHS:
            return True
    return False


def git_info() -> dict:
    """``{"git_sha": <full sha or None>, "git_dirty": bool}`` via subprocess.

    ``git_dirty`` is True when the working tree has any staged or unstaged
    change (``git status --porcelain`` non-empty) OUTSIDE of
    :data:`_DIRTY_EXCLUDE_PATHS`. Those two paths (``results/runs.jsonl``,
    ``EXPERIMENT_LOG.md``) are append-only outputs the run itself writes, so
    counting them would make every run after the first look dirty. Runs against
    the repo root so the result is independent of the process working directory.
    """
    def _run(args: list[str]) -> tuple[int, str]:
        proc = subprocess.run(
            args, cwd=str(_REPO_ROOT), capture_output=True, text=True
        )
        return proc.returncode, proc.stdout

    rc_sha, out_sha = _run(["git", "rev-parse", "HEAD"])
    git_sha = out_sha.strip() if rc_sha == 0 else None
    # -uall lists untracked files individually; without it git collapses an
    # untracked directory to one "?? dir/" line, which the path-level exclusion
    # in _dirty_from_porcelain cannot match (e.g. "?? results/").
    rc_st, out_st = _run(["git", "status", "--porcelain", "-uall"])
    git_dirty = _dirty_from_porcelain(out_st.splitlines()) if rc_st == 0 else False
    return {"git_sha": git_sha, "git_dirty": git_dirty}


def iceberg_snapshot_id(warehouse: str | Path, table: str) -> int:
    """Current snapshot ID of a Hadoop-catalog Iceberg table, WITHOUT Spark.

    For catalog table ``local.gold.x`` the physical metadata dir is
    ``<warehouse>/gold/x/metadata/``. Read ``version-hint.text`` -> ``N``, then
    parse ``vN.metadata.json`` -> ``"current-snapshot-id"``. The catalog name
    (first dotted component, e.g. ``local``) maps to the warehouse root and is
    stripped from the path. This powers the stale-cache guard in a pure-numpy
    process (no JVM).
    """
    parts = table.split(".")
    # Drop the catalog name; the rest is the namespace/table path under warehouse.
    rel = Path(*parts[1:])
    meta_dir = Path(warehouse) / rel / "metadata"
    version = int((meta_dir / "version-hint.text").read_text().strip())
    metadata = json.loads((meta_dir / f"v{version}.metadata.json").read_text())
    return int(metadata["current-snapshot-id"])


def splits_block(splits_path: str | Path) -> dict:
    """``{version, frozen_at, file_hash}`` from the frozen ``configs/splits.yaml``."""
    p = Path(splits_path)
    data = yaml.safe_load(p.read_text())
    return {
        "version": data["version"],
        "frozen_at": data["frozen_at"],
        "file_hash": sha256_file(p),
    }


def hardware_string() -> str:
    """Compact hardware descriptor, e.g. ``"arm64 · Darwin"``."""
    return f"{platform.machine()} · {platform.system()}"


# --- integrity guards --------------------------------------------------------


def check_test_dirty(eval_split: str, git_dirty: bool) -> None:
    """Refuse to record a TEST-split run from a dirty working tree.

    Invariant #1/#3: a TEST number must be reproducible from a committed SHA.
    """
    if eval_split == "test" and git_dirty:
        raise RuntimeError(
            "Refusing to record a TEST-split run with a dirty git working tree: "
            "TEST results must be reproducible from a commit (CLAUDE.md invariant "
            "#1/#3). Commit your changes, or run against VAL for iteration."
        )


def check_stale_cache(
    manifest_snapshot_ids: dict[str, int],
    warehouse: str | Path,
    allow_stale: bool = False,
) -> None:
    """Verify the cache's snapshot IDs still match the live Iceberg tables.

    ``manifest_snapshot_ids`` maps full table names (as stored in the cache
    manifest) to the snapshot ID captured at extract time. Each is re-verified
    against the live table via :func:`iceberg_snapshot_id`. Any mismatch — or a
    missing warehouse dir (pure-cache scenario) — raises unless ``allow_stale``.
    """
    if allow_stale:
        return
    warehouse = Path(warehouse)
    if not warehouse.exists():
        raise RuntimeError(
            f"Stale-cache guard: warehouse {warehouse} does not exist, so cache "
            "snapshot IDs cannot be verified against live tables. Pass "
            "--allow-stale only for tests / offline replays."
        )
    for table, cached_sid in manifest_snapshot_ids.items():
        live_sid = iceberg_snapshot_id(warehouse, table)
        if live_sid != cached_sid:
            raise RuntimeError(
                f"Stale-cache guard: {table} live snapshot {live_sid} != cached "
                f"snapshot {cached_sid}. Re-run `make eval-extract` before scoring "
                "(pass --allow-stale only for tests / offline replays)."
            )


# --- record assembly / append ------------------------------------------------


def build_record(
    *,
    kind: str,
    run_id: str,
    run_ts: str,
    git_sha: str | None,
    git_dirty: bool,
    config_path: str | Path,
    config_hash: str,
    splits: dict,
    dataset_manifest_hash: str,
    iceberg_snapshots: dict,
    contracts: dict,
    protocol: dict,
    model: dict,
    seeds: dict,
    metrics: dict,
    beyond_accuracy: dict,
    per_user_artifact: str,
    wall_clock_s: float,
    hardware: str,
) -> dict:
    """Assemble the schema-version-1 ``kind="eval"`` run record (exact schema
    from UPGRADE_PLAN.md §8 "runs.jsonl record")."""
    return {
        "schema_version": record_schema_version,
        "kind": kind,
        "run_id": run_id,
        "run_ts": run_ts,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "splits": splits,
        "dataset_manifest_hash": dataset_manifest_hash,
        "iceberg_snapshots": iceberg_snapshots,
        "contracts": contracts,
        "protocol": protocol,
        "model": model,
        "seeds": seeds,
        "metrics": metrics,
        "beyond_accuracy": beyond_accuracy,
        "per_user_artifact": per_user_artifact,
        "wall_clock_s": wall_clock_s,
        "hardware": hardware,
    }


def append_record(record: dict, results_path: str | Path) -> None:
    """Append one compact JSON line to ``results_path`` (create parent if needed).

    Opens in append mode ONLY (CLAUDE.md invariant #3 — never any other write
    mode), flushes, and ``os.fsync`` so the record is durable before returning.
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), sort_keys=False)
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def dataset_manifest_hash(manifest_path: str | Path) -> str:
    """``sha256:`` of ``data/MANIFEST.md`` (path configurable for tests)."""
    return sha256_file(manifest_path)


def config_hash(config_path: str | Path) -> str:
    """``sha256:`` of a config file's raw bytes."""
    return sha256_file(config_path)


# Default provenance paths (overridable by callers / tests).
DEFAULT_SPLITS_PATH = _REPO_ROOT / "configs" / "splits.yaml"
DEFAULT_MANIFEST_PATH = _REPO_ROOT / "data" / "MANIFEST.md"
