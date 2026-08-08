"""CLI: n* routing grid on TEST, by recomposition of recorded per-user outputs
(Phase 6, T27; UPGRADE_PLAN.md §9 exhibit 1 "n* slider").

    uv run python -m batch_recsys_lab.policy.grid_test \
        --config configs/policy_grid_test.yaml [--dry-run]

This is the TEST-side *exhibit* for the VAL-only policy selection performed by
:mod:`batch_recsys_lab.policy.select` (results/policy_select_val.json, winner
B/inf). It mirrors select.py's composition mechanics: because every n_star in
the grid lands on a frozen segment edge (``eval.protocol.SEGMENT_LABELS`` =
0 / 1-4 / 5-9 / 10-19 / 20+), "n_train < n_star" is constant within each
bucket, so a user's routing arm is read off the artifact ``segment`` column —
no eval cache, no model code, no re-scoring.

Frozen-TEST justification (CLAUDE.md invariant #1). Nothing here consults
ground truth, refits a model, or scores an item: every number is an arithmetic
recomposition of per-user metric values that the three one-shot TEST eval runs
already computed and committed to ``results/runs.jsonl``. The grid therefore
adds no TEST information beyond what those recorded runs already published;
n* itself was and remains chosen on VAL.

Identity anchors (hard, checked before anything is written): each variant's
n_star=inf cell routes every user to the blend arm, so it must equal the blend
record's recorded global + per-segment floats EXACTLY; each variant's
n_star=0 cell routes everyone to that variant's high arm, so it must equal the
ALS (A) / pop-t12m (B) record's floats EXACTLY. Same per-user vectors + same
bootstrap seed => bit-identical floats. Any mismatch aborts with no write.

Row-order discipline — deliberate deviation from ``select._align``. select.py
reindexes every arm to sorted-user_id order, which is fine there because it
only reports means (order-invariant). Bootstrap CIs are NOT order-invariant:
:func:`eval.bootstrap.resample_matrix` draws *positions*, so permuting the
value vector changes the resampled means and hence the percentile CIs. To
reproduce the recorded floats we must keep the artifacts' original row order.
This module therefore performs select.py's user-set equality assertion and
then additionally requires identical row ORDER across the three artifacts
(they are all written from the same ``gt.user_idx`` of the same split, so this
holds; if it ever stops holding, the identity checks could not pass anyway and
the failure should be loud and early).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from batch_recsys_lab.contracts.engine import _resolve_run_id
from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.bootstrap import ci_mean, segment_cis
from batch_recsys_lab.policy.select import (
    SEGMENT_LABELS,
    _compose_hybrid,
    _n_star_label,
    _resolve_artifact_path,
    _resolve_n_star,
)

PROVENANCE_NOTE = (
    "Derived exhibit: pure recomposition of the per-user metric vectors recorded "
    "by the one-shot TEST eval runs named in source_run_ids. No re-scoring, no "
    "refitting, no new ground-truth consultation — every cell is an arithmetic "
    "regrouping (by the frozen segment edges) of values already committed to "
    "results/runs.jsonl, so the frozen-TEST invariant (CLAUDE.md #1) is not "
    "touched. n* was selected on VAL only (see val_selection_ref)."
)


# --- artifact loading / alignment --------------------------------------------


def _load_artifact(path: str, metrics: list[str]) -> dict:
    """user_ids + segments + one float64 vector per metric, in ARTIFACT ROW ORDER."""
    table = pq.read_table(path, columns=["user_id", "segment", *metrics])
    return {
        "user_ids": [str(u) for u in table.column("user_id").to_pylist()],
        "segments": np.asarray([str(s) for s in table.column("segment").to_pylist()]),
        "values": {
            m: np.asarray(table.column(m).to_pylist(), dtype=np.float64) for m in metrics
        },
    }


def _align_strict(arms: dict[str, dict]) -> np.ndarray:
    """Assert all arms cover the identical user set (select._align's contract)
    AND are in the identical row order (required for CI-float identity; see the
    module docstring). Returns the shared segment vector."""
    names = list(arms)
    base = names[0]
    base_ids = set(arms[base]["user_ids"])
    for n in names[1:]:
        ids = set(arms[n]["user_ids"])
        assert ids == base_ids, (
            f"artifact user set mismatch: {base} has {len(base_ids)} users, "
            f"{n} has {len(ids)}; symmetric diff size = {len(ids ^ base_ids)}"
        )
    for n in names[1:]:
        assert arms[n]["user_ids"] == arms[base]["user_ids"], (
            f"artifact row-order mismatch between {base} and {n}: identity with the "
            "recorded bootstrap CIs requires the original per-user row order "
            "(resample indices are positional)."
        )
        assert np.array_equal(arms[n]["segments"], arms[base]["segments"]), (
            f"segment vector mismatch between {base} and {n}"
        )
    return arms[base]["segments"]


# --- per-cell metric blocks ---------------------------------------------------


def _metric_blocks(
    values: dict[str, np.ndarray],
    segments: np.ndarray,
    n_resamples: int,
    seed: int,
) -> tuple[dict, dict]:
    """(global, per_segment) blocks in the exact shape eval records use
    (``eval.harness.run_eval``: ci_mean for global, segment_cis per segment)."""
    global_block: dict[str, dict] = {}
    per_segment: dict[str, dict] = {}
    for name, vec in values.items():
        global_block[name] = ci_mean(vec, n_resamples=n_resamples, seed=seed)
        seg = segment_cis(vec, segments, n_resamples=n_resamples, seed=seed)
        for label, d in seg.items():
            blk = per_segment.setdefault(label, {"n_users": d["n_users"]})
            blk[name] = {"value": d["value"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]}
    ordered = {lbl: per_segment[lbl] for lbl in SEGMENT_LABELS if lbl in per_segment}
    ordered.update({k: v for k, v in per_segment.items() if k not in ordered})
    return global_block, ordered


# --- identity checks ----------------------------------------------------------


def _diff_metric_blocks(
    cell_global: dict,
    cell_segments: dict,
    record: dict,
    metrics: list[str],
    tag: str,
) -> list[str]:
    """Exact-float comparison of a cell against a recorded eval record's metrics.
    Returns a list of human-readable mismatches (empty == identical)."""
    diffs: list[str] = []
    rec_global = record["metrics"]["global"]
    rec_segments = record["metrics"]["per_segment"]
    for m in metrics:
        got, want = cell_global.get(m), rec_global.get(m)
        for field in ("value", "ci_lo", "ci_hi"):
            g = None if got is None else got.get(field)
            w = None if want is None else want.get(field)
            if g != w:
                diffs.append(f"{tag} global {m}.{field}: cell={g!r} record={w!r}")
    if set(cell_segments) != set(rec_segments):
        diffs.append(
            f"{tag} segment label sets differ: cell={sorted(cell_segments)} "
            f"record={sorted(rec_segments)}"
        )
    for label in sorted(set(cell_segments) & set(rec_segments)):
        if cell_segments[label]["n_users"] != rec_segments[label]["n_users"]:
            diffs.append(
                f"{tag} segment {label} n_users: cell="
                f"{cell_segments[label]['n_users']} record={rec_segments[label]['n_users']}"
            )
        for m in metrics:
            got, want = cell_segments[label].get(m), rec_segments[label].get(m)
            for field in ("value", "ci_lo", "ci_hi"):
                g = None if got is None else got.get(field)
                w = None if want is None else want.get(field)
                if g != w:
                    diffs.append(
                        f"{tag} segment {label} {m}.{field}: cell={g!r} record={w!r}"
                    )
    return diffs


# --- grid ---------------------------------------------------------------------


def build_grid(config: dict, results_path: Path) -> dict:
    """Compute the full n* x variant grid + identity checks. Writes nothing."""
    t0 = time.monotonic()
    run_ids: dict[str, str] = config["run_ids"]
    metrics: list[str] = list(config["metrics"])
    n_star_grid = config["n_star_grid"]
    variants: dict[str, dict] = config["variants"]
    boot = config.get("bootstrap", {})
    n_resamples = int(boot.get("n_resamples", 1000))
    seed = int(boot.get("seed", 20260805))
    split = config.get("split", "test")

    records: dict[str, dict] = {}
    arms: dict[str, dict] = {}
    artifact_paths: dict[str, str] = {}
    for key, run_id in run_ids.items():
        artifact_path, _model = _resolve_artifact_path(run_id, results_path)
        rec = _find_record(run_id, results_path)
        if rec["protocol"]["eval_split"] != split:
            raise ValueError(
                f"{key} ({run_id}) is an eval on split {rec['protocol']['eval_split']!r}, "
                f"expected {split!r}"
            )
        rec_seed = (rec.get("seeds") or {}).get("bootstrap")
        if int(rec_seed) != seed:
            raise ValueError(
                f"{key} ({run_id}) recorded bootstrap seed {rec_seed} != config seed {seed}; "
                "identity with the recorded floats is impossible"
            )
        records[key] = rec
        artifact_paths[key] = artifact_path
        arms[key] = _load_artifact(artifact_path, metrics)

    segments = _align_strict(arms)
    n_users = len(segments)
    expected = config.get("expected_n_users")
    if expected is not None and n_users != int(expected):
        raise ValueError(f"user count {n_users} != expected_n_users {expected}")

    cells: list[dict] = []
    for variant_name, spec in variants.items():
        low_key, high_key = spec["low"], spec["high"]
        for n_star in n_star_grid:
            composed = {
                m: _compose_hybrid(
                    arms[low_key]["values"][m], arms[high_key]["values"][m], segments, n_star
                )
                for m in metrics
            }
            g, ps = _metric_blocks(composed, segments, n_resamples, seed)
            cells.append(
                {
                    "variant": variant_name,
                    "low": low_key,
                    "high": high_key,
                    "n_star": n_star,
                    "n_star_label": _n_star_label(n_star),
                    "low_share": float(
                        np.mean(_compose_hybrid(
                            np.ones(n_users), np.zeros(n_users), segments, n_star
                        ))
                    ),
                    "global": g,
                    "per_segment": ps,
                }
            )

    ident_cfg = config.get("identity_checks", {})
    inf_ref = ident_cfg.get("inf_reference")
    zero_ref = ident_cfg.get("zero_reference", {})
    diffs: list[str] = []
    checked = {"inf": [], "zero": []}
    for cell in cells:
        n_star_f = _resolve_n_star(cell["n_star"])
        if n_star_f == float("inf") and inf_ref:
            tag = f"{cell['variant']}/inf vs {inf_ref}({run_ids[inf_ref]})"
            diffs += _diff_metric_blocks(
                cell["global"], cell["per_segment"], records[inf_ref], metrics, tag
            )
            checked["inf"].append(f"{cell['variant']}/inf=={inf_ref}")
        elif n_star_f == 0.0 and cell["variant"] in zero_ref:
            ref = zero_ref[cell["variant"]]
            tag = f"{cell['variant']}/0 vs {ref}({run_ids[ref]})"
            diffs += _diff_metric_blocks(
                cell["global"], cell["per_segment"], records[ref], metrics, tag
            )
            checked["zero"].append(f"{cell['variant']}/0=={ref}")

    if not checked["inf"] or not checked["zero"]:
        raise RuntimeError(
            "identity checks did not run: the grid must contain an inf cell and a "
            f"0 cell per variant (ran inf={checked['inf']}, zero={checked['zero']})"
        )

    return {
        "cells": cells,
        "metrics": metrics,
        "n_users": n_users,
        "split": split,
        "seed": seed,
        "n_resamples": n_resamples,
        "artifact_paths": artifact_paths,
        "identity_diffs": diffs,
        "identity_checked": checked,
        "wall_clock_s": round(time.monotonic() - t0, 3),
    }


def _find_record(run_id: str, results_path: Path) -> dict:
    match = None
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("kind") == "eval" and rec.get("run_id") == run_id:
            match = rec
    if match is None:
        raise ValueError(f"no eval record with run_id={run_id!r} in {results_path}")
    return match


# --- record assembly ----------------------------------------------------------


def build_record(config: dict, config_path: Path, grid: dict) -> dict:
    run_ids = config["run_ids"]
    git = runlog.git_info()
    run_id, run_ts = _resolve_run_id(None)
    val_cfg = config.get("val_selection", {})
    val_output = val_cfg.get("output", "results/policy_select_val.json")
    return {
        "schema_version": runlog.record_schema_version,
        "kind": "policy_grid",
        "run_id": run_id,
        "run_ts": run_ts,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "config_path": str(config_path),
        "config_hash": runlog.config_hash(config_path),
        "derived": True,
        "provenance_note": PROVENANCE_NOTE,
        "source_run_ids": run_ids,
        "source_artifact_sha256s": {
            key: runlog.sha256_file(grid["artifact_paths"][key]) for key in run_ids
        },
        "split": grid["split"],
        "n_users": grid["n_users"],
        "metrics": grid["metrics"],
        "seeds": {"bootstrap": grid["seed"]},
        "bootstrap": {"n_resamples": grid["n_resamples"], "seed": grid["seed"]},
        "n_star_grid": [c["n_star"] for c in grid["cells"] if c["variant"] == grid["cells"][0]["variant"]],
        "grid": [
            {
                "variant": c["variant"],
                "low": c["low"],
                "high": c["high"],
                "n_star": c["n_star"],
                "n_star_label": c["n_star_label"],
                "low_share": c["low_share"],
                "global": c["global"],
                "per_segment": c["per_segment"],
            }
            for c in grid["cells"]
        ],
        "identity_checks": {
            "inf_matches_blend": True,
            "zero_matches_high": True,
            "checked": grid["identity_checked"],
        },
        "val_selection_ref": {
            "output": val_output,
            "winner": val_cfg.get("winner"),
            "sha256": runlog.sha256_file(val_output),
        },
        "wall_clock_s": grid["wall_clock_s"],
        "hardware": runlog.hardware_string(),
    }


# --- CLI ----------------------------------------------------------------------


def _print_grid(grid: dict) -> None:
    metrics = grid["metrics"]
    header = f"{'variant':<8}{'n*':<6}{'low%':>7}"
    for m in metrics:
        header += f"{m:>36}"
    print(header)
    for c in grid["cells"]:
        row = f"{c['variant']:<8}{c['n_star_label']:<6}{100 * c['low_share']:>6.1f}%"
        for m in metrics:
            d = c["global"][m]
            row += f"  {d['value']:.6f} [{d['ci_lo']:.6f}, {d['ci_hi']:.6f}]"
        print(row)
    print()
    print("per-segment NDCG@10 (segment order 0 / 1-4 / 5-9 / 10-19 / 20+):")
    for c in grid["cells"]:
        vals = "".join(
            f"{c['per_segment'][lbl]['ndcg@10']['value']:>11.6f}"
            if lbl in c["per_segment"] and "ndcg@10" in c["per_segment"][lbl]
            else f"{'n/a':>11}"
            for lbl in SEGMENT_LABELS
        )
        print(f"  {c['variant']}/{c['n_star_label']:<5}{vals}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.policy.grid_test")
    parser.add_argument("--config", default="configs/policy_grid_test.yaml")
    parser.add_argument("--results", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute + print + report identity checks, append nothing",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    results_path = Path(args.results or config.get("results_path", "results/runs.jsonl"))

    try:
        grid = build_grid(config, results_path)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report + non-zero exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_grid(grid)
    print(
        f"\nusers={grid['n_users']}  cells={len(grid['cells'])}  "
        f"bootstrap: n_resamples={grid['n_resamples']} seed={grid['seed']}"
    )

    diffs = grid["identity_diffs"]
    print("\nidentity checks:")
    for name in grid["identity_checked"]["inf"] + grid["identity_checked"]["zero"]:
        print(f"  ran {name}")
    if diffs:
        print(f"  FAIL: {len(diffs)} float mismatch(es) — nothing written", file=sys.stderr)
        for d in diffs[:20]:
            print(f"    {d}", file=sys.stderr)
        if len(diffs) > 20:
            print(f"    … {len(diffs) - 20} more", file=sys.stderr)
        return 1
    print("  PASS: all degenerate cells are bit-identical to their source records")

    if args.dry_run:
        print("\n--dry-run: no record appended")
        return 0

    record = build_record(config, config_path, grid)
    runlog.append_record(record, results_path)
    print(f"\nappended kind=policy_grid run_id={record['run_id']} -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
