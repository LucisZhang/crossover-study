"""CLI: n*/variant selection for the history-depth routing policy, VAL-only
(Phase 4, T13; docs/engineering-log/UPGRADE_PLAN.md §6.4).

    uv run python -m batch_recsys_lab.policy.select \
        --config configs/policy_select_val.yaml \
        [--results results/runs.jsonl]

Composes hybrid per-user NDCG@10 *without re-scoring* by exploiting the fact
that the n* grid in the config is chosen to align exactly with the frozen
segment buckets (``eval.protocol.SEGMENT_LABELS`` = 0 / 1-4 / 5-9 / 10-19 /
20+): for every n_star in the grid, "n_train < n_star" is constant within
each segment bucket (no bucket straddles a grid edge), so a user's routing
arm can be read off their ``segment`` column exactly, with no need to reload
``ds.n_train`` from the eval cache.

Objective (owner-approved, pre-declared in docs/engineering-log/EXPERIMENT_LOG.md T13 entry BEFORE
this script is run): unweighted mean of the five segment-mean NDCG@10 values
("segment_weighted_ndcg10_unweighted_mean" in the config).

Winner rule (pre-declared, see docs/engineering-log/EXPERIMENT_LOG.md T13 entry): argmax objective
over the 2 variants x 5 n_star grid; ties -> prefer variant B (pop-t12m as the
warm/high component) and, among remaining ties, the n_star closest to
infinity (more blend coverage / simpler policy).

T9-3b (ML-32M, docs/engineering-log/EXPERIMENT_LOG.md Rule S5) extends this module with two
opt-in, default-preserving config keys so the Amazon config's output stays
byte-identical:

- ``route_by: n_train`` — instead of reading the coarse five-segment
  bucket column off the per-user artifact (exact only because the Amazon
  n_star grid was chosen to align with the segment edges), route each user
  directly by ``n_train[user_idx] < n_star`` where ``n_train`` is loaded
  from the eval cache named by the config's ``cache_dir`` key, joined on
  the per-user artifact's own ``user_idx`` column. This supports arbitrary
  n_star edges (e.g. 50, 100) that do not align with any segment bucket.
- ``objective: global_ndcg10_mean`` — the plain mean of the composed
  per-user metric over all users (the Rule S5 "maximizes global VAL
  NDCG@10"), computed alongside (not instead of) the default segment
  objective so both are available on every cell.

Tie rule and output shape are unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from batch_recsys_lab.eval.harness import _resolve_cache_dir

# Segment -> minimum n_train value in that bucket (frozen boundaries; see
# eval/protocol.py SEGMENT_LABELS / _SEGMENT_EDGES). Used only to decide, for
# a given n_star, whether an entire segment routes to the low (blend) arm.
SEGMENT_MIN_N_TRAIN = {"0": 0, "1-4": 1, "5-9": 5, "10-19": 10, "20+": 20}
SEGMENT_LABELS = ("0", "1-4", "5-9", "10-19", "20+")


def _resolve_n_star(raw) -> float:
    if raw is None:
        return float("inf")
    if isinstance(raw, str) and raw.strip().lower() in ("inf", "infinity", "null", "none"):
        return float("inf")
    return float(raw)


def _n_star_label(raw) -> str:
    return "inf" if _resolve_n_star(raw) == float("inf") else str(raw)


def _resolve_artifact_path(run_id: str, results_path: Path) -> tuple[str, str]:
    """Return (artifact_path, model_name) for the last kind="eval" record
    matching ``run_id`` in the results log (mirrors eval.compare._resolve_arm)."""
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
    return match["per_user_artifact"], match.get("model", {}).get("name")


def _load_artifact(path: str, metric: str) -> dict:
    table = pq.read_table(path)
    user_ids = [str(u) for u in table.column("user_id").to_pylist()]
    segments = np.asarray([str(s) for s in table.column("segment").to_pylist()])
    values = np.asarray(table.column(metric).to_pylist(), dtype=np.float64)
    user_idx = np.asarray(table.column("user_idx").to_pylist(), dtype=np.int64)
    return {"user_ids": user_ids, "segments": segments, "values": values, "user_idx": user_idx}


def _load_n_train(cache_dir: str) -> np.ndarray:
    """Load ``n_train.npy`` from the eval cache named by the config's
    ``cache_dir`` key (T9-3b ``route_by: n_train``), resolving the single
    snapshot subdir exactly as ``eval.harness`` does."""
    resolved = _resolve_cache_dir(cache_dir)
    return np.load(resolved / "n_train.npy", allow_pickle=False)


def _align(arms: dict[str, dict]) -> dict[str, dict]:
    """Assert all arms cover the identical user set, then reindex every arm's
    arrays to a single common (sorted) user_id order (compare.py precedent:
    same-split runs must share their user universe exactly)."""
    names = list(arms)
    base_ids = set(arms[names[0]]["user_ids"])
    for n in names[1:]:
        ids = set(arms[n]["user_ids"])
        assert ids == base_ids, (
            f"artifact user set mismatch: {names[0]} has {len(base_ids)} users, "
            f"{n} has {len(ids)}; symmetric diff size = {len(ids ^ base_ids)}"
        )

    order = sorted(base_ids)
    aligned = {}
    for n in names:
        pos = {uid: i for i, uid in enumerate(arms[n]["user_ids"])}
        idx = np.array([pos[uid] for uid in order], dtype=np.int64)
        aligned[n] = {
            "segments": arms[n]["segments"][idx],
            "values": arms[n]["values"][idx],
            "user_idx": arms[n]["user_idx"][idx],
        }
    aligned["_order"] = order
    return aligned


def _segment_means(values: np.ndarray, segments: np.ndarray) -> dict[str, float]:
    out = {}
    for lbl in SEGMENT_LABELS:
        mask = segments == lbl
        out[lbl] = float(values[mask].mean()) if np.any(mask) else float("nan")
    return out


def _objective(segment_means: dict[str, float]) -> float:
    return float(np.mean([segment_means[lbl] for lbl in SEGMENT_LABELS]))


def _compose_hybrid(
    low_values: np.ndarray,
    high_values: np.ndarray,
    segments: np.ndarray,
    n_star,
) -> np.ndarray:
    n_star_f = _resolve_n_star(n_star)
    low_mask = np.array(
        [SEGMENT_MIN_N_TRAIN[s] < n_star_f for s in segments], dtype=bool
    )
    return np.where(low_mask, low_values, high_values)


def _compose_hybrid_n_train(
    low_values: np.ndarray,
    high_values: np.ndarray,
    user_n_train: np.ndarray,
    n_star,
) -> np.ndarray:
    """T9-3b ``route_by: n_train``: route each user directly by
    ``n_train[user_idx] < n_star`` (no segment-bucket approximation), so
    n_star grid values that do not align with a segment edge (e.g. 50, 100)
    are exact."""
    n_star_f = _resolve_n_star(n_star)
    low_mask = user_n_train.astype(np.float64) < n_star_f
    return np.where(low_mask, low_values, high_values)


def _global_objective(values: np.ndarray) -> float:
    """T9-3b ``objective: global_ndcg10_mean``: the plain mean of the
    composed per-user metric over all users (Rule S5's "maximizes global
    VAL NDCG@10"), with no segment weighting."""
    return float(np.mean(values))


def select(config: dict, results_path: Path) -> dict:
    run_ids = config["run_ids"]
    metric = config.get("metric", "ndcg@10")
    n_star_grid = config["n_star_grid"]
    variants = config["variants"]
    route_by = config.get("route_by", "segment")
    objective_name = config.get("objective", "segment_weighted_ndcg10_unweighted_mean")

    raw_arms = {}
    model_names = {}
    for key, run_id in run_ids.items():
        artifact_path, model_name = _resolve_artifact_path(run_id, results_path)
        raw_arms[key] = _load_artifact(artifact_path, metric)
        model_names[key] = model_name

    aligned = _align(raw_arms)
    segments = aligned[next(iter(run_ids))]["segments"]  # identical across arms post-align

    n_train_by_user_idx = None
    user_n_train = None
    if route_by == "n_train":
        if "cache_dir" not in config:
            raise ValueError("route_by: n_train requires a 'cache_dir' config key")
        n_train_by_user_idx = _load_n_train(config["cache_dir"])
        user_idx = aligned[next(iter(run_ids))]["user_idx"]  # identical across arms post-align
        user_n_train = n_train_by_user_idx[user_idx]
    elif route_by != "segment":
        raise ValueError(f"unsupported route_by: {route_by!r}")

    grid = []
    best = None
    for variant_name, spec in variants.items():
        low_key, high_key = spec["low"], spec["high"]
        low_values = aligned[low_key]["values"]
        high_values = aligned[high_key]["values"]
        for n_star in n_star_grid:
            if route_by == "n_train":
                composed = _compose_hybrid_n_train(low_values, high_values, user_n_train, n_star)
            else:
                composed = _compose_hybrid(low_values, high_values, segments, n_star)
            seg_means = _segment_means(composed, segments)
            if objective_name == "global_ndcg10_mean":
                obj = _global_objective(composed)
            else:
                obj = _objective(seg_means)
            cell = {
                "variant": variant_name,
                "low": low_key,
                "high": high_key,
                "n_star": n_star,
                "n_star_label": _n_star_label(n_star),
                "segment_means": seg_means,
                "objective": obj,
            }
            grid.append(cell)

    # Winner rule (pre-declared, see module docstring / docs/engineering-log/EXPERIMENT_LOG.md):
    # argmax objective; ties -> prefer variant B, then n_star closest to inf.
    max_obj = max(c["objective"] for c in grid)
    tied = [c for c in grid if c["objective"] == max_obj]
    if len(tied) == 1:
        winner = tied[0]
    else:
        tied_b = [c for c in tied if c["variant"] == "B"]
        pool = tied_b if tied_b else tied
        pool_sorted = sorted(pool, key=lambda c: -_resolve_n_star(c["n_star"]))
        winner = pool_sorted[0]

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    result = {
        "run_ids": run_ids,
        "model_names": model_names,
        "objective": config.get("objective", "segment_weighted_ndcg10_unweighted_mean"),
        "metric": metric,
        "grid": grid,
        "winner": winner,
        "git_sha": git_sha,
    }
    return result


def _print_grid(result: dict) -> None:
    print(f"{'variant':<8}{'n_star':<8}" + "".join(f"{lbl:>10}" for lbl in SEGMENT_LABELS) + f"{'objective':>12}")
    for c in result["grid"]:
        row = f"{c['variant']:<8}{c['n_star_label']:<8}"
        row += "".join(f"{c['segment_means'][lbl]:>10.6f}" for lbl in SEGMENT_LABELS)
        row += f"{c['objective']:>12.6f}"
        print(row)
    w = result["winner"]
    print(
        f"\nWINNER: variant={w['variant']} n_star={w['n_star_label']} "
        f"objective={w['objective']:.6f} (low={w['low']}, high={w['high']})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.policy.select")
    parser.add_argument("--config", default="configs/policy_select_val.yaml")
    parser.add_argument("--results", default="results/runs.jsonl")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    results_path = Path(args.results)

    try:
        result = select(config, results_path)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report + non-zero exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_grid(result)

    output_path = Path(config.get("output_path", "results/policy_select_val.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    print(f"\nwrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
