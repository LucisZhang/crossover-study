"""CLI: paired-bootstrap delta between two per-user eval artifacts (Phase 2, T5).

    uv run python -m batch_recsys_lab.eval.compare \
        --config configs/compare_pop_t12m_vs_alltime_test.yaml \
        [--results results/runs.jsonl]

Compare config::

    kind: paired_delta
    a: {run_id: <id>}   # or {artifact: <path>}
    b: {run_id: <id>}
    metrics: [recall@20, ndcg@10, ...]
    bootstrap: {n_resamples: 1000, seed: 20260805}

Both per-user artifacts are inner-joined on ``user_id``; for same-split runs the
overlap must equal both lengths (asserted). The same paired resample-index matrix
is applied to both arms per metric (:func:`bootstrap.paired_delta_ci`) — globally
and within each segment — and a ``kind="paired_delta"`` record is appended. A
difference is claimed only where the CI excludes 0. Self-comparison yields
``delta == 0`` exactly for every metric (append-only / determinism sanity).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from batch_recsys_lab.contracts.engine import _resolve_run_id
from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.bootstrap import paired_delta_ci


def _resolve_arm(arm: dict, results_path: str | Path) -> dict:
    """Resolve one compare arm to ``{run_id, model, artifact}``.

    Prefers an explicit ``artifact`` path; otherwise scans the results JSONL for
    the last ``kind="eval"`` record whose ``run_id`` matches (last match wins).
    """
    if arm.get("artifact"):
        return {"run_id": arm.get("run_id"), "model": arm.get("model"), "artifact": arm["artifact"]}

    run_id = arm.get("run_id")
    if not run_id:
        raise ValueError("compare arm needs either 'artifact' or 'run_id'")

    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"results log {results_path} not found for run_id resolution")

    match: dict | None = None
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("kind") == "eval" and rec.get("run_id") == run_id:
            match = rec  # last match wins
    if match is None:
        raise ValueError(f"no eval record with run_id={run_id!r} in {results_path}")
    return {
        "run_id": run_id,
        "model": match.get("model", {}).get("name"),
        "artifact": match["per_user_artifact"],
    }


def _load_artifact(path: str | Path) -> dict:
    table = pq.read_table(path)
    cols = {name: table.column(name) for name in table.column_names}
    user_ids = [str(u) for u in cols["user_id"].to_pylist()]
    segments = [str(s) for s in cols["segment"].to_pylist()]
    return {"user_ids": user_ids, "segments": segments, "table": table}


def compare(
    config: dict,
    config_path: str | Path,
    results_path: str | Path,
) -> dict:
    a_arm = _resolve_arm(config["a"], results_path)
    b_arm = _resolve_arm(config["b"], results_path)
    metrics = list(config["metrics"])
    boot = config.get("bootstrap", {})
    n_resamples = int(boot.get("n_resamples", 1000))
    boot_seed = int(boot.get("seed", 20260805))

    a = _load_artifact(a_arm["artifact"])
    b = _load_artifact(b_arm["artifact"])

    b_pos = {uid: i for i, uid in enumerate(b["user_ids"])}
    a_keep = [i for i, uid in enumerate(a["user_ids"]) if uid in b_pos]
    common_ids = [a["user_ids"][i] for i in a_keep]
    n_common = len(common_ids)

    # Same-split runs share their user universe exactly.
    assert n_common == len(a["user_ids"]) == len(b["user_ids"]), (
        f"artifact user overlap {n_common} != a={len(a['user_ids'])} / b={len(b['user_ids'])}; "
        "compare is only defined for same-split runs"
    )

    b_keep = [b_pos[uid] for uid in common_ids]
    segments = np.asarray([a["segments"][i] for i in a_keep])

    def _col(art: dict, idx: list[int], metric: str) -> np.ndarray:
        return np.asarray(art["table"].column(metric).to_pylist(), dtype=np.float64)[idx]

    global_deltas: dict[str, dict] = {}
    seg_deltas: dict[str, dict] = {}
    seen_labels: list[str] = []
    for lbl in segments:
        if lbl not in seen_labels:
            seen_labels.append(lbl)

    for metric in metrics:
        av = _col(a, a_keep, metric)
        bv = _col(b, b_keep, metric)
        global_deltas[metric] = paired_delta_ci(av, bv, n_resamples=n_resamples, seed=boot_seed)
        for lbl in seen_labels:
            mask = segments == lbl
            d = paired_delta_ci(av[mask], bv[mask], n_resamples=n_resamples, seed=boot_seed)
            seg_deltas.setdefault(lbl, {})[metric] = d

    git = runlog.git_info()
    run_id, run_ts = _resolve_run_id(None)

    record = {
        "schema_version": runlog.record_schema_version,
        "kind": "paired_delta",
        "run_id": run_id,
        "run_ts": run_ts,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "config_path": str(config_path),
        "config_hash": runlog.config_hash(config_path),
        "a": a_arm,
        "b": b_arm,
        "n_common_users": n_common,
        "deltas": {"global": global_deltas, "per_segment": seg_deltas},
    }
    runlog.append_record(record, results_path)
    _print_summary(record)
    return record


def _print_summary(record: dict) -> None:
    a, b = record["a"], record["b"]
    print(
        f"[paired_delta] a={a.get('model')}({a.get('run_id')}) - "
        f"b={b.get('model')}({b.get('run_id')})  n_common={record['n_common_users']}"
    )
    for metric, d in record["deltas"]["global"].items():
        star = "*" if d["excludes_zero"] else " "
        print(
            f"  {star} {metric:>12}  delta={d['delta']:+.4f} "
            f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.eval.compare")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", default="results/runs.jsonl")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    try:
        compare(config, config_path=config_path, results_path=args.results)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report + non-zero exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
