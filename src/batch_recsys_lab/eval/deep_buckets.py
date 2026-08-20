"""CLI: deep history-depth buckets (Phase 8, T8-3; UPGRADE_PLAN.md §8b).

    uv run python -m batch_recsys_lab.eval.deep_buckets \
        --config configs/deep_buckets_test.yaml [--dry-run]

Phase 4 reported the frozen five segments (0 / 1-4 / 5-9 / 10-19 / **20+**) and
found the ALS-vs-pop-t12m deficit narrowing monotonically with history depth
without ever crossing. "20+" is open-ended, so the obvious question — does the
deficit keep narrowing inside it? — was unanswerable from the recorded blocks.
This module splits it into 20-49 / 50-99 / 100+ (:data:`eval.protocol
.DEEP_BUCKET_LABELS`) by regrouping the *already persisted* per-user metric
values.

**Exploratory, not confirmatory.** The boundaries were fixed in
``EXPERIMENT_LOG.md`` (2026-08-17) before any per-bucket outcome was computed,
but they were *motivated* by the observed narrowing, so the record carries
``exploratory_derived: true`` and every bucket reports its user count and CI
widths. Thin buckets are expected; they are disclosed, not smoothed.

**Self-check (the thing that makes this trustworthy).** Buckets 0 / 1-4 / 5-9 /
10-19 are the SAME user sets as the corresponding frozen segments, so their
recomposed means must equal the values already committed in each arm's eval
record. The check runs before anything is emitted and fails the run on any
mismatch beyond ``self_check.tolerance``. Only the MEANS are compared: the
recorded per-segment CIs come from ``segment_cis``'s first-appearance ordinals,
whereas each deep bucket seeds off its own fixed bucket ordinal (see
:data:`SEED_SCHEME`), so the CI percentiles legitimately differ — a deliberate
choice to keep the deep-bucket seeding scheme uniform across all seven buckets
rather than special-casing four of them to reproduce old floats.

Nothing here fits or scores a model: the ``top50`` column is not even read.
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
from batch_recsys_lab.eval.cell_stats import cell_block
from batch_recsys_lab.eval.harness import _resolve_cache_dir
from batch_recsys_lab.eval.protocol import (
    DEEP_BUCKET_LABELS,
    SEGMENT_LABELS,
    deep_bucket_of,
    segment_of,
)
from batch_recsys_lab.eval.regime_map import _find_eval_record
from batch_recsys_lab.features import item_train_stats
from batch_recsys_lab.policy.select import _resolve_artifact_path

# The first four deep buckets coincide with these frozen segments by construction.
SELF_CHECK_LABELS = ("0", "1-4", "5-9", "10-19")
DEFAULT_TOLERANCE = 1e-9

SEED_SCHEME = (
    "np.random.default_rng([base_seed, bucket_ordinal]) per bucket, where "
    "base_seed=20260805 and bucket_ordinal indexes DEEP_BUCKET_LABELS "
    "('0','1-4','5-9','10-19','20-49','50-99','100+'). Users are resampled WITHIN "
    "the bucket; the same matrix serves both arms' CIs and the paired delta "
    "(see eval/cell_stats.py). Same shape as segment_cis's [seed, ordinal] scheme, "
    "with the ordinal taken from the fixed label tuple instead of first appearance."
)

PROVENANCE_NOTE = (
    "Derived, EXPLORATORY exhibit (Phase 8 T8-3): the per-user metric vectors already "
    "committed by the one-shot eval runs named in source_run_ids, regrouped by "
    "gold.user_stats.n_train into seven depth buckets. No re-scoring, no refitting, no "
    "new ground-truth consultation, no threshold tuning (boundaries preregistered in "
    "EXPERIMENT_LOG.md 2026-08-17). The four buckets that coincide with frozen segments "
    "are asserted equal to the recorded per-segment means before anything is emitted."
)


def _load_arm(path: str, metrics: tuple[str, ...]) -> dict:
    """user_id/user_idx + the requested metric columns, in ARTIFACT ROW ORDER
    (bootstrap resample indices are positional — grid_test's discipline)."""
    table = pq.read_table(path, columns=["user_id", "user_idx", "segment", *metrics])
    return {
        "user_ids": [str(u) for u in table.column("user_id").to_pylist()],
        "user_idx": np.asarray(table.column("user_idx").to_pylist(), dtype=np.int64),
        "segments": np.asarray([str(s) for s in table.column("segment").to_pylist()]),
        "values": {
            m: np.asarray(table.column(m).to_pylist(), dtype=np.float64) for m in metrics
        },
    }


def build_buckets(config: dict, results_path: Path) -> dict:
    """Compute every bucket block + the self-check. Writes nothing."""
    t0 = time.monotonic()
    run_ids: dict[str, str] = dict(config["run_ids"])
    arm_order = list(run_ids)
    delta_pair = (config["delta"]["minuend"], config["delta"]["subtrahend"])
    for key in delta_pair:
        if key not in run_ids:
            raise ValueError(f"delta arm {key!r} is not one of run_ids {arm_order}")
    metrics = tuple(config.get("metrics", ("ndcg@10", "recall@20")))
    boot = config.get("bootstrap", {})
    n_resamples = int(boot.get("n_resamples", 1000))
    base_seed = int(boot.get("seed", 20260805))
    split = config.get("split", "test")
    tolerance = float((config.get("self_check") or {}).get("tolerance", DEFAULT_TOLERANCE))
    # Opt-in (T9-3c): add the §5(e) two-sided ASL p-value to every bucket's delta
    # block, drawn off that bucket's own child seed. Default False keeps the
    # committed T8-3 record shape and values byte-identical.
    asl_p_values = bool(config.get("asl_p_values", False))

    cache_dir = _resolve_cache_dir(config["cache_dir"])
    cache_manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
    five_core_table = config.get("five_core_table", item_train_stats.FIVE_CORE)
    cache_sid = int(cache_manifest["snapshot_ids"][five_core_table])

    records: dict[str, dict] = {}
    artifact_paths: dict[str, str] = {}
    arms: dict[str, dict] = {}
    for key, run_id in run_ids.items():
        artifact_path, _model = _resolve_artifact_path(run_id, results_path)
        rec = _find_eval_record(run_id, results_path)
        if rec["protocol"]["eval_split"] != split:
            raise ValueError(
                f"{key} ({run_id}) is an eval on split {rec['protocol']['eval_split']!r}, "
                f"expected {split!r}"
            )
        rec_sids = rec.get("iceberg_snapshots") or {}
        if int(rec_sids.get(five_core_table, -1)) != cache_sid:
            raise RuntimeError(
                f"{key} ({run_id}) was scored on {five_core_table} snapshot "
                f"{rec_sids.get(five_core_table)} != cache snapshot {cache_sid}"
            )
        records[key] = rec
        artifact_paths[key] = artifact_path
        arms[key] = _load_arm(artifact_path, metrics)

    base = arm_order[0]
    for key in arm_order[1:]:
        if arms[key]["user_ids"] != arms[base]["user_ids"]:
            raise RuntimeError(
                f"artifact user set/order mismatch between {base} and {key}: "
                "per-bucket resampling is positional, so identical row order is required"
            )
        if not np.array_equal(arms[key]["segments"], arms[base]["segments"]):
            raise RuntimeError(f"segment vector mismatch between {base} and {key}")
    user_idx = arms[base]["user_idx"]
    segments = arms[base]["segments"]
    n_users = int(len(user_idx))
    expected = config.get("expected_n_users")
    if expected is not None and n_users != int(expected):
        raise ValueError(f"user count {n_users} != expected_n_users {expected}")

    n_train = np.load(cache_dir / "n_train.npy", allow_pickle=False)
    depth = n_train[user_idx]
    recomputed = np.asarray([str(s) for s in segment_of(depth)])
    if not np.array_equal(recomputed, segments):
        raise RuntimeError(
            "artifact segment column != segment_of(n_train[user_idx]) — the cache and "
            "the artifacts disagree about user history depth"
        )
    bucket_labels = np.asarray([str(s) for s in deep_bucket_of(depth)])

    # --- self-check: the four coinciding buckets must reproduce recorded means ---
    checks: list[dict] = []
    failures: list[str] = []
    for label in SELF_CHECK_LABELS:
        mask = bucket_labels == label
        for key in arm_order:
            rec_blk = (records[key]["metrics"]["per_segment"] or {}).get(label)
            if rec_blk is None:
                failures.append(f"{key}: no recorded per_segment block for segment {label!r}")
                continue
            if int(rec_blk["n_users"]) != int(mask.sum()):
                failures.append(
                    f"{key} segment {label}: recomposed n_users {int(mask.sum())} != "
                    f"recorded {rec_blk['n_users']}"
                )
            for m in metrics:
                if m not in rec_blk:
                    failures.append(f"{key} segment {label}: metric {m} not in record")
                    continue
                got = float(arms[key]["values"][m][mask].mean())
                want = float(rec_blk[m]["value"])
                diff = abs(got - want)
                checks.append(
                    {
                        "bucket": label,
                        "arm": key,
                        "metric": m,
                        "recomposed": got,
                        "recorded": want,
                        "abs_diff": diff,
                        "within_tolerance": bool(diff <= tolerance),
                    }
                )
                if diff > tolerance:
                    failures.append(
                        f"{key} segment {label} {m}: recomposed {got!r} != recorded "
                        f"{want!r} (|diff|={diff:.3e} > {tolerance:.1e})"
                    )
    if failures:
        raise RuntimeError(
            "deep-bucket self-check FAILED — nothing computed further:\n  "
            + "\n  ".join(failures)
        )

    # --- per-bucket blocks ---
    buckets: list[dict] = []
    for b_ord, label in enumerate(DEEP_BUCKET_LABELS):
        mask = bucket_labels == label
        arm_values = {
            key: {m: arms[key]["values"][m][mask] for m in metrics} for key in arm_order
        }
        block = cell_block(
            arm_values,
            metrics,
            delta_pair,
            [base_seed, b_ord],
            n_resamples,
            asl_p_values=asl_p_values,
        )
        buckets.append(
            {
                "bucket": label,
                "bucket_ordinal": b_ord,
                "user_share": (block["n_users"] / n_users) if n_users else 0.0,
                "n_train_min": int(depth[mask].min()) if mask.any() else None,
                "n_train_max": int(depth[mask].max()) if mask.any() else None,
                **block,
            }
        )

    return {
        "split": split,
        "n_users": n_users,
        "arm_order": arm_order,
        "delta_pair": list(delta_pair),
        "metrics": list(metrics),
        "seed": base_seed,
        "n_resamples": n_resamples,
        "asl_p_values": asl_p_values,
        "cache_dir": str(cache_dir),
        "cache_manifest": cache_manifest,
        "artifact_paths": artifact_paths,
        "records": records,
        "self_check": {
            "labels": list(SELF_CHECK_LABELS),
            "tolerance": tolerance,
            "compared": "per-segment MEANS only (see module docstring on CI seeding)",
            "n_comparisons": len(checks),
            "max_abs_diff": max((c["abs_diff"] for c in checks), default=0.0),
            "passed": True,
            "comparisons": checks,
        },
        "buckets": buckets,
        "wall_clock_s": round(time.monotonic() - t0, 3),
    }


def build_record(config: dict, config_path: Path, out: dict) -> dict:
    git = runlog.git_info()
    run_id, run_ts = _resolve_run_id(None)
    splits_path = config.get("splits_path") or runlog.DEFAULT_SPLITS_PATH
    return {
        "schema_version": runlog.record_schema_version,
        "kind": "deep_buckets",
        "run_id": run_id,
        "run_ts": run_ts,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "config_path": str(config_path),
        "config_hash": runlog.config_hash(config_path),
        "derived": True,
        "exploratory_derived": True,
        "provenance_note": PROVENANCE_NOTE,
        "splits": runlog.splits_block(splits_path),
        "dataset_manifest_hash": runlog.dataset_manifest_hash(runlog.DEFAULT_MANIFEST_PATH),
        "iceberg_snapshots": out["cache_manifest"]["snapshot_ids"],
        "contracts": out["cache_manifest"]["contract_identities"],
        "source_run_ids": dict(config["run_ids"]),
        "source_artifact_sha256s": {
            key: runlog.sha256_file(path) for key, path in out["artifact_paths"].items()
        },
        "protocol": {
            "eval_split": out["split"],
            "n_users": out["n_users"],
            "metrics": out["metrics"],
            "depth_source": "gold.user_stats.n_train (eval cache n_train.npy)",
            "recomposition": "regrouping of persisted per-user metric values; top50 unused",
        },
        "buckets_spec": {
            "labels": list(DEEP_BUCKET_LABELS),
            "frozen_segments": list(SEGMENT_LABELS),
            "preregistered": "EXPERIMENT_LOG.md 2026-08-17 (before any per-bucket outcome)",
        },
        "seeds": {"bootstrap": out["seed"]},
        "bootstrap": {
            "n_resamples": out["n_resamples"],
            "seed": out["seed"],
            "scheme": SEED_SCHEME,
            "resampling": "within-bucket users, with replacement",
        },
        "delta": {"minuend": out["delta_pair"][0], "subtrahend": out["delta_pair"][1]},
        "results": {"self_check": out["self_check"], "buckets": out["buckets"]},
        "wall_clock_s": out["wall_clock_s"],
        "hardware": runlog.hardware_string(),
    }


def _print_report(out: dict) -> None:
    a_key, b_key = out["delta_pair"]
    sc = out["self_check"]
    print(
        f"deep buckets · split={out['split']} users={out['n_users']}  "
        f"self-check: {sc['n_comparisons']} comparisons, max |diff| = {sc['max_abs_diff']:.3e} "
        f"(tolerance {sc['tolerance']:.1e}) PASS"
    )
    for m in out["metrics"]:
        print(f"\n--- {m} ---")
        print(
            f"  {'bucket':<9}{'users':>9}{'share':>8}{'n_train':>12}"
            f"{'  ' + b_key:>26}{'  ' + a_key:>26}{'  delta [CI]':>34}{'  ne0':>5}"
        )
        for blk in out["buckets"]:
            if blk["n_users"] == 0:
                print(f"  {blk['bucket']:<9}{0:>9}  (empty bucket)")
                continue
            b = blk["arms"][b_key][m]
            a = blk["arms"][a_key][m]
            d = blk["delta"][m]
            rng = f"{blk['n_train_min']}-{blk['n_train_max']}"
            print(
                f"  {blk['bucket']:<9}{blk['n_users']:>9}{blk['user_share']:>8.4f}{rng:>12}"
                f"  {b['value']:.6f} [{b['ci_lo']:.6f},{b['ci_hi']:.6f}]"
                f"  {a['value']:.6f} [{a['ci_lo']:.6f},{a['ci_hi']:.6f}]"
                f"  {d['delta']:+.6f} [{d['ci_lo']:+.6f},{d['ci_hi']:+.6f}]"
                f"  {'yes' if d['excludes_zero'] else 'no':>4}"
            )
        print("  CI widths: " + "  ".join(
            f"{blk['bucket']}="
            + (f"{blk['delta'][m]['ci_width']:.6f}" if blk["n_users"] else "n/a")
            for blk in out["buckets"]
        ))
    print(f"\nwall_clock_s={out['wall_clock_s']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.eval.deep_buckets")
    parser.add_argument("--config", default="configs/deep_buckets_test.yaml")
    parser.add_argument("--results", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute + print everything, append nothing to the results log",
    )
    parser.add_argument("--json-out", default=None, help="also dump the full record JSON here")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    results_path = Path(args.results or config.get("results_path", "results/runs.jsonl"))
    dry_run = args.dry_run or bool(config.get("dry_run", False))

    try:
        out = build_buckets(config, results_path)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report + non-zero exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_report(out)

    record = build_record(config, config_path, out)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(record, indent=2))
        print(f"record JSON written to {args.json_out}")

    if dry_run:
        print("\n--dry-run: no record appended")
        return 0

    runlog.check_test_dirty(out["split"], record["git_dirty"])
    runlog.append_record(record, results_path)
    print(f"\nappended kind=deep_buckets run_id={record['run_id']} -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
