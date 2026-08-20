"""CLI: run one eval config -> append a ``kind="eval"`` record (Phase 2, T5).

    uv run python -m batch_recsys_lab.eval.run_eval \
        --config configs/eval_pop_t12m_test.yaml \
        [--results results/runs.jsonl] [--allow-stale]

Exits non-zero on any guard failure (stale cache, TEST-split dirty tree) or
config/scoring error, so ``make eval`` fails loudly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.harness import run_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.eval.run_eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", default="results/runs.jsonl")
    parser.add_argument("--allow-stale", action="store_true")
    parser.add_argument(
        "--splits-path",
        default=None,
        help=(
            "Path to the frozen splits YAML. Precedence: this flag, then the "
            "config's own 'splits_path' key (if present), then the default "
            f"({runlog.DEFAULT_SPLITS_PATH})."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help=(
            "Path to the dataset manifest. Precedence: this flag, then the "
            "config's own 'manifest_path' key (if present), then the default "
            f"({runlog.DEFAULT_MANIFEST_PATH})."
        ),
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    # Precedence: explicit CLI flag > config-carried path > lab default. A
    # config may declare its own splits_path/manifest_path (mirrors the
    # dataset_manifest_path pattern in churn_contrast.build_churn_contrast)
    # so an ML-32M config can't silently fall through to the Amazon defaults.
    splits_path = args.splits_path or config.get("splits_path") or str(
        runlog.DEFAULT_SPLITS_PATH
    )
    manifest_path = args.manifest_path or config.get("manifest_path") or str(
        runlog.DEFAULT_MANIFEST_PATH
    )

    try:
        run_eval(
            config,
            config_path=config_path,
            results_path=args.results,
            allow_stale=args.allow_stale,
            splits_path=splits_path,
            manifest_path=manifest_path,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report + non-zero exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
