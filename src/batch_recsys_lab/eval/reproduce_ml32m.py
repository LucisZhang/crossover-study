"""``make reproduce-ml32m`` — snapshot-pinned re-run of the two ML-32M TEST
records that feed the T9-3c confirmatory ladder, plus a re-derivation check of
the confirmatory verdict itself (Phase 9, T9-3c; mirrors eval/reproduce.py's
`reproduce-headline` pattern exactly).

    uv run python -m batch_recsys_lab.eval.reproduce_ml32m \\
        [--m-star-headline configs/headline_ml32m_m_star.yaml] \\
        [--p-star-headline configs/headline_ml32m_p_star.yaml] \\
        [--confirmatory-config configs/confirmatory_ml32m_test.yaml] \\
        [--committed-confirmatory results/confirmatory_ml32m_test.json]

What it does
------------
1. Re-runs M* (item-kNN-t12m) and P* (pop-t12m) TEST evals via
   :func:`batch_recsys_lab.eval.reproduce.reproduce`, EXACTLY the
   `reproduce-headline` machinery -- pinned Iceberg time travel at each
   record's own `iceberg_snapshots`, the ORIGINAL config file (hash-verified),
   re-run into a scratch cache (`cache_repro_root`, never the live one), and
   field-wise diffed against the recorded record on
   :data:`eval.reproduce.FIELDS_COMPARED`. Both calls pass ``append=False``:
   NEITHER ever appends to results/runs.jsonl -- these are re-derivations of
   already-recorded records, not new experiments.
2. Re-runs `confirmatory_ml32m.run_confirmatory` on the SAME committed
   `configs/confirmatory_ml32m_test.yaml` -- which itself only reads the
   (unchanged, since step 1 never wrote) `results/runs.jsonl` records named in
   that config, so this is deterministic post-processing, not a second
   ingestion of the reproduced arrays -- and diffs the recomputed `verdict`
   block against the committed `results/confirmatory_ml32m_test.json`,
   excluding the volatile top-level fields `generated_ts`/`git_sha`/
   `git_dirty`/`wall_clock_s`/`hardware` (mirrors eval/reproduce.py's
   `FIELDS_EXCLUDED` convention; none of those fields live inside the
   `verdict` block itself, so the exclusion list is documentation, not an
   active filter, but is asserted here for parity with the Amazon pattern).
3. Exits 0 iff both eval reproductions are `byte_exact` AND the confirmatory
   verdict block is identical.

This module writes no file (besides the scratch reproduce cache under
`cache_repro_root`) and never touches `results/runs.jsonl`,
`data/MANIFEST.md`/`data/MANIFEST_ML32M.md`, or `EXPERIMENT_LOG.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from batch_recsys_lab.eval import reproduce as reproduce_mod
from batch_recsys_lab.eval.confirmatory_ml32m import run_confirmatory

REPO_ROOT = reproduce_mod.REPO_ROOT
DEFAULT_M_STAR = REPO_ROOT / "configs" / "headline_ml32m_m_star.yaml"
DEFAULT_P_STAR = REPO_ROOT / "configs" / "headline_ml32m_p_star.yaml"
DEFAULT_CONFIRMATORY_CONFIG = REPO_ROOT / "configs" / "confirmatory_ml32m_test.yaml"
DEFAULT_COMMITTED_CONFIRMATORY = REPO_ROOT / "results" / "confirmatory_ml32m_test.json"

#: Mirrors eval/reproduce.py's FIELDS_EXCLUDED convention: run-of-invocation
#: identity/timing fields, not content. None of these live inside the
#: "verdict" block confirmatory_ml32m.py emits (see its module docstring for
#: the full record shape) -- listed for parity/documentation, and to make an
#: accidental future addition of a volatile field under "verdict" a one-line
#: fix here rather than a silent false mismatch.
CONFIRMATORY_VOLATILE_TOP_LEVEL = ("generated_ts", "git_sha", "git_dirty", "wall_clock_s", "hardware")


def compare_verdict(committed: dict, candidate: dict) -> list[dict]:
    """Field-level diff of the "verdict" block only (§ task scope)."""
    out: list[dict] = []
    reproduce_mod._walk(
        "verdict",
        committed.get("verdict", reproduce_mod.MISSING),
        candidate.get("verdict", reproduce_mod.MISSING),
        out,
    )
    return out


def reproduce_ml32m(
    m_star_headline: str | Path = DEFAULT_M_STAR,
    p_star_headline: str | Path = DEFAULT_P_STAR,
    confirmatory_config: str | Path = DEFAULT_CONFIRMATORY_CONFIG,
    committed_confirmatory: str | Path = DEFAULT_COMMITTED_CONFIRMATORY,
    root: str | Path = REPO_ROOT,
    master: str = "local[10]",
    driver_memory: str = "8g",
) -> dict:
    """Full reproduce-ml32m flow. Returns a summary dict (never appended anywhere)."""
    root = Path(root)

    print("[reproduce-ml32m] reproducing M* (item-kNN-t12m) ...")
    m_star_rec = reproduce_mod.reproduce(
        headline_path=m_star_headline,
        root=root,
        master=master,
        driver_memory=driver_memory,
        append=False,
    )
    print("[reproduce-ml32m] reproducing P* (pop-t12m) ...")
    p_star_rec = reproduce_mod.reproduce(
        headline_path=p_star_headline,
        root=root,
        master=master,
        driver_memory=driver_memory,
        append=False,
    )
    eval_byte_exact = (
        m_star_rec["verdict"] == "byte_exact" and p_star_rec["verdict"] == "byte_exact"
    )

    print("[reproduce-ml32m] re-deriving the confirmatory verdict (results/runs.jsonl unchanged) ...")
    config_path = Path(confirmatory_config)
    config = yaml.safe_load(config_path.read_text())
    results_path = root / config.get("results_path", "results/runs.jsonl")
    candidate = run_confirmatory(config, config_path, results_path)

    committed_path = Path(committed_confirmatory)
    if not committed_path.exists():
        raise FileNotFoundError(
            f"{committed_path} not found; run `make confirmatory-ml32m` once to commit it "
            "before reproduce-ml32m can compare against it"
        )
    committed = json.loads(committed_path.read_text())
    verdict_diff = compare_verdict(committed, candidate)
    confirmatory_verdict_identical = not verdict_diff

    overall_byte_exact = eval_byte_exact and confirmatory_verdict_identical
    summary = {
        "m_star": {
            "run_id": m_star_rec["reproduces_run_id"],
            "verdict": m_star_rec["verdict"],
            "diff": m_star_rec["diff"],
        },
        "p_star": {
            "run_id": p_star_rec["reproduces_run_id"],
            "verdict": p_star_rec["verdict"],
            "diff": p_star_rec["diff"],
        },
        "eval_byte_exact": eval_byte_exact,
        "confirmatory_verdict_identical": confirmatory_verdict_identical,
        "confirmatory_verdict_diff": verdict_diff,
        "confirmatory_volatile_fields_excluded": list(CONFIRMATORY_VOLATILE_TOP_LEVEL),
        "verdict": "byte_exact" if overall_byte_exact else "mismatch",
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.eval.reproduce_ml32m")
    parser.add_argument("--m-star-headline", default=str(DEFAULT_M_STAR))
    parser.add_argument("--p-star-headline", default=str(DEFAULT_P_STAR))
    parser.add_argument("--confirmatory-config", default=str(DEFAULT_CONFIRMATORY_CONFIG))
    parser.add_argument("--committed-confirmatory", default=str(DEFAULT_COMMITTED_CONFIRMATORY))
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    args = parser.parse_args(argv)

    try:
        summary = reproduce_ml32m(
            m_star_headline=args.m_star_headline,
            p_star_headline=args.p_star_headline,
            confirmatory_config=args.confirmatory_config,
            committed_confirmatory=args.committed_confirmatory,
            master=args.master,
            driver_memory=args.driver_memory,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report + non-zero exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0 if summary["verdict"] == "byte_exact" else 1


if __name__ == "__main__":
    sys.exit(main())
