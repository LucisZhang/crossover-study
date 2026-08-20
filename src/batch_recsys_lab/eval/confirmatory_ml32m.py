"""CLI: the T9-3c CONFIRMATORY analysis of the ML-32M TEST ladder (Phase 9, §8c).

    uv run python -m batch_recsys_lab.eval.confirmatory_ml32m \
        --config configs/confirmatory_ml32m_test.yaml

This module decides nothing on its own. Every rule it applies is quoted from the
committed T9-3b preregistration in ``EXPERIMENT_LOG.md`` (2026-08-20, §5
"Multiplicity policy" and §7 "Decision rules"), plus the 2026-08-21 "VAL ladder
complete" entry that froze M\\* / P\\* and fixed the ASL floor's reporting format.
Its whole job is to apply those rules **mechanically** to per-user artifacts that
already exist, so that the verdict is a function of the record rather than of a
judgment call made after seeing the numbers.

What it computes
----------------
* **Family P (primary confirmatory, §5b).** M\\* vs P\\*, paired NDCG@10 delta on
  every *populated* deep bucket (:data:`eval.protocol.DEEP_BUCKET_LABELS`) plus
  one global test — m <= 8 — Benjamini-Hochberg at FDR 0.05 *within the family*.
* **Family S1 (secondary, §5c).** The same bucket family for each other arm vs
  P\\*, BH within that arm.
* **Family S2 (secondary, §5d).** Regime-map cells (``CELL_AXES``) per arm vs
  P\\*, recomposed through the committed T8-1 machinery, BH across all cells of
  that arm.
* **The 5-segment comparability exhibit (§5a).** Paired deltas on the frozen
  Phase 4 segments, **no BH**, labeled comparability-only — it exists so the
  ML-32M numbers can sit beside the Amazon ones, not to support a claim.
* **The §7 classifier.** D1/D2/D3/D5 verdict token + the D4 flag, derived from
  Family P's BH outcome and point estimates alone.

Discipline this module inherits rather than reinvents
-----------------------------------------------------
* Bucket deltas, their CIs and their p-values come from
  :func:`eval.deep_buckets.build_buckets` -> :func:`eval.cell_stats.cell_block`
  — the same 1,000 resamples, base seed 20260805, per-bucket child seeds
  ``default_rng([20260805, bucket_ordinal])``, and the same self-check that the
  four segment-coinciding buckets reproduce each source record's recorded means.
  Cell deltas come from :func:`eval.regime_map.build_regime_map`, including its
  identity anchor. A lineage mismatch aborts inside those functions, before any
  output exists.
* p-values are §5(e)'s two-sided ASL, computed by
  :func:`eval.bootstrap.asl_p_value` from the SAME resample matrix as the CI
  beside it (``cell_block(asl_p_values=True)``). No model is re-evaluated; this
  is post-processing of draws, not a second look at TEST.
* Recall@20 is carried through every family as the §5(g) metric-robustness
  **label** and is corrected in its own separate BH family. It is never
  confirmatory: the confirmatory criterion is BH-corrected NDCG@10.
* Uncorrected p-values are published beside the corrected ones (§5h).

Append-only
-----------
The driver writes exactly one JSON file (default
``results/confirmatory_ml32m_test.json``) and **never** touches
``results/runs.jsonl``, ``EXPERIMENT_LOG.md`` or any manifest. It reads the
results log; it does not open it for writing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from batch_recsys_lab.eval import deep_buckets, regime_map, runlog
from batch_recsys_lab.eval.bootstrap import asl_p_value, paired_delta_ci, paired_delta_resamples
from batch_recsys_lab.eval.cell_stats import cell_block
from batch_recsys_lab.eval.deep_buckets import _load_arm
from batch_recsys_lab.eval.protocol import DEEP_BUCKET_LABELS, SEGMENT_LABELS
from batch_recsys_lab.eval.regime_map import _find_eval_record

# --- preregistered constants. Not configurable: they ARE the preregistration ---

#: §5(b)/(c)/(d): "Benjamini-Hochberg at FDR 0.05".
FDR_ALPHA = 0.05
#: §2 / §6: every VAL and TEST run draws 1,000 resamples off base seed 20260805,
#: and §6 makes seed 20260805 "the sole per-user artifact used for all paired
#: TEST deltas and for every segment-level and cell-level inference".
PREREG_N_RESAMPLES = 1000
PREREG_BASE_SEED = 20260805
#: §5(b): the confirmatory metric. §5(g): the reported-not-confirmatory one.
CONFIRMATORY_METRIC = "ndcg@10"
ROBUSTNESS_METRIC = "recall@20"
#: §1 / §7 D1: n* is drawn from the deep-bucket lower edges.
BUCKET_LOWER_EDGE = {
    "0": 0,
    "1-4": 1,
    "5-9": 5,
    "10-19": 10,
    "20-49": 20,
    "50-99": 50,
    "100+": 100,
}
GLOBAL_LABEL = "global"
P_STAR_KEY = "p_star"

#: §6 seed discipline: "Seed 20260805 is the sole per-user artifact used for all
#: paired TEST deltas and for every segment-level and cell-level inference. Seeds
#: 20260806 and 20260807 are stability evidence only ... they never enter a
#: paired CI, a p-value, or a BH family."
PRIMARY_MODEL_SEED = 20260805
STABILITY_MODEL_SEEDS = (20260806, 20260807)

VERDICT_TOKENS = {
    "D1": "D1_CROSSOVER",
    "D2": "D2_GLOBAL_ONLY_WIN",
    "D3": "D3_DOUBLE_NULL",
    "D5": "D5_MIXED",
}
D4_FLAG_TOKEN = "D4_SIGNIFICANT_NEGATIVES"

PREREG_REFS = {
    "preregistration": (
        "EXPERIMENT_LOG.md - 'Phase 9 T9-3b preregistration' (2026-08-20): "
        "§5 multiplicity policy, §7 decision rules D1-D5"
    ),
    "selections": (
        "EXPERIMENT_LOG.md - 'Phase 9 T9-3b VAL ladder complete' (2026-08-21): "
        "M* = item-kNN-t12m (n50/365d), P* = pop-t12m, ASL floor reported as 2/1001"
    ),
    "confirmatory_metric": CONFIRMATORY_METRIC,
    "robustness_metric": ROBUSTNESS_METRIC,
    "fdr_alpha": FDR_ALPHA,
    "bh_rule": (
        "p sorted ascending with (p, label) as the deterministic tie-break; "
        "q_(i) = min_{j>=i} (m * p_(j) / j), clipped at 1.0 (monotone from the "
        "tail); a test is BH-significant iff q <= 0.05"
    ),
    "asl_rule": (
        "§5(e): p = min(1, 2*min((1+#{D<=0})/(B+1), (1+#{D>=0})/(B+1))), B=1000, "
        "computed from the same resample matrix as the CI beside it"
    ),
    "win_rule": "§5(f): a win requires BH-significant AND delta > 0; a loss, BH-significant AND delta < 0",
}


# --- Benjamini-Hochberg -------------------------------------------------------


def benjamini_hochberg(
    labeled_p: list[tuple[str, float]], alpha: float = FDR_ALPHA
) -> dict[str, dict]:
    """BH step-up adjusted q-values at FDR ``alpha`` over ONE family.

    ``labeled_p`` is ``[(label, p_value), ...]``; the family is exactly what is
    passed in (§5: "within Family P", "within that arm's bucket family",
    "across all cells per arm") — this function never decides membership.

    Sorting is by ``(p, label)`` so that tied p-values get a deterministic,
    input-order-independent rank. Adjusted values are computed from the tail::

        q_(i) = min( 1.0, min_{j >= i} ( m * p_(j) / j ) )

    which is what makes BH a *step-up* procedure: a test can be rejected on the
    strength of a deeper-ranked test even when its own ``m*p/i`` exceeds alpha.
    Significance is then simply ``q <= alpha`` (equivalent to the classical
    largest-i-with-p_(i) <= i*alpha/m formulation, and it also publishes the q).

    Returns ``label -> {p_value, rank, m, q_value, significant}``.
    """
    items = sorted(labeled_p, key=lambda t: (float(t[1]), str(t[0])))
    m = len(items)
    if m == 0:
        return {}
    labels = [str(lbl) for lbl, _ in items]
    if len(set(labels)) != m:
        raise ValueError(f"benjamini_hochberg needs unique labels, got {labels}")

    q_sorted = [0.0] * m
    running = math.inf
    for i in range(m - 1, -1, -1):
        raw = m * float(items[i][1]) / (i + 1)
        running = min(running, raw)
        q_sorted[i] = min(1.0, running)

    return {
        labels[i]: {
            "p_value": float(items[i][1]),
            "rank": i + 1,
            "m": m,
            "q_value": float(q_sorted[i]),
            "significant": bool(q_sorted[i] <= alpha),
        }
        for i in range(m)
    }


def asl_floor(n_resamples: int = PREREG_N_RESAMPLES) -> float:
    """Smallest attainable two-sided ASL: ``2 * 1/(B+1)`` (= 2/1001 at B=1000)."""
    return min(1.0, 2.0 * (1 / (n_resamples + 1)))


def p_display(p: float, n_resamples: int = PREREG_N_RESAMPLES) -> str:
    """Reporting format. The 2026-08-21 selections entry's clarification:
    §5(e)'s "report as < 0.001" describes the ONE-sided resolution (1/1001);
    the two-sided floor is 2/1001, and a floor-attaining p is reported as
    ``"2/1001 (floor)"`` rather than as a spuriously exact decimal."""
    floor = asl_floor(n_resamples)
    if math.isclose(float(p), floor, rel_tol=1e-12, abs_tol=0.0):
        return f"2/{n_resamples + 1} (floor)"
    return f"{float(p):.6f}"


# --- test rows ----------------------------------------------------------------


def _row(
    *,
    unit: str,
    label: str,
    n_users: int,
    delta_block: dict,
    ordinal: int | None = None,
    extra: dict | None = None,
) -> dict:
    """One test: point estimate, CI, uncorrected p (§5h), and its provenance."""
    p = float(delta_block["p_value"])
    delta = float(delta_block["delta"])
    row = {
        "unit": unit,
        "label": label,
        "ordinal": ordinal,
        "n_users": int(n_users),
        "delta": delta,
        "ci_lo": float(delta_block["ci_lo"]),
        "ci_hi": float(delta_block["ci_hi"]),
        "ci_width": float(delta_block["ci_width"]),
        "ci_excludes_zero": bool(delta_block["excludes_zero"]),
        "direction": "positive" if delta > 0 else ("negative" if delta < 0 else "zero"),
        # §5(h): uncorrected numbers are published, clearly marked as uncorrected.
        "p_value_uncorrected": p,
        "p_display_uncorrected": p_display(p),
        "significant_uncorrected": bool(p <= FDR_ALPHA),
        "at_asl_floor": p_display(p).endswith("(floor)"),
    }
    if int(n_users) == 1:
        # Not a rule, a fact: with one user every paired resample is the same
        # number, so the ASL degenerates to the floor whatever the effect size.
        # Disclosed on the row rather than silently corrected or thresholded.
        row["degenerate_single_user"] = True
    if extra:
        row.update(extra)
    return row


def _apply_bh(rows: list[dict], alpha: float = FDR_ALPHA) -> dict:
    """BH within this family; annotates rows in place and returns a summary."""
    adjusted = benjamini_hochberg(
        [(r["label"], r["p_value_uncorrected"]) for r in rows], alpha=alpha
    )
    for r in rows:
        a = adjusted[r["label"]]
        r["bh_rank"] = a["rank"]
        r["bh_m"] = a["m"]
        r["q_value"] = a["q_value"]
        r["bh_significant"] = a["significant"]
        # §5(f): direction is read off the point estimate, symmetric by construction.
        r["bh_win"] = bool(a["significant"] and r["delta"] > 0)
        r["bh_loss"] = bool(a["significant"] and r["delta"] < 0)
    return {
        "alpha": alpha,
        "m_tests": len(rows),
        "n_bh_significant": sum(1 for r in rows if r["bh_significant"]),
        "n_bh_wins": sum(1 for r in rows if r["bh_win"]),
        "n_bh_losses": sum(1 for r in rows if r["bh_loss"]),
        "n_uncorrected_significant": sum(1 for r in rows if r["significant_uncorrected"]),
    }


# --- §7 classifier ------------------------------------------------------------


def classify_verdict(bucket_rows: list[dict], global_row: dict | None) -> dict:
    """Apply §7 D1-D5 to Family P's BH-corrected NDCG@10 rows. Mechanical.

    ``bucket_rows`` are the populated deep-bucket tests (any order; re-sorted
    here by depth). ``global_row`` is the all-users test.

    * **D1** requires (i) >= 1 BH-significant POSITIVE *depth bucket* and
      (ii) coherence: a shallowest bucket ``b`` such that every populated bucket
      at or above ``b`` has point estimate ``delta > 0`` and no BH-significant
      negative sits at or above ``b``. ``n* = lower edge of b``.
    * **D2** the global test is a BH-significant win but no depth bucket is.
    * **D3** no BH-significant positive test anywhere in Family P.
    * **D4** is a FLAG, not a branch: any BH-significant negative, reported with
      the same emphasis as a win.
    * **D5** (i) holds, (ii) fails.

    Reading forced by the text: (i) counts *depth buckets only*, because D2 is
    defined as exactly the case where the global test is significant and no
    individual depth bucket is — counting the global test in (i) would make D2
    unreachable. Unpopulated buckets contribute no test (§5b) and are therefore
    skipped by the "every bucket at or above b" quantifier.
    """
    order = {lbl: i for i, lbl in enumerate(DEEP_BUCKET_LABELS)}
    rows = sorted(bucket_rows, key=lambda r: order[r["label"]])

    positives = [r["label"] for r in rows if r["bh_win"]]
    negatives = [r["label"] for r in rows if r["bh_loss"]]
    global_win = bool(global_row and global_row["bh_win"])
    global_loss = bool(global_row and global_row["bh_loss"])
    d4_flag = bool(negatives or global_loss)

    # (ii) coherence: scan candidate b from shallowest to deepest.
    coherence = None
    for i, cand in enumerate(rows):
        above = rows[i:]
        if all(r["delta"] > 0 for r in above) and not any(r["bh_loss"] for r in above):
            coherence = {
                "bucket": cand["label"],
                "n_star": BUCKET_LOWER_EDGE[cand["label"]],
                "buckets_at_or_above": [r["label"] for r in above],
                # PREREG AMBIGUITY, disclosed not resolved: §7 D1 states (i) and
                # (ii) as INDEPENDENT conditions, so a BH-significant positive
                # bucket *below* the coherent region literally satisfies (i)
                # while the claimed n* rests on buckets that are individually
                # non-significant. The literal text is applied (the verdict is
                # D1); this field says whether the claim's own region carries
                # the significance, so a reader is never misled by the token.
                "contains_significant_positive": any(r["bh_win"] for r in above),
                "significant_positive_buckets_at_or_above": [
                    r["label"] for r in above if r["bh_win"]
                ],
            }
            break

    if positives:
        if coherence is not None:
            verdict, code = VERDICT_TOKENS["D1"], "D1"
            headline = (
                "CROSSOVER at n* = "
                f"{coherence['n_star']} — personalization wins from a definable "
                "history depth onward and keeps winning"
            )
        else:
            verdict, code = VERDICT_TOKENS["D5"], "D5"
            headline = (
                "significant per-depth wins exist but do not form a crossover under "
                "the preregistered definition — no n* is claimed"
            )
    elif global_win:
        verdict, code = VERDICT_TOKENS["D2"], "D2"
        headline = (
            "personalization wins on average on the low-churn catalog, but no "
            "history-depth threshold is identified"
        )
    else:
        verdict, code = VERDICT_TOKENS["D3"], "D3"
        headline = "popularity dominates even where the catalog holds still"

    caveats: list[str] = []
    if code == "D1" and not coherence["contains_significant_positive"]:
        caveats.append(
            "D1 by the literal §7 text: conditions (i) and (ii) are stated "
            "independently, and the BH-significant positive bucket(s) "
            f"{positives} lie BELOW the coherent region "
            f"{coherence['buckets_at_or_above']}, which contains no individually "
            "BH-significant win. The n* claimed here rests on point estimates "
            "plus a significance result from a shallower depth. Report this "
            "explicitly; do not present it as a clean crossover."
        )

    return {
        "verdict": verdict,
        "verdict_code": code,
        "headline": headline,
        "caveats": caveats,
        "n_star": coherence["n_star"] if code == "D1" else None,
        "crossover_bucket": coherence["bucket"] if code == "D1" else None,
        "condition_i_significant_positive_buckets": positives,
        "condition_i_met": bool(positives),
        "condition_ii_coherence": coherence,
        "condition_ii_met": coherence is not None,
        "global_bh_significant_positive": global_win,
        "global_bh_significant_negative": global_loss,
        "d4_flag": d4_flag,
        "d4_token": D4_FLAG_TOKEN if d4_flag else None,
        "bh_significant_negative_buckets": negatives,
        "buckets_considered": [r["label"] for r in rows],
        "rule_source": PREREG_REFS["preregistration"],
        "notes": [
            "D1 condition (i) counts depth buckets only; the global test is D2's "
            "business (see classify_verdict docstring).",
            "§7 states D1(i) and D1(ii) independently; when the coherent region "
            "carries no BH-significant win of its own, the verdict is still D1 by "
            "the committed text and the fact is disclosed under caveats and under "
            "condition_ii_coherence.contains_significant_positive.",
            "Unpopulated buckets yield no test (§5b) and are skipped by the "
            "coherence quantifier; they are listed under excluded_buckets.",
        ],
    }


# --- family builders ----------------------------------------------------------


def _bucket_rows(bucket_blocks: list[dict], metric: str, delta_label: str) -> tuple[list, list]:
    """(rows for populated buckets, disclosure records for empty ones) — §5(b):
    "A bucket with zero TEST users yields no test and is excluded with its count
    disclosed; that exclusion is driven by user counts, not by outcomes"."""
    rows, excluded = [], []
    for blk in bucket_blocks:
        if blk["n_users"] == 0:
            excluded.append(
                {"label": blk["bucket"], "n_users": 0, "reason": "zero TEST users in bucket"}
            )
            continue
        rows.append(
            _row(
                unit="deep_bucket",
                label=blk["bucket"],
                n_users=blk["n_users"],
                delta_block=blk["delta"][metric],
                ordinal=blk["bucket_ordinal"],
                extra={
                    "user_share": blk["user_share"],
                    "n_train_min": blk["n_train_min"],
                    "n_train_max": blk["n_train_max"],
                    "seed_entropy": blk["seed_entropy"],
                    "delta_label": delta_label,
                },
            )
        )
    return rows, excluded


def _cell_rows(cells: dict, metric: str, delta_label: str) -> tuple[list, list]:
    """Regime-map cells -> rows, one BH family per arm across BOTH CELL_AXES."""
    rows, excluded = [], []
    for axis in regime_map.CELL_AXES:
        for cell in cells[axis]:
            label = f"{axis}|{cell['segment']}|{cell['bucket']}"
            if cell["n_users"] == 0:
                excluded.append(
                    {"label": label, "n_users": 0, "reason": "zero users with GT in this cell"}
                )
                continue
            rows.append(
                _row(
                    unit="regime_cell",
                    label=label,
                    n_users=cell["n_users"],
                    delta_block=cell["delta"][metric],
                    extra={
                        "axis": axis,
                        "segment": cell["segment"],
                        "bucket": cell["bucket"],
                        "gt_interactions": cell["gt_interactions"],
                        "user_share": cell["user_share"],
                        "gt_share": cell["gt_share"],
                        "seed_entropy": cell["seed_entropy"],
                        "delta_label": delta_label,
                    },
                )
            )
    return rows, excluded


def _metric_robustness(primary_rows: list[dict], secondary_rows: list[dict]) -> dict:
    """§5(g): a claim is labeled "metric-robust" only when Recall@20 agrees in
    SIGN and SIGNIFICANCE. Computed per test, then aggregated over exactly the
    tests that carry the claim (the BH-significant NDCG@10 wins). Recall@20 has
    its own BH family — mixing metrics into one family would silently change the
    confirmatory correction."""
    by_label = {r["label"]: r for r in secondary_rows}
    per_test = {}
    for r in primary_rows:
        other = by_label.get(r["label"])
        per_test[r["label"]] = {
            "primary_delta": r["delta"],
            "primary_bh_significant": r["bh_significant"],
            "secondary_delta": None if other is None else other["delta"],
            "secondary_bh_significant": None if other is None else other["bh_significant"],
            "sign_agrees": None if other is None else (r["direction"] == other["direction"]),
            "significance_agrees": (
                None if other is None else (r["bh_significant"] == other["bh_significant"])
            ),
        }
    claim_labels = [r["label"] for r in primary_rows if r["bh_win"]]
    agreeing = [
        lbl
        for lbl in claim_labels
        if per_test[lbl]["sign_agrees"] and per_test[lbl]["significance_agrees"]
    ]
    return {
        "definition": (
            f"§5(g) label only, never confirmatory: {ROBUSTNESS_METRIC} must agree in "
            f"sign and BH-significance (its own BH family) with {CONFIRMATORY_METRIC}"
        ),
        "claim_labels": claim_labels,
        "agreeing_labels": agreeing,
        "metric_robust": bool(claim_labels) and len(agreeing) == len(claim_labels),
        "per_test": per_test,
    }


def _global_and_segment_blocks(
    arm_key: str,
    arm_path: str,
    p_path: str,
    metrics: tuple[str, ...],
    n_resamples: int,
    base_seed: int,
) -> dict:
    """The all-users test + the 5-segment comparability exhibit.

    ``build_buckets`` covers the seven depth buckets; the global test (§5b's
    "+ one global test") and the frozen-segment exhibit (§5a) are computed here
    from the same two artifacts.

    **Seed entropy, disclosed:** the preregistration fixes child seeds for
    per-segment / per-cell inference but never names the GLOBAL cell's entropy.
    This uses the scalar base seed ``[20260805]`` — identical stream to
    ``default_rng(20260805)`` — which is exactly ``eval/compare.py``'s committed
    convention for a global paired delta, so this global block is reproducible
    by an independent ``kind="paired_delta"`` run. The 5 frozen segments use the
    same ``compare.py`` convention (scalar base seed per segment mask) because
    the exhibit's only purpose is comparability with the Amazon-side
    ``paired_delta`` records, which were produced that way.
    """
    a = _load_arm(arm_path, metrics)
    b = _load_arm(p_path, metrics)
    if a["user_ids"] != b["user_ids"]:
        raise RuntimeError(
            f"artifact user set/order mismatch between {arm_key} and {P_STAR_KEY}: "
            "paired resampling is positional"
        )
    glob = cell_block(
        {arm_key: a["values"], P_STAR_KEY: b["values"]},
        metrics,
        (arm_key, P_STAR_KEY),
        [base_seed],
        n_resamples,
        asl_p_values=True,
    )
    glob["bucket"] = GLOBAL_LABEL

    segments = a["segments"]
    seg_blocks = []
    for label in SEGMENT_LABELS:
        mask = segments == label
        n = int(mask.sum())
        block = {"segment": label, "n_users": n, "delta": {}}
        for m in metrics:
            if n == 0:
                block["delta"][m] = None
                continue
            d = paired_delta_ci(
                a["values"][m][mask], b["values"][m][mask], n_resamples=n_resamples, seed=base_seed
            )
            d["ci_width"] = float(d["ci_hi"] - d["ci_lo"])
            d["p_value"] = asl_p_value(
                paired_delta_resamples(
                    a["values"][m][mask],
                    b["values"][m][mask],
                    n_resamples=n_resamples,
                    seed=base_seed,
                )
            )
            block["delta"][m] = d
        seg_blocks.append(block)
    return {"global": glob, "segments": seg_blocks, "n_users": len(a["user_ids"])}


def _deep_bucket_config(arm_key: str, arm_run_id: str, p_run_id: str, config: dict) -> dict:
    """Config for :func:`eval.deep_buckets.build_buckets`, ML-32M-namespaced."""
    cfg = {
        "run_ids": {arm_key: arm_run_id, P_STAR_KEY: p_run_id},
        "delta": {"minuend": arm_key, "subtrahend": P_STAR_KEY},
        "split": config.get("split", "test"),
        "metrics": list(config["metrics"]),
        "bootstrap": {
            "n_resamples": config["bootstrap"]["n_resamples"],
            "seed": config["bootstrap"]["seed"],
        },
        "cache_dir": config["cache_dir"],
        "five_core_table": config["five_core_table"],
        "asl_p_values": True,
    }
    if config.get("self_check"):
        cfg["self_check"] = config["self_check"]
    if config.get("expected_n_users") is not None:
        cfg["expected_n_users"] = config["expected_n_users"]
    return cfg


def _regime_map_config(arm_key: str, arm_run_id: str, p_run_id: str, config: dict) -> dict:
    """Config for :func:`eval.regime_map.build_regime_map`, ML-32M-namespaced.

    Every knob the ML-32M lane needs (``cache_dir``, ``five_core_table``,
    ``item_stats_dir``, ``splits_path``) already existed on the committed T8-1
    module and defaults to the Amazon values; only the ASL opt-in is new.
    """
    cfg = {
        "run_ids": {arm_key: arm_run_id, P_STAR_KEY: p_run_id},
        "delta": {"minuend": arm_key, "subtrahend": P_STAR_KEY},
        "split": config.get("split", "test"),
        "cell_metrics": list(config["metrics"]),
        "k_list": list(config.get("k_list", (10, 20, 50))),
        "bootstrap": {
            "n_resamples": config["bootstrap"]["n_resamples"],
            "seed": config["bootstrap"]["seed"],
        },
        "cache_dir": config["cache_dir"],
        "five_core_table": config["five_core_table"],
        "item_stats_dir": config["item_stats_dir"],
        "splits_path": config["splits_path"],
        "asl_p_values": True,
    }
    if config.get("identity_check"):
        cfg["identity_check"] = config["identity_check"]
    if config.get("expected_n_users") is not None:
        cfg["expected_n_users"] = config["expected_n_users"]
    return cfg


def _bucket_family(
    arm_key: str,
    arm_run_id: str,
    p_run_id: str,
    config: dict,
    results_path: Path,
    include_global: bool,
) -> dict:
    """One arm's deep-bucket family (Family P when ``arm_key`` is M*, else S1)."""
    metrics = tuple(config["metrics"])
    n_resamples = int(config["bootstrap"]["n_resamples"])
    base_seed = int(config["bootstrap"]["seed"])

    out = deep_buckets.build_buckets(
        _deep_bucket_config(arm_key, arm_run_id, p_run_id, config), results_path
    )
    extra = _global_and_segment_blocks(
        arm_key,
        out["artifact_paths"][arm_key],
        out["artifact_paths"][P_STAR_KEY],
        metrics,
        n_resamples,
        base_seed,
    )
    delta_label = f"{arm_key}_minus_{P_STAR_KEY}"

    families: dict[str, dict] = {}
    excluded_disclosed: list[dict] = []
    for metric in metrics:
        rows, excluded = _bucket_rows(out["buckets"], metric, delta_label)
        if include_global:
            rows.append(
                _row(
                    unit="global",
                    label=GLOBAL_LABEL,
                    n_users=extra["global"]["n_users"],
                    delta_block=extra["global"]["delta"][metric],
                    extra={
                        "seed_entropy": extra["global"]["seed_entropy"],
                        "delta_label": delta_label,
                    },
                )
            )
        summary = _apply_bh(rows)
        families[metric] = {
            "rows": rows,
            "bh": summary,
            "excluded_buckets": excluded,
            "confirmatory": metric == CONFIRMATORY_METRIC,
        }
        excluded_disclosed = excluded

    robustness = (
        _metric_robustness(
            families[CONFIRMATORY_METRIC]["rows"], families[ROBUSTNESS_METRIC]["rows"]
        )
        if ROBUSTNESS_METRIC in families and CONFIRMATORY_METRIC in families
        else None
    )
    return {
        "arm": arm_key,
        "run_id": arm_run_id,
        "comparator_run_id": p_run_id,
        "delta_label": delta_label,
        "n_users": out["n_users"],
        "axis": "deep_bucket + global",
        "includes_global_test": include_global,
        "metrics": {k: v for k, v in families.items()},
        "excluded_buckets": excluded_disclosed,
        "metric_robustness": robustness,
        "self_check": {
            "n_comparisons": out["self_check"]["n_comparisons"],
            "max_abs_diff": out["self_check"]["max_abs_diff"],
            "tolerance": out["self_check"]["tolerance"],
            "passed": out["self_check"]["passed"],
        },
        "segment_comparability": {
            "label": "COMPARABILITY ONLY — frozen Phase 4 five-segment axis, NO BH "
            "correction, not a confirmatory family (§5a)",
            "seed_convention": "scalar base seed per segment mask (eval/compare.py convention)",
            "segments": extra["segments"],
        },
        "artifact_paths": out["artifact_paths"],
        "artifact_sha256s": {
            k: runlog.sha256_file(p) for k, p in out["artifact_paths"].items()
        },
        "cache_manifest": out["cache_manifest"],
    }


def _cell_family(
    arm_key: str, arm_run_id: str, p_run_id: str, config: dict, results_path: Path
) -> dict:
    """One arm's Family S2: regime-map cells vs P*, BH across ALL cells of the arm."""
    metrics = tuple(config["metrics"])
    out = regime_map.build_regime_map(
        _regime_map_config(arm_key, arm_run_id, p_run_id, config), results_path
    )
    delta_label = f"{arm_key}_minus_{P_STAR_KEY}"
    families: dict[str, dict] = {}
    excluded_disclosed: list[dict] = []
    for metric in metrics:
        rows, excluded = _cell_rows(out["cells"], metric, delta_label)
        families[metric] = {
            "rows": rows,
            "bh": _apply_bh(rows),
            "excluded_cells": excluded,
            "confirmatory": False,
        }
        excluded_disclosed = excluded
    return {
        "arm": arm_key,
        "run_id": arm_run_id,
        "comparator_run_id": p_run_id,
        "delta_label": delta_label,
        "axis": f"regime_map cells ({' x '.join(regime_map.CELL_AXES)}) x frozen segments",
        "n_users": out["n_users"],
        "metrics": families,
        "excluded_cells": excluded_disclosed,
        "identity_check": {
            "n_comparisons": out["identity_check"]["n_comparisons"],
            "max_abs_diff": out["identity_check"]["max_abs_diff"],
            "all_bit_identical": out["identity_check"]["all_bit_identical"],
            "passed": out["identity_check"]["passed"],
        },
        "item_stats_manifest": out["item_stats_manifest"],
        "gate": out["gate"],
    }


# --- driver -------------------------------------------------------------------


def _validate_config(config: dict) -> None:
    """Refuse to run under inference parameters the preregistration did not fix.

    The FDR level, the resample count, the base seed and the confirmatory metric
    are preregistered constants, not knobs. A config that disagrees with them is
    a protocol deviation, and the right response is to abort rather than to
    quietly produce a differently-corrected verdict.
    """
    boot = config.get("bootstrap") or {}
    problems = []
    if int(boot.get("n_resamples", -1)) != PREREG_N_RESAMPLES:
        problems.append(f"bootstrap.n_resamples must be {PREREG_N_RESAMPLES} (§5e)")
    if int(boot.get("seed", -1)) != PREREG_BASE_SEED:
        problems.append(f"bootstrap.seed must be {PREREG_BASE_SEED} (§6 seed discipline)")
    if float(config.get("fdr_alpha", FDR_ALPHA)) != FDR_ALPHA:
        problems.append(f"fdr_alpha is preregistered at {FDR_ALPHA} and may not be changed")
    metrics = list(config.get("metrics") or [])
    if not metrics or metrics[0] != CONFIRMATORY_METRIC:
        problems.append(f"metrics[0] must be the confirmatory metric {CONFIRMATORY_METRIC!r} (§5b)")
    if ROBUSTNESS_METRIC not in metrics:
        problems.append(f"metrics must include {ROBUSTNESS_METRIC!r} for the §5g label")
    if config.get("split", "test") != "test":
        problems.append("this driver reports the frozen TEST split (§6)")

    arms = config.get("arms") or {}
    if not arms.get("m_star") or not arms.get("p_star"):
        problems.append("arms.m_star and arms.p_star are required")
    run_ids = {"m_star": arms.get("m_star"), "p_star": arms.get("p_star")}
    run_ids.update(config.get("secondary_arms") or {})
    placeholders = [k for k, v in run_ids.items() if str(v).upper().startswith("PLACEHOLDER")]
    if placeholders:
        problems.append(
            "unreplaced PLACEHOLDER run_id(s): "
            + ", ".join(sorted(placeholders))
            + " — fill in the TEST run_ids from results/runs.jsonl first"
        )
    if P_STAR_KEY in (config.get("secondary_arms") or {}):
        problems.append(f"{P_STAR_KEY!r} is the comparator and cannot also be a secondary arm")
    if problems:
        raise ValueError("confirmatory config rejected:\n  - " + "\n  - ".join(problems))


def validate_seed_discipline(
    run_ids: dict[str, str], results_path: Path, exemptions: dict | None = None
) -> dict:
    """§6: refuse to treat a stability-seed run as an inference artifact.

    Every configured ``run_id`` is resolved to its ``kind="eval"`` record (a
    missing record aborts) and its ``seeds.model`` is checked. **Default-deny**:

    * ``None`` (deterministic arm — popularity, kNN, content, blend, hybrid) or
      exactly ``20260805`` -> accepted;
    * ``20260806`` / ``20260807`` -> **rejected, and not exemptable**. These are
      the ALS family's stability seeds and §6 bars them from every paired CI,
      p-value and BH family. This is the check the guard exists for;
    * any other seed -> rejected **unless** ``exemptions[arm]`` declares that
      exact integer. The one real case is the A0 random floor, which records
      ``seeds.model: 13`` on both datasets (Amazon and the ML-32M VAL ladder); it
      has a single TEST record and no second-seed ambiguity, so a value-pinned
      declaration in the config is the honest way through. An exemption whose
      value does not match the record still aborts, and no exemption can reach a
      stability seed because that branch is tested first.

    Returns an audit block for the output record. Raises ``ValueError`` on any
    violation, listing all of them rather than the first.
    """
    declared = {str(k): int(v) for k, v in (exemptions or {}).items()}
    audit: dict[str, dict] = {}
    problems: list[str] = []
    for arm, run_id in run_ids.items():
        rec = _find_eval_record(run_id, Path(results_path))
        raw = (rec.get("seeds") or {}).get("model")
        seed = None if raw is None else int(raw)
        entry = {"run_id": run_id, "model_seed": seed, "model": (rec.get("model") or {}).get("name")}
        if seed is None:
            entry["status"] = "deterministic (no model seed recorded)"
        elif seed == PRIMARY_MODEL_SEED:
            entry["status"] = "primary inference seed"
        elif seed in STABILITY_MODEL_SEEDS:
            entry["status"] = "REJECTED: stability seed"
            problems.append(
                f"{arm} ({run_id}) carries seeds.model={seed}, a §6 STABILITY seed. "
                "Stability seeds contribute the reported mean±sd and nothing else; "
                "they may never enter a paired CI, a p-value or a BH family. Use the "
                f"{PRIMARY_MODEL_SEED} record for this arm. No exemption can authorize this."
            )
        elif declared.get(arm) == seed:
            entry["status"] = f"exempted by config (model_seed_exemptions.{arm}={seed})"
        else:
            entry["status"] = "REJECTED: unexpected model seed"
            problems.append(
                f"{arm} ({run_id}) carries seeds.model={seed}, which is neither the "
                f"primary inference seed {PRIMARY_MODEL_SEED} nor null. If this arm is "
                "legitimately seeded off-protocol (the A0 random floor records 13), "
                f"declare it explicitly as model_seed_exemptions.{arm}: {seed}."
            )
        audit[arm] = entry
    if problems:
        raise ValueError("seed discipline violated (§6):\n  - " + "\n  - ".join(problems))
    return {
        "rule": (
            "§6: seed 20260805 is the sole per-user artifact for every paired delta, "
            "p-value and BH family; 20260806/20260807 are stability evidence only"
        ),
        "primary_model_seed": PRIMARY_MODEL_SEED,
        "stability_model_seeds": list(STABILITY_MODEL_SEEDS),
        "exemptions_declared": declared,
        "per_arm": audit,
        "passed": True,
    }


def run_confirmatory(config: dict, config_path: Path, results_path: Path) -> dict:
    """Compute every family, classify, and return the full evidence record."""
    t0 = time.monotonic()
    _validate_config(config)
    arms = config["arms"]
    m_star, p_star = arms["m_star"], arms["p_star"]
    secondary = dict(config.get("secondary_arms") or {})
    include_global_s1 = bool(config.get("s1_include_global_test", True))

    # §6 seed discipline, checked against the records BEFORE any computation, so
    # a stability-seed artifact can never reach a BH family.
    seed_audit = validate_seed_discipline(
        {"m_star": m_star, P_STAR_KEY: p_star, **secondary},
        results_path,
        config.get("model_seed_exemptions"),
    )

    # Family P — the ONLY confirmatory family (§5b).
    family_p = _bucket_family(
        "m_star", m_star, p_star, config, results_path, include_global=True
    )
    conf = family_p["metrics"][CONFIRMATORY_METRIC]
    verdict = classify_verdict(
        [r for r in conf["rows"] if r["unit"] == "deep_bucket"],
        next((r for r in conf["rows"] if r["unit"] == "global"), None),
    )
    verdict["metric_robustness"] = family_p["metric_robustness"]

    # Family S1 — every other arm vs P*, BH within that arm (§5c).
    family_s1 = {
        name: _bucket_family(
            name, run_id, p_star, config, results_path, include_global=include_global_s1
        )
        for name, run_id in secondary.items()
    }

    # Family S2 — regime-map cells per arm vs P*, BH across all cells (§5d).
    # "each arm" includes M*, so it leads the mapping.
    family_s2: dict[str, dict] = {}
    if config.get("run_family_s2", True):
        s2_arms = {"m_star": m_star, **secondary}
        for name, run_id in s2_arms.items():
            family_s2[name] = _cell_family(name, run_id, p_star, config, results_path)

    git = runlog.git_info()
    record = {
        "kind": "confirmatory_ml32m",
        "generated_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "config_path": str(config_path),
        "config_hash": runlog.config_hash(config_path),
        "derived": True,
        "appends_to_runs_jsonl": False,
        "provenance_note": (
            "Derived CONFIRMATORY analysis (Phase 9 T9-3c): the per-user metric vectors "
            "already committed by the one-shot ML-32M TEST eval runs named in "
            "source_run_ids, regrouped and paired-bootstrapped through the committed "
            "T8-1/T8-3 machinery. No re-scoring, no refitting, no ground-truth "
            "consultation, no threshold fitting. Every rule applied is quoted from the "
            "committed T9-3b preregistration; this file appends nothing to "
            "results/runs.jsonl."
        ),
        "preregistration": PREREG_REFS,
        "splits": runlog.splits_block(config["splits_path"]),
        "dataset_manifest_path": config["dataset_manifest_path"],
        "dataset_manifest_hash": runlog.dataset_manifest_hash(config["dataset_manifest_path"]),
        "iceberg_snapshots": family_p["cache_manifest"]["snapshot_ids"],
        "contracts": family_p["cache_manifest"].get("contract_identities"),
        "source_run_ids": {"m_star": m_star, "p_star": p_star, **secondary},
        "protocol": {
            "eval_split": config.get("split", "test"),
            "n_users": family_p["n_users"],
            "metrics": list(config["metrics"]),
            "confirmatory_metric": CONFIRMATORY_METRIC,
            "robustness_metric": ROBUSTNESS_METRIC,
            "seed": int(config["bootstrap"]["seed"]),
            "n_resamples": int(config["bootstrap"]["n_resamples"]),
            "seed_discipline": (
                "§6: seed 20260805 is the sole per-user artifact used for every paired "
                "delta, p-value and BH family; 20260806/20260807 are stability evidence only"
            ),
            "s1_include_global_test": include_global_s1,
        },
        "seed_discipline_audit": seed_audit,
        "verdict": verdict,
        "families": {
            "P": {
                "role": "PRIMARY CONFIRMATORY (§5b) — M* vs P*, BH at FDR 0.05 within the family",
                **family_p,
            },
            "S1": {
                "role": "SECONDARY (§5c) — each other arm vs P* on the deep buckets, "
                "BH within that arm; may not be promoted to the headline (§7)",
                "arms": family_s1,
            },
            "S2": {
                "role": "SECONDARY (§5d) — regime-map cells per arm vs P*, BH across all "
                "cells of that arm; may not be promoted to the headline (§7)",
                "arms": family_s2,
            },
        },
        "wall_clock_s": round(time.monotonic() - t0, 3),
        "hardware": runlog.hardware_string(),
    }
    return record


# --- reporting ----------------------------------------------------------------


def _fmt_rows(rows: list[dict]) -> list[str]:
    order = {lbl: i for i, lbl in enumerate(DEEP_BUCKET_LABELS)}
    rows = sorted(rows, key=lambda r: (order.get(r["label"], 999), r["label"]))
    lines = [
        "| cell | users | delta | 95% CI | p (uncorr) | q (BH) | BH sig | verdict |",
        "|---|---:|---:|---|---|---:|:---:|---|",
    ]
    for r in rows:
        mark = "WIN" if r["bh_win"] else ("LOSS" if r["bh_loss"] else "—")
        lines.append(
            f"| {r['label']} | {r['n_users']} | {r['delta']:+.6f} | "
            f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}] | {r['p_display_uncorrected']} | "
            f"{r['q_value']:.6f} | {'yes' if r['bh_significant'] else 'no'} | {mark} |"
        )
    return lines


def print_report(record: dict) -> None:
    v = record["verdict"]
    print(f"# T9-3c confirmatory analysis — ML-32M {record['protocol']['eval_split'].upper()}")
    print(f"\n**VERDICT: {v['verdict']}** — {v['headline']}")
    if v["verdict_code"] == "D1":
        print(f"n* = {v['n_star']} (lower edge of bucket {v['crossover_bucket']})")
    for caveat in v.get("caveats") or []:
        print(f"\n> CAVEAT: {caveat}")
    if v["d4_flag"]:
        print(
            f"**{v['d4_token']}** — BH-significant negative buckets: "
            f"{', '.join(v['bh_significant_negative_buckets']) or 'global'}"
        )
    mr = v.get("metric_robustness") or {}
    if mr.get("claim_labels"):
        print(
            f"metric-robustness (§5g label, not confirmatory): "
            f"{'metric-robust' if mr['metric_robust'] else 'NOT metric-robust'} "
            f"({len(mr['agreeing_labels'])}/{len(mr['claim_labels'])} claim cells agree on "
            f"{ROBUSTNESS_METRIC})"
        )

    fam_p = record["families"]["P"]
    for metric, blk in fam_p["metrics"].items():
        tag = "CONFIRMATORY" if blk["confirmatory"] else "reported only (§5g)"
        print(f"\n## Family P — {fam_p['delta_label']} — {metric} [{tag}]")
        print(
            f"BH at FDR {blk['bh']['alpha']}: m={blk['bh']['m_tests']} tests, "
            f"{blk['bh']['n_bh_wins']} win(s), {blk['bh']['n_bh_losses']} loss(es); "
            f"uncorrected-significant: {blk['bh']['n_uncorrected_significant']} (§5h)"
        )
        print("\n".join(_fmt_rows(blk["rows"])))
        if blk["excluded_buckets"]:
            print(
                "excluded (zero TEST users, §5b): "
                + ", ".join(e["label"] for e in blk["excluded_buckets"])
            )

    for name, fam in record["families"]["S1"]["arms"].items():
        blk = fam["metrics"][CONFIRMATORY_METRIC]
        print(f"\n## Family S1 (secondary) — {fam['delta_label']} — {CONFIRMATORY_METRIC}")
        print(
            f"BH at FDR {blk['bh']['alpha']}: m={blk['bh']['m_tests']}, "
            f"{blk['bh']['n_bh_wins']} win(s), {blk['bh']['n_bh_losses']} loss(es)"
        )
        print("\n".join(_fmt_rows(blk["rows"])))

    for name, fam in record["families"]["S2"]["arms"].items():
        blk = fam["metrics"][CONFIRMATORY_METRIC]
        print(
            f"\n## Family S2 (secondary) — {fam['delta_label']} — {CONFIRMATORY_METRIC}: "
            f"m={blk['bh']['m_tests']} cells, {blk['bh']['n_bh_wins']} win(s), "
            f"{blk['bh']['n_bh_losses']} loss(es), "
            f"{len(blk['excluded_cells'])} empty cell(s) excluded"
        )

    print("\n## 5-segment comparability exhibit (NO BH — comparability only, §5a)")
    print("| segment | users | delta ndcg@10 | 95% CI | p (uncorrected) |")
    print("|---|---:|---:|---|---|")
    for seg in fam_p["segment_comparability"]["segments"]:
        d = (seg["delta"] or {}).get(CONFIRMATORY_METRIC)
        if d is None:
            print(f"| {seg['segment']} | 0 | — | — | — |")
            continue
        print(
            f"| {seg['segment']} | {seg['n_users']} | {d['delta']:+.6f} | "
            f"[{d['ci_lo']:+.6f}, {d['ci_hi']:+.6f}] | {p_display(d['p_value'])} |"
        )
    print(f"\nwall_clock_s={record['wall_clock_s']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.eval.confirmatory_ml32m")
    parser.add_argument("--config", default="configs/confirmatory_ml32m_test.yaml")
    parser.add_argument("--results", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--no-write", action="store_true", help="print the report, write no JSON"
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    results_path = Path(args.results or config.get("results_path", "results/runs.jsonl"))

    try:
        record = run_confirmatory(config, config_path, results_path)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report + non-zero exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_report(record)

    if args.no_write:
        print("\n--no-write: no JSON written")
        return 0
    out_path = Path(args.json_out or config.get("out_path", "results/confirmatory_ml32m_test.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=_json_default))
    print(f"\nevidence JSON written to {out_path} (results/runs.jsonl untouched)")
    return 0


def _json_default(obj: object):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


if __name__ == "__main__":
    sys.exit(main())
