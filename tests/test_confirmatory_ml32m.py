"""Tests for the T9-3c confirmatory driver (Phase 9; docs/engineering-log/UPGRADE_PLAN.md §8c).

The driver's whole value is that the verdict is a *mechanical* function of the
committed T9-3b preregistration applied to recorded artifacts. So the things
worth grading are the mechanisms, not the plumbing:

* the BH step-up arithmetic, against a case worked by hand in the comments;
* the §7 D1-D5 classifier, one fixture per verdict class (including the
  interleaved D5 that the preregistration explicitly forbids reporting as
  either D1 or D3, and the uniform-win case that the committed text makes
  **D1 with n\\* = 0**, not a special case);
* empty-bucket exclusion with the count disclosed (§5b);
* the ASL floor's reporting format, fixed by the 2026-08-21 selections entry;
* an end-to-end Family P over synthetic per-user artifacts, which is also the
  only way to prove the p-values come off the same child-seeded resample matrix
  as the CIs beside them.

No Spark, no real artifacts, no network. Family S2 is exercised at the
row-assembly level with a synthetic regime-map ``cells`` dict — running the real
recomposition would need a full eval cache and item_train_stats parquet, and the
kernel it would re-test is already graded by tests/test_regime_recompose.py.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batch_recsys_lab.eval.bootstrap import (
    asl_p_value,
    paired_delta_ci,
    paired_delta_resamples,
)
from batch_recsys_lab.eval.cell_stats import cell_block
from batch_recsys_lab.eval.confirmatory_ml32m import (
    CONFIRMATORY_METRIC,
    FDR_ALPHA,
    ROBUSTNESS_METRIC,
    _apply_bh,
    _bucket_rows,
    _cell_rows,
    _validate_config,
    asl_floor,
    benjamini_hochberg,
    classify_verdict,
    p_display,
    print_report,
    run_confirmatory,
    validate_seed_discipline,
)
from batch_recsys_lab.eval.protocol import DEEP_BUCKET_LABELS, segment_of

FIVE_CORE = "local.gold_ml32m.interactions_5core"
SNAPSHOT_ID = 3433604384732745693


# --- Benjamini-Hochberg -------------------------------------------------------


def test_bh_worked_example_step_up_rejects_above_the_naive_line():
    """Hand-worked, m = 4, alpha = 0.05, p = [0.001, 0.008, 0.039, 0.041].

    Raw m*p/i by rank:
        i=1: 4*0.001/1 = 0.004
        i=2: 4*0.008/2 = 0.016
        i=3: 4*0.039/3 = 0.052   <- above alpha on its own
        i=4: 4*0.041/4 = 0.041

    Monotone from the tail (q_(i) = min_{j>=i} raw_(j)):
        q4 = 0.041
        q3 = min(0.052, 0.041) = 0.041
        q2 = min(0.016, 0.041) = 0.016
        q1 = min(0.004, 0.016) = 0.004

    All four are significant at q <= 0.05. Note rank 3: its own naive
    comparison p_(3)=0.039 > 3*0.05/4 = 0.0375 FAILS, yet BH rejects it because
    rank 4 clears the line (0.041 <= 0.05) and BH is a step-up procedure. A
    driver that implemented the per-rank check without the tail-minimum would
    silently drop this test from the family — which on Family P could be the
    difference between D1 and D5.
    """
    out = benjamini_hochberg(
        [("a", 0.001), ("b", 0.008), ("c", 0.039), ("d", 0.041)], alpha=0.05
    )
    assert out["a"]["q_value"] == pytest.approx(0.004)
    assert out["b"]["q_value"] == pytest.approx(0.016)
    assert out["c"]["q_value"] == pytest.approx(0.041)
    assert out["d"]["q_value"] == pytest.approx(0.041)
    assert all(out[k]["significant"] for k in "abcd")
    assert [out[k]["rank"] for k in "abcd"] == [1, 2, 3, 4]
    assert {out[k]["m"] for k in "abcd"} == {4}


def test_bh_worked_example_partial_rejection():
    """Hand-worked, m = 4, p = [0.01, 0.04, 0.05, 0.60].

    raw: 0.04, 0.08, 0.0667, 0.60
    monotone from tail: q4=0.60, q3=0.0667, q2=min(0.08,0.0667)=0.0667, q1=0.04.
    Only the first test survives q <= 0.05 — the 0.04 and 0.05 raw p-values,
    which would both pass an uncorrected 0.05 screen, do not.
    """
    out = benjamini_hochberg([("a", 0.01), ("b", 0.04), ("c", 0.05), ("d", 0.60)])
    assert out["a"]["q_value"] == pytest.approx(0.04)
    assert out["b"]["q_value"] == pytest.approx(0.2 / 3)
    assert out["c"]["q_value"] == pytest.approx(0.2 / 3)
    assert out["d"]["q_value"] == pytest.approx(0.60)
    assert [out[k]["significant"] for k in "abcd"] == [True, False, False, False]


def test_bh_ties_share_a_q_value_and_sort_deterministically():
    """Tied p-values must not depend on input order: the sort key is (p, label)
    and the tail-minimum gives tied tests identical q."""
    # m=3, p=(0.02, 0.02, 0.9): raw = 0.06, 0.03, 0.9; monotone from the tail
    # gives q1 = min(0.06, 0.03) = 0.03 = q2, so the tie is resolved in favour of
    # BOTH tied tests (the step-up property again), while 0.9 stays out.
    a = benjamini_hochberg([("z", 0.02), ("a", 0.02), ("m", 0.9)])
    b = benjamini_hochberg([("m", 0.9), ("a", 0.02), ("z", 0.02)])
    assert a == b
    assert a["a"]["q_value"] == a["z"]["q_value"] == pytest.approx(0.03)
    assert a["a"]["rank"] == 1 and a["z"]["rank"] == 2  # label tie-break
    assert a["a"]["significant"] and a["z"]["significant"]
    assert not a["m"]["significant"]


def test_bh_family_size_matters():
    """The same p-value is significant in a family of 1 and not in a family of 8
    — which is exactly why §5 defines family membership before the numbers."""
    assert benjamini_hochberg([("only", 0.04)])["only"]["significant"] is True
    family = [(f"b{i}", p) for i, p in enumerate([0.04, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])]
    assert benjamini_hochberg(family)["b0"]["significant"] is False


def test_bh_empty_family_and_duplicate_labels():
    assert benjamini_hochberg([]) == {}
    with pytest.raises(ValueError, match="unique labels"):
        benjamini_hochberg([("a", 0.1), ("a", 0.2)])


# --- ASL floor reporting ------------------------------------------------------


def test_asl_floor_is_two_over_b_plus_one_and_is_reported_as_such():
    """The 2026-08-21 selections entry: §5(e)'s "< 0.001" describes the ONE-sided
    resolution 1/1001; the two-sided floor is 2/1001 and is reported verbatim."""
    assert asl_floor(1000) == pytest.approx(2 / 1001)
    all_positive = np.linspace(0.1, 0.2, 1000)  # no resampled delta <= 0
    p = asl_p_value(all_positive)
    assert p == pytest.approx(2 / 1001)
    assert p_display(p) == "2/1001 (floor)"
    assert p_display(0.0032) == "0.003200"
    # One resample on the other side already lifts p off the floor (4/1001) and
    # must NOT be reported as the floor.
    assert p_display(4 / 1001) == "0.003996"


def test_asl_floor_surfaces_on_row_annotations():
    rows = [
        {"label": "x", "p_value_uncorrected": 2 / 1001, "delta": 0.1},
        {"label": "y", "p_value_uncorrected": 0.4, "delta": -0.1},
    ]
    for r in rows:
        r["at_asl_floor"] = p_display(r["p_value_uncorrected"]).endswith("(floor)")
    assert rows[0]["at_asl_floor"] is True
    assert rows[1]["at_asl_floor"] is False


# --- §7 classifier ------------------------------------------------------------


def _stub(label: str, delta: float, significant: bool, unit: str = "deep_bucket") -> dict:
    return {
        "unit": unit,
        "label": label,
        "delta": delta,
        "bh_significant": significant,
        "bh_win": bool(significant and delta > 0),
        "bh_loss": bool(significant and delta < 0),
    }


def test_d1_crossover_at_n_star_twenty():
    """Shallow buckets lose, 20-49 and 100+ win: the canonical crossover.
    Coherence's shallowest qualifying bucket is 20-49 -> n* = 20."""
    rows = [
        _stub("0", -0.30, True),
        _stub("1-4", -0.20, True),
        _stub("5-9", -0.10, False),
        _stub("10-19", -0.02, False),
        _stub("20-49", +0.05, True),
        _stub("50-99", +0.06, False),
        _stub("100+", +0.09, True),
    ]
    v = classify_verdict(rows, _stub("global", -0.10, True, unit="global"))
    assert v["verdict"] == "D1_CROSSOVER"
    assert v["n_star"] == 20
    assert v["crossover_bucket"] == "20-49"
    assert v["condition_ii_coherence"]["buckets_at_or_above"] == ["20-49", "50-99", "100+"]
    # D4 is a flag, not a competing branch: significant losses coexist with D1
    # because they sit BELOW the crossover (§7 D1(ii) scopes to "at or above b").
    assert v["d4_flag"] is True
    assert v["bh_significant_negative_buckets"] == ["0", "1-4"]


def test_d1_with_n_star_zero_uniform_win():
    """Every bucket positive: the committed text makes this D1 with n* = 0 (0 is
    a member of the preregistered edge set {0,1,5,10,20,50,100}), not a separate
    "global win" case and not an unclassifiable one."""
    rows = [_stub(lbl, +0.05, lbl == "10-19") for lbl in DEEP_BUCKET_LABELS]
    v = classify_verdict(rows, _stub("global", +0.05, True, unit="global"))
    assert v["verdict"] == "D1_CROSSOVER"
    assert v["n_star"] == 0
    assert v["crossover_bucket"] == "0"
    assert v["d4_flag"] is False


def test_d1_n_star_takes_shallowest_populated_bucket_when_zero_is_empty():
    """An unpopulated bucket yields no test (§5b) and is skipped by the coherence
    quantifier — so n* falls to the shallowest bucket that actually exists."""
    rows = [_stub(lbl, +0.05, lbl == "100+") for lbl in DEEP_BUCKET_LABELS if lbl != "0"]
    v = classify_verdict(rows, _stub("global", +0.05, False, unit="global"))
    assert v["verdict"] == "D1_CROSSOVER"
    assert v["n_star"] == 1
    assert v["crossover_bucket"] == "1-4"


def test_d2_global_only_win():
    """No depth bucket is BH-significant; the global test is a significant win.
    Pre-named so it "cannot be silently rounded into D1" (§7 D2)."""
    rows = [_stub(lbl, +0.01, False) for lbl in DEEP_BUCKET_LABELS]
    v = classify_verdict(rows, _stub("global", +0.01, True, unit="global"))
    assert v["verdict"] == "D2_GLOBAL_ONLY_WIN"
    assert v["n_star"] is None
    assert v["condition_i_met"] is False
    # The point estimates are all positive, so coherence *would* hold — D2 still
    # wins because condition (i) needs a BH-significant positive BUCKET.
    assert v["condition_ii_met"] is True


def test_d3_double_null():
    rows = [_stub(lbl, -0.05, False) for lbl in DEEP_BUCKET_LABELS]
    v = classify_verdict(rows, _stub("global", -0.05, False, unit="global"))
    assert v["verdict"] == "D3_DOUBLE_NULL"
    assert v["n_star"] is None
    assert v["d4_flag"] is False


def test_d3_holds_even_with_significant_negatives_plus_d4_flag():
    """A double null with losses is still D3; D4 rides alongside as a flag."""
    rows = [_stub(lbl, -0.05, True) for lbl in DEEP_BUCKET_LABELS]
    v = classify_verdict(rows, _stub("global", -0.05, True, unit="global"))
    assert v["verdict"] == "D3_DOUBLE_NULL"
    assert v["d4_flag"] is True
    assert v["d4_token"] == "D4_SIGNIFICANT_NEGATIVES"
    assert len(v["bh_significant_negative_buckets"]) == 7


def test_d5_interleaved_significant_positive_and_negative():
    """§7 D5, the interleaved case: a significant win at 5-9 and a significant
    LOSS at 100+. Every candidate window contains 100+, so no coherent b exists.
    D5 "may not be reported as D1, and may not be reported as D3"."""
    rows = [
        _stub("0", -0.30, True),
        _stub("1-4", -0.02, False),
        _stub("5-9", +0.08, True),
        _stub("10-19", +0.04, False),
        _stub("20-49", +0.03, False),
        _stub("50-99", +0.02, False),
        _stub("100+", -0.07, True),
    ]
    v = classify_verdict(rows, _stub("global", -0.05, False, unit="global"))
    assert v["verdict"] == "D5_MIXED"
    assert v["condition_i_met"] is True
    assert v["condition_ii_met"] is False
    assert v["n_star"] is None
    assert v["condition_i_significant_positive_buckets"] == ["5-9"]
    assert v["d4_flag"] is True


def test_d5_when_a_deeper_bucket_has_a_negative_point_estimate():
    """The other D5 arm of §7: no significant negative anywhere, but the DEEPEST
    bucket has a negative POINT ESTIMATE, so every candidate window contains it
    and the "keeps winning" clause fails on point estimates alone."""
    rows = [
        _stub("0", -0.10, False),
        _stub("1-4", -0.05, False),
        _stub("5-9", +0.01, False),
        _stub("10-19", +0.06, True),
        _stub("20-49", +0.02, False),
        _stub("50-99", +0.01, False),
        _stub("100+", -0.001, False),  # negative point estimate, not significant
    ]
    v = classify_verdict(rows, _stub("global", +0.001, False, unit="global"))
    assert v["verdict"] == "D5_MIXED"
    assert v["condition_ii_met"] is False
    assert v["d4_flag"] is False  # D5 does not imply D4


def test_d1_literal_reading_when_the_coherent_region_holds_no_significant_win():
    """PREREG AMBIGUITY, pinned rather than silently resolved.

    §7 states D1(i) ("at least one BH-significant positive bucket in Family P")
    and D1(ii) (the coherence condition) as INDEPENDENT clauses. Here the only
    significant win is at 10-19, and the only coherent region is {100+} because
    50-99 dips negative. The literal text yields D1 with n* = 100 even though no
    bucket at or above 100 is individually significant.

    The driver applies the literal text — but must say so: the coherence block
    reports ``contains_significant_positive: False`` and a caveat is emitted. If
    the owner intends the coupled reading (the significant win must sit inside
    the region), that is a preregistration amendment, not a code fix.
    """
    rows = [
        _stub("0", -0.10, False),
        _stub("1-4", -0.05, False),
        _stub("5-9", +0.01, False),
        _stub("10-19", +0.06, True),
        _stub("20-49", +0.02, False),
        _stub("50-99", -0.001, False),
        _stub("100+", +0.05, False),
    ]
    v = classify_verdict(rows, _stub("global", +0.001, False, unit="global"))
    assert v["verdict"] == "D1_CROSSOVER"
    assert v["n_star"] == 100
    assert v["condition_ii_coherence"]["contains_significant_positive"] is False
    assert v["caveats"] and "literal §7 text" in v["caveats"][0]


def test_d1_clean_crossover_reports_no_caveat():
    rows = [_stub(lbl, +0.05, lbl in ("20-49", "100+")) for lbl in DEEP_BUCKET_LABELS]
    rows[0] = _stub("0", -0.2, True)
    rows[1] = _stub("1-4", -0.1, False)
    v = classify_verdict(rows, _stub("global", -0.01, False, unit="global"))
    assert v["verdict"] == "D1_CROSSOVER"
    assert v["n_star"] == 5
    assert v["condition_ii_coherence"]["contains_significant_positive"] is True
    assert v["caveats"] == []


# --- empty-bucket exclusion ---------------------------------------------------


def _bucket_block(label: str, ordinal: int, n_users: int, delta: float, p: float) -> dict:
    if n_users == 0:
        return {
            "bucket": label,
            "bucket_ordinal": ordinal,
            "n_users": 0,
            "user_share": 0.0,
            "n_train_min": None,
            "n_train_max": None,
            "delta": {CONFIRMATORY_METRIC: None},
            "seed_entropy": [20260805, ordinal],
        }
    return {
        "bucket": label,
        "bucket_ordinal": ordinal,
        "n_users": n_users,
        "user_share": 0.1,
        "n_train_min": 1,
        "n_train_max": 2,
        "delta": {
            CONFIRMATORY_METRIC: {
                "delta": delta,
                "ci_lo": delta - 0.01,
                "ci_hi": delta + 0.01,
                "ci_width": 0.02,
                "excludes_zero": True,
                "p_value": p,
            }
        },
        "seed_entropy": [20260805, ordinal],
    }


def test_empty_buckets_are_excluded_from_the_family_and_disclosed():
    """§5(b): "A bucket with zero TEST users yields no test and is excluded with
    its count disclosed" — the exclusion must shrink m (BH is family-size
    dependent) and must still appear in the output."""
    blocks = [
        _bucket_block("0", 0, 10, -0.2, 0.01),
        _bucket_block("1-4", 1, 0, 0.0, 1.0),
        _bucket_block("5-9", 2, 0, 0.0, 1.0),
        _bucket_block("10-19", 3, 5, 0.1, 0.02),
        _bucket_block("20-49", 4, 7, 0.2, 0.03),
        _bucket_block("50-99", 5, 0, 0.0, 1.0),
        _bucket_block("100+", 6, 3, 0.3, 0.04),
    ]
    rows, excluded = _bucket_rows(blocks, CONFIRMATORY_METRIC, "m_star_minus_p_star")
    assert [r["label"] for r in rows] == ["0", "10-19", "20-49", "100+"]
    assert [e["label"] for e in excluded] == ["1-4", "5-9", "50-99"]
    assert all(e["n_users"] == 0 for e in excluded)
    summary = _apply_bh(rows)
    assert summary["m_tests"] == 4  # not 7 — empty buckets never entered the family
    assert all(r["bh_m"] == 4 for r in rows)


def test_single_user_bucket_is_annotated_as_degenerate():
    """One user => every paired resample is the same number => the ASL pins to
    the floor regardless of effect size. Disclosed on the row, not thresholded
    away (inventing a minimum-n rule would be un-preregistered scope)."""
    rows, _ = _bucket_rows(
        [_bucket_block("100+", 6, 1, 0.3, 2 / 1001)], CONFIRMATORY_METRIC, "d"
    )
    assert rows[0]["degenerate_single_user"] is True
    assert rows[0]["at_asl_floor"] is True


# --- cell_block ASL opt-in ----------------------------------------------------


def test_cell_block_asl_is_off_by_default_and_matches_the_ci_draws_when_on():
    """The opt-in must (a) leave the committed key set untouched when off and
    (b) when on, return exactly asl_p_value(paired_delta_resamples(...)) at the
    SAME child seed — §5(e)'s "same seed, same resample matrix" is the whole
    reason the p-values cost no second look at TEST."""
    rng = np.random.default_rng(7)
    a = rng.random(40)
    b = rng.random(40)
    arm_values = {"arm": {"ndcg@10": a}, "pop": {"ndcg@10": b}}
    entropy = [20260805, 4]

    off = cell_block(arm_values, ("ndcg@10",), ("arm", "pop"), entropy, 200)
    on = cell_block(arm_values, ("ndcg@10",), ("arm", "pop"), entropy, 200, asl_p_values=True)

    assert "p_value" not in off["delta"]["ndcg@10"]
    for key in ("delta", "ci_lo", "ci_hi", "ci_width", "excludes_zero"):
        assert on["delta"]["ndcg@10"][key] == off["delta"]["ndcg@10"][key]
    expected = asl_p_value(paired_delta_resamples(a, b, n_resamples=200, seed=entropy))
    assert on["delta"]["ndcg@10"]["p_value"] == expected
    # and the CI itself is the one paired_delta_ci draws at that entropy
    assert on["delta"]["ndcg@10"]["ci_lo"] == paired_delta_ci(
        a, b, n_resamples=200, seed=entropy
    )["ci_lo"]


def test_global_cell_entropy_equals_compare_pys_scalar_seed():
    """PREREG GAP, pinned: §6 fixes child seeds for per-segment / per-cell
    inference but never names the GLOBAL cell's entropy. The driver uses
    ``[20260805]``, which NumPy's SeedSequence treats identically to the scalar
    ``20260805`` — i.e. exactly eval/compare.py's committed convention — so the
    global block is independently reproducible by a ``kind="paired_delta"`` run.
    If that NumPy equivalence ever changed, this test fails loudly rather than
    the global test silently drifting off the comparison record."""
    rng = np.random.default_rng(3)
    a, b = rng.random(50), rng.random(50)
    via_cell = cell_block(
        {"arm": {"ndcg@10": a}, "pop": {"ndcg@10": b}},
        ("ndcg@10",),
        ("arm", "pop"),
        [20260805],
        200,
        asl_p_values=True,
    )["delta"]["ndcg@10"]
    via_compare = paired_delta_ci(a, b, n_resamples=200, seed=20260805)
    assert via_cell["delta"] == via_compare["delta"]
    assert via_cell["ci_lo"] == via_compare["ci_lo"]
    assert via_cell["ci_hi"] == via_compare["ci_hi"]
    assert via_cell["p_value"] == asl_p_value(
        paired_delta_resamples(a, b, n_resamples=200, seed=20260805)
    )


# --- config validation --------------------------------------------------------


def _valid_config(**overrides) -> dict:
    cfg = {
        "split": "test",
        "arms": {"m_star": "run-m", "p_star": "run-p"},
        "secondary_arms": {"blend": "run-b"},
        "metrics": ["ndcg@10", "recall@20"],
        "fdr_alpha": 0.05,
        "bootstrap": {"n_resamples": 1000, "seed": 20260805},
    }
    cfg.update(overrides)
    return cfg


def test_validate_config_accepts_the_preregistered_shape():
    _validate_config(_valid_config())


@pytest.mark.parametrize(
    "override, message",
    [
        ({"bootstrap": {"n_resamples": 2000, "seed": 20260805}}, "n_resamples"),
        ({"bootstrap": {"n_resamples": 1000, "seed": 20260806}}, "seed"),
        ({"fdr_alpha": 0.10}, "fdr_alpha"),
        ({"metrics": ["recall@20", "ndcg@10"]}, "confirmatory metric"),
        ({"metrics": ["ndcg@10"]}, "recall@20"),
        ({"split": "val"}, "frozen TEST"),
    ],
)
def test_validate_config_rejects_protocol_deviations(override, message):
    with pytest.raises(ValueError, match=message):
        _validate_config(_valid_config(**override))


def test_validate_config_rejects_unreplaced_placeholders():
    cfg = _valid_config(arms={"m_star": "PLACEHOLDER_A4_KNN_T12M_TEST", "p_star": "run-p"})
    with pytest.raises(ValueError, match="PLACEHOLDER"):
        _validate_config(cfg)


# --- §6 seed discipline guard -------------------------------------------------


def _seed_log(tmp_path, arms: dict[str, int | None]) -> "Path":
    """A minimal results log: one kind="eval" record per arm, carrying only what
    the seed guard reads."""
    path = tmp_path / "runs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "kind": "eval",
                    "run_id": f"run-{arm}",
                    "model": {"name": arm},
                    "seeds": {"bootstrap": 20260805, "model": seed},
                }
            )
            for arm, seed in arms.items()
        )
        + "\n"
    )
    return path


def test_seed_guard_accepts_primary_and_deterministic_arms(tmp_path):
    log = _seed_log(tmp_path, {"m_star": None, "p_star": None, "als": 20260805})
    audit = validate_seed_discipline(
        {"m_star": "run-m_star", "p_star": "run-p_star", "als": "run-als"}, log
    )
    assert audit["passed"] is True
    assert audit["per_arm"]["als"]["model_seed"] == 20260805
    assert "deterministic" in audit["per_arm"]["p_star"]["status"]


@pytest.mark.parametrize("stability_seed", [20260806, 20260807])
def test_seed_guard_rejects_a_stability_seed_artifact(tmp_path, stability_seed):
    """§6: 20260806/20260807 are stability evidence only and "never enter a
    paired CI, a p-value, or a BH family". Pasting one in as the ALS inference
    artifact is exactly the mistake this guard exists to stop."""
    log = _seed_log(tmp_path, {"p_star": None, "als": stability_seed})
    with pytest.raises(ValueError, match="STABILITY seed"):
        validate_seed_discipline({"p_star": "run-p_star", "als": "run-als"}, log)


def test_seed_guard_will_not_let_an_exemption_launder_a_stability_seed(tmp_path):
    """An exemption must never be able to reach a stability seed — the stability
    branch is checked first, so declaring one is not a way around §6."""
    log = _seed_log(tmp_path, {"als": 20260806})
    with pytest.raises(ValueError, match="STABILITY seed"):
        validate_seed_discipline({"als": "run-als"}, log, exemptions={"als": 20260806})


def test_seed_guard_rejects_an_undeclared_off_protocol_seed(tmp_path):
    """Default-deny: the A0 random floor records seeds.model=13 on both datasets,
    so an unqualified allowlist would abort on a legitimate arm. The seed must be
    declared with its exact value instead of the rule being widened."""
    log = _seed_log(tmp_path, {"random": 13})
    with pytest.raises(ValueError, match="model_seed_exemptions.random: 13"):
        validate_seed_discipline({"random": "run-random"}, log)

    audit = validate_seed_discipline({"random": "run-random"}, log, exemptions={"random": 13})
    assert "exempted by config" in audit["per_arm"]["random"]["status"]
    assert audit["exemptions_declared"] == {"random": 13}

    # A declaration that does not match the record still aborts.
    with pytest.raises(ValueError, match="unexpected model seed|neither the"):
        validate_seed_discipline({"random": "run-random"}, log, exemptions={"random": 7})


def test_seed_guard_aborts_on_a_run_id_absent_from_the_log(tmp_path):
    log = _seed_log(tmp_path, {"p_star": None})
    with pytest.raises(ValueError, match="no eval record with run_id"):
        validate_seed_discipline({"m_star": "run-missing"}, log)


def test_seed_guard_runs_inside_the_driver(tmp_path):
    """End-to-end: flipping the M* record to a stability seed must abort the
    whole driver before any family is computed."""
    config, config_path, results = _write_fixture(tmp_path)
    lines = [json.loads(line) for line in results.read_text().splitlines()]
    for rec in lines:
        if rec["run_id"] == "test-m-star":
            rec["seeds"] = {"bootstrap": 20260805, "model": 20260807}
    results.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    with pytest.raises(ValueError, match="STABILITY seed"):
        run_confirmatory(config, config_path, results)


# --- end-to-end Family P over synthetic per-user artifacts --------------------
#
# Six populated depth buckets and one deliberately EMPTY one (50-99). The arm
# loses on every shallow bucket and wins on 20-49 / 100+ — i.e. the fixture is
# the crossover shape, and the driver has to find n* = 20 on its own.

DEPTHS = [0, 0, 0, 0, 2, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30, 45, 120, 150, 200, 300]
ARM_BY_BUCKET = {
    "0": 0.10,
    "1-4": 0.15,
    "5-9": 0.20,
    "10-19": 0.30,
    "20-49": 0.70,
    "100+": 0.80,
}
POP_VALUE = 0.50


def _synthetic_values(metric_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depths = np.asarray(DEPTHS, dtype=np.int64)
    labels = np.asarray(
        [
            "0" if d == 0 else "1-4" if d <= 4 else "5-9" if d <= 9 else "10-19"
            if d <= 19
            else "20-49"
            if d <= 49
            else "50-99"
            if d <= 99
            else "100+"
            for d in depths
        ]
    )
    # Small per-user jitter keeps the bootstrap non-degenerate while preserving
    # each bucket's sign; the sign is what the verdict depends on.
    jitter = np.arange(len(depths)) * 0.001
    arm = np.asarray([ARM_BY_BUCKET[lbl] for lbl in labels]) + jitter
    pop = np.full(len(depths), POP_VALUE) + jitter
    return depths, arm * metric_scale, pop * metric_scale


def _write_fixture(tmp_path):
    depths, arm_ndcg, pop_ndcg = _synthetic_values(1.0)
    _, arm_rec, pop_rec = _synthetic_values(0.8)  # same signs -> metric-robust
    n = len(depths)
    user_ids = [f"u{i:03d}" for i in range(n)]
    user_idx = np.arange(n)[::-1].copy()  # non-identity: exercises the join
    n_train_by_idx = np.zeros(n, dtype=np.int64)
    n_train_by_idx[user_idx] = depths
    segments = np.asarray([str(s) for s in segment_of(depths)])

    cache_dir = tmp_path / "cache_ml32m" / str(SNAPSHOT_ID)
    cache_dir.mkdir(parents=True)
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(
            {
                "snapshot_ids": {FIVE_CORE: SNAPSHOT_ID},
                "contract_identities": {"gold_ml32m.interactions_5core": "sha256:deadbeef"},
            }
        )
    )
    np.save(cache_dir / "n_train.npy", n_train_by_idx)

    def _write_arm(name, ndcg, recall):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(
            pa.table(
                {
                    "user_id": pa.array(user_ids, type=pa.string()),
                    "user_idx": pa.array(user_idx, type=pa.int64()),
                    "segment": pa.array(list(segments), type=pa.string()),
                    "ndcg@10": pa.array(ndcg, type=pa.float64()),
                    "recall@20": pa.array(recall, type=pa.float64()),
                }
            ),
            path,
        )
        return path

    def _per_segment(ndcg, recall):
        out = {}
        for label in ("0", "1-4", "5-9", "10-19", "20+"):
            mask = segments == label
            if not mask.any():
                continue
            out[label] = {
                "n_users": int(mask.sum()),
                "ndcg@10": {"value": float(ndcg[mask].mean())},
                "recall@20": {"value": float(recall[mask].mean())},
            }
        return out

    arm_path = _write_arm("m_star", arm_ndcg, arm_rec)
    pop_path = _write_arm("p_star", pop_ndcg, pop_rec)
    results = tmp_path / "runs.jsonl"
    results.write_text(
        "\n".join(
            json.dumps(
                {
                    "kind": "eval",
                    "run_id": run_id,
                    "model": {"name": run_id},
                    "per_user_artifact": str(path),
                    "protocol": {"eval_split": "test"},
                    "iceberg_snapshots": {FIVE_CORE: SNAPSHOT_ID},
                    "metrics": {"per_segment": _per_segment(ndcg, recall)},
                }
            )
            for run_id, path, ndcg, recall in [
                ("test-m-star", arm_path, arm_ndcg, arm_rec),
                ("test-p-star", pop_path, pop_ndcg, pop_rec),
            ]
        )
        + "\n"
    )

    splits = tmp_path / "splits_ml32m.yaml"
    splits.write_text("version: 1\nfrozen_at: '2026-08-19T00:00:00Z'\n")
    manifest = tmp_path / "MANIFEST_ML32M.md"
    manifest.write_text("# ML-32M manifest fixture\n")
    config = {
        "kind": "confirmatory_ml32m",
        "split": "test",
        "arms": {"m_star": "test-m-star", "p_star": "test-p-star"},
        "secondary_arms": {},
        "metrics": ["ndcg@10", "recall@20"],
        "fdr_alpha": 0.05,
        "bootstrap": {"n_resamples": 1000, "seed": 20260805},
        "run_family_s2": False,
        "cache_dir": str(tmp_path / "cache_ml32m"),
        "five_core_table": FIVE_CORE,
        "item_stats_dir": str(tmp_path / "item_stats"),
        "splits_path": str(splits),
        "dataset_manifest_path": str(manifest),
        "expected_n_users": n,
        "self_check": {"tolerance": 1.0e-9},
        "results_path": str(results),
    }
    config_path = tmp_path / "confirmatory.yaml"
    config_path.write_text(json.dumps(config))  # YAML is a superset of JSON
    return config, config_path, results


def test_family_p_end_to_end_finds_the_crossover(tmp_path):
    config, config_path, results = _write_fixture(tmp_path)
    record = run_confirmatory(config, config_path, results)

    fam = record["families"]["P"]
    conf = fam["metrics"][CONFIRMATORY_METRIC]
    labels = [r["label"] for r in conf["rows"]]
    assert "50-99" not in labels  # empty bucket excluded (§5b)
    assert [e["label"] for e in conf["excluded_buckets"]] == ["50-99"]
    assert conf["bh"]["m_tests"] == 7  # 6 populated buckets + 1 global
    assert conf["confirmatory"] is True
    assert fam["metrics"][ROBUSTNESS_METRIC]["confirmatory"] is False

    by_label = {r["label"]: r for r in conf["rows"]}
    assert by_label["0"]["delta"] < 0 and by_label["20-49"]["delta"] > 0
    # Every user in a bucket sits on one side, so each bucket's 1,000 resampled
    # deltas are one-signed and the ASL pins to its floor.
    assert by_label["20-49"]["p_display_uncorrected"] == "2/1001 (floor)"
    assert by_label["20-49"]["at_asl_floor"] is True
    # Uncorrected values are published beside the corrected ones (§5h).
    assert by_label["20-49"]["significant_uncorrected"] is True
    assert by_label["20-49"]["q_value"] <= FDR_ALPHA

    v = record["verdict"]
    assert v["verdict"] == "D1_CROSSOVER"
    assert v["n_star"] == 20
    assert v["crossover_bucket"] == "20-49"
    assert v["d4_flag"] is True  # shallow buckets are significant losses
    assert v["metric_robustness"]["metric_robust"] is True

    # The comparability exhibit is present, on the FROZEN five segments, and is
    # never BH-corrected.
    segs = fam["segment_comparability"]["segments"]
    assert [s["segment"] for s in segs] == ["0", "1-4", "5-9", "10-19", "20+"]
    assert all("q_value" not in (s["delta"][CONFIRMATORY_METRIC] or {}) for s in segs)
    assert "COMPARABILITY ONLY" in fam["segment_comparability"]["label"]

    # Provenance: the record carries the ML-32M manifest hash, never MANIFEST.md.
    assert record["dataset_manifest_path"].endswith("MANIFEST_ML32M.md")
    assert record["iceberg_snapshots"][FIVE_CORE] == SNAPSHOT_ID
    assert record["appends_to_runs_jsonl"] is False


def test_report_renders_without_crashing(tmp_path, capsys):
    """The markdown report is the human-facing half of the deliverable; a format
    crash discovered during the one-shot TEST reporting run would be a bad time
    to find it."""
    config, config_path, results = _write_fixture(tmp_path)
    record = run_confirmatory(config, config_path, results)
    print_report(record)
    out = capsys.readouterr().out
    assert "D1_CROSSOVER" in out
    assert "n* = 20" in out
    assert "2/1001 (floor)" in out
    assert "COMPARABILITY ONLY" not in out  # exhibit is printed under its own heading
    assert "NO BH" in out


# --- Family S2 row assembly ---------------------------------------------------


def _cell(axis: str, segment: str, bucket: str, n_users: int, delta: float, p: float) -> dict:
    block = {
        "axis": axis,
        "segment": segment,
        "bucket": bucket,
        "n_users": n_users,
        "gt_interactions": n_users * 2,
        "user_share": 0.1,
        "gt_share": 0.1,
        "seed_entropy": [20260805, 0, 0, 0],
        "delta": {
            CONFIRMATORY_METRIC: None
            if n_users == 0
            else {
                "delta": delta,
                "ci_lo": delta - 0.01,
                "ci_hi": delta + 0.01,
                "ci_width": 0.02,
                "excludes_zero": True,
                "p_value": p,
            }
        },
    }
    return block


def test_family_s2_pools_both_axes_into_one_bh_family_and_drops_empty_cells():
    """§5(d): "BH at FDR 0.05 across all cells per arm". The family is one arm's
    cells pooled across BOTH committed CELL_AXES; cells with no users carry no
    test and are disclosed."""
    cells = {
        "support": [
            _cell("support", "0", "zero", 4, -0.1, 0.01),
            _cell("support", "0", "low", 0, 0.0, 1.0),
            _cell("support", "20+", "high", 9, 0.2, 0.02),
        ],
        "recency": [
            _cell("recency", "20+", "<=90d", 6, 0.3, 0.03),
            _cell("recency", "20+", "absent", 0, 0.0, 1.0),
        ],
    }
    rows, excluded = _cell_rows(cells, CONFIRMATORY_METRIC, "als_minus_p_star")
    assert [r["label"] for r in rows] == [
        "support|0|zero",
        "support|20+|high",
        "recency|20+|<=90d",
    ]
    assert [e["label"] for e in excluded] == ["support|0|low", "recency|20+|absent"]
    summary = _apply_bh(rows)
    assert summary["m_tests"] == 3
    assert all(r["unit"] == "regime_cell" for r in rows)
    assert summary["n_bh_wins"] == 2 and summary["n_bh_losses"] == 1


def test_driver_never_writes_to_the_results_log(tmp_path):
    config, config_path, results = _write_fixture(tmp_path)
    before = results.read_bytes()
    run_confirmatory(config, config_path, results)
    assert results.read_bytes() == before


def test_p_values_are_reproducible_run_to_run(tmp_path):
    """Invariant #2: same seed, same numbers. The p-values are drawn from the
    child-seeded matrices, so two runs must agree bit-for-bit."""
    config, config_path, results = _write_fixture(tmp_path)
    a = run_confirmatory(config, config_path, results)
    b = run_confirmatory(config, config_path, results)
    rows_a = a["families"]["P"]["metrics"][CONFIRMATORY_METRIC]["rows"]
    rows_b = b["families"]["P"]["metrics"][CONFIRMATORY_METRIC]["rows"]
    assert [r["p_value_uncorrected"] for r in rows_a] == [
        r["p_value_uncorrected"] for r in rows_b
    ]
    assert [r["q_value"] for r in rows_a] == [r["q_value"] for r in rows_b]
