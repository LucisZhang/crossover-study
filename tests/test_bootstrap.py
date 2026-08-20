"""Tests for eval/bootstrap.py (Phase 2, T3; UPGRADE_PLAN.md §8)."""

from __future__ import annotations

import numpy as np
import pytest

from batch_recsys_lab.eval.bootstrap import (
    MAX_RESAMPLE_ELEMENTS,
    RESAMPLE_BLOCK_ELEMENTS,
    asl_p_value,
    ci_mean,
    paired_delta_ci,
    paired_delta_resamples,
    resample_matrix,
    segment_cis,
)

SEED = 20260805


def test_ci_mean_constant_vector_is_degenerate():
    values = np.full(50, 3.7)
    result = ci_mean(values, n_resamples=200, seed=SEED)
    expected = float(np.mean(values))
    assert result["value"] == expected
    assert result["ci_lo"] == result["ci_hi"] == expected


def test_resample_matrix_determinism():
    m1 = resample_matrix(100, 1000, SEED)
    m2 = resample_matrix(100, 1000, SEED)
    assert np.array_equal(m1, m2)


def test_ci_mean_determinism():
    rng = np.random.default_rng(1)
    values = rng.random(200)
    r1 = ci_mean(values, n_resamples=1000, seed=SEED)
    r2 = ci_mean(values, n_resamples=1000, seed=SEED)
    assert r1 == r2


def test_ci_mean_with_precomputed_resamples_matches_internal():
    rng = np.random.default_rng(2)
    values = rng.random(150)
    resamples = resample_matrix(values.shape[0], 500, SEED)
    r_precomputed = ci_mean(values, resamples=resamples)
    r_internal = ci_mean(values, n_resamples=500, seed=SEED)
    assert r_precomputed == r_internal


def test_paired_delta_self_is_exactly_zero():
    rng = np.random.default_rng(3)
    v = rng.random(80)
    result = paired_delta_ci(v, v, n_resamples=300, seed=SEED)
    assert result["delta"] == 0.0
    assert result["ci_lo"] == 0.0
    assert result["ci_hi"] == 0.0
    assert result["excludes_zero"] is False


def test_paired_delta_determinism():
    rng = np.random.default_rng(4)
    a = rng.random(80)
    b = rng.random(80)
    r1 = paired_delta_ci(a, b, n_resamples=500, seed=SEED)
    r2 = paired_delta_ci(a, b, n_resamples=500, seed=SEED)
    assert r1 == r2


def test_paired_delta_constant_offset_is_exact_and_degenerate():
    rng = np.random.default_rng(5)
    b = rng.random(100)
    a = b + 0.5
    result = paired_delta_ci(a, b, n_resamples=500, seed=SEED)
    assert abs(result["delta"] - 0.5) < 1e-9
    assert abs(result["ci_lo"] - 0.5) < 1e-9
    assert abs(result["ci_hi"] - 0.5) < 1e-9
    assert result["excludes_zero"] is True


def test_ci_mean_brackets_analytic_mean_small_case():
    # values=[0,1]: analytic mean 0.5. Resample means take values in
    # {0, 0.5, 1} with binomial(2,0.5)-ish structure; the CI must bracket 0.5
    # and stay within [0, 1].
    values = np.array([0.0, 1.0])
    result = ci_mean(values, n_resamples=5000, seed=SEED)
    assert result["value"] == 0.5
    assert 0.0 <= result["ci_lo"] <= 0.5 <= result["ci_hi"] <= 1.0


def test_segment_cis_reports_correct_n_users_and_values():
    labels = np.array(["a", "a", "a", "b", "b", "c"])
    values = np.array([1.0, 1.0, 1.0, 5.0, 7.0, 9.0])
    result = segment_cis(values, labels, n_resamples=500, seed=SEED)

    assert set(result.keys()) == {"a", "b", "c"}
    assert result["a"]["n_users"] == 3
    assert result["a"]["value"] == 1.0
    assert result["a"]["ci_lo"] == result["a"]["ci_hi"] == 1.0  # constant segment
    assert result["b"]["n_users"] == 2
    assert result["b"]["value"] == 6.0


def test_segment_cis_one_user_segment_is_degenerate():
    labels = np.array(["a", "a", "solo"])
    values = np.array([1.0, 2.0, 42.0])
    result = segment_cis(values, labels, n_resamples=500, seed=SEED)
    assert result["solo"]["n_users"] == 1
    assert result["solo"]["value"] == 42.0
    assert result["solo"]["ci_lo"] == result["solo"]["ci_hi"] == 42.0


def test_segment_cis_are_independent_across_segments():
    # A segment's CI must not change when another segment's values change.
    labels = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    values_1 = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    values_2 = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0])

    result_1 = segment_cis(values_1, labels, n_resamples=500, seed=SEED)
    result_2 = segment_cis(values_2, labels, n_resamples=500, seed=SEED)

    assert result_1["a"] == result_2["a"]
    assert result_1["b"] != result_2["b"]


# --- Chunked (large-n) path -------------------------------------------------
# Above MAX_RESAMPLE_ELEMENTS the resample matrix is drawn in row blocks off one
# rng instead of one call. These tests force that path with tiny thresholds and
# assert byte-identical output, which empirically pins the NumPy property the
# design rests on: sequential Generator.integers block draws concatenate to the
# same bit stream as a single big draw. If one of these ever fails on a NumPy
# upgrade, the chunked path's seed semantics must be re-declared (nothing
# recorded depends on the chunked stream today) — not silently patched.


def test_default_thresholds_keep_realistic_sizes_on_single_call_path():
    # The 5-core headline size (1000 x 228,153) must stay below the switch, or
    # recorded results / reproduce-headline byte_exact would move.
    assert 1000 * 228_153 < MAX_RESAMPLE_ELEMENTS
    assert RESAMPLE_BLOCK_ELEMENTS < MAX_RESAMPLE_ELEMENTS


def test_ci_mean_chunked_matches_single_call_byte_for_byte():
    rng = np.random.default_rng(6)
    values = rng.random(40)
    n_resamples = 251  # not a multiple of block_rows -> ragged final block

    single = ci_mean(values, n_resamples=n_resamples, seed=SEED)
    chunked = ci_mean(
        values,
        n_resamples=n_resamples,
        seed=SEED,
        max_resample_elements=100,  # 251*40 = 10,040 > 100 -> chunked
        resample_block_elements=100,  # block_rows = 100 // 40 = 2
    )
    assert chunked == single
    assert chunked["ci_lo"] == single["ci_lo"]
    assert chunked["ci_hi"] == single["ci_hi"]
    assert chunked["value"] == single["value"]


def test_ci_mean_chunked_single_row_blocks_match():
    # block_rows floors to 1 when n_users exceeds the block budget: the extreme
    # of the chunking (one integers() call per resample) must still match.
    rng = np.random.default_rng(7)
    values = rng.random(64)
    single = ci_mean(values, n_resamples=97, seed=SEED)
    chunked = ci_mean(
        values,
        n_resamples=97,
        seed=SEED,
        max_resample_elements=1,
        resample_block_elements=1,
    )
    assert chunked == single


def test_segment_cis_chunked_matches_single_call_byte_for_byte():
    rng = np.random.default_rng(8)
    labels = np.array(["a"] * 30 + ["b"] * 20 + ["solo"])
    values = rng.random(51)

    single = segment_cis(values, labels, n_resamples=133, seed=SEED)
    chunked = segment_cis(
        values,
        labels,
        n_resamples=133,
        seed=SEED,
        max_resample_elements=100,  # 133*30 and 133*20 both exceed it
        resample_block_elements=100,  # block_rows = 3 (seg a) / 5 (seg b)
    )
    assert chunked == single
    # Per-segment [seed, ordinal] streams are preserved, not collapsed.
    assert single["a"]["ci_lo"] != single["b"]["ci_lo"]
    assert chunked["solo"] == single["solo"]  # 1-user segment skips both paths


# --------------------------------------------------------------------------- #
# Bootstrap achieved-significance level (Phase 9, T9-3b preregistration §5e):
#
#   p = min(1, 2 * min((1 + #{D_b <= 0}) / (B+1), (1 + #{D_b >= 0}) / (B+1)))
#
# Pure post-processing of resample draws that already exist — never a second
# evaluation of TEST.
# --------------------------------------------------------------------------- #


def test_asl_p_value_hand_computed_fixture():
    # B = 10.  #{D <= 0} = 3 (-2, -1, 0), #{D >= 0} = 8 (0 and the seven > 0).
    # p = min(1, 2 * min(4/11, 9/11)) = 8/11.
    deltas = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert asl_p_value(deltas) == pytest.approx(8 / 11)


def test_asl_p_value_one_sided_extremes_are_add_one_smoothed():
    # All deltas strictly positive: #{D <= 0} = 0 -> 2 * 1/11.  Never exactly 0.
    assert asl_p_value(np.arange(1.0, 11.0)) == pytest.approx(2 / 11)
    # Mirror image, same magnitude: the p-value is direction-symmetric (§5f).
    assert asl_p_value(-np.arange(1.0, 11.0)) == pytest.approx(2 / 11)


def test_asl_p_value_is_capped_at_one():
    # A delta distribution straddling zero evenly doubles past 1 and is clipped.
    deltas = np.array([-1.0, -1.0, 1.0, 1.0])
    # #{<=0} = 2, #{>=0} = 2 -> 2 * 3/5 = 1.2 -> 1.0
    assert asl_p_value(deltas) == 1.0


def test_asl_p_value_all_zero_deltas_is_one():
    assert asl_p_value(np.zeros(1000)) == 1.0


def test_asl_p_value_resolution_floor_at_b_1000():
    # §5e's stated floor 1/1001 is the ONE-SIDED tail resolution; the two-sided
    # p bottoms out at 2/1001. Pinned so the reporting rule is not mis-applied.
    assert asl_p_value(np.ones(1000)) == pytest.approx(2 / 1001)


def test_asl_p_value_rejects_empty_and_non_finite_input():
    with pytest.raises(ValueError):
        asl_p_value(np.array([]))
    with pytest.raises(ValueError):
        asl_p_value(np.array([1.0, np.nan, -1.0]))


def test_asl_p_value_uses_the_same_draws_as_the_paired_ci():
    rng = np.random.default_rng(11)
    a = rng.random(120)
    b = rng.random(120)
    deltas = paired_delta_resamples(a, b, n_resamples=1000, seed=SEED)
    ci = paired_delta_ci(a, b, n_resamples=1000, seed=SEED)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # Same seed, same resample matrix: the CI is a function of these very draws.
    assert (float(lo), float(hi)) == (ci["ci_lo"], ci["ci_hi"])
    assert 0.0 < asl_p_value(deltas) <= 1.0

    # A constant positive offset: every resampled delta is +0.5, so the CI
    # excludes zero AND the ASL sits at its two-sided floor — CI and p-value
    # agree because they read the same draws.
    shifted = paired_delta_ci(b + 0.5, b, n_resamples=1000, seed=SEED)
    assert shifted["excludes_zero"] is True
    assert asl_p_value(
        paired_delta_resamples(b + 0.5, b, n_resamples=1000, seed=SEED)
    ) == pytest.approx(2 / 1001)


def test_paired_delta_resamples_is_deterministic_and_shaped():
    rng = np.random.default_rng(12)
    a, b = rng.random(40), rng.random(40)
    first = paired_delta_resamples(a, b, n_resamples=250, seed=SEED)
    second = paired_delta_resamples(a, b, n_resamples=250, seed=SEED)
    assert first.shape == (250,)
    assert np.array_equal(first, second)
