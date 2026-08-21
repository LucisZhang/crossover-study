"""Tests for the ALS time-decay arm (Phase 8, T8-2; docs/engineering-log/EXPERIMENT_LOG.md 2026-08-17).

No Spark: the decay weighting is pure numpy (``models.als.time_decay_confidence``
+ ``models.als_train.training_ratings``) and the identity rules are pure hashing.

The load-bearing test here is (b): ``half_life_days`` must be invisible to the
param hash unless ``weighting == "time_decay"``, or every ALS artifact directory,
``als_manifest.json`` and recorded run in ``results/runs.jsonl`` would silently
stop resolving. The expected hashes below were computed on the pre-change code
(commit 9b7065c) and are hardcoded, not recomputed from the implementation under
test — a hash test that calls the code it guards proves nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from batch_recsys_lab.eval.harness import _build_model
from batch_recsys_lab.models.als import (
    ALSRecommender,
    als_param_hash,
    canonical_params,
    time_decay_confidence,
)
from batch_recsys_lab.models.als_train import training_ratings

# The Phase 3 chosen ALS config (rank=128, reg=0.01, alpha=10, max_iter=25) at
# each headline seed, and the tests/test_als.py toy params — hashes as produced
# BEFORE half_life_days existed.
FROZEN_HASHES = {
    (128, 0.01, 10.0, 25, "binary", 20260805): "36cb7b0cd328",
    (128, 0.01, 10.0, 25, "binary", 20260806): "bf9b729fd658",
    (128, 0.01, 10.0, 25, "binary", 20260807): "afb53ab82d49",
    (128, 0.01, 10.0, 15, "binary", 20260805): "1cf43592e59f",
    (128, 0.01, 10.0, 25, "rating", 20260805): "21146f8954a6",
    (128, 0.01, 10.0, 15, "rating", 20260805): "a5062ca20904",
    (3, 0.1, 40.0, 5, "binary", 7): "9afcef9ee7c5",
}


def _params(key) -> dict:
    rank, reg, alpha, iters, weighting, seed = key
    return dict(
        rank=rank,
        reg_param=reg,
        alpha=alpha,
        max_iter=iters,
        weighting=weighting,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# (a) decay formula: r(0) == 1, r(half-life) == 0.5, monotone, float32         #
# --------------------------------------------------------------------------- #
def test_decay_anchor_points_and_dtype():
    hl = 365.0
    r = time_decay_confidence(np.array([0.0, hl, 2 * hl, 3 * hl]), hl)
    assert r.dtype == np.float32
    assert r[0] == np.float32(1.0)  # age 0 == the binary baseline, exactly
    assert np.isclose(r[1], 0.5, atol=1e-6)
    assert np.isclose(r[2], 0.25, atol=1e-6)
    assert np.isclose(r[3], 0.125, atol=1e-6)


@pytest.mark.parametrize("hl", [90.0, 365.0, 1460.0])  # the preregistered grid
def test_decay_is_strictly_monotone_and_bounded(hl):
    ages = np.linspace(0.0, 2500.0, 501, dtype=np.float64)
    r = time_decay_confidence(ages, hl)
    assert np.all(np.diff(r) < 0)  # strictly decreasing in age
    assert r.max() <= 1.0 and r.min() > 0.0  # decay only REMOVES confidence
    # Fractional ages are honoured (no day bucketing).
    assert time_decay_confidence(np.array([0.5 * hl]), hl)[0] > time_decay_confidence(
        np.array([0.51 * hl]), hl
    )[0]


def test_decay_rejects_bad_inputs():
    with pytest.raises(ValueError):
        time_decay_confidence(np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        time_decay_confidence(np.array([1.0]), -5.0)
    with pytest.raises(ValueError):
        time_decay_confidence(np.array([-1.0]), 365.0)  # age before train_end
    with pytest.raises(ValueError):
        time_decay_confidence(np.array([np.nan]), 365.0)


# --------------------------------------------------------------------------- #
# (b) hash back-compat: binary/rating identity is byte-identical to pre-T8-2   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", list(FROZEN_HASHES))
def test_existing_param_hashes_unchanged(key):
    """Hardcoded pre-change hashes — the artifact/manifest identity contract."""
    assert als_param_hash(_params(key)) == FROZEN_HASHES[key]


@pytest.mark.parametrize("key", list(FROZEN_HASHES))
def test_half_life_is_ignored_for_binary_and_rating(key):
    """A stray half_life_days must not perturb a non-decay config's identity —
    neither the canonical dict nor the hash."""
    base = _params(key)
    canon = canonical_params(**base)
    assert "half_life_days" not in canon
    assert canonical_params(**base, half_life_days=365.0) == canon
    assert als_param_hash({**base, "half_life_days": 365.0}) == FROZEN_HASHES[key]


def test_recommender_params_for_binary_are_unchanged():
    model = ALSRecommender(**_params((3, 0.1, 40.0, 5, "binary", 7)))
    assert model.params == {
        "rank": 3,
        "reg_param": 0.1,
        "alpha": 40.0,
        "max_iter": 5,
        "weighting": "binary",
        "seed": 7,
    }


# --------------------------------------------------------------------------- #
# (c) time_decay identity: half-life bearing, distinct from binary            #
# --------------------------------------------------------------------------- #
def test_time_decay_hash_varies_with_half_life_and_differs_from_binary():
    base = dict(rank=128, reg_param=0.01, alpha=10.0, max_iter=25, seed=20260805)
    hashes = {
        hl: als_param_hash({**base, "weighting": "time_decay", "half_life_days": hl})
        for hl in (90, 365, 1460)
    }
    assert len(set(hashes.values())) == 3  # each grid point gets its own artifact
    binary = als_param_hash({**base, "weighting": "binary"})
    assert binary not in set(hashes.values())
    assert binary == FROZEN_HASHES[(128, 0.01, 10.0, 25, "binary", 20260805)]
    # int/float drift on the same half-life is the same identity.
    assert hashes[365] == als_param_hash(
        {**base, "weighting": "time_decay", "half_life_days": 365.0}
    )


def test_time_decay_canonical_params_carry_the_half_life():
    canon = canonical_params(
        rank=8, reg_param=0.05, alpha=40.0, max_iter=12, weighting="time_decay", seed=13,
        half_life_days=90,
    )
    assert canon["half_life_days"] == 90.0
    assert isinstance(canon["half_life_days"], float)


@pytest.mark.parametrize("hl", [None, 0, -1, float("inf")])
def test_time_decay_requires_a_positive_finite_half_life(hl):
    kw = dict(rank=8, reg_param=0.05, alpha=40.0, max_iter=12, weighting="time_decay", seed=13)
    with pytest.raises(ValueError):
        canonical_params(**kw, half_life_days=hl)


# --------------------------------------------------------------------------- #
# (d) harness wiring                                                          #
# --------------------------------------------------------------------------- #
def test_build_model_threads_half_life_through():
    model_cfg = {
        "name": "als",
        "params": {
            "rank": 128,
            "reg_param": 0.01,
            "alpha": 10.0,
            "max_iter": 25,
            "weighting": "time_decay",
            "half_life_days": 90,
        },
    }
    model = _build_model(model_cfg, {"model": 20260805})
    assert model.params["half_life_days"] == 90.0
    assert model.half_life_days == 90.0

    # …and a binary config built through the same path keeps its frozen hash.
    binary_cfg = {
        "name": "als",
        "params": {
            "rank": 128,
            "reg_param": 0.01,
            "alpha": 10.0,
            "max_iter": 25,
            "weighting": "binary",
        },
    }
    binary = _build_model(binary_cfg, {"model": 20260805})
    assert als_param_hash(binary.params) == FROZEN_HASHES[
        (128, 0.01, 10.0, 25, "binary", 20260805)
    ]


# --------------------------------------------------------------------------- #
# (e) Step A rating construction (pure numpy half of als_train)                #
# --------------------------------------------------------------------------- #
def _cache(tmp_path: Path, ages: np.ndarray | None, n_pairs: int) -> Path:
    cache_dir = tmp_path / "cache" / "424242"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "train_user_idx.npy", np.arange(n_pairs, dtype=np.int32))
    np.save(cache_dir / "train_item_idx.npy", np.zeros(n_pairs, dtype=np.int32))
    np.save(cache_dir / "train_rating.npy", np.full(n_pairs, 4.0, dtype=np.float32))
    if ages is not None:
        np.save(cache_dir / "train_age_days.npy", ages.astype(np.float32))
    (cache_dir / "cache_manifest.json").write_text(json.dumps({"schema_version": 2}))
    return cache_dir


def test_training_ratings_time_decay_matches_the_formula(tmp_path):
    ages = np.array([0.0, 365.0, 730.0, 1000.0], dtype=np.float32)
    cache_dir = _cache(tmp_path, ages, len(ages))
    r = training_ratings(cache_dir, "time_decay", len(ages), half_life_days=365.0)
    assert r.dtype == np.float32
    assert np.allclose(r, time_decay_confidence(ages, 365.0))
    assert r[0] == np.float32(1.0)


def test_training_ratings_binary_and_rating_are_untouched(tmp_path):
    cache_dir = _cache(tmp_path, None, 4)
    assert np.array_equal(
        training_ratings(cache_dir, "binary", 4), np.ones(4, dtype=np.float32)
    )
    assert np.array_equal(
        training_ratings(cache_dir, "rating", 4), np.full(4, 4.0, dtype=np.float32)
    )


def test_training_ratings_missing_age_file_names_the_make_target(tmp_path):
    cache_dir = _cache(tmp_path, None, 4)
    with pytest.raises(FileNotFoundError, match="make extract-age"):
        training_ratings(cache_dir, "time_decay", 4, half_life_days=365.0)


def test_training_ratings_rejects_misaligned_age_array(tmp_path):
    cache_dir = _cache(tmp_path, np.zeros(3, dtype=np.float32), 4)
    with pytest.raises(ValueError, match="not aligned"):
        training_ratings(cache_dir, "time_decay", 4, half_life_days=365.0)
