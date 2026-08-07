"""Tests for the history-depth routing policy (Phase 4, T13).

Builds a tiny in-test ``EvalDataset`` (mirrors ``tests/test_recommenders.py``)
so these run in plain ``pytest`` against pure numpy/scipy code, no Spark/cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from batch_recsys_lab.eval.dataset import EvalDataset
from batch_recsys_lab.models.base import Recommender
from batch_recsys_lab.policy.hybrid import HybridRecommender


def _toy_dataset() -> EvalDataset:
    # 4 users x 5 items. n_train: u0=0, u1=2, u2=7, u3=25 (spans every
    # SEGMENT_LABELS bucket: 0, 1-4, 5-9, 20+).
    item_ids = np.array(["i0", "i1", "i2", "i3", "i4"], dtype=object)
    user_ids = np.array(["u0", "u1", "u2", "u3"], dtype=object)

    n_train = np.array([0, 2, 7, 25], dtype=np.int32)
    train_csr = sp.csr_matrix((4, 5), dtype=np.float32)  # exclusion irrelevant here

    low_pop = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)  # "low" arm
    high_pop = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)  # "high" arm

    return EvalDataset(
        cache_dir=Path("/nonexistent"),
        manifest={},
        item_ids=item_ids,
        user_ids=user_ids,
        n_train=n_train,
        train_csr=train_csr,
        pop={("low", 0): low_pop, ("high", 0): high_pop},
        item_category_codes=None,
        category_names=[],
        gt={},
    )


_LOW_CFG = {"name": "popularity", "params": {"as_of": "low", "window_days": 0}}
_HIGH_CFG = {"name": "popularity", "params": {"as_of": "high", "window_days": 0}}


def test_hybrid_satisfies_protocol():
    ds = _toy_dataset()
    model = HybridRecommender(n_star=10, low=_LOW_CFG, high=_HIGH_CFG)
    model.fit(ds)
    assert isinstance(model, Recommender)
    assert isinstance(model.name, str)
    assert isinstance(model.params, dict)


def test_routing_below_and_above_n_star():
    ds = _toy_dataset()
    # n_star=10 -> n_train {0,2,7} < 10 route to low; {25} >= 10 routes to high.
    model = HybridRecommender(n_star=10, low=_LOW_CFG, high=_HIGH_CFG)
    model.fit(ds)
    scores = model.score_batch(np.array([0, 1, 2, 3]))

    low_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    high_vec = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)

    np.testing.assert_array_equal(scores[0], low_vec)   # n_train=0  < 10
    np.testing.assert_array_equal(scores[1], low_vec)   # n_train=2  < 10
    np.testing.assert_array_equal(scores[2], low_vec)   # n_train=7  < 10
    np.testing.assert_array_equal(scores[3], high_vec)  # n_train=25 >= 10


def test_n_star_one_only_strict_cold_gets_low():
    ds = _toy_dataset()
    model = HybridRecommender(n_star=1, low=_LOW_CFG, high=_HIGH_CFG)
    model.fit(ds)
    scores = model.score_batch(np.array([0, 1, 2, 3]))

    low_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    high_vec = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)

    np.testing.assert_array_equal(scores[0], low_vec)   # n_train=0  < 1
    np.testing.assert_array_equal(scores[1], high_vec)  # n_train=2  >= 1
    np.testing.assert_array_equal(scores[2], high_vec)  # n_train=7  >= 1
    np.testing.assert_array_equal(scores[3], high_vec)  # n_train=25 >= 1


@pytest.mark.parametrize("n_star", [None, "inf"])
def test_n_star_inf_routes_everyone_to_low(n_star):
    ds = _toy_dataset()
    model = HybridRecommender(n_star=n_star, low=_LOW_CFG, high=_HIGH_CFG)
    model.fit(ds)
    scores = model.score_batch(np.array([0, 1, 2, 3]))

    low_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    for row in scores:
        np.testing.assert_array_equal(row, low_vec)


def test_score_batch_returns_fresh_writable_array():
    ds = _toy_dataset()
    model = HybridRecommender(n_star=10, low=_LOW_CFG, high=_HIGH_CFG)
    model.fit(ds)
    scores = model.score_batch(np.array([0, 1, 2, 3]))
    assert scores.flags.writeable
    original = scores.copy()
    scores[0, 0] = -np.inf  # must not alias any cached vector / other row
    np.testing.assert_array_equal(scores[1], original[1])
    assert scores.dtype == np.float32


def test_hybrid_registered_in_build_model():
    from batch_recsys_lab.eval.harness import _build_model

    model_cfg = {
        "name": "hybrid",
        "params": {"n_star": 10, "low": _LOW_CFG, "high": _HIGH_CFG},
    }
    model = _build_model(model_cfg, seeds={})
    assert isinstance(model, HybridRecommender)
