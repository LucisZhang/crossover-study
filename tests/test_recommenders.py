"""Tests for the three baseline ``Recommender`` implementations (Phase 2, T4).

Builds a tiny in-test ``EvalDataset`` directly (no Spark, no cache files) so
these run in plain ``pytest`` against pure numpy/scipy code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from batch_recsys_lab.eval.dataset import EvalDataset
from batch_recsys_lab.models.base import Recommender
from batch_recsys_lab.models.popularity import PopularityRecommender
from batch_recsys_lab.models.popularity_category import PopularityCategoryRecommender
from batch_recsys_lab.models.random_rec import RandomRecommender


def _toy_dataset() -> EvalDataset:
    # 3 users x 5 items, 2 categories: items 0,1,2 -> cat 0; items 3,4 -> cat 1.
    item_ids = np.array(["i0", "i1", "i2", "i3", "i4"], dtype=object)
    user_ids = np.array(["u0", "u1", "u2"], dtype=object)
    item_category_codes = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    category_names = ["cat0", "cat1"]

    # u0: 2 interactions in cat0 (items 0,1), 1 in cat1 (item 3) -> n_train=3
    # u1: 0 TRAIN interactions (strict-cold)
    # u2: 1 interaction in cat1 (item 4) -> n_train=1
    train_dense = np.array(
        [
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    train_csr = sp.csr_matrix(train_dense)
    n_train = train_dense.sum(axis=1).astype(np.int32)

    pop_vec = np.array([0.5, 0.3, 0.1, 0.05, 0.05], dtype=np.float32)

    return EvalDataset(
        cache_dir=Path("/nonexistent"),
        manifest={},
        item_ids=item_ids,
        user_ids=user_ids,
        n_train=n_train,
        train_csr=train_csr,
        pop={("train_end", 0): pop_vec},
        item_category_codes=item_category_codes,
        category_names=category_names,
        gt={},
    )


# --- protocol conformance -----------------------------------------------------


def test_all_models_satisfy_protocol():
    ds = _toy_dataset()
    for model in (
        RandomRecommender(seed=42),
        PopularityRecommender(as_of="train_end", window_days=0),
        PopularityCategoryRecommender(as_of="train_end", window_days=0),
    ):
        model.fit(ds)
        assert isinstance(model, Recommender)
        assert isinstance(model.name, str)
        assert isinstance(model.params, dict)


# --- RandomRecommender ---------------------------------------------------------


def test_random_scores_are_per_user_reproducible_across_batches():
    ds = _toy_dataset()
    model = RandomRecommender(seed=7)
    model.fit(ds)

    # user 0 alone
    row_alone = model.score_batch(np.array([0]))[0]
    # user 0 as part of a larger, differently-ordered batch
    row_in_batch = model.score_batch(np.array([2, 1, 0]))[2]

    np.testing.assert_array_equal(row_alone, row_in_batch)


def test_random_different_seeds_differ():
    ds = _toy_dataset()
    m1 = RandomRecommender(seed=1)
    m2 = RandomRecommender(seed=2)
    m1.fit(ds)
    m2.fit(ds)

    row1 = m1.score_batch(np.array([0]))[0]
    row2 = m2.score_batch(np.array([0]))[0]
    assert not np.array_equal(row1, row2)


def test_random_shape_and_dtype():
    ds = _toy_dataset()
    model = RandomRecommender(seed=0)
    model.fit(ds)
    out = model.score_batch(np.array([0, 1]))
    assert out.shape == (2, 5)
    assert out.dtype == np.float32


# --- PopularityRecommender ------------------------------------------------------


def test_popularity_scores_equal_pop_vector_for_every_user():
    ds = _toy_dataset()
    model = PopularityRecommender(as_of="train_end", window_days=0)
    model.fit(ds)
    out = model.score_batch(np.array([0, 1, 2]))
    pop_vec = ds.pop[("train_end", 0)]
    for row in out:
        np.testing.assert_array_equal(row, pop_vec)
    assert out.dtype == np.float32


def test_popularity_returns_writable_fresh_copy_each_call():
    ds = _toy_dataset()
    model = PopularityRecommender(as_of="train_end", window_days=0)
    model.fit(ds)

    out1 = model.score_batch(np.array([0]))
    assert out1.flags.writeable
    out1[0, 0] = -999.0  # mutate

    out2 = model.score_batch(np.array([0]))
    # mutating out1 must not have corrupted the cached vector / out2.
    assert out2[0, 0] != -999.0
    np.testing.assert_array_equal(out2[0], ds.pop[("train_end", 0)])


def test_popularity_missing_key_raises_clear_keyerror():
    ds = _toy_dataset()
    model = PopularityRecommender(as_of="val_end", window_days=90)
    with pytest.raises(KeyError, match="train_end"):
        model.fit(ds)


# --- PopularityCategoryRecommender ----------------------------------------------


def test_popularity_category_hand_computed_scores():
    ds = _toy_dataset()
    model = PopularityCategoryRecommender(as_of="train_end", window_days=0)
    model.fit(ds)

    out = model.score_batch(np.array([0, 1, 2]))
    pop = np.array([0.5, 0.3, 0.1, 0.05, 0.05], dtype=np.float32)

    # u0: cat0 count=2, cat1 count=1, total=3 -> w = [2/3, 1/3]
    # items 0,1,2 -> cat0 weight 2/3; items 3,4 -> cat1 weight 1/3.
    w0 = np.array([2 / 3, 2 / 3, 2 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    expected_u0 = w0 * pop
    np.testing.assert_allclose(out[0], expected_u0, rtol=1e-5)

    # u1: n_train == 0 -> strict-cold fallback to raw global pop vector.
    np.testing.assert_array_equal(out[1], pop)

    # u2: cat0 count=0, cat1 count=1, total=1 -> w = [0, 1]
    w2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    expected_u2 = w2 * pop
    np.testing.assert_allclose(out[2], expected_u2, rtol=1e-5)


def test_popularity_category_weights_sum_to_one_for_warm_users():
    ds = _toy_dataset()
    model = PopularityCategoryRecommender(as_of="train_end", window_days=0)
    model.fit(ds)

    sub = ds.train_csr[np.array([0, 2])]
    counts = np.asarray((sub @ model._cat_indicator).todense())
    row_sums = counts.sum(axis=1)
    weights = counts / row_sums[:, None]
    np.testing.assert_allclose(weights.sum(axis=1), [1.0, 1.0])


def test_popularity_category_shape_dtype():
    ds = _toy_dataset()
    model = PopularityCategoryRecommender(as_of="train_end", window_days=0)
    model.fit(ds)
    out = model.score_batch(np.array([0, 1, 2]))
    assert out.shape == (3, 5)
    assert out.dtype == np.float32
