"""Tests for the item-kNN trailing-window arm (Phase 8, T8-2; kNN-t12m).

Pure numpy/scipy. The toy fixture is the same 3-users x 4-items matrix
``tests/test_item_knn.py`` hand-computes, with per-pair ages attached so the
window can be shown to change the similarity matrix in exactly the way the
preregistration describes: item-side co-occurrence is filtered, user profile
vectors at scoring time are NOT.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from batch_recsys_lab.eval.dataset import EvalDataset, attach_train_pairs, load_dataset
from batch_recsys_lab.eval.harness import _build_model
from batch_recsys_lab.models.item_knn import (
    ItemKNNRecommender,
    build_similarity,
    build_windowed_csr,
)

# TRAIN pairs (user, item) with ages in days before train_end:
#   u0: {0 @ 10d, 1 @ 10d, 2 @ 400d}
#   u1: {0 @ 20d, 1 @ 30d}
#   u2: {1 @ 500d, 2 @ 5d, 3 @ 5d}
PAIR_USERS = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
PAIR_ITEMS = np.array([0, 1, 2, 0, 1, 1, 2, 3], dtype=np.int32)
PAIR_AGES = np.array([10.0, 10.0, 400.0, 20.0, 30.0, 500.0, 5.0, 5.0], dtype=np.float32)
SHAPE = (3, 4)


def _full_csr() -> sp.csr_matrix:
    return build_windowed_csr(PAIR_USERS, PAIR_ITEMS, PAIR_AGES, np.inf, SHAPE)


def _ds(train_csr: sp.csr_matrix, *, with_pairs: bool = True) -> EvalDataset:
    n_users, n_items = train_csr.shape
    return EvalDataset(
        cache_dir=None,
        manifest={},
        item_ids=np.array([str(i) for i in range(n_items)], dtype=object),
        user_ids=np.array([str(u) for u in range(n_users)], dtype=object),
        n_train=np.asarray(train_csr.sum(axis=1)).ravel().astype(np.int32),
        train_csr=train_csr,
        item_category_codes=np.zeros(n_items, dtype=np.int32),
        train_user_idx=PAIR_USERS if with_pairs else None,
        train_item_idx=PAIR_ITEMS if with_pairs else None,
        train_age_days=PAIR_AGES if with_pairs else None,
    )


# --------------------------------------------------------------------------- #
# 1. windowed CSR: strict `<` boundary, same shape, binary                     #
# --------------------------------------------------------------------------- #
def test_windowed_csr_keeps_only_in_window_pairs():
    csr = build_windowed_csr(PAIR_USERS, PAIR_ITEMS, PAIR_AGES, 365.0, SHAPE)
    assert csr.shape == SHAPE
    assert csr.dtype == np.float32
    assert set(zip(*csr.nonzero())) == {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3)}
    assert np.all(csr.data == 1.0)
    # The two stale pairs (u0,i2 @400d and u2,i1 @500d) are gone.
    assert csr[0, 2] == 0.0 and csr[2, 1] == 0.0


def test_window_boundary_is_strict_less_than():
    """age == window is OUT; one epsilon younger is IN. This is what makes the
    window identical to the frozen pop-t12m boundary (ts > as_of - window)."""
    users = np.array([0, 0], dtype=np.int32)
    items = np.array([0, 1], dtype=np.int32)
    ages = np.array([365.0, np.nextafter(np.float32(365.0), np.float32(0.0))], dtype=np.float32)
    csr = build_windowed_csr(users, items, ages, 365.0, (1, 2))
    assert csr[0, 0] == 0.0  # exactly at the boundary -> excluded
    assert csr[0, 1] == 1.0  # just inside -> kept


def test_windowed_csr_wide_window_reproduces_the_full_train_matrix():
    full = _full_csr()
    wide = build_windowed_csr(PAIR_USERS, PAIR_ITEMS, PAIR_AGES, 10_000.0, SHAPE)
    assert np.array_equal(wide.indptr, full.indptr)
    assert np.array_equal(wide.indices, full.indices)
    assert np.array_equal(wide.data, full.data)


def test_windowed_csr_rejects_misaligned_columns():
    with pytest.raises(ValueError, match="disagree in length"):
        build_windowed_csr(PAIR_USERS, PAIR_ITEMS, PAIR_AGES[:-1], 365.0, SHAPE)


# --------------------------------------------------------------------------- #
# 2. fit(): similarity from the window, scoring from the FULL profile          #
# --------------------------------------------------------------------------- #
def test_similarity_is_built_from_the_window_only():
    ds = _ds(_full_csr())
    windowed = ItemKNNRecommender(top_n=10, train_window_days=365).fit(ds)
    expected = build_similarity(
        build_windowed_csr(PAIR_USERS, PAIR_ITEMS, PAIR_AGES, 365.0, SHAPE), top_n=10
    )
    assert np.allclose(windowed.S.toarray(), expected.toarray())

    # It really differs from the all-history matrix: items 1 and 2 no longer
    # co-occur (u0's i2 and u2's i1 are both out of window).
    all_history = ItemKNNRecommender(top_n=10).fit(_ds(_full_csr()))
    assert all_history.S.toarray()[1, 2] > 0.0
    assert windowed.S.toarray()[1, 2] == 0.0


def test_scoring_uses_the_full_train_profile_not_the_window():
    ds = _ds(_full_csr())
    model = ItemKNNRecommender(top_n=10, train_window_days=365).fit(ds)
    scores = model.score_batch(np.array([0, 1, 2]))
    # Reference: FULL train rows (including the out-of-window interactions)
    # multiplied by the WINDOWED similarity matrix.
    expected = _full_csr().toarray() @ model.S.toarray()
    assert np.allclose(scores, expected, atol=1e-6)
    # u2's profile still contains item 1 (age 500d) even though that pair built
    # no similarity: dropping it from the profile would change u2's scores.
    profile_dropped = build_windowed_csr(
        PAIR_USERS, PAIR_ITEMS, PAIR_AGES, 365.0, SHAPE
    ).toarray() @ model.S.toarray()
    assert not np.allclose(scores[2], profile_dropped[2])


def test_window_zero_is_byte_identical_to_current_behavior():
    train_csr = _full_csr()
    baseline = ItemKNNRecommender(top_n=10).fit(_ds(train_csr, with_pairs=False))
    explicit_zero = ItemKNNRecommender(top_n=10, train_window_days=0).fit(
        _ds(train_csr, with_pairs=False)
    )
    for attr in ("indptr", "indices", "data"):
        assert np.array_equal(
            getattr(baseline.S, attr), getattr(explicit_zero.S, attr)
        )
    # …and the recorded params stay exactly what pre-Phase-8 records carry.
    assert baseline.params == explicit_zero.params
    assert baseline.params == {"top_n": 10, "shrinkage": 0.0, "block_size": 8192}


def test_window_is_recorded_in_params_when_in_force():
    model = ItemKNNRecommender(top_n=50, train_window_days=365)
    assert model.params["train_window_days"] == 365


def test_negative_window_is_rejected():
    with pytest.raises(ValueError, match="train_window_days"):
        ItemKNNRecommender(top_n=10, train_window_days=-1)


# --------------------------------------------------------------------------- #
# 3. harness wiring + the missing-age-array error path                         #
# --------------------------------------------------------------------------- #
def test_build_model_threads_train_window_days():
    cfg = {
        "name": "item_knn",
        "params": {"top_n": 50, "shrinkage": 0.0, "train_window_days": 365},
    }
    model = _build_model(cfg, {"model": None})
    assert model.train_window_days == 365
    # Absent -> 0, i.e. the pre-Phase-8 arm.
    legacy = _build_model({"name": "item_knn", "params": {"top_n": 50}}, {"model": None})
    assert legacy.train_window_days == 0
    assert "train_window_days" not in legacy.params


def _write_min_cache(cache_dir: Path, ages: np.ndarray | None) -> Path:
    """Smallest cache `load_dataset` accepts, carrying the toy TRAIN pairs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cache_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"item_id": pa.array([str(i) for i in range(SHAPE[1])])}),
        cache_dir / "item_ids.parquet",
    )
    pq.write_table(
        pa.table({"user_id": pa.array([str(u) for u in range(SHAPE[0])])}),
        cache_dir / "user_ids.parquet",
    )
    np.save(cache_dir / "n_train.npy", np.zeros(SHAPE[0], dtype=np.int32))
    np.save(cache_dir / "train_user_idx.npy", PAIR_USERS)
    np.save(cache_dir / "train_item_idx.npy", PAIR_ITEMS)
    np.save(cache_dir / "item_category_codes.npy", np.zeros(SHAPE[1], dtype=np.int32))
    (cache_dir / "item_category_names.json").write_text(json.dumps(["__unknown__"]))
    (cache_dir / "cache_manifest.json").write_text(json.dumps({"schema_version": 2}))
    if ages is not None:
        np.save(cache_dir / "train_age_days.npy", ages)
    return cache_dir


def test_missing_age_array_is_a_hard_error_naming_the_make_target(tmp_path):
    ds = load_dataset(_write_min_cache(tmp_path / "cache", None))
    assert ds.train_age_days is None
    with pytest.raises(FileNotFoundError, match="make extract-age"):
        ItemKNNRecommender(top_n=10, train_window_days=365).fit(ds)
    # A window of 0 does not need the file at all.
    ItemKNNRecommender(top_n=10).fit(ds)


def test_loader_exposes_ages_when_present(tmp_path):
    cache_dir = _write_min_cache(tmp_path / "cache", PAIR_AGES)
    lazy = load_dataset(cache_dir)
    assert lazy.train_age_days is None  # optional: not loaded by default
    eager = load_dataset(cache_dir, with_train_pairs=True)
    assert np.array_equal(eager.train_age_days, PAIR_AGES)
    assert np.array_equal(eager.train_user_idx, PAIR_USERS)
    attach_train_pairs(lazy, require_age=True)
    assert np.array_equal(lazy.train_age_days, PAIR_AGES)


def test_loader_rejects_a_misaligned_age_array(tmp_path):
    cache_dir = _write_min_cache(tmp_path / "cache", PAIR_AGES[:-1])
    with pytest.raises(ValueError, match="not aligned"):
        load_dataset(cache_dir, with_train_pairs=True)
