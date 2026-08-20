"""Tests for the T9-3b ``policy.select`` extensions (Rule S5, ML-32M):
``route_by: n_train`` and ``objective: global_ndcg10_mean``. Default-path
(no route_by/objective keys) behavior must stay byte-identical to the
Amazon T13 config's segment-based routing.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batch_recsys_lab.policy.select import (
    _compose_hybrid_n_train,
    _global_objective,
    select,
)

# 6 users spanning every frozen segment bucket, plus a straddling n_train
# value (50) that does NOT align with any segment edge — exercises exactly
# why route_by: n_train exists.
USER_IDS = ["u0", "u1", "u2", "u3", "u4", "u5"]
USER_IDX = [3, 0, 5, 1, 4, 2]  # deliberately non-identity, tests the join
SEGMENTS = ["0", "1-4", "5-9", "10-19", "20+", "20+"]
N_TRAIN_BY_IDX = np.array([70, 0, 2, 7, 25, 12], dtype=np.int32)  # indexed by user_idx

LOW_VALUES = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
HIGH_VALUES = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55])

RUN_IDS = {
    "knn_t12m": "20260101T000000Z-aaaaaaa",
    "pop_t12m": "20260101T000001Z-bbbbbbb",
}


def _write_arm(tmp_path, key: str, values: np.ndarray):
    path = tmp_path / f"{key}.parquet"
    cols = {
        "user_id": pa.array(USER_IDS, type=pa.string()),
        "user_idx": pa.array(USER_IDX, type=pa.int32()),
        "segment": pa.array(SEGMENTS, type=pa.string()),
        "ndcg@10": pa.array(values, type=pa.float64()),
    }
    pq.write_table(pa.table(cols), path)
    return path


def _write_results(tmp_path, low_path, high_path):
    results = tmp_path / "runs.jsonl"
    lines = [
        json.dumps(
            {
                "kind": "eval",
                "run_id": RUN_IDS["knn_t12m"],
                "model": {"name": "knn_t12m"},
                "per_user_artifact": str(low_path),
            }
        ),
        json.dumps(
            {
                "kind": "eval",
                "run_id": RUN_IDS["pop_t12m"],
                "model": {"name": "pop_t12m"},
                "per_user_artifact": str(high_path),
            }
        ),
    ]
    results.write_text("\n".join(lines) + "\n")
    return results


def _write_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cache_manifest.json").write_text("{}")
    np.save(cache_dir / "n_train.npy", N_TRAIN_BY_IDX)
    return cache_dir


# --- _compose_hybrid_n_train: unit-level routing math ------------------------


def test_compose_hybrid_n_train_edge_at_fifty():
    user_n_train = N_TRAIN_BY_IDX[USER_IDX]  # [7, 70, 12, 0, 25, 2]
    composed = _compose_hybrid_n_train(LOW_VALUES, HIGH_VALUES, user_n_train, 50)
    # n_train < 50 -> low; n_train >= 50 -> high. Only u1 (n_train=70) is high.
    expected = np.array(
        [LOW_VALUES[0], HIGH_VALUES[1], LOW_VALUES[2], LOW_VALUES[3], LOW_VALUES[4], LOW_VALUES[5]]
    )
    np.testing.assert_array_equal(composed, expected)


def test_compose_hybrid_n_train_inf_routes_everyone_low():
    user_n_train = N_TRAIN_BY_IDX[USER_IDX]
    composed = _compose_hybrid_n_train(LOW_VALUES, HIGH_VALUES, user_n_train, None)
    np.testing.assert_array_equal(composed, LOW_VALUES)


# --- _global_objective --------------------------------------------------------


def test_global_objective_is_plain_mean():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    assert _global_objective(values) == pytest.approx(0.25)


# --- select(): route_by=n_train + objective=global_ndcg10_mean end-to-end ----


def test_select_n_train_routing_end_to_end(tmp_path):
    low_path = _write_arm(tmp_path, "knn_t12m", LOW_VALUES)
    high_path = _write_arm(tmp_path, "pop_t12m", HIGH_VALUES)
    results = _write_results(tmp_path, low_path, high_path)
    cache_dir = _write_cache(tmp_path)

    config = {
        "run_ids": dict(RUN_IDS),
        "n_star_grid": [1, 5, 10, 20, 50, 100, None],
        "variants": {"A": {"low": "knn_t12m", "high": "pop_t12m"}},
        "route_by": "n_train",
        "cache_dir": str(cache_dir),
        "objective": "global_ndcg10_mean",
        "metric": "ndcg@10",
    }
    result = select(config, results)

    cells = {c["n_star_label"]: c for c in result["grid"]}
    # user_idx order (post-align, sorted by user_id u0..u5) maps to n_train
    # [7, 70, 12, 0, 25, 2] via USER_IDX; verify a straddling edge (50) and
    # the segment-approximation-would-differ edge (10, since u2 has
    # n_train=12 in segment "5-9" whose min is 5 < 10, but 12 >= 10 so exact
    # n_train routing puts u2 in the HIGH arm while segment routing on "5-9"
    # would have put it LOW).
    n_star_50 = cells["50"]
    composed_50 = _compose_hybrid_n_train(LOW_VALUES, HIGH_VALUES, N_TRAIN_BY_IDX[USER_IDX], 50)
    assert n_star_50["objective"] == pytest.approx(_global_objective(composed_50))

    n_star_10 = cells["10"]
    composed_10 = _compose_hybrid_n_train(LOW_VALUES, HIGH_VALUES, N_TRAIN_BY_IDX[USER_IDX], 10)
    assert n_star_10["objective"] == pytest.approx(_global_objective(composed_10))
    # exact n_train routing differs from a naive segment routing at n_star=10
    # for at least one user (u2: n_train=12, segment "5-9"): confirms this
    # test would fail if route_by silently fell back to segment routing.
    assert composed_10[2] == HIGH_VALUES[2]

    assert result["objective"] == "global_ndcg10_mean"


def test_select_n_train_routing_requires_cache_dir(tmp_path):
    low_path = _write_arm(tmp_path, "knn_t12m", LOW_VALUES)
    high_path = _write_arm(tmp_path, "pop_t12m", HIGH_VALUES)
    results = _write_results(tmp_path, low_path, high_path)

    config = {
        "run_ids": dict(RUN_IDS),
        "n_star_grid": [10],
        "variants": {"A": {"low": "knn_t12m", "high": "pop_t12m"}},
        "route_by": "n_train",
    }
    with pytest.raises(ValueError, match="cache_dir"):
        select(config, results)


# --- default path: segment routing / segment-weighted objective unchanged ---


def test_select_default_path_matches_segment_routing(tmp_path):
    low_path = _write_arm(tmp_path, "knn_t12m", LOW_VALUES)
    high_path = _write_arm(tmp_path, "pop_t12m", HIGH_VALUES)
    results = _write_results(tmp_path, low_path, high_path)

    config = {
        "run_ids": dict(RUN_IDS),
        "n_star_grid": [1, 5, 10, 20, None],
        "variants": {"A": {"low": "knn_t12m", "high": "pop_t12m"}},
        "metric": "ndcg@10",
    }
    result = select(config, results)
    assert result["objective"] == "segment_weighted_ndcg10_unweighted_mean"

    # Segment routing at n_star=10: SEGMENT_MIN_N_TRAIN["5-9"]=5 < 10 -> low,
    # so this is the exact regime where segment and n_train routing would
    # diverge for u2 (n_train=12) -- default path must still use the
    # segment bucket, not n_train.
    cell_10 = next(c for c in result["grid"] if c["n_star_label"] == "10")
    seg_means = cell_10["segment_means"]
    # segment "5-9" (u2 only) routes LOW under the default segment rule.
    assert seg_means["5-9"] == pytest.approx(LOW_VALUES[USER_IDS.index("u2")])
