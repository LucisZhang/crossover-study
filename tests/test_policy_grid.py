"""Tests for the n* TEST grid recomposition (Phase 6, T27).

Pure numpy/pyarrow: builds a tiny synthetic per-user substrate (3 arms x 6
users spanning every frozen segment bucket) plus a synthetic results log, so
the routing composition and the hard identity assertions are exercised without
any Spark, cache, or real artifact.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batch_recsys_lab.eval.bootstrap import ci_mean, segment_cis
from batch_recsys_lab.policy import grid_test
from batch_recsys_lab.policy.select import _compose_hybrid

METRICS = ["ndcg@10", "recall@20"]
SEED = 20260805
N_RESAMPLES = 25  # keep the synthetic bootstrap cheap; protocol shape is identical

# 6 users, one per frozen bucket edge (two in 1-4 so segments are not singletons).
USER_IDS = ["u0", "u1", "u2", "u3", "u4", "u5"]
SEGMENTS = np.array(["0", "1-4", "1-4", "5-9", "10-19", "20+"])

ARM_VALUES = {
    "blend_a30": {
        "ndcg@10": np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60]),
        "recall@20": np.array([0.11, 0.21, 0.31, 0.41, 0.51, 0.61]),
    },
    "als_chosen": {
        "ndcg@10": np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.90]),
        "recall@20": np.array([0.00, 0.03, 0.05, 0.07, 0.09, 0.91]),
    },
    "pop_t12m": {
        "ndcg@10": np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55]),
        "recall@20": np.array([0.06, 0.16, 0.26, 0.36, 0.46, 0.56]),
    },
}

RUN_IDS = {
    "blend_a30": "20260101T000000Z-aaaaaaa",
    "als_chosen": "20260101T000001Z-bbbbbbb",
    "pop_t12m": "20260101T000002Z-ccccccc",
}


def _write_arm(tmp_path, key: str):
    path = tmp_path / f"{key}.parquet"
    cols = {
        "user_id": pa.array(USER_IDS, type=pa.string()),
        "segment": pa.array(list(SEGMENTS), type=pa.string()),
    }
    for m in METRICS:
        cols[m] = pa.array(ARM_VALUES[key][m], type=pa.float64())
    pq.write_table(pa.table(cols), path)
    return path


def _metrics_block(key: str) -> dict:
    """The metrics block a real eval run would have recorded for this arm."""
    g, ps = grid_test._metric_blocks(ARM_VALUES[key], SEGMENTS, N_RESAMPLES, SEED)
    return {"global": g, "per_segment": ps}


def _substrate(tmp_path) -> tuple:
    results = tmp_path / "runs.jsonl"
    lines = []
    for key, run_id in RUN_IDS.items():
        path = _write_arm(tmp_path, key)
        lines.append(
            json.dumps(
                {
                    "kind": "eval",
                    "run_id": run_id,
                    "model": {"name": key},
                    "protocol": {"eval_split": "test"},
                    "seeds": {"bootstrap": SEED},
                    "metrics": _metrics_block(key),
                    "per_user_artifact": str(path),
                }
            )
        )
    results.write_text("\n".join(lines) + "\n")
    config = {
        "run_ids": dict(RUN_IDS),
        "n_star_grid": [0, 1, 5, 10, 20, None],
        "variants": {
            "A": {"low": "blend_a30", "high": "als_chosen"},
            "B": {"low": "blend_a30", "high": "pop_t12m"},
        },
        "metrics": METRICS,
        "identity_checks": {
            "inf_reference": "blend_a30",
            "zero_reference": {"A": "als_chosen", "B": "pop_t12m"},
        },
        "bootstrap": {"n_resamples": N_RESAMPLES, "seed": SEED},
        "split": "test",
        "expected_n_users": len(USER_IDS),
    }
    return config, results


# --- routing composition ------------------------------------------------------


@pytest.mark.parametrize(
    ("n_star", "expected_low_mask"),
    [
        (0, [False, False, False, False, False, False]),  # everyone -> high arm
        (1, [True, False, False, False, False, False]),   # only the cold bucket
        (5, [True, True, True, False, False, False]),     # 0 + 1-4
        (10, [True, True, True, True, False, False]),     # + 5-9
        (20, [True, True, True, True, True, False]),      # + 10-19
        (None, [True, True, True, True, True, True]),     # inf -> everyone low
    ],
)
def test_segment_edge_routing(n_star, expected_low_mask):
    low = ARM_VALUES["blend_a30"]["ndcg@10"]
    high = ARM_VALUES["pop_t12m"]["ndcg@10"]
    composed = _compose_hybrid(low, high, SEGMENTS, n_star)
    expected = np.where(np.array(expected_low_mask), low, high)
    np.testing.assert_array_equal(composed, expected)


def test_grid_shape_and_low_share(tmp_path):
    config, results = _substrate(tmp_path)
    grid = grid_test.build_grid(config, results)
    assert len(grid["cells"]) == 12
    assert grid["n_users"] == 6
    labels = [(c["variant"], c["n_star_label"]) for c in grid["cells"]]
    assert labels == [
        (v, lbl) for v in ("A", "B") for lbl in ("0", "1", "5", "10", "20", "inf")
    ]
    shares = {
        (c["variant"], c["n_star_label"]): c["low_share"] for c in grid["cells"]
    }
    assert shares[("B", "0")] == 0.0
    assert shares[("B", "inf")] == 1.0
    assert shares[("B", "1")] == pytest.approx(1 / 6)
    assert shares[("B", "5")] == pytest.approx(3 / 6)


def test_identity_checks_pass_on_consistent_substrate(tmp_path):
    config, results = _substrate(tmp_path)
    grid = grid_test.build_grid(config, results)
    assert grid["identity_diffs"] == []
    assert grid["identity_checked"]["inf"] == ["A/inf==blend_a30", "B/inf==blend_a30"]
    assert grid["identity_checked"]["zero"] == ["A/0==als_chosen", "B/0==pop_t12m"]

    # The inf cell must literally be the blend arm's own recorded floats.
    cell = next(c for c in grid["cells"] if (c["variant"], c["n_star_label"]) == ("B", "inf"))
    expected = ci_mean(ARM_VALUES["blend_a30"]["ndcg@10"], N_RESAMPLES, SEED)
    assert cell["global"]["ndcg@10"] == expected
    seg = segment_cis(ARM_VALUES["blend_a30"]["recall@20"], SEGMENTS, N_RESAMPLES, SEED)
    assert cell["per_segment"]["5-9"]["recall@20"]["value"] == seg["5-9"]["value"]


def test_identity_check_detects_perturbed_record(tmp_path):
    config, results = _substrate(tmp_path)
    lines = [json.loads(x) for x in results.read_text().splitlines()]
    for rec in lines:
        if rec["run_id"] == RUN_IDS["pop_t12m"]:
            rec["metrics"]["global"]["ndcg@10"]["value"] += 1e-12
    results.write_text("\n".join(json.dumps(r) for r in lines) + "\n")

    grid = grid_test.build_grid(config, results)
    assert grid["identity_diffs"], "a 1e-12 perturbation must fail the exact-float check"
    assert any("B/0" in d and "ndcg@10.value" in d for d in grid["identity_diffs"])


# --- guards -------------------------------------------------------------------


def test_row_order_mismatch_is_rejected(tmp_path):
    config, results = _substrate(tmp_path)
    # Reverse one arm's rows: same user set, different order -> CIs would not be
    # reproducible, so alignment must refuse.
    path = tmp_path / "pop_t12m.parquet"
    table = pq.read_table(path)
    pq.write_table(table.take(list(reversed(range(table.num_rows)))), path)
    with pytest.raises(AssertionError, match="row-order mismatch"):
        grid_test.build_grid(config, results)


def test_user_set_mismatch_is_rejected(tmp_path):
    config, results = _substrate(tmp_path)
    path = tmp_path / "als_chosen.parquet"
    table = pq.read_table(path).slice(0, 5)
    pq.write_table(table, path)
    with pytest.raises(AssertionError, match="user set mismatch"):
        grid_test.build_grid(config, results)


def test_seed_mismatch_is_rejected(tmp_path):
    config, results = _substrate(tmp_path)
    config["bootstrap"] = {"n_resamples": N_RESAMPLES, "seed": 12345}
    with pytest.raises(ValueError, match="recorded bootstrap seed"):
        grid_test.build_grid(config, results)


def test_expected_n_users_guard(tmp_path):
    config, results = _substrate(tmp_path)
    config["expected_n_users"] = 999
    with pytest.raises(ValueError, match="expected_n_users"):
        grid_test.build_grid(config, results)


def test_shipped_config_matches_canonical_test_run_ids():
    import yaml

    cfg = yaml.safe_load(open("configs/policy_grid_test.yaml").read())
    canonical = yaml.safe_load(open("configs/crossover_test.yaml").read())
    assert cfg["split"] == "test"
    assert cfg["bootstrap"] == {"n_resamples": 1000, "seed": 20260805}
    assert cfg["n_star_grid"] == [0, 1, 5, 10, 20, None]
    flat = json.dumps(canonical)
    for run_id in cfg["run_ids"].values():
        assert run_id in flat, f"{run_id} is not one of the canonical TEST run_ids"
