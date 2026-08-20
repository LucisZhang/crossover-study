"""Tests for implicit-ALS Step B (rescoring) and Step A (Spark training)
(Phase 2, T2/T3/T4).

The no-Spark tests build a hand-written fake artifact (U/V npys + als_manifest)
plus a minimal ``EvalDataset`` in ``tmp_path`` and exercise the pure-numpy
``ALSRecommender`` end-to-end (score math, cold collapse, buffer freshness,
identity guards, registry wiring, param hashing). The single ``spark``-marked
test trains real MLlib ALS on a two-cluster synthetic cache and asserts factor
alignment + cluster separation for both weighting modes.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from batch_recsys_lab.eval.dataset import EvalDataset
from batch_recsys_lab.eval.harness import _build_model
from batch_recsys_lab.models.als import (
    FIVE_CORE_TABLE,
    ALSRecommender,
    als_param_hash,
    artifact_dir,
    canonical_params,
    five_core_snapshot_id,
    sha256_file,
)

# Identity params reused across the no-Spark tests.
_PARAMS = dict(rank=3, reg_param=0.1, alpha=40.0, max_iter=5, weighting="binary", seed=7)
_SNAP = 999_123


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _make_ds(manifest: dict, n_users: int, n_items: int) -> EvalDataset:
    """Minimal EvalDataset: ALS only reads manifest + user_ids/item_ids lengths."""
    return EvalDataset(
        cache_dir=None,
        manifest=manifest,
        item_ids=np.array([str(i) for i in range(n_items)], dtype=object),
        user_ids=np.array([str(u) for u in range(n_users)], dtype=object),
        n_train=np.zeros(n_users, dtype=np.int32),
        train_csr=sp.csr_matrix((n_users, n_items), dtype=np.float32),
    )


def _write_artifact(
    factors_root: Path,
    U: np.ndarray,
    V: np.ndarray,
    params: dict = _PARAMS,
    snap: int = _SNAP,
    manifest_overrides: dict | None = None,
) -> Path:
    """Hand-write a valid artifact dir at the canonical param_hash path.

    ``manifest_overrides`` is merged into the manifest AFTER the honest fields,
    so a test can tamper params/seed while keeping the directory at the real
    param_hash (to exercise the fit-time identity assertions)."""
    canon = canonical_params(**params)
    param_hash = als_param_hash(canon)
    adir = artifact_dir(factors_root, snap, param_hash)
    adir.mkdir(parents=True, exist_ok=True)
    np.save(adir / "user_factors.npy", U.astype(np.float32), allow_pickle=False)
    np.save(adir / "item_factors.npy", V.astype(np.float32), allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "params": canon,
        "seed": canon["seed"],
        "weighting": canon["weighting"],
        "param_hash": param_hash,
        "snapshot_ids": {FIVE_CORE_TABLE: snap},
        "five_core_snapshot_id": snap,
        "n_users": U.shape[0],
        "n_items": V.shape[0],
        "user_factors_sha256": sha256_file(adir / "user_factors.npy"),
        "item_factors_sha256": sha256_file(adir / "item_factors.npy"),
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    (adir / "als_manifest.json").write_text(json.dumps(manifest))
    return adir


# --------------------------------------------------------------------------- #
# (g) als_param_hash: stable + order-independent                              #
# --------------------------------------------------------------------------- #
def test_param_hash_stable_and_order_independent():
    h1 = als_param_hash(_PARAMS)
    # Same values, different insertion order + int/float drift on equal values.
    reordered = dict(
        seed=7, weighting="binary", max_iter=5, alpha=40, reg_param=0.1, rank=3.0
    )
    h2 = als_param_hash(reordered)
    assert h1 == h2
    assert len(h1) == 12 and all(c in "0123456789abcdef" for c in h1)
    # A different param changes the hash.
    changed = {**_PARAMS, "rank": 4}
    assert als_param_hash(changed) != h1


# --------------------------------------------------------------------------- #
# (a) score_batch == U @ V.T, row i <-> user idx i                            #
# --------------------------------------------------------------------------- #
def test_score_batch_equals_u_at_vt(tmp_path):
    rng = np.random.default_rng(0)
    U = rng.standard_normal((5, 3)).astype(np.float32)
    V = rng.standard_normal((4, 3)).astype(np.float32)
    _write_artifact(tmp_path, U, V)
    ds = _make_ds({"snapshot_ids": {FIVE_CORE_TABLE: _SNAP}}, 5, 4)

    model = ALSRecommender(**_PARAMS, factors_root=tmp_path).fit(ds)
    idx = np.array([0, 2, 4, 1, 3])
    scores = model.score_batch(idx)
    assert scores.shape == (5, 4)
    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, (U[idx] @ V.T), rtol=1e-6, atol=1e-6)
    # row alignment: single-user request equals the multi-user row for that user.
    # Tight tolerance, not bitwise equality: BLAS backends may pick different
    # kernel paths for 1-row vs batched GEMM (observed 1-ulp float32 drift on
    # x86 OpenBLAS; Apple Accelerate happened to be bitwise-identical).
    np.testing.assert_allclose(
        model.score_batch(np.array([2]))[0], scores[1], rtol=1e-6, atol=1e-6
    )


# --------------------------------------------------------------------------- #
# (b) cold user (zero factor row) -> all-zero scores                          #
# --------------------------------------------------------------------------- #
def test_cold_user_zero_row_scores_all_zero(tmp_path):
    U = np.ones((3, 3), dtype=np.float32)
    U[1] = 0.0  # user 1 is cold: zero factor row (segment-0 collapse)
    V = np.ones((4, 3), dtype=np.float32)
    _write_artifact(tmp_path, U, V)
    ds = _make_ds({"snapshot_ids": {FIVE_CORE_TABLE: _SNAP}}, 3, 4)

    model = ALSRecommender(**_PARAMS, factors_root=tmp_path).fit(ds)
    scores = model.score_batch(np.array([0, 1, 2]))
    assert np.all(scores[1] == 0.0)
    assert not np.all(scores[0] == 0.0)


# --------------------------------------------------------------------------- #
# (c) returned buffer is fresh + writable; mutation does not leak             #
# --------------------------------------------------------------------------- #
def test_score_buffer_fresh_and_writable(tmp_path):
    rng = np.random.default_rng(1)
    U = rng.standard_normal((3, 3)).astype(np.float32)
    V = rng.standard_normal((4, 3)).astype(np.float32)
    _write_artifact(tmp_path, U, V)
    ds = _make_ds({"snapshot_ids": {FIVE_CORE_TABLE: _SNAP}}, 3, 4)

    model = ALSRecommender(**_PARAMS, factors_root=tmp_path).fit(ds)
    out1 = model.score_batch(np.array([0]))
    assert out1.flags.writeable
    out1[0, 0] = -999.0
    out2 = model.score_batch(np.array([0]))
    assert out2[0, 0] != -999.0  # mutation did not corrupt U/V or a cached buffer


# --------------------------------------------------------------------------- #
# (d) manifest param/seed mismatch raises                                     #
# --------------------------------------------------------------------------- #
def test_manifest_param_seed_mismatch_raises(tmp_path):
    U = np.ones((3, 3), dtype=np.float32)
    V = np.ones((4, 3), dtype=np.float32)
    # Directory stays at the real param_hash, but the manifest records a wrong seed.
    tampered = {**canonical_params(**_PARAMS), "seed": 999}
    _write_artifact(tmp_path, U, V, manifest_overrides={"params": tampered})
    ds = _make_ds({"snapshot_ids": {FIVE_CORE_TABLE: _SNAP}}, 3, 4)

    model = ALSRecommender(**_PARAMS, factors_root=tmp_path)
    with pytest.raises(ValueError, match="param/seed mismatch"):
        model.fit(ds)


def test_snapshot_mismatch_raises(tmp_path):
    U = np.ones((3, 3), dtype=np.float32)
    V = np.ones((4, 3), dtype=np.float32)
    _write_artifact(tmp_path, U, V)
    # Cache manifest advertises a DIFFERENT five-core snapshot than the artifact.
    ds = _make_ds({"snapshot_ids": {FIVE_CORE_TABLE: _SNAP + 1}}, 3, 4)
    # Different snapshot -> different artifact dir -> not found (train first).
    model = ALSRecommender(**_PARAMS, factors_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="als-train"):
        model.fit(ds)


def test_factor_shape_mismatch_raises(tmp_path):
    U = np.ones((3, 3), dtype=np.float32)
    V = np.ones((4, 3), dtype=np.float32)
    _write_artifact(tmp_path, U, V)
    # ds claims 5 users but the artifact only has 3 factor rows.
    ds = _make_ds({"snapshot_ids": {FIVE_CORE_TABLE: _SNAP}}, 5, 4)
    model = ALSRecommender(**_PARAMS, factors_root=tmp_path)
    with pytest.raises(ValueError, match="user_factors shape"):
        model.fit(ds)


# --------------------------------------------------------------------------- #
# (e) missing artifact raises with the make als-train hint                    #
# --------------------------------------------------------------------------- #
def test_missing_artifact_raises_with_hint(tmp_path):
    ds = _make_ds({"snapshot_ids": {FIVE_CORE_TABLE: _SNAP}}, 3, 4)
    model = ALSRecommender(**_PARAMS, factors_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="make als-train"):
        model.fit(ds)


# --------------------------------------------------------------------------- #
# (f) _build_model("als") wiring + missing seeds.model raises                 #
# --------------------------------------------------------------------------- #
def test_build_model_als_wiring(tmp_path):
    model_cfg = {
        "name": "als",
        "params": {
            "rank": 3,
            "reg_param": 0.1,
            "alpha": 40.0,
            "max_iter": 5,
            "weighting": "binary",
            "factors_root": str(tmp_path),
        },
    }
    model = _build_model(model_cfg, {"model": 7})
    assert isinstance(model, ALSRecommender)
    assert model.name == "als"
    assert model.rank == 3 and model.seed == 7
    assert model.factors_root == str(tmp_path)
    # params echoed into the record carry the six identity keys.
    assert set(model.params) >= set(_PARAMS)


def test_five_core_snapshot_id_default_table():
    manifest = {"snapshot_ids": {FIVE_CORE_TABLE: 42}}
    assert five_core_snapshot_id(manifest) == 42


def test_five_core_snapshot_id_non_default_table():
    ml32m_table = "local.gold_ml32m.interactions_5core"
    manifest = {"snapshot_ids": {ml32m_table: 4242}}
    assert five_core_snapshot_id(manifest, ml32m_table) == 4242
    # The default-table lookup must NOT silently succeed against a manifest
    # that only carries the ML-32M key.
    with pytest.raises(KeyError):
        five_core_snapshot_id(manifest)


def test_build_model_als_wiring_non_default_five_core_table(tmp_path):
    ml32m_table = "local.gold_ml32m.interactions_5core"
    model_cfg = {
        "name": "als",
        "params": {
            "rank": 3,
            "reg_param": 0.1,
            "alpha": 40.0,
            "max_iter": 5,
            "weighting": "binary",
            "factors_root": str(tmp_path),
        },
    }
    model = _build_model(model_cfg, {"model": 7}, {"five_core": ml32m_table})
    assert model.five_core_table == ml32m_table
    # Not part of the identity/param hash — Amazon and ML-32M runs with
    # identical model params must still hash identically.
    assert "five_core_table" not in model.params


def test_build_model_als_missing_seed_raises():
    model_cfg = {
        "name": "als",
        "params": {"rank": 3, "reg_param": 0.1, "alpha": 40.0, "max_iter": 5, "weighting": "binary"},
    }
    with pytest.raises(ValueError, match="seeds.model"):
        _build_model(model_cfg, {"model": None})


def test_fit_echoes_provenance_into_params(tmp_path):
    U = np.ones((3, 3), dtype=np.float32)
    V = np.ones((4, 3), dtype=np.float32)
    adir = _write_artifact(tmp_path, U, V)
    ds = _make_ds({"snapshot_ids": {FIVE_CORE_TABLE: _SNAP}}, 3, 4)
    model = ALSRecommender(**_PARAMS, factors_root=tmp_path).fit(ds)
    am = json.loads((adir / "als_manifest.json").read_text())
    assert model.params["param_hash"] == als_param_hash(_PARAMS)
    assert model.params["user_factors_sha256"] == am["user_factors_sha256"]
    assert model.params["item_factors_sha256"] == am["item_factors_sha256"]


# --------------------------------------------------------------------------- #
# Spark-marked: real MLlib ALS training on a two-cluster synthetic cache       #
# --------------------------------------------------------------------------- #
def _write_cluster_cache(cache_dir: Path, weighting: str) -> None:
    """30 users x 12 items, two disjoint clusters. Cluster A = users 0..14 /
    items 0..5; cluster B = users 15..29 / items 6..11. Every user interacts
    with items 0..4 (resp 6..10) of its cluster; the 6th cluster item (5 / 11)
    is a partial hold-out seen only by even-indexed users, so it is trained yet
    absent from odd users' histories."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    users, items, ratings = [], [], []
    rng = np.random.default_rng(2026)

    def add(u, i):
        users.append(u)
        items.append(i)
        ratings.append(float(rng.integers(3, 6)))  # 3..5

    for u in range(15):  # cluster A
        for i in range(5):  # items 0..4 always
            add(u, i)
        if u % 2 == 0:  # item 5 partial hold-out
            add(u, 5)
    for u in range(15, 30):  # cluster B
        for i in range(6, 11):  # items 6..10 always
            add(u, i)
        if u % 2 == 0:  # item 11 partial hold-out
            add(u, 11)

    np.save(cache_dir / "train_user_idx.npy", np.array(users, dtype=np.int32))
    np.save(cache_dir / "train_item_idx.npy", np.array(items, dtype=np.int32))
    np.save(cache_dir / "train_rating.npy", np.array(ratings, dtype=np.float32))
    manifest = {
        "schema_version": 2,
        "snapshot_ids": {FIVE_CORE_TABLE: 424242, "local.gold.user_stats": 1},
        "n_users": 30,
        "catalog_size": 12,
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest))


@pytest.mark.spark
@pytest.mark.parametrize("weighting", ["binary", "rating"])
def test_train_als_cluster_separation(spark, tmp_path, weighting):
    from batch_recsys_lab.models.als_train import train_als

    cache_dir = tmp_path / "cache" / "424242"
    _write_cluster_cache(cache_dir, weighting)
    factors_root = tmp_path / "als"

    adir = train_als(
        spark,
        cache_dir,
        rank=8,
        reg_param=0.05,
        alpha=40.0,
        max_iter=12,
        weighting=weighting,
        seed=13,
        factors_root=factors_root,
        git_sha="testsha",
    )

    U = np.load(adir / "user_factors.npy")
    V = np.load(adir / "item_factors.npy")
    assert U.shape == (30, 8)
    assert V.shape == (12, 8)

    am = json.loads((adir / "als_manifest.json").read_text())
    assert am["weighting"] == weighting
    assert am["n_train_pairs"] == len(np.load(cache_dir / "train_user_idx.npy"))
    assert am["param_hash"] == als_param_hash(
        dict(rank=8, reg_param=0.05, alpha=40.0, max_iter=12, weighting=weighting, seed=13)
    )

    scores = U @ V.T  # (30, 12)
    A_users, B_users = np.arange(15), np.arange(15, 30)
    A_items, B_items = np.arange(0, 6), np.arange(6, 12)

    # Cluster separation: same-cluster mean score exceeds cross-cluster mean.
    assert scores[np.ix_(A_users, A_items)].mean() > scores[np.ix_(A_users, B_items)].mean()
    assert scores[np.ix_(B_users, B_items)].mean() > scores[np.ix_(B_users, A_items)].mean()

    # Held-out in-cluster item beats cross-cluster: odd users never saw item 5 (11)
    # yet score it above their cross-cluster block.
    odd_A = A_users[A_users % 2 == 1]
    odd_B = B_users[B_users % 2 == 1]
    assert scores[odd_A, 5].mean() > scores[np.ix_(odd_A, B_items)].mean()
    assert scores[odd_B, 11].mean() > scores[np.ix_(odd_B, A_items)].mean()


@pytest.mark.spark
def test_train_als_idempotent_skip(spark, tmp_path):
    from batch_recsys_lab.models.als_train import train_als

    cache_dir = tmp_path / "cache" / "424242"
    _write_cluster_cache(cache_dir, "binary")
    factors_root = tmp_path / "als"
    kw = dict(rank=8, reg_param=0.05, alpha=40.0, max_iter=6, weighting="binary", seed=1)

    adir = train_als(spark, cache_dir, factors_root=factors_root, git_sha="s", **kw)
    sha_before = json.loads((adir / "als_manifest.json").read_text())["user_factors_sha256"]
    # Second call must skip (idempotent) and leave the artifact byte-identical.
    adir2 = train_als(spark, cache_dir, factors_root=factors_root, git_sha="s", **kw)
    assert adir2 == adir
    sha_after = json.loads((adir / "als_manifest.json").read_text())["user_factors_sha256"]
    assert sha_before == sha_after
