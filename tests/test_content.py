"""Tests for the MiniLM content recommenders (Phase 4, T11).

Builds a fake artifact + tiny in-test ``EvalDataset`` (no torch, no real
embedding download) so these run in plain ``pytest`` against pure numpy/scipy
code. Mirrors ``tests/test_recommenders.py``'s fixture style.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from batch_recsys_lab.eval.dataset import EvalDataset
from batch_recsys_lab.models.base import Recommender
from batch_recsys_lab.models.content import ContentRecommender, l2_normalize_rows
from batch_recsys_lab.models.content_blend import (
    ContentPopBlendRecommender,
    minmax_1d,
)
from batch_recsys_lab.models.minilm_embed import item_ids_sha256, sha256_file

SNAPSHOT_ID = 999
RECIPE_HASH = "deadbeef1234"

# 6 items x 8 dims. Item 3 is the empty-text / zero-vector case.
RAW_EMBEDDINGS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # i0
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # i1
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # i2
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # i3 -- zero vector (empty text)
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # i4
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # i5
    ],
    dtype=np.float32,
)

ITEM_IDS = ["i0", "i1", "i2", "i3", "i4", "i5"]


def _write_fake_artifact(adir: Path, item_ids=ITEM_IDS, embeddings=RAW_EMBEDDINGS) -> None:
    adir.mkdir(parents=True, exist_ok=True)
    emb_fp16 = embeddings.astype(np.float16)
    emb_path = adir / "embeddings.npy"
    np.save(emb_path, emb_fp16, allow_pickle=False)
    embeddings_sha256 = sha256_file(emb_path)
    manifest = {
        "schema_version": 1,
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "recipe_id": "v1_title_brand_cat_features",
        "recipe_hash": RECIPE_HASH,
        "recipe_hash_short": RECIPE_HASH,
        "five_core_snapshot_id": SNAPSHOT_ID,
        "item_ids_sha256": item_ids_sha256(item_ids),
        "embeddings_sha256": embeddings_sha256,
        "embedding_dim": embeddings.shape[1],
        "embedding_dtype": "float16",
        "row_count": len(item_ids),
    }
    (adir / "minilm_manifest.json").write_text(json.dumps(manifest, indent=2))


def _toy_dataset() -> EvalDataset:
    item_ids = np.array(ITEM_IDS, dtype=object)
    user_ids = np.array(["u0", "u1", "u2", "u3"], dtype=object)

    # u0: TRAIN items i0, i1 (2 items)          -> mean-pool of e0, e1
    # u1: TRAIN item i4 (1 item)                -> mean-pool of e4 alone
    # u2: 0 TRAIN items                          -> cold user
    # u3: TRAIN item i3 (the zero-vector item)   -> zero profile
    train_dense = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    train_csr = sp.csr_matrix(train_dense)
    n_train = train_dense.sum(axis=1).astype(np.int32)

    pop_vec = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02], dtype=np.float32)

    return EvalDataset(
        cache_dir=Path("/nonexistent"),
        manifest={"snapshot_ids": {"local.gold.interactions_5core": SNAPSHOT_ID}},
        item_ids=item_ids,
        user_ids=user_ids,
        n_train=n_train,
        train_csr=train_csr,
        pop={("train_end", 0): pop_vec},
        item_category_codes=None,
        category_names=[],
        gt={},
    )


@pytest.fixture()
def artifact_root(tmp_path) -> Path:
    root = tmp_path / "minilm"
    _write_fake_artifact(root / str(SNAPSHOT_ID) / RECIPE_HASH)
    return root


def _content_model(artifact_root) -> ContentRecommender:
    return ContentRecommender(recipe_hash=RECIPE_HASH, artifact_root=artifact_root)


# --- protocol conformance -----------------------------------------------------


def test_content_models_satisfy_protocol(artifact_root):
    ds = _toy_dataset()
    content = _content_model(artifact_root)
    content.fit(ds)
    assert isinstance(content, Recommender)
    assert isinstance(content.name, str)
    assert isinstance(content.params, dict)

    blend = ContentPopBlendRecommender(
        alpha=0.5, as_of="train_end", window_days=0,
        recipe_hash=RECIPE_HASH, artifact_root=artifact_root,
    )
    blend.fit(ds)
    assert isinstance(blend, Recommender)
    assert isinstance(blend.name, str)
    assert isinstance(blend.params, dict)


# --- ContentRecommender: hand-computed scores ---------------------------------


def test_content_hand_computed_scores(artifact_root):
    ds = _toy_dataset()
    model = _content_model(artifact_root)
    model.fit(ds)

    E_norm = l2_normalize_rows(RAW_EMBEDDINGS.astype(np.float32))
    out = model.score_batch(np.array([0, 1, 2, 3]))

    # u0: mean(e0, e1) normalized, cosine against all items.
    profile0 = l2_normalize_rows(((RAW_EMBEDDINGS[0] + RAW_EMBEDDINGS[1]) / 2.0)[None, :])[0]
    expected0 = profile0 @ E_norm.T
    np.testing.assert_allclose(out[0], expected0, rtol=1e-5, atol=1e-6)

    # u1: single TRAIN item i4 -> profile == normalized e4.
    profile1 = l2_normalize_rows(RAW_EMBEDDINGS[4][None, :])[0]
    expected1 = profile1 @ E_norm.T
    np.testing.assert_allclose(out[1], expected1, rtol=1e-5, atol=1e-6)

    # u2: cold user (0 TRAIN items) -> all-zero scores.
    np.testing.assert_array_equal(out[2], np.zeros(6, dtype=np.float32))

    # u3: TRAIN item is the zero-vector item -> zero profile -> all-zero scores.
    np.testing.assert_array_equal(out[3], np.zeros(6, dtype=np.float32))

    assert out.dtype == np.float32
    assert out.shape == (4, 6)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()


def test_content_fresh_writable_array(artifact_root):
    ds = _toy_dataset()
    model = _content_model(artifact_root)
    model.fit(ds)

    out1 = model.score_batch(np.array([0]))
    assert out1.flags.writeable
    original = out1[0, 0]
    out1[0, 0] = -999.0

    out2 = model.score_batch(np.array([0]))
    assert out2[0, 0] == pytest.approx(original)
    assert out2[0, 0] != -999.0


def test_content_determinism(artifact_root):
    ds = _toy_dataset()
    model = _content_model(artifact_root)
    model.fit(ds)
    out1 = model.score_batch(np.array([0, 1, 2, 3]))
    out2 = model.score_batch(np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(out1, out2)


# --- ContentRecommender: identity guards ---------------------------------------


def test_content_wrong_item_count_raises(tmp_path):
    ds = _toy_dataset()
    root = tmp_path / "minilm"
    # Manifest built for only 5 items -- row_count mismatch vs ds (6 items).
    _write_fake_artifact(
        root / str(SNAPSHOT_ID) / RECIPE_HASH,
        item_ids=ITEM_IDS[:5],
        embeddings=RAW_EMBEDDINGS[:5],
    )
    model = _content_model(root)
    with pytest.raises(ValueError, match="make embed-items"):
        model.fit(ds)


def test_content_tampered_item_ids_sha_raises(tmp_path):
    ds = _toy_dataset()
    adir = tmp_path / "minilm" / str(SNAPSHOT_ID) / RECIPE_HASH
    _write_fake_artifact(adir)
    man_path = adir / "minilm_manifest.json"
    man = json.loads(man_path.read_text())
    man["item_ids_sha256"] = "0" * 64
    man_path.write_text(json.dumps(man))

    model = _content_model(tmp_path / "minilm")
    with pytest.raises(ValueError, match="make embed-items"):
        model.fit(ds)


def test_content_tampered_embeddings_sha_raises(tmp_path):
    ds = _toy_dataset()
    adir = tmp_path / "minilm" / str(SNAPSHOT_ID) / RECIPE_HASH
    _write_fake_artifact(adir)
    man_path = adir / "minilm_manifest.json"
    man = json.loads(man_path.read_text())
    man["embeddings_sha256"] = "0" * 64
    man_path.write_text(json.dumps(man))

    model = _content_model(tmp_path / "minilm")
    with pytest.raises(ValueError, match="make embed-items"):
        model.fit(ds)


# --- ContentPopBlendRecommender -------------------------------------------------


def test_blend_hand_computed(artifact_root):
    ds = _toy_dataset()
    alpha = 0.4
    model = ContentPopBlendRecommender(
        alpha=alpha, as_of="train_end", window_days=0,
        recipe_hash=RECIPE_HASH, artifact_root=artifact_root,
    )
    model.fit(ds)

    out = model.score_batch(np.array([0, 1, 2, 3]))

    pop_vec = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02], dtype=np.float32)
    pop_normed = minmax_1d(np.log1p(pop_vec))

    content_model = _content_model(artifact_root)
    content_model.fit(ds)
    content_scores = content_model.score_batch(np.array([0, 1, 2, 3]))

    # u0, u1: non-constant content rows -> per-row minmax then blended.
    for i in (0, 1):
        row = content_scores[i]
        lo, hi = row.min(), row.max()
        content_normed = (row - lo) / (hi - lo)
        expected = alpha * content_normed + (1 - alpha) * pop_normed
        np.testing.assert_allclose(out[i], expected, rtol=1e-5, atol=1e-6)

    # u2 (cold, all-zero content), u3 (zero-vector TRAIN item, all-zero content):
    # constant content rows -> content term is zero -> pure (1-alpha)*pop_normed.
    for i in (2, 3):
        expected = (1 - alpha) * pop_normed
        np.testing.assert_array_equal(out[i], expected)

    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert out.dtype == np.float32
    assert out.shape == (4, 6)


def test_blend_fresh_writable_array(artifact_root):
    ds = _toy_dataset()
    model = ContentPopBlendRecommender(
        alpha=0.5, as_of="train_end", window_days=0,
        recipe_hash=RECIPE_HASH, artifact_root=artifact_root,
    )
    model.fit(ds)

    out1 = model.score_batch(np.array([0]))
    assert out1.flags.writeable
    original = out1[0, 0]
    out1[0, 0] = -999.0

    out2 = model.score_batch(np.array([0]))
    assert out2[0, 0] == pytest.approx(original)
    assert out2[0, 0] != -999.0


def test_blend_determinism(artifact_root):
    ds = _toy_dataset()
    model = ContentPopBlendRecommender(
        alpha=0.5, as_of="train_end", window_days=0,
        recipe_hash=RECIPE_HASH, artifact_root=artifact_root,
    )
    model.fit(ds)
    out1 = model.score_batch(np.array([0, 1, 2, 3]))
    out2 = model.score_batch(np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(out1, out2)
