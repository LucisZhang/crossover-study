"""MiniLM content-similarity recommender — Step B: pure-numpy rescoring of a
persisted embedding artifact (Phase 4, T11; docs/engineering-log/UPGRADE_PLAN.md §8 "Architecture").

This module contains NO ``torch``/``sentence_transformers`` import. It is the
eval-time half of the T10 two-step design mirrored from ALS:

* **Step A** (``models.minilm_embed``) embeds item text once with MiniLM and
  persists ``embeddings.npy`` (fp16, row-aligned to ``ds.item_ids``) plus a
  ``minilm_manifest.json`` under a snapshot-and-recipe-hash-keyed artifact
  directory.
* **Step B** (this module) is the ``Recommender`` the harness fits: it loads
  the artifact, builds a user profile as the mean of the user's TRAIN items'
  L2-normalized embeddings, and scores by cosine similarity (profile . E^T,
  both L2-normalized). The harness stays contractually torch-free.

Cold-start collapse (intentional, documented — mirrors ALS's segment-0 note):
users with zero TRAIN interactions get an all-zero profile vector (row-sum-0
normalization short-circuits to zero, not NaN), hence a score of exactly 0 for
every catalog item. Items whose recipe text was empty (e.g. missing title)
embed to the zero vector at Step A; those rows stay zero after L2
normalization (guarded explicitly, never divide by a zero norm).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from batch_recsys_lab.eval.dataset import EvalDataset
from batch_recsys_lab.models.als import FIVE_CORE_TABLE, five_core_snapshot_id, sha256_file
from batch_recsys_lab.models.minilm_embed import item_ids_sha256

DEFAULT_ARTIFACT_ROOT = "data/eval/minilm"


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; zero-norm rows stay exactly zero (no NaN)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return mat / safe


def artifact_dir(artifact_root: str | Path, snapshot_id: int, recipe_hash: str) -> Path:
    """``<artifact_root>/<five_core_snapshot_id>/<recipe_hash>/`` — the directory
    holding ``embeddings.npy``, ``minilm_manifest.json``."""
    return Path(artifact_root) / str(snapshot_id) / str(recipe_hash)


class ContentRecommender:
    """MiniLM content-similarity scorer over a persisted embedding artifact
    (``Recommender`` protocol).

    Instantiated from the eval config's ``model.params``; ``fit`` loads +
    identity-validates the artifact Step A wrote for the cache's snapshot and
    the requested ``recipe_hash``, then L2-normalizes the embedding matrix
    once. ``score_batch`` builds each user's mean-pooled normalized profile
    from TRAIN items and scores by cosine similarity against every item.

    Cold users (0 TRAIN items) score 0 for every item by construction — see
    the module docstring's cold-start note; this mirrors ALS's segment-0
    collapse.
    """

    name = "content"

    def __init__(
        self,
        recipe_hash: str,
        artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
        five_core_table: str = FIVE_CORE_TABLE,
    ):
        self.recipe_hash = str(recipe_hash)
        self.artifact_root = str(artifact_root)
        # Not part of self.params — see ALSRecommender's five_core_table note.
        self.five_core_table = str(five_core_table)
        self.params = {"recipe_hash": self.recipe_hash, "artifact_root": self.artifact_root}
        self._E_norm: np.ndarray | None = None
        self._train_csr: sp.csr_matrix | None = None

    def fit(self, ds: EvalDataset) -> "ContentRecommender":
        """Resolve + load + identity-validate the persisted MiniLM artifact for
        this cache snapshot + recipe hash.

        Raises a clear error (pointing at ``make embed-items``) if no artifact
        exists, or if the manifest's row count / item_ids hash / embeddings
        hash disagree with what this cache actually contains.
        """
        manifest = ds.manifest
        snap = five_core_snapshot_id(manifest, self.five_core_table)
        adir = artifact_dir(self.artifact_root, snap, self.recipe_hash)
        man_path = adir / "minilm_manifest.json"
        emb_path = adir / "embeddings.npy"

        if not man_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                f"MiniLM artifact not found at {adir} (expected minilm_manifest.json + "
                f"embeddings.npy). Build it first:  make embed-items  (Step A embeds item "
                f"text with MiniLM and persists the artifact this scorer loads)."
            )

        man = json.loads(man_path.read_text())

        n_items = len(ds.item_ids)
        if int(man.get("row_count", -1)) != n_items:
            raise ValueError(
                f"MiniLM artifact row_count mismatch at {adir}: manifest "
                f"{man.get('row_count')!r} != cache item count {n_items!r}. "
                f"Re-run `make embed-items` against the current cache."
            )

        recomputed_item_ids_sha256 = item_ids_sha256(list(ds.item_ids))
        if man.get("item_ids_sha256") != recomputed_item_ids_sha256:
            raise ValueError(
                f"MiniLM artifact item_ids_sha256 mismatch at {adir}: manifest "
                f"{man.get('item_ids_sha256')!r} != recomputed {recomputed_item_ids_sha256!r}. "
                f"Re-run `make embed-items` against the current cache."
            )

        recomputed_embeddings_sha256 = sha256_file(emb_path)
        if man.get("embeddings_sha256") != recomputed_embeddings_sha256:
            raise ValueError(
                f"MiniLM artifact embeddings_sha256 mismatch at {adir}: manifest "
                f"{man.get('embeddings_sha256')!r} != recomputed "
                f"{recomputed_embeddings_sha256!r}. Re-run `make embed-items` to regenerate "
                f"a consistent artifact."
            )

        E = np.load(emb_path, allow_pickle=False).astype(np.float32, copy=False)
        if E.shape[0] != n_items:
            raise ValueError(
                f"MiniLM embeddings shape {E.shape} row count != cache item count {n_items}."
            )
        self._E_norm = l2_normalize_rows(E)
        self._train_csr = ds.train_csr

        self.params = {
            **self.params,
            "five_core_snapshot_id": snap,
            "embeddings_sha256": man.get("embeddings_sha256"),
            "item_ids_sha256": man.get("item_ids_sha256"),
            "embedding_dim": int(man.get("embedding_dim", E.shape[1])),
        }
        return self

    def score_batch(self, user_idx: np.ndarray) -> np.ndarray:
        """Score every catalog item for a batch of users -> (B, I) float32.

        Profile = L2-normalize(mean of TRAIN items' L2-normalized embeddings);
        scores = profile . E_norm^T. Cold users (0 TRAIN items) get an
        all-zero profile and therefore score 0 for every item — the
        intentional cold-start collapse documented at module level. Returns a
        fresh, writable float32 array (the harness masks TRAIN-seen items in
        place afterward).
        """
        if self._E_norm is None or self._train_csr is None:
            raise RuntimeError("ContentRecommender.score_batch called before fit().")
        user_idx = np.asarray(user_idx)
        sub = self._train_csr[user_idx]  # (B, I) binary csr
        counts = np.asarray(sub.sum(axis=1)).ravel()
        safe_counts = np.where(counts == 0, 1.0, counts)
        # (B, I) @ (I, D) -> (B, D) fresh dense array (sparse @ dense yields ndarray).
        summed = sub @ self._E_norm
        profiles = np.asarray(summed, dtype=np.float32) / safe_counts[:, None]
        profiles[counts == 0] = 0.0
        profiles = l2_normalize_rows(profiles)
        return (profiles @ self._E_norm.T).astype(np.float32)
