"""Content + popularity blend recommender (Phase 4, T11; docs/engineering-log/UPGRADE_PLAN.md §8).

Composes a :class:`~batch_recsys_lab.models.content.ContentRecommender`
internally and blends its per-user cosine scores with a global popularity
vector (mirrors the crossover framing: content should help most where
popularity alone is weakest — early history — and matter least at the tail
where popularity dominates).

Score: ``alpha * minmax_per_user(content_scores) + (1 - alpha) * pop_normed``,
where ``pop_normed = minmax(log1p(pop_vector))`` is computed once in ``fit``.

Per-user min-max guard: if a user's content row is constant (max == min —
covers the cold-user all-zero row from :class:`ContentRecommender`, and any
degenerate row), that row's content term is left at zero, so the blend
degenerates to pure popularity ranking for that user. No NaN/inf anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from batch_recsys_lab.eval.dataset import EvalDataset
from batch_recsys_lab.models.als import FIVE_CORE_TABLE
from batch_recsys_lab.models.content import DEFAULT_ARTIFACT_ROOT, ContentRecommender


def minmax_1d(vec: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D vector to [0, 1]; constant vectors map to all-zero
    (no NaN/inf from a zero range)."""
    lo = float(vec.min())
    hi = float(vec.max())
    if hi == lo:
        return np.zeros_like(vec, dtype=np.float32)
    return ((vec - lo) / (hi - lo)).astype(np.float32)


def minmax_per_row(mat: np.ndarray) -> np.ndarray:
    """Min-max normalize each row of a 2D array independently to [0, 1]; rows
    that are constant (max == min) map to all-zero for that row."""
    lo = mat.min(axis=1, keepdims=True)
    hi = mat.max(axis=1, keepdims=True)
    rng = hi - lo
    out = np.zeros_like(mat, dtype=np.float32)
    nonconstant = (rng[:, 0] != 0)
    if np.any(nonconstant):
        out[nonconstant] = (mat[nonconstant] - lo[nonconstant]) / rng[nonconstant]
    return out


class ContentPopBlendRecommender:
    """Blends per-user min-max-normalized content cosine scores with a global
    min-max-normalized log-popularity vector (``Recommender`` protocol)."""

    name = "content_pop_blend"

    def __init__(
        self,
        alpha: float,
        as_of: str,
        window_days: int,
        recipe_hash: str,
        artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
        five_core_table: str = FIVE_CORE_TABLE,
    ):
        self.alpha = float(alpha)
        self.as_of = str(as_of)
        self.window_days = int(window_days)
        self.recipe_hash = str(recipe_hash)
        self.artifact_root = str(artifact_root)
        # Not part of self.params — see ALSRecommender's five_core_table note.
        self.five_core_table = str(five_core_table)
        self.params = {
            "alpha": self.alpha,
            "as_of": self.as_of,
            "window_days": self.window_days,
            "recipe_hash": self.recipe_hash,
            "artifact_root": self.artifact_root,
        }
        self._content = ContentRecommender(
            recipe_hash=self.recipe_hash,
            artifact_root=self.artifact_root,
            five_core_table=self.five_core_table,
        )
        self._pop_normed: np.ndarray | None = None

    def fit(self, ds: EvalDataset) -> "ContentPopBlendRecommender":
        self._content.fit(ds)

        key = (self.as_of, self.window_days)
        if key not in ds.pop:
            available = sorted(ds.pop.keys())
            raise KeyError(
                f"ContentPopBlendRecommender: no popularity vector for {key!r}; "
                f"available keys: {available}"
            )
        pop_vec = np.asarray(ds.pop[key], dtype=np.float32)
        self._pop_normed = minmax_1d(np.log1p(pop_vec))

        self.params = {**self.params, "content_params": self._content.params}
        return self

    def score_batch(self, user_idx: np.ndarray) -> np.ndarray:
        if self._pop_normed is None:
            raise RuntimeError("ContentPopBlendRecommender.score_batch called before fit().")
        user_idx = np.asarray(user_idx)
        content_scores = self._content.score_batch(user_idx)  # fresh (B, I) float32
        content_normed = minmax_per_row(content_scores)
        pop_row = self._pop_normed[None, :]  # broadcasts, not aliased into output
        out = self.alpha * content_normed + (1.0 - self.alpha) * pop_row
        return out.astype(np.float32, copy=False)
