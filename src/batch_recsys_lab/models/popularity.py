"""Global-popularity baseline (Phase 2, T4).

Every user gets the same score row: the precomputed ``(as_of, window_days)``
popularity vector from ``ds.pop`` (built by ``features.gold.build_popularity``,
leak-free at ``as_of``). No personalization.
"""

from __future__ import annotations

import numpy as np

from batch_recsys_lab.eval.dataset import EvalDataset


class PopularityRecommender:
    """Broadcasts one global popularity vector to every user in the batch."""

    def __init__(self, as_of: str, window_days: int):
        self.as_of = as_of
        self.window_days = window_days
        self.name = "popularity"
        self.params = {"as_of": as_of, "window_days": window_days}
        self._vector: np.ndarray | None = None

    def fit(self, ds: EvalDataset) -> None:
        key = (self.as_of, self.window_days)
        if key not in ds.pop:
            available = sorted(ds.pop.keys())
            raise KeyError(
                f"PopularityRecommender: no popularity vector for {key!r}; "
                f"available keys: {available}"
            )
        self._vector = np.asarray(ds.pop[key], dtype=np.float32)

    def score_batch(self, user_idx: np.ndarray) -> np.ndarray:
        if self._vector is None:
            raise RuntimeError("PopularityRecommender.fit() must be called before score_batch()")
        user_idx = np.asarray(user_idx)
        # Fresh, writable copy per call — the harness masks TRAIN-seen items
        # in place, and callers must not corrupt the cached vector or alias
        # rows across batches.
        return np.tile(self._vector, (len(user_idx), 1)).astype(np.float32, copy=False)
