"""Random-scoring baseline (Phase 2, T4).

Scores are seeded per-user so a user's score row is identical regardless of
which batch (or batch order) it is requested in — required for reproducible
metrics when the harness batches users in arbitrary order. We pay for this
reproducibility with a fresh ``default_rng`` construction per user per batch
(cheap relative to Spark/IO elsewhere in the pipeline; B x I generation cost
is acceptable at the eval batch sizes used here).
"""

from __future__ import annotations

import numpy as np

from batch_recsys_lab.eval.dataset import EvalDataset


class RandomRecommender:
    """Uniform random scores, reproducible per-user via a seeded substream."""

    def __init__(self, seed: int):
        self.seed = seed
        self.name = "random"
        self.params = {"seed": seed}
        self._n_items: int | None = None

    def fit(self, ds: EvalDataset) -> None:
        self._n_items = len(ds.item_ids)

    def score_batch(self, user_idx: np.ndarray) -> np.ndarray:
        if self._n_items is None:
            raise RuntimeError("RandomRecommender.fit() must be called before score_batch()")
        user_idx = np.asarray(user_idx)
        out = np.empty((len(user_idx), self._n_items), dtype=np.float32)
        for row, u in enumerate(user_idx):
            # Per-user substream: [seed, user_idx] uniquely determines the
            # stream regardless of batch composition/position.
            rng = np.random.default_rng([self.seed, int(u)])
            out[row] = rng.random(self._n_items, dtype=np.float32)
        return out
