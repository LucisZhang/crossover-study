"""Category-conditioned popularity baseline (Phase 2, T4).

``score(u, i) = w_u[cat(i)] * pop[i]`` where ``w_u`` is user ``u``'s TRAIN
interaction-count distribution over categories (normalized to sum 1). Users
with ``n_train == 0`` have no TRAIN signal to build ``w_u`` from and fall back
to the raw global popularity vector, unweighted — this degradation to plain
popularity is the strict-cold exhibit for this model (there is no
personalization signal available for a user with zero TRAIN history).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from batch_recsys_lab.eval.dataset import EvalDataset


class PopularityCategoryRecommender:
    """Popularity reweighted by each user's TRAIN category distribution."""

    def __init__(self, as_of: str, window_days: int):
        self.as_of = as_of
        self.window_days = window_days
        self.name = "popularity_category"
        self.params = {
            "as_of": as_of,
            "window_days": window_days,
            "fallback": "global_pop",
        }
        self._pop: np.ndarray | None = None
        self._train_csr: sp.csr_matrix | None = None
        self._cat_indicator: sp.csr_matrix | None = None
        self._category_codes: np.ndarray | None = None

    def fit(self, ds: EvalDataset) -> None:
        key = (self.as_of, self.window_days)
        if key not in ds.pop:
            available = sorted(ds.pop.keys())
            raise KeyError(
                f"PopularityCategoryRecommender: no popularity vector for {key!r}; "
                f"available keys: {available}"
            )
        self._pop = np.asarray(ds.pop[key], dtype=np.float32)
        self._train_csr = ds.train_csr
        self._category_codes = np.asarray(ds.item_category_codes)

        n_items = len(ds.item_ids)
        n_categories = len(ds.category_names)
        # (I, C) one-hot category indicator: train_csr[user_idx] @ indicator
        # gives each user's raw interaction count per category in one sparse
        # matmul over the whole requested batch.
        rows = np.arange(n_items)
        data = np.ones(n_items, dtype=np.float32)
        self._cat_indicator = sp.csr_matrix(
            (data, (rows, self._category_codes)), shape=(n_items, n_categories)
        )

    def score_batch(self, user_idx: np.ndarray) -> np.ndarray:
        if self._pop is None:
            raise RuntimeError(
                "PopularityCategoryRecommender.fit() must be called before score_batch()"
            )
        user_idx = np.asarray(user_idx)
        sub = self._train_csr[user_idx]  # (B, I) sparse
        counts = sub @ self._cat_indicator  # (B, C) sparse category counts
        counts = np.asarray(counts.todense(), dtype=np.float32)

        row_sums = counts.sum(axis=1, keepdims=True)
        warm = row_sums[:, 0] > 0
        weights = np.zeros_like(counts)
        weights[warm] = counts[warm] / row_sums[warm]

        # Gather w_u[cat(i)] per item, then scale by the global pop vector.
        scores = weights[:, self._category_codes] * self._pop[np.newaxis, :]

        cold = ~warm
        if np.any(cold):
            # Strict-cold fallback: no TRAIN category signal, use raw pop.
            scores[cold] = self._pop[np.newaxis, :]

        return scores.astype(np.float32, copy=False)
