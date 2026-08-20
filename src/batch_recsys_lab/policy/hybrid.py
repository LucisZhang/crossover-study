"""``HybridRecommender`` — history-depth routing policy (Phase 4, T13;
UPGRADE_PLAN.md §6.4 "routing policy").

Routes each eval user to one of two component ``Recommender``s based on their
TRAIN interaction count (``ds.n_train``), then assembles a single (B, I) score
matrix so the harness can score/mask/rank a hybrid exactly like any other
model.

Routing convention (pinned; matches the §6.4 framing "blend below n*, X
above"):

    n_train[u] <  n_star  -> scored by ``low``   (e.g. blend; helps early history)
    n_train[u] >= n_star  -> scored by ``high``  (e.g. ALS / pop-t12m; warm users)

``n_star = None`` (YAML ``null``/``"inf"``) means every user is routed to
``low`` — the hybrid degenerates to the low-side model everywhere (the
n*=infinity grid point in T13's selection).

Both components are built via ``eval.harness._build_model`` from small
``{name, params}`` config dicts, so this module accepts the exact same model
config shape used elsewhere in the harness (no separate registry to keep in
sync). Import is intentionally *inside* the constructor (not at module level)
to avoid a circular import: ``harness.py`` imports this module's
``HybridRecommender`` to register ``"hybrid"`` in its own ``_build_model``.
"""

from __future__ import annotations

import numpy as np

from batch_recsys_lab.eval.dataset import EvalDataset


def _resolve_n_star(n_star) -> float:
    """Normalize the config's n_star to a float threshold; None/"inf" -> +inf
    (route everyone to ``low``)."""
    if n_star is None:
        return float("inf")
    if isinstance(n_star, str) and n_star.strip().lower() in ("inf", "infinity", "null", "none"):
        return float("inf")
    return float(n_star)


class HybridRecommender:
    """Routes users to ``low`` (n_train < n_star) or ``high`` (n_train >=
    n_star) and assembles their scores into one fresh (B, I) matrix
    (``Recommender`` protocol)."""

    name = "hybrid"

    def __init__(
        self, n_star, low: dict, high: dict, seeds: dict | None = None, tables: dict | None = None
    ):
        # Deferred import — see module docstring (avoids harness<->policy cycle).
        from batch_recsys_lab.eval.harness import _build_model

        self.n_star_raw = n_star
        self.n_star = _resolve_n_star(n_star)
        self.low_cfg = dict(low)
        self.high_cfg = dict(high)
        seeds = seeds or {}
        self._low = _build_model(self.low_cfg, seeds, tables)
        self._high = _build_model(self.high_cfg, seeds, tables)
        self.params = {
            "n_star": None if self.n_star == float("inf") else self.n_star,
            "low": {"name": self.low_cfg["name"], "params": self.low_cfg.get("params") or {}},
            "high": {"name": self.high_cfg["name"], "params": self.high_cfg.get("params") or {}},
        }
        self._n_train: np.ndarray | None = None

    def fit(self, ds: EvalDataset) -> None:
        self._low.fit(ds)
        self._high.fit(ds)
        self._n_train = np.asarray(ds.n_train)

    def score_batch(self, user_idx: np.ndarray) -> np.ndarray:
        if self._n_train is None:
            raise RuntimeError("HybridRecommender.score_batch called before fit().")
        user_idx = np.asarray(user_idx)
        n_train_batch = self._n_train[user_idx]
        low_mask = n_train_batch < self.n_star
        high_mask = ~low_mask

        n_items = None
        out = None

        if np.any(low_mask):
            low_users = user_idx[low_mask]
            low_scores = self._low.score_batch(low_users)
            n_items = low_scores.shape[1]
            out = np.empty((len(user_idx), n_items), dtype=np.float32)
            out[low_mask] = low_scores

        if np.any(high_mask):
            high_users = user_idx[high_mask]
            high_scores = self._high.score_batch(high_users)
            if out is None:
                n_items = high_scores.shape[1]
                out = np.empty((len(user_idx), n_items), dtype=np.float32)
            out[high_mask] = high_scores

        # user_idx is never empty in the harness's batching loop, so out is
        # always assigned above; this guards a degenerate empty-batch caller.
        if out is None:
            out = np.empty((0, 0), dtype=np.float32)

        return out.astype(np.float32, copy=False)
