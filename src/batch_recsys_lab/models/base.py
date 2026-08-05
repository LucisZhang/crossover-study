"""``Recommender`` protocol (Phase 2, T4; UPGRADE_PLAN.md §8 "Architecture").

Every model in this package is scored by ``eval.harness`` the same way: fit
once against an ``EvalDataset``, then produce raw (unmasked) scores for a
batch of user indices. The harness — not the model — is responsible for
masking TRAIN-seen items after ``score_batch`` returns.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from batch_recsys_lab.eval.dataset import EvalDataset


@runtime_checkable
class Recommender(Protocol):
    """Fit-then-score interface consumed by ``eval.harness``.

    ``name`` and ``params`` are echoed into the run record (``results/runs.jsonl``);
    ``params`` must be JSON-serializable.
    """

    name: str
    params: dict

    def fit(self, ds: EvalDataset) -> None:
        """Precompute whatever state is needed to score batches."""
        ...

    def score_batch(self, user_idx: np.ndarray) -> np.ndarray:
        """Score a batch of users against the full catalog.

        Parameters
        ----------
        user_idx : (B,) int array of row indices into ``ds.user_ids``.

        Returns
        -------
        (B, I) float32 array of raw scores, one row per requested user, one
        column per catalog item (``ds.item_ids`` order). No TRAIN-seen-item
        exclusion happens here — the harness masks in place afterward, so the
        returned array must be a fresh, writable buffer (never a view/broadcast
        that aliases other rows or a cached vector).
        """
        ...
