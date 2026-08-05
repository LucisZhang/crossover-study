"""Eval protocol config + history-depth segment labeling (Phase 2, T1).

``segment_of`` buckets a user's TRAIN-interaction count (``gold.user_stats.n_train``)
into the five frozen segments used throughout the eval harness and the routing
policy (UPGRADE_PLAN.md §8, owner decision "All five segments reported").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEGMENT_LABELS = ("0", "1-4", "5-9", "10-19", "20+")

# Upper-bound edges (inclusive) for the first four buckets; anything >= the last
# edge falls into "20+". np.searchsorted with side="right" on these edges gives
# the bucket index directly.
_SEGMENT_EDGES = np.array([0, 4, 9, 19], dtype=np.int64)


def segment_of(n_train: np.ndarray) -> np.ndarray:
    """Vectorized mapping from ``n_train`` (int) to a segment label string.

    0 -> "0"; 1..4 -> "1-4"; 5..9 -> "5-9"; 10..19 -> "10-19"; >=20 -> "20+".
    Returns an object array of the same shape as ``n_train``.
    """
    n = np.asarray(n_train)
    idx = np.searchsorted(_SEGMENT_EDGES, n, side="left")
    # searchsorted(side="left") on edges [0,4,9,19]:
    #   n==0 -> 0; 1<=n<=4 -> 1; 5<=n<=9 -> 2; 10<=n<=19 -> 3; n>=20 -> 4.
    labels = np.array(SEGMENT_LABELS, dtype=object)
    return labels[idx]


@dataclass(frozen=True)
class EvalProtocol:
    """The evaluation protocol knobs (UPGRADE_PLAN.md §8 "Config YAML" block)."""

    eval_split: str  # "val" | "test"
    knowledge_cutoff: str  # "train_end"
    k_list: tuple[int, ...] = (10, 20, 50)
    batch_size: int = 1024
