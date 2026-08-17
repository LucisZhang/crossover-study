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


# --- exploratory deep depth buckets (Phase 8, T8-3) ---------------------------
#
# The frozen five segments above are the CONFIRMATORY axis: every eval record,
# the crossover chart and the routing policy are reported on them. The seven
# buckets below split the open-ended "20+" segment into 20-49 / 50-99 / 100+ and
# are labelled EXPLORATORY/DERIVED (UPGRADE_PLAN §8b T8-1/T8-3 preregistration,
# 2026-08-17): the boundaries were fixed in EXPERIMENT_LOG.md *before* any
# per-bucket outcome was computed, but they were motivated by an observed
# pattern (the monotonically narrowing ALS deficit), so they carry no
# confirmatory weight. The first four labels are identical to SEGMENT_LABELS by
# construction, which is what makes the deep-bucket recomposition checkable
# against the recorded per-segment numbers.
DEEP_BUCKET_LABELS = ("0", "1-4", "5-9", "10-19", "20-49", "50-99", "100+")

_DEEP_BUCKET_EDGES = np.array([0, 4, 9, 19, 49, 99], dtype=np.int64)


def deep_bucket_of(n_train: np.ndarray) -> np.ndarray:
    """Vectorized mapping from ``n_train`` to a :data:`DEEP_BUCKET_LABELS` string.

    0 -> "0"; 1..4 -> "1-4"; 5..9 -> "5-9"; 10..19 -> "10-19"; 20..49 -> "20-49";
    50..99 -> "50-99"; >=100 -> "100+". Same ``searchsorted(side="left")``
    construction as :func:`segment_of`, so the first four buckets partition
    exactly the same users as the corresponding frozen segments.
    """
    n = np.asarray(n_train)
    idx = np.searchsorted(_DEEP_BUCKET_EDGES, n, side="left")
    labels = np.array(DEEP_BUCKET_LABELS, dtype=object)
    return labels[idx]


@dataclass(frozen=True)
class EvalProtocol:
    """The evaluation protocol knobs (UPGRADE_PLAN.md §8 "Config YAML" block)."""

    eval_split: str  # "val" | "test"
    knowledge_cutoff: str  # "train_end"
    k_list: tuple[int, ...] = (10, 20, 50)
    batch_size: int = 1024
