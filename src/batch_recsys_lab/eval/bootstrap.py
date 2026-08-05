"""User-bootstrap confidence intervals (Phase 2, T3; UPGRADE_PLAN.md §8 "Bootstrap").

1,000 user-index resamples with replacement, drawn via
``np.random.default_rng(seed).integers``, percentile (2.5th/97.5th, linear
interpolation — ``np.percentile`` default) CIs of resampled means. The SAME
resample-index matrix underlies single-model CIs (:func:`ci_mean`) and paired
deltas (:func:`paired_delta_ci`) — sharing indices across the two arms of a
paired comparison is what makes the delta's CI valid (it cancels shared
resampling noise rather than compounding it). Determinism (frozen seed) is an
invariant (CLAUDE.md invariant #2): the same seed must reproduce byte-identical
resample matrices and CIs run to run.

Segment CIs (:func:`segment_cis`) resample *within* each segment's own user
subpopulation, using a per-segment seed derived from the base seed and the
segment's ordinal position (its index in the sorted-by-first-appearance label
order) via ``np.random.default_rng([seed, segment_ordinal])`` — NumPy's
``SeedSequence`` accepts a sequence of integers as entropy, so this gives each
segment an independent, deterministic sub-stream without needing a global
counter or hashing scheme.
"""

from __future__ import annotations

import numpy as np

DEFAULT_N_RESAMPLES = 1000
DEFAULT_SEED = 20260805


def resample_matrix(n_users: int, n_resamples: int, seed: int) -> np.ndarray:
    """(n_resamples, n_users) int64 user-index matrix, drawn with replacement.

    ``np.random.default_rng(seed).integers(0, n_users, size=(n_resamples, n_users))``.
    This is the single source of resample indices shared by :func:`ci_mean` and
    :func:`paired_delta_ci` so that paired comparisons use identical resamples
    on both arms.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_users, size=(n_resamples, n_users), dtype=np.int64)


def ci_mean(
    values: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    resamples: np.ndarray | None = None,
) -> dict:
    """Bootstrap 95% CI of the mean of ``values`` (one row per user).

    ``ci_lo``/``ci_hi`` are the 2.5th/97.5th percentiles (linear interpolation,
    ``np.percentile`` default) of the resampled means. If ``resamples`` (a
    precomputed :func:`resample_matrix`) is given, it is used as-is; otherwise
    a fresh matrix is built from ``(n_resamples, seed)``.
    """
    values = np.asarray(values)
    if resamples is None:
        resamples = resample_matrix(values.shape[0], n_resamples, seed)
    resampled_means = values[resamples].mean(axis=1)
    lo, hi = np.percentile(resampled_means, [2.5, 97.5])
    return {"value": float(np.mean(values)), "ci_lo": float(lo), "ci_hi": float(hi)}


def paired_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Bootstrap 95% CI of ``mean(a) - mean(b)`` for user-aligned vectors ``a``, ``b``.

    ``a`` and ``b`` must already be aligned to the same users in the same
    order (caller's responsibility). Resample user indices ONCE (a shared
    :func:`resample_matrix`) and apply that same index set to both arms per
    resample, so shared resampling noise cancels rather than compounds.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    resamples = resample_matrix(a.shape[0], n_resamples, seed)
    resampled_deltas = a[resamples].mean(axis=1) - b[resamples].mean(axis=1)
    lo, hi = np.percentile(resampled_deltas, [2.5, 97.5])
    delta = float(np.mean(a) - np.mean(b))
    return {
        "delta": delta,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def segment_cis(
    values: np.ndarray,
    segment_labels: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict]:
    """Per-segment bootstrap CIs, resampling within each segment's own users.

    Segment ordinals are assigned by order of first appearance in
    ``segment_labels`` (deterministic given the input array). Each segment's
    resample matrix is seeded via ``np.random.default_rng([seed, ordinal])``
    so segments are independent, deterministic sub-streams of the base seed —
    a segment's CI never changes when another segment's values change.

    Segments with 0 users are omitted (never occurs from an array pass, kept
    for API completeness against a caller-supplied segment universe). Segments
    with exactly 1 user get a degenerate CI (``ci_lo == ci_hi == value``).
    """
    values = np.asarray(values)
    segment_labels = np.asarray(segment_labels)

    seen: list = []
    for label in segment_labels:
        if label not in seen:
            seen.append(label)

    out: dict[str, dict] = {}
    for ordinal, label in enumerate(seen):
        mask = segment_labels == label
        seg_values = values[mask]
        n_users = int(seg_values.shape[0])
        if n_users == 0:
            continue
        if n_users == 1:
            v = float(seg_values[0])
            out[label] = {"n_users": n_users, "value": v, "ci_lo": v, "ci_hi": v}
            continue
        rng = np.random.default_rng([seed, ordinal])
        resamples = rng.integers(0, n_users, size=(n_resamples, n_users), dtype=np.int64)
        resampled_means = seg_values[resamples].mean(axis=1)
        lo, hi = np.percentile(resampled_means, [2.5, 97.5])
        out[label] = {
            "n_users": n_users,
            "value": float(np.mean(seg_values)),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
        }
    return out
