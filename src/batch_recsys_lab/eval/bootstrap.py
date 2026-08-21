"""User-bootstrap confidence intervals (Phase 2, T3; docs/engineering-log/UPGRADE_PLAN.md §8 "Bootstrap").

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

**Sequence seeds.** Every ``seed`` parameter here is passed straight to
``np.random.default_rng``, which accepts a *sequence* of integers as
``SeedSequence`` entropy as readily as a single int. Callers that need an
independent, deterministic sub-stream per analysis cell therefore pass a list —
``[base_seed, axis_ordinal, …]`` — exactly as :func:`segment_cis` does
internally for its per-segment ordinals (Phase 8 T8-1/T8-3 use the same trick
for their per-cell / per-bucket streams). Nothing about the draw changes; only
the entropy does, so a cell's CI never moves when another cell's values move.

At un-cored scale the ``(n_resamples, n_users)`` int64 index matrix stops
fitting in RAM (1000 x 2.5M x 8B = 20GB on a 16GB laptop), so
:func:`ci_mean` and :func:`segment_cis` switch to a row-blocked draw above
``MAX_RESAMPLE_ELEMENTS``. The switch is bit-neutral, not just
statistically equivalent: sequential ``Generator.integers`` block calls from
ONE rng consume the same bit stream as a single big call, so the chunked path
returns byte-identical CIs (pinned by ``tests/test_bootstrap.py``). Every
5-core size stays below the threshold (1000 x 228,153 = 2.3e8), so recorded
results — and ``make reproduce-headline`` byte_exact — are untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Anything ``np.random.default_rng`` accepts as entropy: a single int, or a
# sequence of ints for a deterministic sub-stream (see the module docstring).
SeedLike = int | Sequence[int]

DEFAULT_N_RESAMPLES = 1000
DEFAULT_SEED = 20260805

# Switch threshold: above this many index elements (n_resamples * n_users) the
# resample matrix is drawn in row blocks instead of one call. 5e8 int64 = 4GB.
MAX_RESAMPLE_ELEMENTS = 500_000_000
# Block size for the chunked path: 5e7 int64 = 400MB of indices per block (plus
# the same again for the gathered values), i.e. 2 rows/block at 2.5M users.
RESAMPLE_BLOCK_ELEMENTS = 50_000_000


def resample_matrix(n_users: int, n_resamples: int, seed: SeedLike) -> np.ndarray:
    """(n_resamples, n_users) int64 user-index matrix, drawn with replacement.

    ``np.random.default_rng(seed).integers(0, n_users, size=(n_resamples, n_users))``.
    This is the single source of resample indices shared by :func:`ci_mean` and
    :func:`paired_delta_ci` so that paired comparisons use identical resamples
    on both arms.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_users, size=(n_resamples, n_users), dtype=np.int64)


def _blocked_resampled_means(
    rng: np.random.Generator,
    values: np.ndarray,
    n_resamples: int,
    block_elements: int,
) -> np.ndarray:
    """Resampled means drawn in row blocks, never materializing the full matrix.

    ``rng`` must be freshly seeded exactly as the single-call path would seed it
    (:func:`resample_matrix`'s ``default_rng(seed)``, or ``default_rng([seed,
    ordinal])`` per segment). Consecutive ``(rows, n_users)`` ``integers`` calls
    consume the same bit stream as one ``(n_resamples, n_users)`` call, so the
    means returned here are byte-identical to the unchunked path — that NumPy
    property is pinned by the chunked-vs-single tests, not assumed.

    Each block's indices are discarded as soon as its means are taken, so peak
    extra memory is ``block_rows * n_users * 8`` bytes twice over (indices plus
    gathered values), independent of ``n_resamples``.
    """
    n_users = int(values.shape[0])
    block_rows = max(1, block_elements // n_users)
    means: list[np.ndarray] = []
    drawn = 0
    while drawn < n_resamples:
        rows = min(block_rows, n_resamples - drawn)
        block = rng.integers(0, n_users, size=(rows, n_users), dtype=np.int64)
        means.append(values[block].mean(axis=1))
        drawn += rows
    return np.concatenate(means)


def ci_mean(
    values: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: SeedLike = DEFAULT_SEED,
    resamples: np.ndarray | None = None,
    max_resample_elements: int = MAX_RESAMPLE_ELEMENTS,
    resample_block_elements: int = RESAMPLE_BLOCK_ELEMENTS,
) -> dict:
    """Bootstrap 95% CI of the mean of ``values`` (one row per user).

    ``ci_lo``/``ci_hi`` are the 2.5th/97.5th percentiles (linear interpolation,
    ``np.percentile`` default) of the resampled means. If ``resamples`` (a
    precomputed :func:`resample_matrix`) is given, it is used as-is; otherwise
    a fresh matrix is built from ``(n_resamples, seed)`` — unless that matrix
    would exceed ``max_resample_elements`` indices, in which case the same
    stream is consumed in ``resample_block_elements``-sized row blocks
    (bit-identical result, bounded memory). The two keyword knobs exist so
    tests can force the chunked path on tiny inputs; callers should not tune
    them per run.
    """
    values = np.asarray(values)
    if resamples is None and n_resamples * values.shape[0] > max_resample_elements:
        resampled_means = _blocked_resampled_means(
            np.random.default_rng(seed), values, n_resamples, resample_block_elements
        )
    else:
        if resamples is None:
            resamples = resample_matrix(values.shape[0], n_resamples, seed)
        resampled_means = values[resamples].mean(axis=1)
    lo, hi = np.percentile(resampled_means, [2.5, 97.5])
    return {"value": float(np.mean(values)), "ci_lo": float(lo), "ci_hi": float(hi)}


def paired_delta_resamples(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: SeedLike = DEFAULT_SEED,
) -> np.ndarray:
    """The ``n_resamples`` paired resampled deltas behind :func:`paired_delta_ci`.

    Exposed so the achieved-significance level (:func:`asl_p_value`) is computed
    from the SAME draws as the CI — same seed, same resample matrix, no second
    evaluation of any model (T9-3b preregistration §5e).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    resamples = resample_matrix(a.shape[0], n_resamples, seed)
    return a[resamples].mean(axis=1) - b[resamples].mean(axis=1)


def asl_p_value(resampled_deltas: np.ndarray) -> float:
    """Two-sided bootstrap achieved-significance level, T9-3b §5(e) verbatim::

        p = min(1, 2 * min((1 + #{D_b <= 0}) / (B+1), (1 + #{D_b >= 0}) / (B+1)))

    Pure post-processing of draws already taken (:func:`paired_delta_resamples`)
    — it costs nothing but arithmetic and cannot constitute a second look at
    TEST. Both tails are add-one ("+1") smoothed, so p is never 0.

    Resolution: with ``B = 1000`` the smallest attainable *one-sided* term is
    ``1/1001 ≈ 0.000999`` (the figure §5e names as the floor) and therefore the
    smallest attainable *two-sided* p is ``2/1001 ≈ 0.001998``. §5e's "report a p
    at the floor as ``< 0.001``" is a REPORTING rule about that one-sided
    resolution; this function returns the two-sided value exactly as specified
    and never rounds.

    ``resampled_deltas`` must be finite: a NaN would silently drop out of both
    ``<= 0`` and ``>= 0`` counts and deflate p.
    """
    d = np.asarray(resampled_deltas, dtype=float).ravel()
    if d.size == 0:
        raise ValueError("asl_p_value needs at least one resampled delta")
    if not np.all(np.isfinite(d)):
        raise ValueError("asl_p_value requires finite resampled deltas")
    b = int(d.size)
    p_le = (1 + int(np.count_nonzero(d <= 0.0))) / (b + 1)
    p_ge = (1 + int(np.count_nonzero(d >= 0.0))) / (b + 1)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def paired_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: SeedLike = DEFAULT_SEED,
) -> dict:
    """Bootstrap 95% CI of ``mean(a) - mean(b)`` for user-aligned vectors ``a``, ``b``.

    ``a`` and ``b`` must already be aligned to the same users in the same
    order (caller's responsibility). Resample user indices ONCE (a shared
    :func:`resample_matrix`) and apply that same index set to both arms per
    resample, so shared resampling noise cancels rather than compounds.

    The returned keys are deliberately unchanged (recorded comparison records
    carry this exact shape); a p-value is obtained by passing
    :func:`paired_delta_resamples` to :func:`asl_p_value`.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    resampled_deltas = paired_delta_resamples(a, b, n_resamples, seed)
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
    seed: SeedLike = DEFAULT_SEED,
    max_resample_elements: int = MAX_RESAMPLE_ELEMENTS,
    resample_block_elements: int = RESAMPLE_BLOCK_ELEMENTS,
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

    A segment whose matrix would exceed ``max_resample_elements`` indices draws
    in row blocks off its own ``[seed, ordinal]`` stream (see :func:`ci_mean`);
    the switch is per segment, so a big segment chunking never perturbs a small
    one's CI.
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
        if n_resamples * n_users > max_resample_elements:
            resampled_means = _blocked_resampled_means(
                rng, seg_values, n_resamples, resample_block_elements
            )
        else:
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
