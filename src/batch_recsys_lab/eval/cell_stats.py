"""Per-cell bootstrap blocks for the Phase 8 derived analyses (T8-1 / T8-3).

Both Phase 8 recompositions — the catalog-learnability regime map
(:mod:`batch_recsys_lab.eval.regime_map`) and the deep depth buckets
(:mod:`batch_recsys_lab.eval.deep_buckets`) — need the identical shape of
evidence for every analysis cell: each arm's mean with a 95% user-bootstrap CI,
plus the PAIRED delta between two arms with its own CI. This module is that one
implementation; the two callers differ only in how they define a cell.

**Within-cell resampling.** Users are resampled *inside* the cell, never
globally: a cell's CI must reflect the uncertainty of the subpopulation the cell
is about. This mirrors :func:`eval.bootstrap.segment_cis` (which does the same
for the frozen five segments) rather than :func:`eval.bootstrap.ci_mean`'s
global draw.

**Seeding.** Each cell draws from its own deterministic sub-stream,
``np.random.default_rng(seed_entropy)`` with ``seed_entropy`` a *list* whose
first element is the frozen base seed (20260805) and whose remaining elements
are the cell's fixed ordinals (see each caller's ``SEED_SCHEME`` string, which
is recorded verbatim in the run record). Consequences that matter:

* a cell's CI never changes when another cell's values change;
* the two arms and the paired delta of one cell share the SAME resample matrix
  (``resample_matrix`` is a pure function of ``(n, n_resamples, entropy)``), so
  the delta's CI cancels shared resampling noise instead of compounding it —
  the property :func:`eval.bootstrap.paired_delta_ci` exists to preserve;
* ordinals, not label strings, carry the entropy, so re-ordering the printed
  output can never move a number.

Cells with zero users are emitted with ``null`` metric blocks rather than
dropped: the cell grid is preregistered, so an empty cell is a finding
("nobody in this segment has ground truth on this kind of item"), not a row to
hide.
"""

from __future__ import annotations

import numpy as np

from batch_recsys_lab.eval.bootstrap import ci_mean, paired_delta_ci, resample_matrix


def _with_width(block: dict, lo_key: str = "ci_lo", hi_key: str = "ci_hi") -> dict:
    """Add ``ci_width`` (hi - lo) to a CI block — reported explicitly because
    thin cells are expected and their width is the honesty signal (§8b T8-3)."""
    out = dict(block)
    out["ci_width"] = float(block[hi_key] - block[lo_key])
    return out


def cell_block(
    arm_values: dict[str, dict[str, np.ndarray]],
    metrics: tuple[str, ...] | list[str],
    delta_pair: tuple[str, str],
    seed_entropy: list[int],
    n_resamples: int,
) -> dict:
    """One cell's evidence block.

    ``arm_values`` maps arm key -> metric name -> the cell's per-user values, all
    arms aligned to the same users in the same order (caller's contract — the
    paired delta is only valid under that alignment). ``delta_pair`` is
    ``(minuend_arm, subtrahend_arm)``; the emitted key is
    ``"<minuend>_minus_<subtrahend>"``.

    Returns ``{"n_users", "arms", "delta", "delta_label", "seed_entropy"}``.
    """
    arm_keys = list(arm_values)
    lengths = {k: len(next(iter(v.values()))) for k, v in arm_values.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"cell arms have unequal user counts: {lengths}")
    n_users = int(next(iter(lengths.values())))

    a_key, b_key = delta_pair
    delta_label = f"{a_key}_minus_{b_key}"

    if n_users == 0:
        return {
            "n_users": 0,
            "arms": {k: {m: None for m in metrics} for k in arm_keys},
            "delta": {m: None for m in metrics},
            "delta_label": delta_label,
            "seed_entropy": list(seed_entropy),
        }

    # ONE matrix for this cell, shared by every arm's CI and by the paired delta.
    resamples = resample_matrix(n_users, n_resamples, seed_entropy)

    arms_out: dict[str, dict] = {}
    for key in arm_keys:
        arms_out[key] = {
            m: _with_width(
                ci_mean(arm_values[key][m], n_resamples=n_resamples, resamples=resamples)
            )
            for m in metrics
        }
    delta_out = {
        m: _with_width(
            paired_delta_ci(
                arm_values[a_key][m],
                arm_values[b_key][m],
                n_resamples=n_resamples,
                seed=seed_entropy,
            )
        )
        for m in metrics
    }
    return {
        "n_users": n_users,
        "arms": arms_out,
        "delta": delta_out,
        "delta_label": delta_label,
        "seed_entropy": list(seed_entropy),
    }
