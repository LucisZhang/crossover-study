"""CLI: catalog-learnability regime map (Phase 8, T8-1; UPGRADE_PLAN.md §8b).

    uv run python -m batch_recsys_lab.eval.regime_map \
        --config configs/regime_map_test.yaml [--dry-run]

**What this measures.** Phases 2-4 *inferred* that the 2023 TEST ground truth
concentrates on items a TRAIN-frozen model could not have learned ("catalog
churn"); `docs/case_study.md` line 178 concedes the inference was never measured.
This module measures it, on two axes crossed:

* the **user** axis — the frozen five history-depth segments;
* the **item** axis — learnability at the train cutoff: TRAIN support
  (zero / low 1-4 / high >=5), recency of the last TRAIN interaction
  (<=90d / 91-365d / >365d before ``train_end``, plus absent-in-TRAIN), and
  first-seen year (<=2019 / 2020 / 2021 / 2022-H1 / post-cutoff).

Every threshold is preregistered in ``EXPERIMENT_LOG.md`` (2026-08-17), fixed
before any per-cell outcome was computed, and anchored to constants already
frozen elsewhere (k=5 of the 5-core; the trailing popularity windows; the frozen
segment edges).

**Frozen-TEST justification (CLAUDE.md invariant #1), same posture as
``policy/grid_test.py``.** Nothing here fits, refits or scores a model. The
per-user ``top50`` lists and the ground-truth pairs were both produced by the
one-shot TEST eval runs named in ``source_run_ids`` and already committed to
``results/runs.jsonl``; this module regroups them arithmetically. The one new
input is a Spark aggregate over the SAME gold table those runs used
(``features/item_train_stats.py``), whose TRAIN columns are computed with
``ts <= train_end`` only.

**Exactness of the restricted metrics.** For any GT subset ``GT_b(u)``:

    recall@K_b   = |{g in GT_b(u) : rank(g) <= K}| / |GT_b(u)|
    ndcg@10_b    = sum_{g in GT_b(u), rank(g)<=10} 1/log2(rank(g)+1)
                   / sum_{i=1..min(|GT_b(u)|,10)} 1/log2(i+1)

with ``rank(g)`` the 1-based position of ``g`` in the arm's stored top-50 (and
"absent from the top-50" contributing nothing). These are **exact**, not
approximations: recall@K needs only membership of the top-K prefix for K <= 50,
and NDCG@10 only scores ranks <= 10. Recall@100 or NDCG@20-beyond-50 could NOT
be recomposed this way, which is why the reported cutoffs stop at 50/10.

A user belongs to cell ``(segment, bucket)`` iff they have >=1 GT item in that
bucket, so the cells of one axis overlap in users (a user with GT on both a
zero-support and a high-support item is in both) — deliberate, and the reason
population shares are reported per cell rather than summed to 1 over users.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from batch_recsys_lab.contracts.engine import _resolve_run_id
from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.cell_stats import cell_block
from batch_recsys_lab.eval.dataset import _build_gt
from batch_recsys_lab.eval.harness import _resolve_cache_dir
from batch_recsys_lab.eval.protocol import (
    DEEP_BUCKET_LABELS,
    SEGMENT_LABELS,
    deep_bucket_of,
    segment_of,
)
from batch_recsys_lab.features import item_train_stats
from batch_recsys_lab.features.splits import load_splits
from batch_recsys_lab.policy.select import _resolve_artifact_path

# --- preregistered item axes (EXPERIMENT_LOG.md 2026-08-17; do not tune) ------

SUPPORT_LABELS = ("zero", "low", "high")
SUPPORT_SPEC = {
    "zero": "n_train_support == 0",
    "low": "1 <= n_train_support <= 4  (below the k=5 core degree)",
    "high": "n_train_support >= 5",
}

RECENCY_LABELS = ("<=90d", "91-365d", ">365d", "absent")
RECENCY_SPEC = {
    "<=90d": "0 <= train_end - last_train_ts <= 90 days",
    "91-365d": "90 days < train_end - last_train_ts <= 365 days",
    ">365d": "train_end - last_train_ts > 365 days",
    "absent": "no TRAIN interaction at all (last_train_ts is NULL)",
}

FIRST_SEEN_LABELS = ("<=2019", "2020", "2021", "2022-H1", "post-cutoff")
FIRST_SEEN_SPEC = {
    "<=2019": "first 5-core interaction in calendar year <= 2019",
    "2020": "first 5-core interaction in 2020",
    "2021": "first 5-core interaction in 2021",
    "2022-H1": "first 5-core interaction in 2022 and <= train_end",
    "post-cutoff": (
        "first 5-core interaction after train_end — an item that did not exist in "
        "TRAIN at all (interaction-based proxy for release date, disclosed as such)"
    ),
}

# Axis ordinals feed the per-cell bootstrap entropy; fixed here, never derived
# from iteration order.
AXES = ("support", "recency", "first_seen")
AXIS_LABELS = {"support": SUPPORT_LABELS, "recency": RECENCY_LABELS, "first_seen": FIRST_SEEN_LABELS}
AXIS_SPEC = {"support": SUPPORT_SPEC, "recency": RECENCY_SPEC, "first_seen": FIRST_SEEN_SPEC}
# Per-cell analysis runs on these two axes (§8b T8-1 acceptance (b)); first_seen
# is a headline/share axis only.
CELL_AXES = ("support", "recency")

# --- identity anchor ----------------------------------------------------------
#
# The degenerate partition (ONE bucket holding the whole catalog) restricts to
# nothing, so the recomposed per-user values must equal the values the eval
# harness itself computed from FULL-CATALOG ranks and committed to the artifact.
# That equality is not a nicety: it simultaneously pins (a) that the artifact's
# top-50 rows are aligned to the cache's GT rows, (b) that "position in top-50"
# really is the full-catalog rank for ranks <= 50 (it is: `topk_indices` and
# `gt_ranks` implement the same ordering rule, and TRAIN-masked items sort after
# every finitely-scored item so they never occupy a top-50 slot), and (c) that
# the restricted formulas reduce to eval/metrics.accuracy_metrics. Same posture
# as policy/grid_test.py's degenerate-cell anchors: it is checked before anything
# is emitted, and a mismatch aborts with no write.
#
# Recall@K is exact integer arithmetic and matches bit-for-bit. NDCG@10 sums the
# same terms in the same order but through np.bincount rather than np.sum, so a
# last-bit difference is admissible — hence a tolerance rather than ==.
IDENTITY_METRICS = ("recall@10", "recall@20", "recall@50", "ndcg@10")
IDENTITY_TOLERANCE = 1e-12

SEED_SCHEME = (
    "np.random.default_rng([base_seed, axis_ordinal, segment_ordinal, bucket_ordinal])"
    " per cell, where base_seed=20260805, axis_ordinal indexes "
    "('support','recency','first_seen'), segment_ordinal indexes the frozen "
    "SEGMENT_LABELS ('0','1-4','5-9','10-19','20+'), and bucket_ordinal indexes that "
    "axis's label tuple. Users are resampled WITHIN the cell; the same matrix serves "
    "both arms' CIs and the paired delta (see eval/cell_stats.py)."
)

PROVENANCE_NOTE = (
    "Derived exhibit (Phase 8 T8-1): arithmetic regrouping of the per-user top-50 "
    "lists and ground-truth pairs already committed by the one-shot eval runs named "
    "in source_run_ids, crossed with a leak-free Spark aggregate over the SAME gold "
    "5-core snapshot (features/item_train_stats.py: TRAIN columns use ts <= train_end "
    "only). No model is fitted or scored, no new ground truth is consulted, and no "
    "threshold is tuned — every bucket edge was preregistered in EXPERIMENT_LOG.md on "
    "2026-08-17 before any per-cell outcome existed. Restricted recall@K (K<=50) and "
    "NDCG@10 are EXACT under top-50 recomposition, not approximations."
)

# Preregistered gate on the zero+low GT share (EXPERIMENT_LOG.md 2026-08-17).
GATE_SPEC = {
    "statistic": "share of eval-split GT interactions on items with n_train_support <= 4",
    "wrong_below": 0.10,
    "supported_at_or_above": 0.25,
    "verdicts": {
        "<0.10": "CHURN DIAGNOSIS WRONG — near-total TRAIN/TEST catalog overlap; STOP, revisit T8-2 design",
        "0.10-0.25": "PARTIAL SUPPORT — T8-2 proceeds with the measured share disclosed as a caveat",
        ">=0.25": "MEASURED AND SUPPORTED — churn diagnosis converts from derived to measured",
    },
}


def gate_verdict(share: float) -> tuple[str, str]:
    """(band, verdict) for the preregistered zero+low gate."""
    if share < GATE_SPEC["wrong_below"]:
        band = "<0.10"
    elif share < GATE_SPEC["supported_at_or_above"]:
        band = "0.10-0.25"
    else:
        band = ">=0.25"
    return band, GATE_SPEC["verdicts"][band]


# --- item bucketing -----------------------------------------------------------


def load_item_stats(stats_dir: str | Path, item_ids: np.ndarray) -> dict:
    """Align the item-stats parquet to catalog order; report coverage.

    Returns ``{"support", "last_train_ms", "first_seen_ms", "manifest",
    "coverage"}``. ``support`` is int64 (0 for a catalog item absent from the
    stats table), ``*_ms`` are int64 epoch milliseconds with ``MISSING_MS``
    (= INT64 min) marking "no such instant".

    Coverage is *asserted*, not assumed: the catalog (distinct ``parent_asin`` of
    ``gold.item_features``) and the stats keys (distinct ``parent_asin`` of
    ``gold.interactions_5core``) are expected to be the same set — the recorded
    ``catalog_size == n_5core_distinct_items`` says the item_features join lost
    nothing — but a silent drift would fabricate zero-support items, i.e. exactly
    the number this task exists to measure. So both directions are counted and
    recorded, and a nonzero ``missing_from_stats`` is a hard failure unless the
    config sets ``allow_missing_item_stats: true``.
    """
    stats_dir = Path(stats_dir)
    manifest = json.loads((stats_dir / item_train_stats.MANIFEST_FILENAME).read_text())
    table = pq.read_table(stats_dir / manifest.get("stats_parquet", item_train_stats.STATS_FILENAME))

    asins = np.asarray(table.column("parent_asin").to_pylist(), dtype=object)
    support_raw = np.asarray(table.column("n_train_support").to_pylist(), dtype=np.int64)
    last_raw = table.column("last_train_ts").cast("int64").to_pylist()
    first_raw = table.column("first_seen_ts").cast("int64").to_pylist()

    MISSING = np.iinfo(np.int64).min
    last_ms_raw = np.asarray([MISSING if v is None else int(v) for v in last_raw], dtype=np.int64)
    first_ms_raw = np.asarray([MISSING if v is None else int(v) for v in first_raw], dtype=np.int64)

    # Positional join: catalog order is the sorted item_ids from the eval cache.
    import pandas as pd

    pos = pd.Index(item_ids).get_indexer(asins)  # -1 == stats row outside the catalog
    n_items = len(item_ids)
    support = np.zeros(n_items, dtype=np.int64)
    last_ms = np.full(n_items, MISSING, dtype=np.int64)
    first_ms = np.full(n_items, MISSING, dtype=np.int64)
    keep = pos >= 0
    support[pos[keep]] = support_raw[keep]
    last_ms[pos[keep]] = last_ms_raw[keep]
    first_ms[pos[keep]] = first_ms_raw[keep]

    covered = np.zeros(n_items, dtype=bool)
    covered[pos[keep]] = True
    coverage = {
        "catalog_size": int(n_items),
        "stats_rows": int(len(asins)),
        "catalog_items_covered": int(covered.sum()),
        "missing_from_stats": int((~covered).sum()),
        "stats_rows_outside_catalog": int((~keep).sum()),
    }
    return {
        "support": support,
        "last_train_ms": last_ms,
        "first_seen_ms": first_ms,
        "missing_ms": int(MISSING),
        "manifest": manifest,
        "coverage": coverage,
    }


def support_codes(support: np.ndarray) -> np.ndarray:
    """0=zero, 1=low (1-4), 2=high (>=5) — :data:`SUPPORT_LABELS` ordinals."""
    s = np.asarray(support)
    codes = np.full(s.shape, 2, dtype=np.int8)
    codes[(s >= 1) & (s <= 4)] = 1
    codes[s == 0] = 0
    return codes


def recency_codes(last_train_ms: np.ndarray, train_end_ms: int, missing_ms: int) -> np.ndarray:
    """0='<=90d', 1='91-365d', 2='>365d', 3='absent' — :data:`RECENCY_LABELS`."""
    last = np.asarray(last_train_ms)
    absent = last == missing_ms
    days = (train_end_ms - last.astype(np.float64)) / 86_400_000.0
    codes = np.full(last.shape, 2, dtype=np.int8)  # >365d
    codes[days <= 365.0] = 1
    codes[days <= 90.0] = 0
    codes[absent] = 3
    return codes


def first_seen_codes(
    first_seen_ms: np.ndarray, train_end_ms: int, missing_ms: int
) -> np.ndarray:
    """0='<=2019' … 4='post-cutoff' — :data:`FIRST_SEEN_LABELS` ordinals.

    A catalog item with no interaction at all cannot exist in ``interactions_5core``
    (that is what put it in the 5-core), so ``missing_ms`` here would mean the
    stats join failed; such items are labelled ``post-cutoff`` only if the config
    tolerated the gap, and the coverage counters above make the gap visible.
    """
    first = np.asarray(first_seen_ms)
    years = np.array(
        [
            -1
            if v == missing_ms
            else datetime.fromtimestamp(v / 1000.0, tz=timezone.utc).year
            for v in first.tolist()
        ],
        dtype=np.int64,
    )
    # Cascade from the coarsest bound down, so each later line overrides the
    # earlier one only for the strictly older years: <=2021 -> 2, then <=2020 -> 1,
    # then <=2019 -> 0. Every year <= 2021 is necessarily <= train_end.
    codes = np.full(first.shape, 4, dtype=np.int8)  # default: post-cutoff
    codes[(years <= 2021) & (years >= 0)] = 2
    codes[(years <= 2020) & (years >= 0)] = 1
    codes[(years <= 2019) & (years >= 0)] = 0
    in_2022_h1 = (years == 2022) & (first <= train_end_ms)
    codes[in_2022_h1] = 3
    codes[(years >= 0) & (first > train_end_ms)] = 4
    return codes


# --- restricted-metric recomposition (the unit-tested core) ------------------


def topk_ranks(
    topk: np.ndarray,
    gt_indptr: np.ndarray,
    gt_items: np.ndarray,
    block: int = 200_000,
) -> np.ndarray:
    """1-based rank of every GT pair within its user's stored top-K list; 0 = miss.

    ``topk`` is the ``(N, K)`` int array of catalog indices per eval user (score
    desc, ties by ascending item index — the harness's stored ``top50``);
    ``gt_indptr``/``gt_items`` are the CSR-ragged GT sets for the SAME N users in
    the SAME order. Returns an int64 array aligned element-for-element with
    ``gt_items``, holding ``position + 1`` where the item appears in the user's
    list and ``0`` where it does not.

    Ranks are computed in row blocks so the ``(pairs, K)`` boolean comparison
    never materializes at full 5-core scale (499k x 50).
    """
    topk = np.asarray(topk)
    gt_indptr = np.asarray(gt_indptr)
    gt_items = np.asarray(gt_items)
    n_users = topk.shape[0]
    if gt_indptr.shape[0] != n_users + 1:
        raise ValueError(
            f"gt_indptr length {gt_indptr.shape[0]} != n_users + 1 ({n_users + 1}); "
            "the GT must be aligned to the artifact's user rows"
        )
    rows = np.repeat(np.arange(n_users, dtype=np.int64), np.diff(gt_indptr))
    ranks = np.zeros(gt_items.shape[0], dtype=np.int64)
    for start in range(0, gt_items.shape[0], block):
        end = min(start + block, gt_items.shape[0])
        eq = topk[rows[start:end]] == gt_items[start:end, None]
        hit = eq.any(axis=1)
        pos = eq.argmax(axis=1) + 1
        ranks[start:end] = np.where(hit, pos, 0)
    return ranks


def restricted_from_ranks(
    ranks: np.ndarray,
    gt_indptr: np.ndarray,
    gt_items: np.ndarray,
    item_code: np.ndarray,
    n_buckets: int,
    k_list: tuple[int, ...] = (10, 20, 50),
    ndcg_k: int = 10,
) -> dict[str, np.ndarray]:
    """Per-(user, bucket) restricted metrics from top-K ranks.

    Returns ``{"gt_count": (N,B) int64, "recall@K": (N,B) float64 for each K,
    "ndcg@<ndcg_k>": (N,B) float64}``. ``item_code`` maps catalog index ->
    bucket ordinal in ``[0, n_buckets)``. Cells with ``gt_count == 0`` are 0.0 in
    every metric (they are excluded from cell membership by the caller, never
    averaged in as zeros).

    Formulas are exactly :func:`eval.metrics.accuracy_metrics`'s, restricted to
    the bucket's GT subset — ``IDCG`` uses ``min(|GT_b(u)|, ndcg_k)`` ideal
    positions, i.e. the best achievable ordering of the bucket's own items.
    """
    ranks = np.asarray(ranks, dtype=np.int64)
    gt_indptr = np.asarray(gt_indptr)
    n_users = gt_indptr.shape[0] - 1
    rows = np.repeat(np.arange(n_users, dtype=np.int64), np.diff(gt_indptr))
    codes = np.asarray(item_code)[np.asarray(gt_items)]
    if codes.size and (codes.min() < 0 or codes.max() >= n_buckets):
        raise ValueError("item_code contains a bucket ordinal outside [0, n_buckets)")
    flat = rows * n_buckets + codes
    size = n_users * n_buckets

    hit = ranks > 0
    gt_count = np.bincount(flat, minlength=size).reshape(n_users, n_buckets)

    out: dict[str, np.ndarray] = {"gt_count": gt_count.astype(np.int64)}
    nonzero = gt_count > 0
    for k in k_list:
        w = (hit & (ranks <= k)).astype(np.float64)
        hits_k = np.bincount(flat, weights=w, minlength=size).reshape(n_users, n_buckets)
        out[f"recall@{k}"] = np.divide(
            hits_k, gt_count, out=np.zeros_like(hits_k), where=nonzero
        )

    # rank 0 means "not in the stored top-K"; substitute a dummy rank of 1 there so
    # log2(rank + 1) is never log2(1) == 0 (the np.where would discard the inf, but
    # only after numpy had already emitted a divide-by-zero warning).
    scoring = hit & (ranks <= ndcg_k)
    safe_ranks = np.where(scoring, ranks, 1)
    gain = np.where(scoring, 1.0 / np.log2(safe_ranks + 1.0), 0.0)
    dcg = np.bincount(flat, weights=gain, minlength=size).reshape(n_users, n_buckets)
    idcg_cum = np.cumsum(1.0 / np.log2(np.arange(1, ndcg_k + 1) + 1.0))
    m_clip = np.minimum(gt_count, ndcg_k)
    idcg = np.where(nonzero, idcg_cum[np.maximum(m_clip, 1) - 1], 0.0)
    out[f"ndcg@{ndcg_k}"] = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0.0)
    return out


def recompose_restricted(
    topk: np.ndarray,
    gt_indptr: np.ndarray,
    gt_items: np.ndarray,
    item_code: np.ndarray,
    n_buckets: int,
    k_list: tuple[int, ...] = (10, 20, 50),
    ndcg_k: int = 10,
    block: int = 200_000,
) -> dict[str, np.ndarray]:
    """:func:`topk_ranks` then :func:`restricted_from_ranks` (the composition the
    unit test brute-forces against a naive per-user loop)."""
    ranks = topk_ranks(topk, gt_indptr, gt_items, block=block)
    return restricted_from_ranks(
        ranks, gt_indptr, gt_items, item_code, n_buckets, k_list=k_list, ndcg_k=ndcg_k
    )


# --- artifact / cache loading -------------------------------------------------


def _load_arm(path: str, metrics: tuple[str, ...]) -> dict:
    """user_idx + segment + top50 (+ any recorded metric columns) in ARTIFACT ROW
    ORDER (grid_test's discipline: bootstrap indices are positional)."""
    table = pq.read_table(path, columns=["user_id", "user_idx", "segment", "top50", *metrics])
    return {
        "user_ids": [str(u) for u in table.column("user_id").to_pylist()],
        "user_idx": np.asarray(table.column("user_idx").to_pylist(), dtype=np.int64),
        "segments": np.asarray([str(s) for s in table.column("segment").to_pylist()]),
        "top50": np.asarray(
            [row for row in table.column("top50").to_pylist()], dtype=np.int32
        ),
        "values": {
            m: np.asarray(table.column(m).to_pylist(), dtype=np.float64) for m in metrics
        },
    }


def _find_eval_record(run_id: str, results_path: Path) -> dict:
    match = None
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("kind") == "eval" and rec.get("run_id") == run_id:
            match = rec  # last match wins
    if match is None:
        raise ValueError(f"no eval record with run_id={run_id!r} in {results_path}")
    return match


def _shares(counts: np.ndarray, labels: tuple[str, ...], total: int) -> dict:
    return {
        label: {"n": int(counts[i]), "share": (float(counts[i]) / total) if total else 0.0}
        for i, label in enumerate(labels)
    }


# --- input-equivalence exception (Phase 8 T8-2) --------------------------------
#
# WHY this exists. The lineage guard in :func:`build_regime_map` requires every
# source eval record to have been scored on the SAME ``gold.interactions_5core``
# snapshot the eval cache carries — the cheapest available proof that the arms,
# the ground truth and the item axis all describe one universe. On the machine of
# record (the Linux box the warehouse was rebuilt on, EXPERIMENT_LOG.md
# 2026-08-17) that guard fires for one record and one only: the preregistered
# popularity comparator was scored ONCE on the Mac, on a frozen TEST split that
# must never be re-scored, and the rebuilt warehouse produces byte-identical data
# under a NEW Iceberg snapshot id. The snapshot id is a *label*; the guard would
# reject on the label while the *bytes* agree.
#
# So the exception is deliberately not a boolean escape hatch. It is a config
# block that must NAME the single (arm, run_id) it covers and the two snapshot
# ids it bridges, and must carry a falsifiable proof that the two universes are
# the same data: the item_train_stats parquet digest, cross-checked against the
# local manifest, against the declared expectation, and against the manifest
# embedded in the earlier regime-map record produced on the OLD snapshot — plus
# that record's own comparator-artifact digest against the local artifact. Every
# one of those is a hard failure, an exception that is declared but never needed
# is a hard failure, and a mismatch on any other arm stays fatal. Presence of the
# block is the authorization; there is no enable flag to leave switched on.
#
# The disclosure object this produces is written verbatim into the run record, so
# the exception can never be exercised without appearing in the published
# evidence.

INPUT_EQUIVALENCE_KEY = "regime_map_input_equivalence"
INPUT_EQUIVALENCE_SCHEMA_VERSION = 1
INPUT_EQUIVALENCE_PROOF_KIND = "item_train_stats_parquet_sha256"
# The only two item_train_stats manifest fields a byte-identical rebuild may
# differ in: the wall clock it was written at, and the snapshot label itself.
# Row count, train_end, splits hash, bucket counts and the parquet digest must
# all match exactly — that is what makes "same data, new label" checkable.
INPUT_EQUIVALENCE_MANIFEST_EXEMPT = ("created_ts", "interactions_5core_snapshot_id")

_IE_FIELDS: dict[str, type] = {
    "schema_version": int,
    "exception_id": str,
    "table": str,
    "record_snapshot_id": int,
    "cache_snapshot_id": int,
    "applies_to": dict,
    "proof": dict,
}
_IE_APPLIES_TO_FIELDS: dict[str, type] = {"arm": str, "run_id": str}
_IE_PROOF_FIELDS: dict[str, type] = {
    "kind": str,
    "reference_regime_map_run_id": str,
    "expected_sha256": str,
}


def _typed_mapping(block: object, spec: dict[str, type], where: str) -> dict:
    """Exact-key, strict-type validation of one config mapping.

    Unknown keys are rejected rather than ignored: a typo in an authorization
    block must fail loudly, not silently widen or narrow what was authorized.
    ``bool`` is not accepted where ``int`` is required (Python would).
    """
    if not isinstance(block, dict):
        raise RuntimeError(f"{where} must be a mapping, got {type(block).__name__}")
    unknown = sorted(set(block) - set(spec))
    if unknown:
        raise RuntimeError(
            f"{where}: unknown key(s) {unknown}; allowed keys are {sorted(spec)}"
        )
    missing = sorted(set(spec) - set(block))
    if missing:
        raise RuntimeError(f"{where}: missing required key(s) {missing}")
    for key, typ in spec.items():
        val = block[key]
        ok = (
            isinstance(val, int) and not isinstance(val, bool)
            if typ is int
            else isinstance(val, typ)
        )
        if not ok:
            raise RuntimeError(
                f"{where}.{key} must be {typ.__name__}, got "
                f"{type(val).__name__} ({val!r})"
            )
    return block


def _parse_input_equivalence(config: dict) -> dict | None:
    """Validate and return the config's input-equivalence block, or ``None``.

    Shape only — the substantive checks (which arm, which snapshots, which
    digests) happen in :func:`_apply_input_equivalence`, after every arm's own
    lineage verdict is known.
    """
    exc = config.get(INPUT_EQUIVALENCE_KEY)
    if exc is None:
        return None
    _typed_mapping(exc, _IE_FIELDS, INPUT_EQUIVALENCE_KEY)
    if exc["schema_version"] != INPUT_EQUIVALENCE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{INPUT_EQUIVALENCE_KEY}.schema_version {exc['schema_version']} != supported "
            f"{INPUT_EQUIVALENCE_SCHEMA_VERSION}"
        )
    _typed_mapping(exc["applies_to"], _IE_APPLIES_TO_FIELDS, f"{INPUT_EQUIVALENCE_KEY}.applies_to")
    _typed_mapping(exc["proof"], _IE_PROOF_FIELDS, f"{INPUT_EQUIVALENCE_KEY}.proof")
    if exc["proof"]["kind"] != INPUT_EQUIVALENCE_PROOF_KIND:
        raise RuntimeError(
            f"{INPUT_EQUIVALENCE_KEY}.proof.kind {exc['proof']['kind']!r} is not the only "
            f"implemented proof {INPUT_EQUIVALENCE_PROOF_KIND!r}"
        )
    if exc["record_snapshot_id"] == exc["cache_snapshot_id"]:
        raise RuntimeError(
            f"{INPUT_EQUIVALENCE_KEY}: record_snapshot_id == cache_snapshot_id "
            f"({exc['record_snapshot_id']}) — there is nothing to except"
        )
    return exc


def _find_regime_map_records(run_id: str, results_path: Path) -> list[dict]:
    """Every ``kind="regime_map"`` record with this ``run_id`` (0, 1 or more).

    Unlike :func:`_find_eval_record` this does NOT take the last match: the
    reference record is evidence, so "how many are there" is itself a check.
    """
    out: list[dict] = []
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("kind") == "regime_map" and rec.get("run_id") == run_id:
            out.append(rec)
    return out


def _apply_input_equivalence(
    exc: dict,
    *,
    mismatches: list[dict],
    five_core_table: str,
    cache_sid: int,
    run_ids: dict[str, str],
    stats_dir: Path,
    stats_manifest: dict,
    artifact_paths: dict[str, str],
    results_path: Path,
) -> dict:
    """Adjudicate the declared exception against every arm's lineage verdict.

    ``mismatches`` holds one entry per arm whose recorded 5-core snapshot differs
    from the cache's. Returns the disclosure block for the run record; every
    failure path raises :class:`RuntimeError` and nothing is assembled or
    appended.
    """
    arm = exc["applies_to"]["arm"]
    declared_run_id = exc["applies_to"]["run_id"]
    record_sid = exc["record_snapshot_id"]

    # -- (a) the exception covers exactly the one (arm, run_id) it names --------
    if exc["table"] != five_core_table:
        raise RuntimeError(
            f"{INPUT_EQUIVALENCE_KEY}.table {exc['table']!r} != the config's lineage table "
            f"{five_core_table!r}"
        )
    if cache_sid != exc["cache_snapshot_id"]:
        raise RuntimeError(
            f"{INPUT_EQUIVALENCE_KEY}.cache_snapshot_id {exc['cache_snapshot_id']} != the "
            f"active eval-cache {five_core_table} snapshot {cache_sid}: the exception was "
            "declared against a different local warehouse"
        )
    if arm not in run_ids:
        raise RuntimeError(
            f"{INPUT_EQUIVALENCE_KEY}.applies_to.arm {arm!r} is not one of run_ids "
            f"{sorted(run_ids)}"
        )
    if run_ids[arm] != declared_run_id:
        raise RuntimeError(
            f"{INPUT_EQUIVALENCE_KEY}.applies_to names run_id {declared_run_id!r} for arm "
            f"{arm!r}, but this config scores {run_ids[arm]!r}"
        )
    others = [m for m in mismatches if m["arm"] != arm]
    if others:
        raise RuntimeError(
            "5-core snapshot mismatch on arm(s) the input-equivalence exception does NOT "
            "cover (it covers only "
            f"{arm}/{declared_run_id}): "
            + "; ".join(
                f"{m['arm']} ({m['run_id']}) scored on {m['recorded_snapshot_id']} != cache "
                f"{cache_sid}"
                for m in others
            )
        )
    covered = [m for m in mismatches if m["arm"] == arm]
    if not covered:
        raise RuntimeError(
            f"{INPUT_EQUIVALENCE_KEY} is declared (exception_id={exc['exception_id']!r}) but "
            f"arm {arm} ({declared_run_id}) already matches the cache snapshot {cache_sid}: an "
            "unused exception is stale authorization and must be removed from the config"
        )
    got_sid = covered[0]["recorded_snapshot_id"]
    if got_sid != record_sid:
        raise RuntimeError(
            f"{arm} ({declared_run_id}) was scored on {five_core_table} snapshot {got_sid}, "
            f"but {INPUT_EQUIVALENCE_KEY}.record_snapshot_id declares {record_sid}"
        )

    # -- (b) the local item-stats parquet is the bytes the proof names ---------
    stats_parquet = Path(stats_dir) / stats_manifest.get(
        "stats_parquet", item_train_stats.STATS_FILENAME
    )
    computed_sha = runlog.sha256_file(stats_parquet)
    local_sha = stats_manifest.get("stats_parquet_sha256")
    expected_sha = exc["proof"]["expected_sha256"]
    if computed_sha != local_sha:
        raise RuntimeError(
            f"item_train_stats parquet {stats_parquet} hashes to {computed_sha} but its local "
            f"manifest records {local_sha}: the local item axis is not the file it claims"
        )
    if computed_sha != expected_sha:
        raise RuntimeError(
            f"item_train_stats parquet {stats_parquet} hashes to {computed_sha} != "
            f"{INPUT_EQUIVALENCE_KEY}.proof.expected_sha256 {expected_sha}: the rebuilt gold "
            "5-core is NOT byte-identical to the one the comparator was scored on"
        )

    # -- (c) the reference regime-map record produced on the OLD snapshot ------
    ref_run_id = exc["proof"]["reference_regime_map_run_id"]
    found = _find_regime_map_records(ref_run_id, results_path)
    if len(found) != 1:
        raise RuntimeError(
            f"expected exactly 1 kind=regime_map record with run_id={ref_run_id!r} in "
            f"{results_path}, found {len(found)}"
        )
    ref = found[0]
    ref_sids = ref.get("iceberg_snapshots") or {}
    if int(ref_sids.get(five_core_table, -1)) != record_sid:
        raise RuntimeError(
            f"reference regime_map {ref_run_id} was computed on {five_core_table} snapshot "
            f"{ref_sids.get(five_core_table)} != declared record_snapshot_id {record_sid}"
        )
    ref_source = (ref.get("source_run_ids") or {}).get(arm)
    if ref_source != declared_run_id:
        raise RuntimeError(
            f"reference regime_map {ref_run_id} sourced arm {arm} from run_id {ref_source!r} "
            f"!= the excepted {declared_run_id!r}"
        )
    ref_manifest = ((ref.get("item_stats") or {}).get("manifest")) or {}
    if int(ref_manifest.get("interactions_5core_snapshot_id", -1)) != record_sid:
        raise RuntimeError(
            f"reference regime_map {ref_run_id} item_stats manifest snapshot "
            f"{ref_manifest.get('interactions_5core_snapshot_id')} != declared "
            f"record_snapshot_id {record_sid}"
        )
    ref_stats_sha = ref_manifest.get("stats_parquet_sha256")
    if ref_stats_sha != expected_sha:
        raise RuntimeError(
            f"reference regime_map {ref_run_id} item_stats parquet sha {ref_stats_sha} != "
            f"{INPUT_EQUIVALENCE_KEY}.proof.expected_sha256 {expected_sha}"
        )
    ref_identity = ((ref.get("results") or {}).get("identity_check") or {}).get("passed")
    if ref_identity is not True:
        raise RuntimeError(
            f"reference regime_map {ref_run_id} identity_check.passed is {ref_identity!r}, "
            "not true: it cannot serve as evidence for anything"
        )

    # -- (d) the two item-stats manifests agree except for time and label ------
    def _strip(manifest: dict) -> dict:
        return {
            k: v for k, v in manifest.items() if k not in INPUT_EQUIVALENCE_MANIFEST_EXEMPT
        }

    local_stripped, ref_stripped = _strip(stats_manifest), _strip(ref_manifest)
    if local_stripped != ref_stripped:
        differing = sorted(
            k
            for k in set(local_stripped) | set(ref_stripped)
            if local_stripped.get(k, "<absent>") != ref_stripped.get(k, "<absent>")
        )
        raise RuntimeError(
            "local and reference item_train_stats manifests differ in field(s) "
            f"{differing} (only {list(INPUT_EQUIVALENCE_MANIFEST_EXEMPT)} may differ): "
            "the rebuilt aggregate is not the same aggregate"
        )

    # -- (e) the comparator's per-user artifact is byte-identical --------------
    local_artifact_sha = runlog.sha256_file(artifact_paths[arm])
    ref_artifact_sha = (ref.get("source_artifact_sha256s") or {}).get(arm)
    if local_artifact_sha != ref_artifact_sha:
        raise RuntimeError(
            f"comparator artifact {artifact_paths[arm]} hashes to {local_artifact_sha} but "
            f"reference regime_map {ref_run_id} recorded {ref_artifact_sha}: the local "
            "artifact is not the one the excepted record was built from"
        )

    return {
        "exception_used": True,
        "declaration": exc,
        "applied_to": {
            "arm": arm,
            "run_id": declared_run_id,
            "record_snapshot_id": record_sid,
            "cache_snapshot_id": cache_sid,
        },
        "validation": {
            "status": "passed",
            "reference_regime_map_run_id": ref_run_id,
            "reference_stats_parquet_sha256": ref_stats_sha,
            "local_manifest_stats_parquet_sha256": local_sha,
            "computed_stats_parquet_sha256": computed_sha,
            "manifests_equal_except": list(INPUT_EQUIVALENCE_MANIFEST_EXEMPT),
            "reference_comparator_artifact_sha256": ref_artifact_sha,
            "local_computed_comparator_artifact_sha256": local_artifact_sha,
        },
    }


# --- the analysis --------------------------------------------------------------


def build_regime_map(config: dict, results_path: Path) -> dict:
    """Compute everything. Writes nothing, appends nothing."""
    t0 = time.monotonic()
    run_ids: dict[str, str] = dict(config["run_ids"])
    arm_order = list(run_ids)
    delta_pair = (config["delta"]["minuend"], config["delta"]["subtrahend"])
    for key in delta_pair:
        if key not in run_ids:
            raise ValueError(f"delta arm {key!r} is not one of run_ids {arm_order}")
    metrics = tuple(config.get("cell_metrics", ("ndcg@10", "recall@20")))
    k_list = tuple(int(k) for k in config.get("k_list", (10, 20, 50)))
    # Columns to read: the per-cell metrics plus whatever the identity anchor needs.
    load_metrics = tuple(dict.fromkeys((*metrics, *IDENTITY_METRICS)))
    identity_tolerance = float(
        (config.get("identity_check") or {}).get("tolerance", IDENTITY_TOLERANCE)
    )
    # Shape-checked up front so a malformed authorization block fails before any
    # I/O, never mid-way through the analysis.
    input_equivalence = _parse_input_equivalence(config)
    boot = config.get("bootstrap", {})
    n_resamples = int(boot.get("n_resamples", 1000))
    base_seed = int(boot.get("seed", 20260805))
    split = config.get("split", "test")

    cache_dir = _resolve_cache_dir(config["cache_dir"])
    cache_manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
    five_core_table = config.get("five_core_table", item_train_stats.FIVE_CORE)
    cache_sid = int(cache_manifest["snapshot_ids"][five_core_table])

    splits_path_cfg = config.get("splits_path")
    splits = load_splits(splits_path_cfg) if splits_path_cfg else load_splits()
    train_end_ms = int(splits.train_end.timestamp() * 1000)

    # --- item stats, snapshot-matched to the cache the arms were scored on ---
    stats_root = Path(config.get("item_stats_dir", item_train_stats.DEFAULT_OUT))
    stats_dir = stats_root if (stats_root / item_train_stats.MANIFEST_FILENAME).exists() else stats_root / str(cache_sid)
    item_ids = np.array(
        pq.read_table(cache_dir / "item_ids.parquet").column(0).to_pylist(), dtype=object
    )
    stats = load_item_stats(stats_dir, item_ids)
    stats_manifest = stats["manifest"]
    if int(stats_manifest["interactions_5core_snapshot_id"]) != cache_sid:
        raise RuntimeError(
            f"item_train_stats snapshot {stats_manifest['interactions_5core_snapshot_id']} != "
            f"eval-cache {five_core_table} snapshot {cache_sid}: the item axis and the "
            "scored universe must come from the same 5-core snapshot"
        )
    if stats_manifest["train_end"] != splits.train_end.isoformat():
        raise RuntimeError(
            f"item_train_stats train_end {stats_manifest['train_end']!r} != frozen "
            f"{splits.train_end.isoformat()!r}"
        )
    if stats["coverage"]["missing_from_stats"] and not config.get("allow_missing_item_stats"):
        raise RuntimeError(
            f"{stats['coverage']['missing_from_stats']} catalog items have no item_train_stats "
            "row; they would be counted as zero-support and inflate the very statistic "
            "this analysis measures. Rebuild `make item-train-stats` against the cache's "
            "snapshot, or set allow_missing_item_stats: true deliberately."
        )

    codes = {
        "support": support_codes(stats["support"]),
        "recency": recency_codes(stats["last_train_ms"], train_end_ms, stats["missing_ms"]),
        "first_seen": first_seen_codes(
            stats["first_seen_ms"], train_end_ms, stats["missing_ms"]
        ),
    }

    # --- arms + ground truth, strictly row-aligned ---
    records: dict[str, dict] = {}
    artifact_paths: dict[str, str] = {}
    arms: dict[str, dict] = {}
    snapshot_mismatches: list[dict] = []
    for key, run_id in run_ids.items():
        artifact_path, _model = _resolve_artifact_path(run_id, results_path)
        rec = _find_eval_record(run_id, results_path)
        if rec["protocol"]["eval_split"] != split:
            raise ValueError(
                f"{key} ({run_id}) is an eval on split {rec['protocol']['eval_split']!r}, "
                f"expected {split!r}"
            )
        rec_sids = rec.get("iceberg_snapshots") or {}
        rec_sid = int(rec_sids.get(five_core_table, -1))
        if rec_sid != cache_sid:
            # Unattested, this is fatal here and now (unchanged). With an
            # input-equivalence exception declared the verdict is deferred: the
            # adjudication below must see EVERY arm's outcome before it can rule
            # that the exception covers exactly the one arm it names.
            if input_equivalence is None:
                raise RuntimeError(
                    f"{key} ({run_id}) was scored on {five_core_table} snapshot "
                    f"{rec_sids.get(five_core_table)} != cache snapshot {cache_sid}"
                )
            snapshot_mismatches.append(
                {"arm": key, "run_id": run_id, "recorded_snapshot_id": rec_sid}
            )
        records[key] = rec
        artifact_paths[key] = artifact_path
        arms[key] = _load_arm(artifact_path, load_metrics)

    # Adjudicated before any downstream computation: an exception that does not
    # hold up aborts with nothing assembled and nothing appended. Every guard
    # after this point (user set/order, segment vector, GT alignment, identity
    # anchor) stays unconditional — the exception buys ONE snapshot label, not
    # any relaxation of the checks that the data actually lines up.
    equivalence_block = None
    if input_equivalence is not None:
        equivalence_block = _apply_input_equivalence(
            input_equivalence,
            mismatches=snapshot_mismatches,
            five_core_table=five_core_table,
            cache_sid=cache_sid,
            run_ids=run_ids,
            stats_dir=stats_dir,
            stats_manifest=stats_manifest,
            artifact_paths=artifact_paths,
            results_path=results_path,
        )

    base = arm_order[0]
    for key in arm_order[1:]:
        if arms[key]["user_ids"] != arms[base]["user_ids"]:
            raise RuntimeError(
                f"artifact user set/order mismatch between {base} and {key}: "
                "per-cell resampling is positional, so identical row order is required"
            )
        if not np.array_equal(arms[key]["segments"], arms[base]["segments"]):
            raise RuntimeError(f"segment vector mismatch between {base} and {key}")
    user_idx = arms[base]["user_idx"]
    segments = arms[base]["segments"]
    n_users = int(len(user_idx))
    expected = config.get("expected_n_users")
    if expected is not None and n_users != int(expected):
        raise ValueError(f"user count {n_users} != expected_n_users {expected}")

    gt_u = np.load(cache_dir / f"{split}_user_idx.npy", allow_pickle=False)
    gt_i = np.load(cache_dir / f"{split}_item_idx.npy", allow_pickle=False)
    gt = _build_gt(gt_u, gt_i)
    if not np.array_equal(np.asarray(gt.user_idx, dtype=np.int64), user_idx):
        raise RuntimeError(
            "GT user rows do not match the artifacts' user_idx column: the cache "
            f"has {len(gt.user_idx)} {split} users, the artifacts {n_users}"
        )
    n_train = np.load(cache_dir / "n_train.npy", allow_pickle=False)
    recomputed = np.asarray([str(s) for s in segment_of(n_train[user_idx])])
    if not np.array_equal(recomputed, segments):
        raise RuntimeError(
            "artifact segment column != segment_of(n_train[user_idx]) — the cache and "
            "the artifacts disagree about user history depth"
        )
    gt_total = int(gt.item_idx.shape[0])

    # --- identity anchor: the degenerate partition must reproduce the record ---
    whole_catalog = np.zeros(len(item_ids), dtype=np.int8)
    identity: list[dict] = []
    identity_failures: list[str] = []
    for key in arm_order:
        whole = recompose_restricted(
            arms[key]["top50"], gt.indptr, gt.item_idx, whole_catalog, 1, k_list=k_list
        )
        if not np.array_equal(whole["gt_count"][:, 0], np.diff(gt.indptr)):
            identity_failures.append(f"{key}: single-bucket gt_count != per-user |GT|")
        for m in IDENTITY_METRICS:
            if m not in whole or m not in arms[key]["values"]:
                continue
            got = whole[m][:, 0]
            want = arms[key]["values"][m]
            diff = float(np.max(np.abs(got - want))) if len(want) else 0.0
            identity.append(
                {
                    "arm": key,
                    "metric": m,
                    "max_abs_diff": diff,
                    "bit_identical": bool(np.array_equal(got, want)),
                    "within_tolerance": bool(diff <= identity_tolerance),
                }
            )
            if diff > identity_tolerance:
                identity_failures.append(
                    f"{key} {m}: recomposed per-user vector differs from the recorded "
                    f"artifact by up to {diff:.3e} (> {identity_tolerance:.1e})"
                )
    if identity_failures:
        raise RuntimeError(
            "regime-map identity anchor FAILED — nothing computed further:\n  "
            + "\n  ".join(identity_failures)
        )
    identity_block = {
        "definition": (
            "restricted metrics over a single bucket containing the whole catalog must "
            "equal the per-user values the eval harness computed from full-catalog ranks"
        ),
        "tolerance": identity_tolerance,
        "n_comparisons": len(identity),
        "max_abs_diff": max((c["max_abs_diff"] for c in identity), default=0.0),
        "all_bit_identical": all(c["bit_identical"] for c in identity),
        "passed": True,
        "comparisons": identity,
    }

    # --- headline (a): exact counts, no bootstrap ---
    distinct_gt_items = np.unique(gt.item_idx)
    headline: dict[str, dict] = {
        "eval_split": split,
        "n_users": n_users,
        "gt_interactions_total": gt_total,
        "distinct_gt_items_total": int(distinct_gt_items.size),
        "catalog_size": int(len(item_ids)),
    }
    for axis in AXES:
        labels = AXIS_LABELS[axis]
        code = codes[axis]
        headline[f"gt_interactions_by_{axis}"] = _shares(
            np.bincount(code[gt.item_idx], minlength=len(labels)), labels, gt_total
        )
        headline[f"distinct_gt_items_by_{axis}"] = _shares(
            np.bincount(code[distinct_gt_items], minlength=len(labels)),
            labels,
            int(distinct_gt_items.size),
        )
        headline[f"catalog_items_by_{axis}"] = _shares(
            np.bincount(code, minlength=len(labels)), labels, int(len(item_ids))
        )

    zero_low_share = (
        headline["gt_interactions_by_support"]["zero"]["share"]
        + headline["gt_interactions_by_support"]["low"]["share"]
    )
    band, verdict = gate_verdict(zero_low_share)
    gate = {
        **GATE_SPEC,
        "measured_share": zero_low_share,
        "band": band,
        "verdict": verdict,
    }

    # --- per-cell restricted metrics ---
    per_arm_restricted = {
        key: {
            axis: recompose_restricted(
                arms[key]["top50"],
                gt.indptr,
                gt.item_idx,
                codes[axis],
                len(AXIS_LABELS[axis]),
                k_list=k_list,
                ndcg_k=10,
            )
            for axis in CELL_AXES
        }
        for key in arm_order
    }

    cells: dict[str, list] = {}
    for axis in CELL_AXES:
        axis_ord = AXES.index(axis)
        labels = AXIS_LABELS[axis]
        gt_count = per_arm_restricted[base][axis]["gt_count"]  # arm-independent
        rows: list[dict] = []
        for seg_ord, seg in enumerate(SEGMENT_LABELS):
            seg_mask = segments == seg
            for b_ord, label in enumerate(labels):
                mask = seg_mask & (gt_count[:, b_ord] > 0)
                arm_values = {
                    key: {m: per_arm_restricted[key][axis][m][mask, b_ord] for m in metrics}
                    for key in arm_order
                }
                block = cell_block(
                    arm_values,
                    metrics,
                    delta_pair,
                    [base_seed, axis_ord, seg_ord, b_ord],
                    n_resamples,
                )
                cell_gt = int(gt_count[mask, b_ord].sum())
                rows.append(
                    {
                        "axis": axis,
                        "segment": seg,
                        "bucket": label,
                        "gt_interactions": cell_gt,
                        "user_share": (block["n_users"] / n_users) if n_users else 0.0,
                        "gt_share": (cell_gt / gt_total) if gt_total else 0.0,
                        **block,
                    }
                )
        cells[axis] = rows

    # --- deliverable (c): max attainable recall for a TRAIN-frozen factor model ---
    support_of_gt = stats["support"][gt.item_idx]
    gt_rows = np.repeat(np.arange(n_users, dtype=np.int64), np.diff(gt.indptr))

    def _ceiling(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {"gt_interactions": 0, "support_ge_1": None, "support_ge_5": None}
        return {
            "gt_interactions": n,
            "support_ge_1": float((support_of_gt[mask] >= 1).sum() / n),
            "support_ge_5": float((support_of_gt[mask] >= 5).sum() / n),
        }

    deep_labels = np.asarray([str(s) for s in deep_bucket_of(n_train[user_idx])])
    ceilings = {
        "definition": (
            "share of the split's GT interactions that lie on items with TRAIN support "
            ">= 1 (a TRAIN-frozen factor model cannot rank an item it never saw) and, as "
            "the 'well-learnable' ceiling, >= 5 (the 5-core degree). This is an upper "
            "bound on recall@K for ANY K, not an achievable score."
        ),
        "global": _ceiling(np.ones(gt_total, dtype=bool)),
        "per_segment": {
            seg: _ceiling(segments[gt_rows] == seg) for seg in SEGMENT_LABELS
        },
        "per_deep_bucket_note": (
            "the deep-bucket slice below uses the T8-3 EXPLORATORY boundaries "
            "(20-49 / 50-99 / 100+) and inherits their exploratory label; the "
            "per_segment slice above is on the frozen confirmatory axis"
        ),
        "per_deep_bucket": {
            lbl: _ceiling(deep_labels[gt_rows] == lbl) for lbl in DEEP_BUCKET_LABELS
        },
    }

    return {
        "split": split,
        "n_users": n_users,
        "gt_interactions_total": gt_total,
        "catalog_size": int(len(item_ids)),
        "arm_order": arm_order,
        "delta_pair": list(delta_pair),
        "metrics": list(metrics),
        "k_list": list(k_list),
        "seed": base_seed,
        "n_resamples": n_resamples,
        "cache_dir": str(cache_dir),
        "cache_manifest": cache_manifest,
        "item_stats_dir": str(stats_dir),
        "item_stats_manifest": stats_manifest,
        "item_stats_coverage": stats["coverage"],
        "artifact_paths": artifact_paths,
        "input_equivalence": equivalence_block,
        "records": records,
        "identity_check": identity_block,
        "headline": headline,
        "gate": gate,
        "cells": cells,
        "ceilings": ceilings,
        "wall_clock_s": round(time.monotonic() - t0, 3),
    }


# --- record assembly ----------------------------------------------------------


def build_record(config: dict, config_path: Path, out: dict) -> dict:
    git = runlog.git_info()
    run_id, run_ts = _resolve_run_id(None)
    splits_path = config.get("splits_path") or runlog.DEFAULT_SPLITS_PATH
    stats_dir = Path(out["item_stats_dir"])
    return {
        "schema_version": runlog.record_schema_version,
        "kind": "regime_map",
        "run_id": run_id,
        "run_ts": run_ts,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "config_path": str(config_path),
        "config_hash": runlog.config_hash(config_path),
        "derived": True,
        "provenance_note": PROVENANCE_NOTE,
        "splits": runlog.splits_block(splits_path),
        "dataset_manifest_hash": runlog.dataset_manifest_hash(runlog.DEFAULT_MANIFEST_PATH),
        "iceberg_snapshots": out["cache_manifest"]["snapshot_ids"],
        "contracts": out["cache_manifest"]["contract_identities"],
        "source_run_ids": dict(config["run_ids"]),
        "source_artifact_sha256s": {
            key: runlog.sha256_file(path) for key, path in out["artifact_paths"].items()
        },
        "item_stats": {
            "dir": str(stats_dir),
            "manifest": out["item_stats_manifest"],
            "manifest_sha256": runlog.sha256_file(
                stats_dir / item_train_stats.MANIFEST_FILENAME
            ),
            "coverage": out["item_stats_coverage"],
        },
        "protocol": {
            "eval_split": out["split"],
            "n_users": out["n_users"],
            "gt_interactions": out["gt_interactions_total"],
            "catalog_size": out["catalog_size"],
            "cell_metrics": out["metrics"],
            "k_list": out["k_list"],
            "recomposition": (
                "restricted recall@K (K<=50) and NDCG@10 from the persisted per-user "
                "top-50 lists; exact, not sampled"
            ),
        },
        "axes": {
            axis: {"labels": list(AXIS_LABELS[axis]), "spec": AXIS_SPEC[axis]}
            for axis in AXES
        },
        "cell_axes": list(CELL_AXES),
        "user_axis": {"labels": list(SEGMENT_LABELS), "source": "gold.user_stats.n_train"},
        "seeds": {"bootstrap": out["seed"]},
        "bootstrap": {
            "n_resamples": out["n_resamples"],
            "seed": out["seed"],
            "scheme": SEED_SCHEME,
            "resampling": "within-cell users, with replacement",
        },
        "delta": {"minuend": out["delta_pair"][0], "subtrahend": out["delta_pair"][1]},
        "results": {
            "identity_check": out["identity_check"],
            "headline": out["headline"],
            "gate": out["gate"],
            "cells": out["cells"],
            "ceilings": out["ceilings"],
        },
        "wall_clock_s": out["wall_clock_s"],
        "hardware": runlog.hardware_string(),
        # Present ONLY when a lineage exception was declared AND used, so a record
        # without one is byte-identical in shape to every record written before
        # this facility existed.
        **(
            {INPUT_EQUIVALENCE_KEY: out["input_equivalence"]}
            if out.get("input_equivalence")
            else {}
        ),
    }


# --- printing -----------------------------------------------------------------


def _print_report(out: dict) -> None:
    split = out["split"]
    print(
        f"regime map · split={split} users={out['n_users']} "
        f"GT interactions={out['gt_interactions_total']} catalog={out['catalog_size']}"
    )
    print(f"item stats: {out['item_stats_dir']}  coverage={out['item_stats_coverage']}")
    ie = out.get("input_equivalence")
    if ie:
        # A lineage exception must never be silent at the console either.
        print(
            f"INPUT-EQUIVALENCE EXCEPTION USED: {ie['declaration']['exception_id']} — arm "
            f"{ie['applied_to']['arm']} ({ie['applied_to']['run_id']}) accepted on record "
            f"snapshot {ie['applied_to']['record_snapshot_id']} vs cache "
            f"{ie['applied_to']['cache_snapshot_id']}; proof vs regime_map "
            f"{ie['validation']['reference_regime_map_run_id']} PASS"
        )
    ic = out["identity_check"]
    print(
        f"identity anchor: {ic['n_comparisons']} per-user vectors vs the recorded "
        f"artifacts, max |diff| = {ic['max_abs_diff']:.3e} (tolerance "
        f"{ic['tolerance']:.1e}), all bit-identical={ic['all_bit_identical']} PASS"
    )

    print("\n--- headline (a): exact GT shares by item bucket ---")
    for axis in AXES:
        labels = AXIS_LABELS[axis]
        gt_blk = out["headline"][f"gt_interactions_by_{axis}"]
        it_blk = out["headline"][f"distinct_gt_items_by_{axis}"]
        cat_blk = out["headline"][f"catalog_items_by_{axis}"]
        print(f"  {axis}:")
        print(f"    {'bucket':<12}{'GT n':>10}{'GT share':>11}{'items n':>10}"
              f"{'item share':>12}{'catalog n':>11}{'cat share':>11}")
        for label in labels:
            print(
                f"    {label:<12}{gt_blk[label]['n']:>10}{gt_blk[label]['share']:>11.4f}"
                f"{it_blk[label]['n']:>10}{it_blk[label]['share']:>12.4f}"
                f"{cat_blk[label]['n']:>11}{cat_blk[label]['share']:>11.4f}"
            )

    g = out["gate"]
    print(
        f"\n--- preregistered gate --- zero+low GT share = {g['measured_share']:.4f} "
        f"(band {g['band']})\n    {g['verdict']}"
    )

    print("\n--- (c) max attainable recall for a TRAIN-frozen factor model ---")
    c = out["ceilings"]
    print(f"  global: support>=1 {c['global']['support_ge_1']:.4f}  "
          f"support>=5 {c['global']['support_ge_5']:.4f}  (GT n={c['global']['gt_interactions']})")
    for name, blk in (("per_segment", c["per_segment"]), ("per_deep_bucket", c["per_deep_bucket"])):
        print(f"  {name}:")
        for label, d in blk.items():
            if d["support_ge_1"] is None:
                print(f"    {label:<10} GT n=0")
                continue
            print(
                f"    {label:<10} GT n={d['gt_interactions']:>8}  "
                f">=1 {d['support_ge_1']:.4f}  >=5 {d['support_ge_5']:.4f}"
            )

    a_key, b_key = out["delta_pair"]
    for axis in out["cells"]:
        print(f"\n--- (b) cells: user segment x {axis} bucket ---")
        print(
            f"  {'seg':<7}{'bucket':<10}{'users':>9}{'u.share':>9}{'GT':>9}{'gt.share':>10}"
            f"{'  ' + b_key + ' ndcg@10':>28}{'  ' + a_key + ' ndcg@10':>28}"
            f"{'  delta [CI]':>34}{'  ne0':>5}"
        )
        for cell in out["cells"][axis]:
            if cell["n_users"] == 0:
                print(f"  {cell['segment']:<7}{cell['bucket']:<10}{0:>9}  (empty cell)")
                continue
            b = cell["arms"][b_key]["ndcg@10"]
            a = cell["arms"][a_key]["ndcg@10"]
            d = cell["delta"]["ndcg@10"]
            print(
                f"  {cell['segment']:<7}{cell['bucket']:<10}{cell['n_users']:>9}"
                f"{cell['user_share']:>9.4f}{cell['gt_interactions']:>9}{cell['gt_share']:>10.4f}"
                f"  {b['value']:.6f} [{b['ci_lo']:.6f},{b['ci_hi']:.6f}]"
                f"  {a['value']:.6f} [{a['ci_lo']:.6f},{a['ci_hi']:.6f}]"
                f"  {d['delta']:+.6f} [{d['ci_lo']:+.6f},{d['ci_hi']:+.6f}]"
                f"  {'yes' if d['excludes_zero'] else 'no':>4}"
            )
    print(f"\nwall_clock_s={out['wall_clock_s']}")


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.eval.regime_map")
    parser.add_argument("--config", default="configs/regime_map_test.yaml")
    parser.add_argument("--results", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute + print everything, append nothing to the results log",
    )
    parser.add_argument("--json-out", default=None, help="also dump the full record JSON here")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    results_path = Path(args.results or config.get("results_path", "results/runs.jsonl"))
    dry_run = args.dry_run or bool(config.get("dry_run", False))

    try:
        out = build_regime_map(config, results_path)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report + non-zero exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_report(out)

    record = build_record(config, config_path, out)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(record, indent=2))
        print(f"record JSON written to {args.json_out}")

    if dry_run:
        print("\n--dry-run: no record appended")
        return 0

    # A TEST-split number must trace to a commit (CLAUDE.md invariant #1/#3).
    runlog.check_test_dirty(out["split"], record["git_dirty"])
    runlog.append_record(record, results_path)
    print(f"\nappended kind=regime_map run_id={record['run_id']} -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
