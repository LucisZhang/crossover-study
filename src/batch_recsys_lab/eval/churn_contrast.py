"""CLI: pre-model catalog-churn statistic for the regime contrast (Phase 9, T9-3a;
UPGRADE_PLAN.md §8c).

    uv run python -m batch_recsys_lab.eval.churn_contrast \
        --config configs/churn_contrast_ml32m.yaml [--dry-run]

**What this measures.** §8c T9-3a: "rerun the T8-1 churn methodology to produce the
churn-contrast statistic (Amazon 41.11% vs ML-32M x%) — this number is the hinge of
the whole contrast and is computed **before** any model evaluation." So this job
computes the *data-side half* of the T8-1 regime map — the share of TEST
ground-truth interactions landing on items a TRAIN-frozen model could not have
learned (TRAIN support zero, or low 1-4) — and nothing else. It requires **no model
artifacts, no eval cache, no per-user top-50 lists**: only the gold tables and the
frozen split.

Methodological identity with T8-1, and the one definitional difference
----------------------------------------------------------------------
T8-1 (``eval/regime_map.py``) takes its ground truth from the snapshot-keyed eval
cache: ``test_user_idx.npy`` / ``test_item_idx.npy``, which ``eval/extract.py``
builds as *5-core rows whose ``split_label(ts) == "test"``, inner-joined to the user
index (``gold.user_stats`` — every 5-core user, so lossless) and to the item index
(the distinct ``parent_asin`` of ``gold.item_features``)*. This job reproduces that
set directly in Spark: TEST-window 5-core rows inner-joined to the ML-32M
``item_features`` catalog. Same rows, one Spark pass instead of a cache round trip.

The item axis is not merely "the same idea" — it is the SAME CODE: the per-item
aggregate comes from ``features/item_train_stats.compute_item_train_stats`` and the
bucket edges from ``eval/regime_map`` (:func:`~batch_recsys_lab.eval.regime_map.support_codes`,
``recency_codes``, ``first_seen_codes``, ``gate_verdict``), imported, not re-derived.
The preregistered bands (<0.10 / 0.10-0.25 / >=0.25, EXPERIMENT_LOG.md 2026-08-17)
are therefore evaluated by the identical function that produced the Amazon 0.4111.

**The one difference, stated explicitly:** T8-1 additionally asserted that the
scored arms' user rows equal the cache's GT users (an alignment guard on model
artifacts). Pre-model there is nothing to align, so that guard has no analogue here.
Nothing else differs — and because the catalog join is the only place the two could
diverge, its size is reported both ways: ``gt_interactions_total`` (catalog-joined,
the T8-1-equivalent figure and the published statistic) alongside
``gt_interactions_all_5core`` and ``catalog_join_loss_interactions``.

**Frozen-TEST posture (CLAUDE.md invariant #1).** This is a counting aggregation
over the ML-32M TEST window, computed before any model exists — it cannot be, and is
not, used to tune or select anything. It is still recorded as a TEST-split number:
the dirty-tree guard applies, so the record traces to a commit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from batch_recsys_lab.contracts.engine import _resolve_run_id
from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.extract import _contract_identity
from batch_recsys_lab.eval.regime_map import (
    AXES,
    AXIS_LABELS,
    AXIS_SPEC,
    GATE_SPEC,
    _shares,
    first_seen_codes,
    gate_verdict,
    recency_codes,
    support_codes,
)
from batch_recsys_lab.features import item_train_stats
from batch_recsys_lab.ingest.download_ml32m import parse_manifest_text
from batch_recsys_lab.features.splits import SplitConfig, load_splits
from batch_recsys_lab.spark_session import get_spark

RECORD_KIND = "churn_contrast"

# INT64 min doubles as "no such instant" (same convention as regime_map).
MISSING_MS = int(np.iinfo(np.int64).min)

PROVENANCE_NOTE = (
    "Data-stage exhibit (Phase 9 T9-3a): the T8-1 churn methodology re-run on a second "
    "dataset, BEFORE any model is trained or scored. The item axis is the same leak-free "
    "aggregate (features/item_train_stats.compute_item_train_stats: TRAIN columns use "
    "ts <= train_end only) and the same preregistered bucket edges and gate bands "
    "(eval/regime_map, EXPERIMENT_LOG.md 2026-08-17), imported rather than re-derived. "
    "Ground truth is the TEST-window 5-core interaction set joined to the item catalog — "
    "identical to the eval cache's TEST pairs that T8-1 consumed; the catalog-join edge is "
    "reported both ways. No model artifact, eval cache or threshold is involved, and "
    "nothing here is tunable: it is an exact count."
)

GT_DEFINITION = {
    "t8_1_definition": (
        "eval-cache TEST pairs: gold.interactions_5core rows with split_label(ts) == "
        "'test', inner-joined to the user index (gold.user_stats — lossless) and the item "
        "index (distinct parent_asin of gold.item_features)"
    ),
    "this_job_definition": (
        "the same set computed directly in Spark: TEST-window interactions_5core rows "
        "inner-joined to the distinct parent_asin of the dataset's gold item_features"
    ),
    "difference": (
        "T8-1 additionally asserted that the scored arms' artifact user rows equal the "
        "cache GT users; pre-model there are no artifacts, so that alignment guard has no "
        "analogue. The only set-level edge is the item-catalog join, whose size is "
        "published here as catalog_join_loss_interactions / _items."
    ),
}


# --- the statistic (pure numpy: unit-tested without Spark) ---------------------


def compute_churn(
    support: np.ndarray,
    last_train_ms: np.ndarray,
    first_seen_ms: np.ndarray,
    gt_counts: np.ndarray,
    train_end_ms: int,
    missing_ms: int = MISSING_MS,
) -> dict:
    """Headline shares + the preregistered gate, from catalog-aligned arrays.

    All four arrays are aligned to the item catalog (one entry per catalog item):
    ``support`` is the item's TRAIN interaction count, ``last_train_ms`` /
    ``first_seen_ms`` are epoch-millisecond instants (``missing_ms`` = none), and
    ``gt_counts`` is the number of TEST ground-truth interactions on that item.

    Returns the T8-1 "headline (a)" block — per axis, the bucket breakdown of GT
    interactions, of distinct GT items, and of the catalog itself — plus the gate
    verdict on the zero+low GT share. Exact integer counting; no sampling, no seed.
    """
    support = np.asarray(support, dtype=np.int64)
    gt_counts = np.asarray(gt_counts, dtype=np.int64)
    if not (
        support.shape == gt_counts.shape == np.shape(last_train_ms) == np.shape(first_seen_ms)
    ):
        raise ValueError(
            "support / last_train_ms / first_seen_ms / gt_counts must all be aligned to "
            "the item catalog (same length)"
        )

    codes = {
        "support": support_codes(support),
        "recency": recency_codes(last_train_ms, train_end_ms, missing_ms),
        "first_seen": first_seen_codes(first_seen_ms, train_end_ms, missing_ms),
    }

    catalog_size = int(support.shape[0])
    gt_total = int(gt_counts.sum())
    has_gt = gt_counts > 0
    distinct_gt_items = int(has_gt.sum())

    headline: dict = {
        "gt_interactions_total": gt_total,
        "distinct_gt_items_total": distinct_gt_items,
        "catalog_size": catalog_size,
    }
    for axis in AXES:
        labels = AXIS_LABELS[axis]
        code = codes[axis]
        n = len(labels)
        # weights=gt_counts is the per-interaction bincount T8-1 computes over the
        # GT pair array, without materializing one row per interaction.
        gt_by_bucket = np.bincount(code, weights=gt_counts, minlength=n).astype(np.int64)
        headline[f"gt_interactions_by_{axis}"] = _shares(gt_by_bucket, labels, gt_total)
        headline[f"distinct_gt_items_by_{axis}"] = _shares(
            np.bincount(code[has_gt], minlength=n), labels, distinct_gt_items
        )
        headline[f"catalog_items_by_{axis}"] = _shares(
            np.bincount(code, minlength=n), labels, catalog_size
        )

    zero_low_share = (
        headline["gt_interactions_by_support"]["zero"]["share"]
        + headline["gt_interactions_by_support"]["low"]["share"]
    )
    band, verdict = gate_verdict(zero_low_share)
    gate = {**GATE_SPEC, "measured_share": zero_low_share, "band": band, "verdict": verdict}
    return {"headline": headline, "gate": gate}


# --- Spark side ----------------------------------------------------------------


def collect_inputs(
    spark: SparkSession,
    splits: SplitConfig,
    five_core_table: str,
    item_features_table: str,
    split: str = "test",
) -> dict:
    """Catalog-aligned item stats + per-item eval-split GT counts, in one session.

    The item aggregate is ``item_train_stats.compute_item_train_stats`` verbatim
    (the T8-1 code path). The GT side labels the 5-core rows with the frozen splits
    and joins them to the item catalog; both the joined and the unjoined totals are
    returned so the catalog-join edge is visible rather than implicit.
    """
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    stats_pdf = item_train_stats.compute_item_train_stats(
        spark, splits, five_core_table
    ).toPandas()

    catalog_pdf = (
        spark.table(item_features_table)
        .select("parent_asin")
        .distinct()
        .orderBy("parent_asin")
        .toPandas()
    )
    item_ids = catalog_pdf["parent_asin"].tolist()

    five_core_rows_total = spark.table(five_core_table).count()

    labeled = (
        spark.table(five_core_table)
        .select("user_id", "parent_asin", "ts")
        .withColumn("split", splits.split_label("ts"))
        .where(F.col("split") == F.lit(split))
        .drop("split")
        .localCheckpoint(eager=True)
    )
    catalog_df = spark.table(item_features_table).select("parent_asin").distinct()
    joined = labeled.join(catalog_df, "parent_asin", "inner")

    all_5core = labeled.agg(
        F.count(F.lit(1)).alias("n"), F.countDistinct("user_id").alias("u")
    ).first()
    in_catalog = joined.agg(
        F.count(F.lit(1)).alias("n"), F.countDistinct("user_id").alias("u")
    ).first()
    gt_pdf = joined.groupBy("parent_asin").agg(F.count(F.lit(1)).alias("gt_n")).toPandas()

    # Positional alignment onto catalog order (same technique as
    # regime_map.load_item_stats): -1 means "row outside the catalog".
    index = pd.Index(item_ids)
    n_items = len(item_ids)

    support = np.zeros(n_items, dtype=np.int64)
    last_ms = np.full(n_items, MISSING_MS, dtype=np.int64)
    first_ms = np.full(n_items, MISSING_MS, dtype=np.int64)
    stats_pos = index.get_indexer(stats_pdf["parent_asin"].to_numpy())
    keep = stats_pos >= 0
    support[stats_pos[keep]] = stats_pdf["n_train_support"].to_numpy(dtype=np.int64)[keep]
    last_raw = stats_pdf["last_train_ms"].tolist()
    first_raw = stats_pdf["first_seen_ms"].tolist()
    last_all = np.asarray(
        [MISSING_MS if v is None or v != v else int(v) for v in last_raw], dtype=np.int64
    )
    first_all = np.asarray(
        [MISSING_MS if v is None or v != v else int(v) for v in first_raw], dtype=np.int64
    )
    last_ms[stats_pos[keep]] = last_all[keep]
    first_ms[stats_pos[keep]] = first_all[keep]

    covered = np.zeros(n_items, dtype=bool)
    covered[stats_pos[keep]] = True

    gt_counts = np.zeros(n_items, dtype=np.int64)
    if len(gt_pdf):
        gt_pos = index.get_indexer(gt_pdf["parent_asin"].to_numpy())
        gt_keep = gt_pos >= 0
        gt_counts[gt_pos[gt_keep]] = gt_pdf["gt_n"].to_numpy(dtype=np.int64)[gt_keep]

    support_all = stats_pdf["n_train_support"].to_numpy(dtype=np.int64)
    return {
        "item_ids": item_ids,
        "support": support,
        "last_train_ms": last_ms,
        "first_seen_ms": first_ms,
        "gt_counts": gt_counts,
        "n_users": int(in_catalog["u"] or 0),
        "gt_interactions_total": int(in_catalog["n"] or 0),
        "gt_interactions_all_5core": int(all_5core["n"] or 0),
        "five_core_rows_total": int(five_core_rows_total),
        "gt_users_all_5core": int(all_5core["u"] or 0),
        "catalog_join_loss_interactions": int((all_5core["n"] or 0) - (in_catalog["n"] or 0)),
        "catalog_join_loss_items": int(len(stats_pdf) - int(keep.sum())),
        "coverage": {
            "catalog_size": n_items,
            "stats_rows": int(len(stats_pdf)),
            "catalog_items_covered": int(covered.sum()),
            "missing_from_stats": int((~covered).sum()),
            "stats_rows_outside_catalog": int((~keep).sum()),
        },
        "support_bucket_counts_all_5core": {
            "zero": int((support_all == 0).sum()),
            "low": int(((support_all >= 1) & (support_all <= 4)).sum()),
            "high": int((support_all >= 5).sum()),
        },
    }


# --- anti-vacuity ------------------------------------------------------------------


def assert_non_vacuous(data: dict, five_core_table: str, split: str) -> None:
    """Refuse to publish a share computed over nothing.

    ``0/0`` is the failure mode this whole job is exposed to: every downstream
    check (bucket cross-checks, gate bands, the contrast against 0.4111) is
    satisfied by an empty input, and ``compute_churn`` would return a clean
    ``measured_share`` of 0.0 that lands in ``results/runs.jsonl`` as a finding.
    An empty 5-core table (gold never built, or built into a different warehouse)
    and an empty eval window (wrong splits file, or ts read as millis when
    MovieLens publishes seconds) are broken runs, not results.
    """
    if data["five_core_rows_total"] == 0:
        raise RuntimeError(
            f"{five_core_table} is EMPTY: there is nothing to compute a churn share "
            "over. Build the ML-32M gold stage first (`make data-ml32m`) and check "
            "that this process points at the same warehouse."
        )
    if data["gt_interactions_total"] == 0:
        raise RuntimeError(
            f"0 {split.upper()} ground-truth interactions after the item-catalog join "
            f"({data['gt_interactions_all_5core']} {split}-window rows in "
            f"{five_core_table}, {data['five_core_rows_total']} rows in total; catalog "
            f"{data['coverage']['catalog_size']} items). The churn share would be 0/0. "
            "Check the frozen split window, the ts unit (ML-32M publishes epoch "
            "SECONDS), and that item_features was built from this 5-core snapshot."
        )


# --- provenance ------------------------------------------------------------------


def load_item_stats_manifest(stats_root: str | Path, snapshot_id: int) -> tuple[Path, dict]:
    """Resolve and read the snapshot-keyed ``item_train_stats`` manifest.

    Accepts either the snapshot subdir itself or its parent root (same resolution
    rule as ``eval/regime_map.py``). The manifest is provenance for the published
    number, so a snapshot mismatch is a hard failure: an item axis built from a
    different 5-core snapshot describes a different catalog.
    """
    root = Path(stats_root)
    stats_dir = root if (root / item_train_stats.MANIFEST_FILENAME).exists() else root / str(snapshot_id)
    manifest_path = stats_dir / item_train_stats.MANIFEST_FILENAME
    if not manifest_path.exists():
        raise RuntimeError(
            f"no item_train_stats manifest at {manifest_path}. Run the ML-32M "
            "item-train-stats build for the live 5-core snapshot first "
            "(`make item-train-stats-ml32m`), or drop item_stats_dir from the config."
        )
    manifest = json.loads(manifest_path.read_text())
    if int(manifest["interactions_5core_snapshot_id"]) != int(snapshot_id):
        raise RuntimeError(
            f"item_train_stats snapshot {manifest['interactions_5core_snapshot_id']} != live "
            f"5-core snapshot {snapshot_id}: rebuild it (`make item-train-stats-ml32m "
            "ITEM_STATS_FLAGS=--force`) so the published item axis matches the counted one"
        )
    return stats_dir, manifest


def verify_dataset_manifest(
    manifest_path: str | Path,
    must_contain: str | None,
    required_files: list[dict] | None = None,
) -> dict:
    """The recorded ``dataset_manifest_hash`` must describe THIS dataset, in full.

    ``dataset_manifest_hash`` is a hash of a *file*; on its own it attests only
    that some markdown existed. A substring probe (``must_contain``) was the first
    version of this guard and it was too weak: a manifest that merely mentions
    ``ml-32m.zip`` in prose would pass while recording no checksum at all. So the
    document is parsed (``ingest/download_ml32m.parse_manifest`` — the same module
    that writes it) and every file the config declares must carry:

    * a 64-hex SHA-256 (our locally computed ground truth),
    * a byte size,
    * for CSVs, a data-row count — the number ``make bronze-verify-ml32m`` gates
      the live bronze tables against,
    * and, where the config declares a published count, exact agreement with it.

    Every problem is collected and reported at once; the caller gets one message
    listing what is missing rather than a whack-a-mole sequence.

    Returns ``{"marker": str, "path": str, "files": {...}}`` for the run record, so
    the per-file hashes travel inside the record and not merely a whole-file digest.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise RuntimeError(f"dataset manifest {path} does not exist")
    text = path.read_text()
    marker = must_contain or ""
    if must_contain and must_contain not in text:
        raise RuntimeError(
            f"{path} does not mention {must_contain!r}: the raw-data SHA-256s for this "
            "dataset are not recorded yet, so the record's dataset_manifest_hash would "
            "attest to a manifest that never saw it. Run `make manifest-ml32m` first "
            "(it writes data/MANIFEST_ML32M.md; data/MANIFEST.md is the Amazon lane's "
            "file and must not be edited — UPGRADE_PLAN.md §8c T9-3a)."
        )

    entries = parse_manifest_text(text)
    verified: dict[str, dict] = {}
    problems: list[str] = []
    for spec in required_files or []:
        filename = spec["filename"]
        needs_rows = spec.get("kind", "csv") == "csv"
        entry = entries.get(filename)
        if entry is None:
            problems.append(f"{filename}: no '### {filename}' entry under '## Files'")
            continue
        sha, size, rows = entry.get("sha256"), entry.get("size"), entry.get("data_rows")
        if not sha:
            problems.append(f"{filename}: no 64-hex SHA-256 line")
        if not size:
            problems.append(f"{filename}: no positive '- Size (bytes): N' line")
        if needs_rows and not rows:
            problems.append(f"{filename}: no positive '- Data rows (excl. header): N' line")
        published = spec.get("published_rows")
        if published is not None and rows is not None and int(rows) != int(published):
            problems.append(
                f"{filename}: manifest records {rows} data rows but the config's "
                f"published count is {published}"
            )
        verified[filename] = {"sha256": sha, "size": size, "data_rows": rows}

    if problems:
        raise RuntimeError(
            f"{path} does not record this dataset completely:\n  - "
            + "\n  - ".join(problems)
            + "\n  Regenerate it with `make manifest-ml32m` after `fetch` has extracted "
            "every declared file. A run record whose dataset_manifest_hash points at an "
            "incomplete manifest cannot be reproduced from it."
        )
    return {"marker": marker, "path": str(path), "files": verified}


def verify_reference(reference: dict, results_path: Path) -> dict:
    """Re-derive the Amazon contrast anchor from the append-only log.

    The contrast is only meaningful if the number it is contrasted against is the
    recorded one, so 0.4111 is not accepted as a config literal: the named
    ``kind="regime_map"`` record is located in ``results/runs.jsonl`` and its
    ``results.gate.measured_share`` must equal the declared value.
    """
    run_id = reference["run_id"]
    found = [
        rec
        for rec in (
            json.loads(line)
            for line in results_path.read_text().splitlines()
            if line.strip()
        )
        if rec.get("kind") == "regime_map" and rec.get("run_id") == run_id
    ]
    if len(found) != 1:
        raise RuntimeError(
            f"expected exactly 1 kind=regime_map record with run_id={run_id!r} in "
            f"{results_path}, found {len(found)}"
        )
    recorded = float(found[0]["results"]["gate"]["measured_share"])
    declared = float(reference["value"])
    if abs(recorded - declared) > 1e-12:
        raise RuntimeError(
            f"reference anchor mismatch: {run_id} records gate.measured_share={recorded!r}, "
            f"config declares {declared!r}"
        )
    band, verdict = gate_verdict(recorded)
    return {
        **reference,
        "value": recorded,
        "band": band,
        "verdict": verdict,
        "verified_against": str(results_path),
    }


# --- the analysis ----------------------------------------------------------------


def build_churn_contrast(config: dict, results_path: Path) -> dict:
    """Compute everything. Writes nothing, appends nothing."""
    t0 = time.monotonic()
    dataset = config["dataset"]
    split = config.get("split", "test")
    five_core_table = config["five_core_table"]
    item_features_table = config["item_features_table"]
    splits_path = config["splits_path"]
    splits = load_splits(splits_path)
    train_end_ms = int(splits.train_end.timestamp() * 1000)

    reference = verify_reference(
        config["reference"],
        Path(config["reference"].get("results_path", results_path)),
    )
    manifest_path = config.get("dataset_manifest_path") or runlog.DEFAULT_MANIFEST_PATH
    manifest = verify_dataset_manifest(
        manifest_path,
        config.get("dataset_manifest_must_contain"),
        config.get("dataset_manifest_required_files"),
    )

    spark = get_spark(
        app_name=f"churn-contrast-{dataset}",
        warehouse=config.get("warehouse", "data/warehouse"),
        master=config.get("master", "local[10]"),
        driver_memory=config.get("driver_memory", "8g"),
    )
    try:
        snapshots = {
            table: item_train_stats._snapshot_id(spark, table)
            for table in (five_core_table, item_features_table)
        }
        contracts = {
            table: _contract_identity(spark, table)
            for table in (five_core_table, item_features_table)
        }
        data = collect_inputs(spark, splits, five_core_table, item_features_table, split)
    finally:
        spark.stop()

    assert_non_vacuous(data, five_core_table, split)

    if data["coverage"]["missing_from_stats"] and not config.get("allow_missing_item_stats"):
        raise RuntimeError(
            f"{data['coverage']['missing_from_stats']} catalog items have no item stats row; "
            "they would be counted as zero-support and inflate the very statistic this job "
            "measures. Rebuild gold, or set allow_missing_item_stats: true deliberately."
        )

    out = compute_churn(
        data["support"],
        data["last_train_ms"],
        data["first_seen_ms"],
        data["gt_counts"],
        train_end_ms,
    )

    # Provenance link to the published item-axis artifact (optional but required
    # when the config names it): same snapshot, same bucket counts.
    item_stats: dict | None = None
    if config.get("item_stats_dir"):
        stats_dir, manifest = load_item_stats_manifest(
            config["item_stats_dir"], snapshots[five_core_table]
        )
        recorded_buckets = manifest.get("support_bucket_counts")
        if recorded_buckets != data["support_bucket_counts_all_5core"]:
            raise RuntimeError(
                f"item_train_stats manifest support_bucket_counts {recorded_buckets} != the "
                f"counts recomputed here {data['support_bucket_counts_all_5core']}: the "
                "published item axis and this aggregation disagree"
            )
        if manifest["train_end"] != splits.train_end.isoformat():
            raise RuntimeError(
                f"item_train_stats train_end {manifest['train_end']!r} != frozen "
                f"{splits.train_end.isoformat()!r}"
            )
        item_stats = {
            "dir": str(stats_dir),
            "manifest": manifest,
            "manifest_sha256": runlog.sha256_file(
                stats_dir / item_train_stats.MANIFEST_FILENAME
            ),
            "crosscheck": "support_bucket_counts match this run's recomputation",
        }

    headline = {
        "eval_split": split,
        "n_users": data["n_users"],
        **out["headline"],
    }
    contrast = {
        "statistic": GATE_SPEC["statistic"],
        "reference": reference,
        "measured": {
            "dataset": dataset,
            "value": out["gate"]["measured_share"],
            "band": out["gate"]["band"],
        },
        "difference_vs_reference": out["gate"]["measured_share"] - reference["value"],
        "framing": (
            "regime contrast, not causal proof: explicit-rating movie data changes several "
            "variables at once (§8b T8-4). MovieLens timestamps are rating-ENTRY times on a "
            "backfilled catalog (cite Sun et al., arXiv:2307.09985)."
        ),
    }
    return {
        "dataset": dataset,
        "split": split,
        "five_core_table": five_core_table,
        "item_features_table": item_features_table,
        "splits_path": str(splits_path),
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_marker": manifest["marker"],
        "dataset_manifest_files": manifest["files"],
        "iceberg_snapshots": snapshots,
        "contracts": contracts,
        "item_stats": item_stats,
        "coverage": data["coverage"],
        "gt_accounting": {
            "gt_interactions_total": data["gt_interactions_total"],
            "gt_interactions_all_5core": data["gt_interactions_all_5core"],
            "five_core_rows_total": data["five_core_rows_total"],
            "catalog_join_loss_interactions": data["catalog_join_loss_interactions"],
            "catalog_join_loss_items": data["catalog_join_loss_items"],
            "n_users": data["n_users"],
            "n_users_all_5core": data["gt_users_all_5core"],
        },
        "headline": headline,
        "gate": out["gate"],
        "contrast": contrast,
        "wall_clock_s": round(time.monotonic() - t0, 3),
    }


# --- record assembly --------------------------------------------------------------


def build_record(config_path: Path, out: dict) -> dict:
    """Assemble the append-only run record (every input it needs is in ``out``,
    which :func:`build_churn_contrast` already resolved from the config)."""
    git = runlog.git_info()
    run_id, run_ts = _resolve_run_id(None)
    manifest_path = out.get("dataset_manifest_path") or runlog.DEFAULT_MANIFEST_PATH
    return {
        "schema_version": runlog.record_schema_version,
        "kind": RECORD_KIND,
        "run_id": run_id,
        "run_ts": run_ts,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "config_path": str(config_path),
        "config_hash": runlog.config_hash(config_path),
        "dataset": out["dataset"],
        "derived": False,
        "provenance_note": PROVENANCE_NOTE,
        "splits": runlog.splits_block(out["splits_path"]),
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_hash": runlog.dataset_manifest_hash(manifest_path),
        "dataset_manifest_marker": out.get("dataset_manifest_marker"),
        # Per-file SHA-256 / size / row counts, verified by verify_dataset_manifest.
        # The whole-file hash alone would not survive a reformat of the manifest;
        # these do, and they are what a reproducer actually needs.
        "dataset_manifest_files": out.get("dataset_manifest_files"),
        "iceberg_snapshots": out["iceberg_snapshots"],
        "contracts": out["contracts"],
        "item_stats": out["item_stats"],
        "protocol": {
            "eval_split": out["split"],
            "five_core_table": out["five_core_table"],
            "item_features_table": out["item_features_table"],
            "gt_definition": GT_DEFINITION,
            "gt_accounting": out["gt_accounting"],
            "item_stats_coverage": out["coverage"],
            "method": (
                "exact counting aggregation; the item axis is "
                "features/item_train_stats.compute_item_train_stats and the bucket edges / "
                "gate bands are eval/regime_map's, imported verbatim"
            ),
        },
        "axes": {
            axis: {"labels": list(AXIS_LABELS[axis]), "spec": AXIS_SPEC[axis]}
            for axis in AXES
        },
        "seeds": {
            "note": "no stochastic step: exact counts over the frozen TEST window, no bootstrap"
        },
        "results": {
            "headline": out["headline"],
            "gate": out["gate"],
            "contrast": out["contrast"],
        },
        "wall_clock_s": out["wall_clock_s"],
        "hardware": runlog.hardware_string(),
    }


# --- printing ---------------------------------------------------------------------


def _print_report(out: dict) -> None:
    acc = out["gt_accounting"]
    print(
        f"churn contrast · dataset={out['dataset']} split={out['split']} "
        f"users={acc['n_users']} GT interactions={acc['gt_interactions_total']} "
        f"catalog={out['headline']['catalog_size']}"
    )
    if out.get("dataset_manifest_files"):
        print(
            f"  dataset manifest: {out['dataset_manifest_path']} "
            f"({', '.join(sorted(out['dataset_manifest_files']))} verified: sha256 + size"
            " + row counts)"
        )
    print(
        f"  5-core rows (all splits): {acc.get('five_core_rows_total')}"
    )
    print(
        f"  catalog join: {acc['gt_interactions_all_5core']} TEST 5-core interactions -> "
        f"{acc['gt_interactions_total']} on catalog items "
        f"(loss {acc['catalog_join_loss_interactions']} interactions / "
        f"{acc['catalog_join_loss_items']} items)"
    )
    if out["item_stats"]:
        print(f"  item stats: {out['item_stats']['dir']} (bucket counts cross-checked)")

    print("\n--- headline: exact GT shares by item bucket ---")
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

    c = out["contrast"]
    ref = c["reference"]
    print("\n--- regime contrast ---")
    print(
        f"  {ref['dataset']:<20} {ref['value']:.4f} (band {ref['band']}, run {ref['run_id']})"
    )
    print(f"  {c['measured']['dataset']:<20} {c['measured']['value']:.4f} (band {c['measured']['band']})")
    print(f"  difference           {c['difference_vs_reference']:+.4f}")
    print(f"  {c['framing']}")
    print(f"\nwall_clock_s={out['wall_clock_s']}")


# --- CLI ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.eval.churn_contrast")
    parser.add_argument("--config", default="configs/churn_contrast_ml32m.yaml")
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
        out = build_churn_contrast(config, results_path)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report + non-zero exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_report(out)

    record = build_record(config_path, out)
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
    print(f"\nappended kind={RECORD_KIND} run_id={record['run_id']} -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
