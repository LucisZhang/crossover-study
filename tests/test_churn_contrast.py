"""Churn-contrast math, anchor verification and record shape (Phase 9, T9-3a).

Pure numpy / filesystem — no Spark, no warehouse, no downloaded data. The Spark
half of the job is covered end to end in ``tests/test_ml32m_pipeline.py``.

The bucketing itself is deliberately NOT re-implemented here: the assertions are
hand-computed from a synthetic catalog, so a drift in the imported
``eval/regime_map`` bucket edges (the preregistered T8-1 axes) would fail these
tests rather than silently redefine the published statistic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from batch_recsys_lab.eval.churn_contrast import (
    MISSING_MS,
    RECORD_KIND,
    build_record,
    compute_churn,
    load_item_stats_manifest,
    verify_dataset_manifest,
    verify_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "churn_contrast_ml32m.yaml"
RESULTS_PATH = REPO_ROOT / "results" / "runs.jsonl"

TRAIN_END = datetime(2022, 6, 30, 23, 59, 59, 999000, tzinfo=timezone.utc)
TRAIN_END_MS = int(TRAIN_END.timestamp() * 1000)
DAY_MS = 86_400_000


def _ms(year, month=1, day=1) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


# --------------------------------------------------------------------------- #
# The statistic.
# --------------------------------------------------------------------------- #


def _micro_catalog():
    """Six catalog items spanning every support / recency / first-seen bucket.

    idx support last TRAIN activity          first seen     GT interactions
     0    0     none (absent)                2023-02  post-cutoff   40
     1    3     30d before the cutoff        2021     low/<=90d      5
     2    4     200d before the cutoff       2020     low/91-365d    5
     3   12     500d before the cutoff       2019     high/>365d    25
     4    9     10d before the cutoff        2022-H1  high/<=90d    25
     5    7     none of the above (no GT)    2020     high/>365d     0
    """
    support = np.array([0, 3, 4, 12, 9, 7], dtype=np.int64)
    last_train = np.array(
        [
            MISSING_MS,
            TRAIN_END_MS - 30 * DAY_MS,
            TRAIN_END_MS - 200 * DAY_MS,
            TRAIN_END_MS - 500 * DAY_MS,
            TRAIN_END_MS - 10 * DAY_MS,
            TRAIN_END_MS - 400 * DAY_MS,
        ],
        dtype=np.int64,
    )
    first_seen = np.array(
        [_ms(2023, 2), _ms(2021), _ms(2020), _ms(2019), _ms(2022, 3), _ms(2020)],
        dtype=np.int64,
    )
    gt_counts = np.array([40, 5, 5, 25, 25, 0], dtype=np.int64)
    return support, last_train, first_seen, gt_counts


def test_compute_churn_shares_are_exact_counts():
    support, last_train, first_seen, gt_counts = _micro_catalog()
    out = compute_churn(support, last_train, first_seen, gt_counts, TRAIN_END_MS)
    headline, gate = out["headline"], out["gate"]

    assert headline["gt_interactions_total"] == 100
    assert headline["distinct_gt_items_total"] == 5
    assert headline["catalog_size"] == 6

    support_gt = headline["gt_interactions_by_support"]
    assert support_gt["zero"] == {"n": 40, "share": 0.40}
    assert support_gt["low"] == {"n": 10, "share": 0.10}
    assert support_gt["high"] == {"n": 50, "share": 0.50}
    # Distinct items and the catalog are counted per item, not per interaction.
    assert [headline["distinct_gt_items_by_support"][b]["n"] for b in ("zero", "low", "high")] == [
        1,
        2,
        2,
    ]
    assert [headline["catalog_items_by_support"][b]["n"] for b in ("zero", "low", "high")] == [
        1,
        2,
        3,
    ]

    recency_gt = headline["gt_interactions_by_recency"]
    assert [recency_gt[b]["n"] for b in ("<=90d", "91-365d", ">365d", "absent")] == [
        30,
        5,
        25,
        40,
    ]
    first_gt = headline["gt_interactions_by_first_seen"]
    assert [
        first_gt[b]["n"] for b in ("<=2019", "2020", "2021", "2022-H1", "post-cutoff")
    ] == [25, 5, 5, 25, 40]

    # zero + low, the T8-1 headline statistic.
    assert gate["measured_share"] == pytest.approx(0.50)
    assert gate["band"] == ">=0.25"


@pytest.mark.parametrize(
    "zero_gt, high_gt, expected_share, expected_band",
    [
        (5, 95, 0.05, "<0.10"),  # near-total TRAIN/TEST catalog overlap
        (10, 90, 0.10, "0.10-0.25"),  # exactly at the lower edge -> partial support
        (24, 76, 0.24, "0.10-0.25"),
        (25, 75, 0.25, ">=0.25"),  # exactly at the upper edge -> supported
        (80, 20, 0.80, ">=0.25"),
    ],
)
def test_preregistered_bands_at_their_edges(zero_gt, high_gt, expected_share, expected_band):
    support = np.array([0, 9], dtype=np.int64)
    last_train = np.array([MISSING_MS, TRAIN_END_MS - DAY_MS], dtype=np.int64)
    first_seen = np.array([_ms(2023, 2), _ms(2020)], dtype=np.int64)
    gt_counts = np.array([zero_gt, high_gt], dtype=np.int64)

    gate = compute_churn(support, last_train, first_seen, gt_counts, TRAIN_END_MS)["gate"]
    assert gate["measured_share"] == pytest.approx(expected_share)
    assert gate["band"] == expected_band
    assert gate["verdict"] == gate["verdicts"][expected_band]


def test_empty_ground_truth_is_all_zero_shares_not_a_divide_by_zero():
    support, last_train, first_seen, _ = _micro_catalog()
    gt_counts = np.zeros_like(support)
    out = compute_churn(support, last_train, first_seen, gt_counts, TRAIN_END_MS)
    assert out["headline"]["gt_interactions_total"] == 0
    assert out["gate"]["measured_share"] == 0.0
    assert out["gate"]["band"] == "<0.10"


def test_misaligned_arrays_are_rejected():
    with pytest.raises(ValueError, match="aligned to the item catalog"):
        compute_churn(
            np.array([0, 1]),
            np.array([MISSING_MS, 0]),
            np.array([0, 0]),
            np.array([1, 2, 3]),
            TRAIN_END_MS,
        )


# --------------------------------------------------------------------------- #
# Contrast anchor.
# --------------------------------------------------------------------------- #


def _anchor_record(run_id: str, share: float) -> str:
    return json.dumps(
        {"kind": "regime_map", "run_id": run_id, "results": {"gate": {"measured_share": share}}}
    )


def test_verify_reference_reads_the_share_off_the_named_record(tmp_path):
    log = tmp_path / "runs.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"kind": "eval", "run_id": "other"}),
                _anchor_record("anchor", 0.4111295514585914),
            ]
        )
        + "\n"
    )
    out = verify_reference(
        {"dataset": "amazon_electronics", "run_id": "anchor", "value": 0.4111295514585914}, log
    )
    assert out["value"] == 0.4111295514585914
    assert out["band"] == ">=0.25"
    assert out["verified_against"] == str(log)


def test_verify_reference_rejects_a_drifted_or_ambiguous_anchor(tmp_path):
    log = tmp_path / "runs.jsonl"
    log.write_text(_anchor_record("anchor", 0.30) + "\n")
    with pytest.raises(RuntimeError, match="reference anchor mismatch"):
        verify_reference({"run_id": "anchor", "value": 0.4111295514585914}, log)

    log.write_text("\n".join([_anchor_record("anchor", 0.30)] * 2) + "\n")
    with pytest.raises(RuntimeError, match="expected exactly 1"):
        verify_reference({"run_id": "anchor", "value": 0.30}, log)

    log.write_text("")
    with pytest.raises(RuntimeError, match="expected exactly 1"):
        verify_reference({"run_id": "anchor", "value": 0.30}, log)


def test_shipped_config_anchor_matches_the_committed_amazon_run():
    # The 0.4111 the whole contrast hangs on must still be the recorded number.
    config = yaml.safe_load(CONFIG_PATH.read_text())
    reference = verify_reference(config["reference"], RESULTS_PATH)
    assert reference["run_id"] == "20260817T095926Z-633d454"
    assert round(reference["value"], 4) == 0.4111
    assert reference["band"] == ">=0.25"


def test_dataset_manifest_must_mention_this_dataset(tmp_path):
    manifest = tmp_path / "MANIFEST.md"
    manifest.write_text("# Amazon Reviews 2023 — Electronics — Data Manifest\n")
    with pytest.raises(RuntimeError, match="ml-32m.zip"):
        verify_dataset_manifest(manifest, "ml-32m.zip")

    manifest.write_text("...\n- URL: https://files.grouplens.org/.../ml-32m.zip\n")
    assert verify_dataset_manifest(manifest, "ml-32m.zip") == "ml-32m.zip"
    # No marker declared => no claim made, nothing to check.
    assert verify_dataset_manifest(manifest, None) == ""
    with pytest.raises(RuntimeError, match="does not exist"):
        verify_dataset_manifest(tmp_path / "nope.md", "ml-32m.zip")


def test_shipped_config_declares_the_manifest_guard():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config["dataset_manifest_path"] == "data/MANIFEST.md"
    assert config["dataset_manifest_must_contain"] == "ml-32m.zip"
    # Live behaviour, whichever state the shared manifest is in: before the owner
    # records the ML-32M SHA-256s the job must refuse; after, it must accept. (As
    # of T9-3a the manifest is still Amazon-only, so this takes the refusal path.)
    manifest_path = REPO_ROOT / config["dataset_manifest_path"]
    marker = config["dataset_manifest_must_contain"]
    if marker in manifest_path.read_text():
        assert verify_dataset_manifest(manifest_path, marker) == marker
    else:
        with pytest.raises(RuntimeError, match="ml-32m.zip"):
            verify_dataset_manifest(manifest_path, marker)


def test_shipped_config_points_at_the_ml32m_lane_only():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config["dataset"] == "ml32m"
    assert config["split"] == "test"
    assert config["splits_path"] == "configs/splits_ml32m.yaml"
    assert config["five_core_table"] == "local.gold_ml32m.interactions_5core"
    assert config["item_features_table"] == "local.gold_ml32m.item_features"
    # No Amazon table may appear anywhere in the ML-32M job's inputs.
    for key in ("five_core_table", "item_features_table"):
        assert ".gold." not in config[key]


# --------------------------------------------------------------------------- #
# Provenance: item-stats manifest resolution + the run record.
# --------------------------------------------------------------------------- #


def _write_stats_manifest(root: Path, snapshot_id: int) -> Path:
    stats_dir = root / str(snapshot_id)
    stats_dir.mkdir(parents=True)
    (stats_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "interactions_5core_snapshot_id": snapshot_id,
                "train_end": TRAIN_END.isoformat(),
                "support_bucket_counts": {"zero": 1, "low": 2, "high": 3},
            }
        )
    )
    return stats_dir


def test_item_stats_manifest_must_match_the_live_snapshot(tmp_path):
    root = tmp_path / "item_train_stats_ml32m"
    stats_dir = _write_stats_manifest(root, 12345)

    resolved, manifest = load_item_stats_manifest(root, 12345)
    assert resolved == stats_dir
    assert manifest["interactions_5core_snapshot_id"] == 12345
    # The snapshot subdir may also be passed directly.
    assert load_item_stats_manifest(stats_dir, 12345)[0] == stats_dir

    with pytest.raises(RuntimeError, match="rebuild it"):
        load_item_stats_manifest(stats_dir, 999)
    with pytest.raises(RuntimeError, match="no item_train_stats manifest"):
        load_item_stats_manifest(root, 999)


def test_record_carries_the_full_provenance_manifest(tmp_path):
    out = {
        "dataset": "ml32m",
        "split": "test",
        "five_core_table": "local.gold_ml32m.interactions_5core",
        "item_features_table": "local.gold_ml32m.item_features",
        "splits_path": str(REPO_ROOT / "configs" / "splits_ml32m.yaml"),
        "dataset_manifest_path": str(REPO_ROOT / "data" / "MANIFEST.md"),
        "dataset_manifest_marker": "ml-32m.zip",
        "iceberg_snapshots": {
            "local.gold_ml32m.interactions_5core": 111,
            "local.gold_ml32m.item_features": 222,
        },
        "contracts": {
            "local.gold_ml32m.interactions_5core": {
                "name": "gold_ml32m_interactions_5core",
                "version": "1",
            }
        },
        "item_stats": None,
        "coverage": {"catalog_size": 3},
        "gt_accounting": {"gt_interactions_total": 4},
        "headline": {"eval_split": "test", "catalog_size": 3},
        "gate": {"measured_share": 0.75, "band": ">=0.25"},
        "contrast": {
            "reference": {"dataset": "amazon_electronics", "value": 0.4111295514585914},
            "measured": {"dataset": "ml32m", "value": 0.75},
        },
        "wall_clock_s": 1.0,
    }
    config_path = tmp_path / "churn.yaml"
    config_path.write_text("dataset: ml32m\n")

    record = build_record(config_path, out)

    assert record["kind"] == RECORD_KIND
    assert record["schema_version"] == 1
    assert record["dataset"] == "ml32m"
    # Invariant #3: config hash, git SHA, dataset manifest hash, snapshot IDs.
    assert record["config_hash"].startswith("sha256:")
    assert record["dataset_manifest_hash"].startswith("sha256:")
    assert record["git_sha"] and isinstance(record["git_dirty"], bool)
    assert record["iceberg_snapshots"] == out["iceberg_snapshots"]
    assert record["contracts"] == out["contracts"]
    # The frozen ML-32M split, not the Amazon one.
    assert record["splits"] == {
        "version": 1,
        "frozen_at": "2026-08-19",
        "file_hash": record["splits"]["file_hash"],
    }
    # The Amazon contrast anchor travels inside the payload.
    assert round(record["results"]["contrast"]["reference"]["value"], 4) == 0.4111
    assert record["results"]["gate"]["measured_share"] == 0.75
    assert "no stochastic step" in record["seeds"]["note"]
    assert record["protocol"]["gt_definition"]["difference"]
    # Serializable as one append-only JSON line.
    assert json.loads(json.dumps(record))["run_id"] == record["run_id"]
