"""reproduce-headline pure-Python unit tests (Phase 5, T18).

No Spark: these cover the record locator, the deterministic-field comparison,
the config-hash refusal, and the pinned-cache guard — everything the orchestrator
does BEFORE and AFTER the one Spark step.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.reproduce import (
    FIELDS_COMPARED,
    compare_cache_dirs,
    compare_per_user,
    diff_records,
    find_record,
    load_headline,
    verify_artifact_hashes,
    verify_config_hash,
)

HEADLINE_ID = "20260807T055333Z-c320c79"


def _record(run_id: str = HEADLINE_ID, kind: str = "eval") -> dict:
    """A miniature record with the same shape as the real headline record."""
    return {
        "schema_version": 1,
        "kind": kind,
        "run_id": run_id,
        "run_ts": "2026-08-07T06:14:09.171133+00:00",
        "git_sha": "c320c793956053347171ff170088607f95d31a5e",
        "git_dirty": False,
        "config_path": "configs/eval_blend_test.yaml",
        "config_hash": "sha256:aaaa",
        "splits": {"version": 1, "frozen_at": "2026-08-05", "file_hash": "sha256:bbbb"},
        "dataset_manifest_hash": "sha256:cccc",
        "iceberg_snapshots": {"local.gold.interactions_5core": 8184397443787800955},
        "contracts": {"local.gold.interactions_5core": {"name": "gold_interactions_5core", "version": "1"}},
        "protocol": {"eval_split": "test", "catalog_size": 368228, "k_list": [10, 20, 50]},
        "model": {"name": "content_pop_blend", "params": {"alpha": 0.3}},
        "seeds": {"bootstrap": 20260805, "model": None},
        "metrics": {
            "global": {"ndcg@10": {"value": 0.005726134272789762, "ci_lo": 0.1, "ci_hi": 0.2}},
            "per_segment": {"1-4": {"n_users": 76989}},
        },
        "beyond_accuracy": {"coverage@10": 0.25, "catalog_gini": 0.9},
        "per_user_artifact": "data/eval/per_user/x.parquet",
        "wall_clock_s": 1234.5,
        "hardware": "arm64 · Darwin",
    }


def _write_log(path, records) -> str:
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))
    return str(path)


# --- record locator ----------------------------------------------------------


def test_find_record_returns_the_single_match(tmp_path):
    log = tmp_path / "runs.jsonl"
    _write_log(log, [_record("other"), _record(), _record("later")])
    assert find_record(log, HEADLINE_ID)["run_id"] == HEADLINE_ID


def test_find_record_missing_run_id_errors(tmp_path):
    log = tmp_path / "runs.jsonl"
    _write_log(log, [_record("other")])
    with pytest.raises(ValueError, match="no kind='eval' record"):
        find_record(log, HEADLINE_ID)


def test_find_record_duplicate_run_id_errors(tmp_path):
    """Two records with the pinned run_id is ambiguous, not 'last wins'."""
    log = tmp_path / "runs.jsonl"
    _write_log(log, [_record(), _record()])
    with pytest.raises(ValueError, match="exactly one run"):
        find_record(log, HEADLINE_ID)


def test_find_record_ignores_other_kinds(tmp_path):
    log = tmp_path / "runs.jsonl"
    _write_log(log, [_record(kind="reproduce")])
    with pytest.raises(ValueError, match="no kind='eval' record"):
        find_record(log, HEADLINE_ID)


def test_find_record_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_record(tmp_path / "nope.jsonl", HEADLINE_ID)


# --- comparison --------------------------------------------------------------


def test_identical_records_are_byte_exact():
    rec = _record()
    assert diff_records(rec, copy.deepcopy(rec)) == []


def test_metric_perturbed_in_the_15th_decimal_is_a_mismatch():
    """A last-bits float change must be caught and NAMED — no tolerance anywhere."""
    rec = _record()
    cand = copy.deepcopy(rec)
    original = rec["metrics"]["global"]["ndcg@10"]["value"]
    perturbed = original + 1e-15
    assert perturbed != original  # the perturbation survives float64
    cand["metrics"]["global"]["ndcg@10"]["value"] = perturbed

    diff = diff_records(rec, cand)
    assert len(diff) == 1
    assert diff[0]["path"] == "metrics.global.ndcg@10.value"
    assert diff[0]["recorded"] == original
    assert diff[0]["candidate"] == perturbed


def test_excluded_fields_differing_is_still_byte_exact():
    rec = _record()
    cand = copy.deepcopy(rec)
    cand.update(
        {
            "run_id": "20260808T000000Z-deadbee",
            "run_ts": "2026-08-08T00:00:00+00:00",
            "wall_clock_s": 999.0,
            "hardware": "x86_64 · Linux",
            "git_sha": "deadbeef",
            "git_dirty": True,
            "config_path": "/somewhere/else/eval_blend_test.yaml",
            "per_user_artifact": "data/eval/cache_repro/per_user/y.parquet",
        }
    )
    assert diff_records(rec, cand) == []


def test_diff_names_nested_paths_and_missing_keys():
    rec = _record()
    cand = copy.deepcopy(rec)
    cand["protocol"]["catalog_size"] = 368229
    cand["contracts"]["local.gold.interactions_5core"]["version"] = "2"
    del cand["beyond_accuracy"]["catalog_gini"]
    cand["protocol"]["k_list"] = [10, 20, 100]

    paths = {d["path"] for d in diff_records(rec, cand)}
    assert paths == {
        "protocol.catalog_size",
        "contracts.local.gold.interactions_5core.version",
        "beyond_accuracy.catalog_gini",
        "protocol.k_list[2]",
    }
    missing = [d for d in diff_records(rec, cand) if d["path"] == "beyond_accuracy.catalog_gini"]
    assert missing[0]["candidate_missing"] is True


def test_diff_flags_type_drift_even_when_equal():
    """0.3 vs 0 -> obviously different; 1 vs 1.0 -> equal but schema drift."""
    rec = _record()
    cand = copy.deepcopy(rec)
    cand["splits"]["version"] = 1.0
    diff = diff_records(rec, cand)
    assert [d["path"] for d in diff] == ["splits.version"]


def test_json_roundtrip_normalizes_numpy_scalars():
    """The candidate is compared as it WOULD be serialized: numpy float64 in,
    plain float out, so equality is equality of the bytes runs.jsonl would get."""
    np = pytest.importorskip("numpy")
    rec = _record()
    cand = copy.deepcopy(rec)
    cand["metrics"]["global"]["ndcg@10"]["value"] = np.float64(
        rec["metrics"]["global"]["ndcg@10"]["value"]
    )
    assert diff_records(rec, cand) == []


def test_fields_compared_cover_every_provenance_field():
    """Guard against a future record field silently escaping comparison."""
    from batch_recsys_lab.eval.reproduce import FIELDS_EXCLUDED

    rec = _record()
    assert set(rec) == set(FIELDS_COMPARED) | set(FIELDS_EXCLUDED)


# --- config-hash refusal -----------------------------------------------------


def test_verify_config_hash_passes_on_unchanged_file(tmp_path):
    cfg = tmp_path / "eval.yaml"
    cfg.write_text("model:\n  name: popularity\n")
    verify_config_hash(cfg, runlog.config_hash(cfg))


def test_verify_config_hash_refuses_changed_file(tmp_path):
    cfg = tmp_path / "eval.yaml"
    cfg.write_text("model:\n  name: popularity\n")
    recorded = runlog.config_hash(cfg)
    cfg.write_text("model:\n  name: popularity\n# touched\n")
    with pytest.raises(RuntimeError, match="config file changed"):
        verify_config_hash(cfg, recorded)


def test_verify_config_hash_refuses_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_config_hash(tmp_path / "gone.yaml", "sha256:whatever")


# --- pinned-cache guard ------------------------------------------------------


def test_check_pinned_cache_passes_on_exact_match():
    expected = {"local.gold.interactions_5core": 8184397443787800955, "local.gold.user_stats": 7}
    runlog.check_pinned_cache(dict(expected), expected)


def test_check_pinned_cache_accepts_string_ids():
    """Manifest IDs may arrive as strings from JSON; the guard compares as ints."""
    runlog.check_pinned_cache({"t": "123"}, {"t": 123})


def test_check_pinned_cache_fails_on_differing_id():
    with pytest.raises(RuntimeError, match=r"cache=2 expected=1"):
        runlog.check_pinned_cache({"t": 2}, {"t": 1})


def test_check_pinned_cache_fails_on_missing_table():
    with pytest.raises(RuntimeError, match="Pinned-cache guard"):
        runlog.check_pinned_cache({"t": 1}, {"t": 1, "u": 2})


# --- headline pin ------------------------------------------------------------


def test_load_headline_reads_the_committed_pin():
    cfg = load_headline()
    assert cfg["headline_run_id"] == HEADLINE_ID
    assert cfg["results_path"] == "results/runs.jsonl"
    assert cfg["cache_repro_root"] == "data/eval/cache_repro"


def test_committed_headline_pin_resolves_to_exactly_one_recorded_run():
    """The pin must name a real, unique record in the committed append-only log."""
    from batch_recsys_lab.eval.reproduce import REPO_ROOT

    cfg = load_headline()
    rec = find_record(REPO_ROOT / cfg["results_path"], cfg["headline_run_id"])
    assert rec["protocol"]["eval_split"] == "test"
    assert set(FIELDS_COMPARED).issubset(rec)


def test_load_headline_requires_a_run_id(tmp_path):
    p = tmp_path / "headline.yaml"
    p.write_text("results_path: results/runs.jsonl\n")
    with pytest.raises(ValueError, match="headline_run_id"):
        load_headline(p)


# --- cache directory comparison ----------------------------------------------


def _make_cache(dir_path, *, pair_order=(0, 1, 2), created_ts="t0", extra=None):
    """Minimal cache dir: one order-invariant vector + one TRAIN pair group."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "cache_manifest.json").write_text(
        json.dumps({"created_ts": created_ts, "snapshot_ids": {"t": 1}})
    )
    np.save(dir_path / "n_train.npy", np.array([2, 1, 0], dtype=np.int32))
    order = list(pair_order)
    users = np.array([0, 0, 1], dtype=np.int32)[order]
    items = np.array([5, 7, 5], dtype=np.int32)[order]
    ratings = np.array([5.0, 4.0, 3.0], dtype=np.float32)[order]
    np.save(dir_path / "train_user_idx.npy", users)
    np.save(dir_path / "train_item_idx.npy", items)
    np.save(dir_path / "train_rating.npy", ratings)
    if extra:
        np.save(dir_path / extra, np.array([1], dtype=np.int32))
    return dir_path


def test_cache_compare_identical_dirs_match(tmp_path):
    a = _make_cache(tmp_path / "a")
    # created_ts differs but cache_manifest.json is excluded by construction.
    b = _make_cache(tmp_path / "b", created_ts="t1")
    res = compare_cache_dirs(a, b)
    assert res["files_match"] is True
    assert res["canonical_match"] is True
    assert res["detail"]["excluded_files"] == ["cache_manifest.json"]


def test_cache_compare_pair_row_order_is_canonical_not_strict(tmp_path):
    """The documented, audited case: identical pair CONTENT written in a different
    Spark output order -> strict sha256 fails, order-normalized digest passes."""
    a = _make_cache(tmp_path / "a")
    b = _make_cache(tmp_path / "b", pair_order=(2, 0, 1))
    res = compare_cache_dirs(a, b)
    assert res["files_match"] is False
    assert res["canonical_match"] is True
    assert set(res["detail"]["sha256_mismatches"]) == {
        "train_user_idx.npy",
        "train_item_idx.npy",
        "train_rating.npy",
    }
    assert res["detail"]["canonical_mismatches"] == []


def test_cache_compare_different_pair_content_fails_both(tmp_path):
    a = _make_cache(tmp_path / "a")
    b = _make_cache(tmp_path / "b")
    np.save(b / "train_item_idx.npy", np.array([5, 7, 9], dtype=np.int32))
    res = compare_cache_dirs(a, b)
    assert res["files_match"] is False
    assert res["canonical_match"] is False
    assert res["detail"]["canonical_mismatches"] == ["train"]


def test_cache_compare_order_invariant_file_change_fails_both(tmp_path):
    """A changed non-pair file is never excused by the canonical relaxation."""
    a = _make_cache(tmp_path / "a")
    b = _make_cache(tmp_path / "b")
    np.save(b / "n_train.npy", np.array([2, 1, 1], dtype=np.int32))
    res = compare_cache_dirs(a, b)
    assert res["files_match"] is False
    assert res["canonical_match"] is False
    assert res["detail"]["sha256_mismatches"] == ["n_train.npy"]


def test_cache_compare_missing_file_fails(tmp_path):
    a = _make_cache(tmp_path / "a", extra="pop_train_end_0.npy")
    b = _make_cache(tmp_path / "b")
    res = compare_cache_dirs(a, b)
    assert res["files_match"] is False
    assert res["canonical_match"] is False
    assert res["detail"]["only_in_original"] == ["pop_train_end_0.npy"]


# --- per-user artifact comparison --------------------------------------------


def _write_per_user(path, user_ids, ndcg, top50=None):
    cols = {
        "user_id": pa.array(user_ids, type=pa.string()),
        "ndcg@10": pa.array(ndcg, type=pa.float64()),
        "top50": pa.array(top50 or [[1, 2] for _ in user_ids], type=pa.list_(pa.int32())),
    }
    pq.write_table(pa.table(cols), path)
    return path


def test_per_user_compare_matches_across_row_order(tmp_path):
    a = _write_per_user(tmp_path / "a.parquet", ["U2", "U1"], [0.5, 0.25])
    b = _write_per_user(tmp_path / "b.parquet", ["U1", "U2"], [0.25, 0.5])
    res = compare_per_user(a, b)
    assert res["match"] is True


def test_per_user_compare_catches_a_last_bits_metric_shift(tmp_path):
    a = _write_per_user(tmp_path / "a.parquet", ["U1", "U2"], [0.25, 0.5])
    b = _write_per_user(tmp_path / "b.parquet", ["U1", "U2"], [0.25, 0.5 + 1e-16])
    res = compare_per_user(a, b)
    assert res["match"] is False
    assert res["detail"]["mismatched_columns"] == ["ndcg@10"]


def test_per_user_compare_catches_topk_change_and_row_count(tmp_path):
    a = _write_per_user(tmp_path / "a.parquet", ["U1", "U2"], [0.25, 0.5])
    b = _write_per_user(
        tmp_path / "b.parquet", ["U1", "U2"], [0.25, 0.5], top50=[[1, 2], [2, 1]]
    )
    assert compare_per_user(a, b)["detail"]["mismatched_columns"] == ["top50"]

    c = _write_per_user(tmp_path / "c.parquet", ["U1"], [0.25])
    assert compare_per_user(a, c)["match"] is False


# --- model artifact hashes ---------------------------------------------------


def _make_minilm_artifact(root, snapshot="99", recipe="abc", payload=b"emb"):
    adir = root / "data/eval/minilm" / snapshot / recipe
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "embeddings.npy").write_bytes(payload)
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    (adir / "minilm_manifest.json").write_text(
        json.dumps(
            {
                "embeddings_sha256": digest,
                "item_ids_sha256": "ii",
                "embedding_dim": 384,
            }
        )
    )
    return adir, digest


def _blend_model(digest, snapshot="99", recipe="abc"):
    return {
        "name": "content_pop_blend",
        "params": {
            "alpha": 0.3,
            "artifact_root": "data/eval/minilm",
            "content_params": {
                "recipe_hash": recipe,
                "artifact_root": "data/eval/minilm",
                "five_core_snapshot_id": int(snapshot),
                "embeddings_sha256": digest,
                "item_ids_sha256": "ii",
                "embedding_dim": 384,
            },
        },
    }


def test_artifact_hashes_match_on_disk(tmp_path):
    _, digest = _make_minilm_artifact(tmp_path)
    res = verify_artifact_hashes(_blend_model(digest), tmp_path)
    assert res["match"] is True
    assert all(res["detail"]["checks"].values())


def test_artifact_hashes_detect_a_tampered_artifact(tmp_path):
    adir, digest = _make_minilm_artifact(tmp_path)
    (adir / "embeddings.npy").write_bytes(b"tampered")
    res = verify_artifact_hashes(_blend_model(digest), tmp_path)
    assert res["match"] is False
    assert res["detail"]["checks"]["recomputed_embeddings_sha256"] is False
    assert res["detail"]["checks"]["manifest_embeddings_sha256"] is True


def test_artifact_hashes_skip_when_absent_or_not_carried(tmp_path):
    assert verify_artifact_hashes(_blend_model("deadbeef"), tmp_path)["match"] is None
    pop = {"name": "popularity", "params": {"as_of": "train_end", "window_days": 365}}
    assert verify_artifact_hashes(pop, tmp_path)["match"] is None
