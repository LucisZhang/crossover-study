"""TracedWriter round-trip + every failure mode of the independent verifier
(Phase 6, T26).

The point of the pair is that they share no code: the writer builds the
manifest, the verifier re-derives the same facts from the documents, the log
and the artifacts. Each test below breaks exactly one link and asserts the
verifier names it.
"""

from __future__ import annotations

import json

import pytest

from batch_recsys_lab.demo.export_core import (
    TraceManifest,
    TracedWriter,
    index_runs,
    iter_leaves,
    jp,
    parse_pointer,
    resolve_pointer,
    sha256_file,
    write_document,
)
from batch_recsys_lab.demo.export_crossover import build as build_crossover
from batch_recsys_lab.demo.export_lineage import build_lineage, build_timetravel
from batch_recsys_lab.demo.export_policy_grid import build as build_policy_grid
from batch_recsys_lab.demo.export_receipts import build as build_receipts
from batch_recsys_lab.demo.export_receipts import closure
from batch_recsys_lab.demo.verify_traceability import verify

SEGMENTS = ["0", "1-4"]
METRICS = ["ndcg@10", "recall@20"]


# --- synthetic evidence -------------------------------------------------------


def _eval_record(run_id, model="popularity", base=0.01):
    def block(mult):
        # Distinct value per metric, so a test that re-points an entry at the
        # wrong metric actually changes the number.
        return {
            m: {"value": base * mult * (i + 1), "ci_lo": base * mult - 0.001, "ci_hi": base * mult + 0.001}
            for i, m in enumerate(METRICS)
        }

    return {
        "schema_version": 1,
        "kind": "eval",
        "run_id": run_id,
        "run_ts": "2026-08-05T00:00:00+00:00",
        "git_sha": "0" * 40,
        "git_dirty": False,
        "config_path": f"configs/{model}.yaml",
        "config_hash": "sha256:" + "a" * 64,
        "splits": {"version": 1, "frozen_at": "2026-08-05", "file_hash": "sha256:" + "b" * 64},
        "dataset_manifest_hash": "sha256:" + "c" * 64,
        "iceberg_snapshots": {"local.gold.interactions_5core": 8184397443787800955},
        "contracts": {},
        "protocol": {"eval_split": "test", "catalog_size": 100, "n_users": 30},
        "model": {"name": model, "params": {"window_days": 365}},
        "seeds": {"bootstrap": 20260805, "model": None},
        "metrics": {
            "global": block(1.0),
            "per_segment": {seg: {"n_users": 10 * (i + 1), **block(i + 1)} for i, seg in enumerate(SEGMENTS)},
        },
        "beyond_accuracy": {},
        "per_user_artifact": f"data/eval/per_user/{run_id}_{model}.parquet",
        "wall_clock_s": 12.5,
        "hardware": "arm64 · Darwin",
    }


def _paired_delta_record(run_id, a_id, b_id):
    def block():
        return {m: {"delta": 0.002, "ci_lo": 0.001, "ci_hi": 0.003, "excludes_zero": True} for m in METRICS}

    return {
        "schema_version": 1,
        "kind": "paired_delta",
        "run_id": run_id,
        "run_ts": "2026-08-05T01:00:00+00:00",
        "git_sha": "1" * 40,
        "git_dirty": False,
        "config_path": "configs/compare.yaml",
        "config_hash": "sha256:" + "d" * 64,
        "a": {"run_id": a_id, "model": "blend", "artifact": "x.parquet"},
        "b": {"run_id": b_id, "model": "popularity", "artifact": "y.parquet"},
        "n_common_users": 30,
        "deltas": {"global": block(), "per_segment": {seg: block() for seg in SEGMENTS}},
    }


def _reproduce_record(run_id, target):
    return {
        "schema_version": 1,
        "kind": "reproduce",
        "run_id": run_id,
        "run_ts": "2026-08-07T00:00:00+00:00",
        "git_sha": "2" * 40,
        "git_dirty": False,
        "reproduces_run_id": target,
        "verdict": "byte_exact",
        "hardware": "arm64 · Darwin",
    }


@pytest.fixture()
def evidence(tmp_path):
    """A miniature repo: runs log, demo/data dir, and the export config."""
    runs_log = tmp_path / "results" / "runs.jsonl"
    runs_log.parent.mkdir(parents=True)
    records = [
        _eval_record("R-blend", model="content_pop_blend", base=0.02),
        _eval_record("R-pop", model="popularity", base=0.01),
        _paired_delta_record("R-delta", "R-blend", "R-pop"),
        _reproduce_record("R-repro", "R-blend"),
    ]
    with open(runs_log, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    data_dir = tmp_path / "demo" / "data"
    cfg = {
        "runs_log": str(runs_log),
        "out_dir": str(data_dir),
        "manifest": str(data_dir / "trace_manifest.json"),
        "headline_run_id": "R-blend",
        "repro_command": "make reproduce-headline",
        "split": "test",
        "segments": SEGMENTS,
        "metrics": METRICS,
        "title": "t",
        "models": [
            {"key": "blend", "label": "blend", "run_id": "R-blend", "highlight": True},
            {"key": "pop", "label": "pop", "run_id": "R-pop"},
        ],
        "paired_deltas": [{"key": "blend_vs_pop", "label": "blend - pop", "run_id": "R-delta"}],
    }
    return {"root": tmp_path, "runs_log": runs_log, "data_dir": data_dir, "cfg": cfg}


def _export(evidence):
    cfg = evidence["cfg"]
    runs = index_runs(cfg["runs_log"])
    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    write_document(build_crossover(cfg, runs), cfg["out_dir"], manifest)
    manifest.write()
    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    seeds = manifest.run_ids() | {"R-blend", "R-repro"}
    run_ids = closure(seeds, runs)
    write_document(build_receipts(cfg, runs, run_ids), cfg["out_dir"], manifest)
    manifest.write()
    return manifest


def _verify(evidence, mode="full"):
    cfg = evidence["cfg"]
    return verify(
        cfg["out_dir"], cfg["manifest"], runs_log=cfg["runs_log"], mode=mode, repo_root=evidence["root"]
    )


def _classes(report):
    return {kind for kind, _ in report.failures}


def _rewrite(path, mutate):
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return doc


# --- pointers -----------------------------------------------------------------


def test_pointer_escaping_round_trip():
    ptr = jp("metrics", "ndcg@10", "a/b", "c~d")
    assert ptr == "/metrics/ndcg@10/a~1b/c~0d"
    assert parse_pointer(ptr) == ["metrics", "ndcg@10", "a/b", "c~d"]
    doc = {"metrics": {"ndcg@10": {"a/b": {"c~d": 1.5}}}}
    assert resolve_pointer(doc, ptr) == 1.5
    assert dict(iter_leaves(doc))[ptr] == 1.5


def test_resolve_pointer_reports_the_failing_step():
    with pytest.raises(KeyError, match="no key 'nope'"):
        resolve_pointer({"a": {"b": 1}}, "/a/nope")


# --- writer -------------------------------------------------------------------


def test_traced_writer_round_trip(evidence):
    _export(evidence)
    rep = _verify(evidence)
    assert rep.ok, rep.failures
    assert rep.counts["numeric_leaves"] > 0
    assert rep.counts["src_runs_record"] == rep.counts["entries"]

    doc = json.loads((evidence["data_dir"] / "crossover.json").read_text())
    # Full precision, straight off the record — not a re-derivation.
    assert doc["models"]["blend"]["global"]["ndcg@10"]["value"] == 0.02
    assert doc["models"]["blend"]["segments"]["1-4"]["n_users"] == 20
    receipts = json.loads((evidence["data_dir"] / "receipts.json").read_text())
    assert receipts["runs"]["R-blend"]["reproduce"] == [{"run_id": "R-repro", "verdict": "byte_exact"}]
    assert receipts["runs"]["R-blend"]["repro_command"] == "make reproduce-headline"
    assert "iceberg_snapshots" in receipts["runs"]["R-blend"]


def test_export_is_byte_stable(evidence):
    _export(evidence)
    before = {p.name: p.read_bytes() for p in evidence["data_dir"].glob("*.json")}
    _export(evidence)
    after = {p.name: p.read_bytes() for p in evidence["data_dir"].glob("*.json")}
    assert before["crossover.json"] == after["crossover.json"]
    assert before["receipts.json"] == after["receipts.json"]


def test_writer_is_write_once_and_self_checks(evidence):
    runs = index_runs(evidence["runs_log"])
    w = TracedWriter("x.json", runs)
    w.copy_from_record("/a", "R-pop", "/metrics/global/ndcg@10/value")
    with pytest.raises(KeyError, match="write-once"):
        w.put_descriptive("/a", 1)
    assert w.untraced_numeric_leaves() == []
    # Anything that bypasses the traced methods is caught before it is written.
    w._set("/sneaky", 0.123)
    assert w.untraced_numeric_leaves() == ["/sneaky"]
    with pytest.raises(AssertionError, match="without a trace entry"):
        write_document(w, evidence["data_dir"], TraceManifest(evidence["cfg"]["manifest"], evidence["runs_log"]))


def test_descriptive_subtree_absorbs_only_declared_leaves(evidence):
    runs = index_runs(evidence["runs_log"])
    w = TracedWriter("x.json", runs)
    w.put_descriptive("/history", [{"price": 9.99}, {"price": 1.0}], subtree=True)
    w.put_descriptive("/one", 3)
    assert w.untraced_numeric_leaves() == []
    manifest = TraceManifest(evidence["cfg"]["manifest"], evidence["runs_log"])
    write_document(w, evidence["data_dir"], manifest)
    manifest.write()
    rep = _verify(evidence)
    assert rep.ok, rep.failures
    assert rep.counts["subtree_absorbed_leaves"] == 2


def test_config_validation(tmp_path):
    from batch_recsys_lab.demo.export_core import load_export_config

    p = tmp_path / "c.yaml"
    p.write_text("split: test\n")
    with pytest.raises(ValueError, match="missing required keys"):
        load_export_config(p)


def test_closure_pulls_in_compared_and_reproduced_runs(evidence):
    runs = index_runs(evidence["runs_log"])
    assert closure({"R-delta"}, runs) == ["R-blend", "R-delta", "R-pop"]
    assert closure({"R-repro"}, runs) == ["R-blend", "R-repro"]
    with pytest.raises(ValueError, match="not in the runs log"):
        closure({"nope"}, runs)


# --- verifier failure modes ---------------------------------------------------


def test_perturbed_value_fails(evidence):
    _export(evidence)
    path = evidence["data_dir"] / "crossover.json"

    def bump(doc):
        doc["models"]["blend"]["global"]["ndcg@10"]["value"] = 0.0200001

    _rewrite(path, bump)
    rep = _verify(evidence)
    assert not rep.ok
    # The document hash moved AND the leaf no longer matches the manifest.
    assert {"FILE_HASH", "DOC_MISMATCH"} <= _classes(rep)


def test_perturbed_value_with_repaired_file_hash_still_fails(evidence):
    """The hash check is a convenience, not the guarantee: repair it and the
    leaf-level exact-match check still catches the edit."""
    manifest = _export(evidence)
    path = evidence["data_dir"] / "crossover.json"
    _rewrite(path, lambda d: d["models"]["blend"]["segments"]["0"].__setitem__("n_users", 11))
    manifest.files["crossover.json"]["sha256"] = sha256_file(path)
    manifest.write()
    rep = _verify(evidence)
    assert _classes(rep) == {"DOC_MISMATCH"}


def test_int_float_restatement_is_a_mismatch(evidence):
    """1 vs 1.0 is not 'equal enough' — the demo must carry the record's type."""
    manifest = _export(evidence)
    path = evidence["data_dir"] / "crossover.json"
    _rewrite(path, lambda d: d["models"]["blend"]["segments"]["0"].__setitem__("n_users", 10.0))
    manifest.files["crossover.json"]["sha256"] = sha256_file(path)
    manifest.write()
    assert "DOC_MISMATCH" in _classes(_verify(evidence))


def test_dropped_manifest_entry_fails(evidence):
    manifest = _export(evidence)
    target = "/models/blend/global/ndcg@10/value"
    manifest.entries = [e for e in manifest.entries if e["pointer"] != target]
    manifest.write()
    rep = _verify(evidence)
    assert "UNCOVERED" in _classes(rep)
    assert any(target in msg for _, msg in rep.failures)


def test_orphan_manifest_entry_fails(evidence):
    manifest = _export(evidence)
    entry = dict(manifest.entries[0])
    entry["pointer"] = "/models/blend/global/ndcg@10/does_not_exist"
    manifest.entries.append(entry)
    manifest.write()
    assert "ORPHAN" in _classes(_verify(evidence))


def test_source_mismatch_fails(evidence):
    """Manifest and document agree with each other but not with the log."""
    manifest = _export(evidence)
    for e in manifest.entries:
        if e["file"] == "crossover.json" and e["pointer"].endswith("/models/pop/global/ndcg@10/value"):
            e["source"]["source_pointer"] = "/metrics/global/recall@20/value"
    manifest.write()
    assert "SOURCE_MISMATCH" in _classes(_verify(evidence))


def test_missing_run_fails(evidence):
    manifest = _export(evidence)
    manifest.entries[0]["source"]["run_id"] = "R-ghost"
    manifest.write()
    rep = _verify(evidence)
    assert {"SOURCE_MISSING", "RECEIPTS"} <= _classes(rep)


def test_stale_runs_log_fails(evidence):
    _export(evidence)
    with open(evidence["runs_log"], "a") as fh:
        fh.write(json.dumps(_eval_record("R-later")) + "\n")
    rep = _verify(evidence)
    assert "STALE" in _classes(rep)
    # …and re-exporting against the moved log clears it.
    _export(evidence)
    assert _verify(evidence).ok


def test_undeclared_file_fails(evidence):
    _export(evidence)
    (evidence["data_dir"] / "stray.json").write_text('{"n": 1}\n')
    assert "FILESET" in _classes(_verify(evidence))


def test_missing_receipt_card_fails(evidence):
    _export(evidence)
    path = evidence["data_dir"] / "receipts.json"
    manifest_path = evidence["data_dir"] / "trace_manifest.json"

    def drop(doc):
        doc["runs"].pop("R-pop")

    _rewrite(path, drop)
    m = json.loads(manifest_path.read_text())
    m["entries"] = [e for e in m["entries"] if not e["pointer"].startswith("/runs/R-pop")]
    m["files"] = [
        {**f, "sha256": sha256_file(path)} if f["name"] == "receipts.json" else f for f in m["files"]
    ]
    manifest_path.write_text(json.dumps(m, indent=2) + "\n")
    rep = _verify(evidence)
    assert "RECEIPTS" in _classes(rep)


def test_missing_manifest_fails(evidence):
    rep = _verify(evidence)
    assert "MANIFEST" in _classes(rep)


# --- the other two source kinds ----------------------------------------------


def test_results_artifact_source_round_trip_and_drift(evidence, tmp_path):
    """A number from a results artifact is evidence only while the artifact
    still hashes to what its anchoring record recorded."""
    artifact = tmp_path / "results" / "lineage.json"
    artifact.write_text(json.dumps({"stages_count": 24, "stages": [{"rows_out": 43886944}]}))
    runs_log = evidence["runs_log"]
    lineage_rec = {
        "schema_version": 1,
        "kind": "lineage",
        "run_id": "R-lineage",
        "run_ts": "2026-08-07T16:09:10+00:00",
        "git_sha": "3" * 40,
        "git_dirty": False,
        "artifact_sha256": sha256_file(artifact),
    }
    with open(runs_log, "a") as fh:
        fh.write(json.dumps(lineage_rec) + "\n")

    cfg = evidence["cfg"]
    runs = index_runs(runs_log)
    manifest = TraceManifest(cfg["manifest"], runs_log)
    w = TracedWriter("lineage.json", runs)
    w.register_artifact("lineage", artifact, run_id="R-lineage", anchor_pointer="/artifact_sha256")
    w.copy_from_artifact("/stages_count", "lineage", "/stages_count")
    w.copy_from_artifact("/rows", "lineage", "/stages/0/rows_out")
    write_document(w, cfg["out_dir"], manifest)
    # receipts must cover the anchoring record too
    write_document(build_receipts(cfg, runs, ["R-blend", "R-repro", "R-lineage"]), cfg["out_dir"], manifest)
    manifest.write()
    rep = _verify(evidence)
    assert rep.ok, rep.failures
    assert rep.counts["src_results_artifact"] == 2

    # Drift the artifact: same pointer, different bytes -> refused.
    artifact.write_text(json.dumps({"stages_count": 24, "stages": [{"rows_out": 43886944}], "x": 1}))
    assert "SOURCE_HASH" in _classes(_verify(evidence))

    # And the writer refuses to bind a drifted artifact in the first place.
    with pytest.raises(ValueError, match="drifted from the record"):
        TracedWriter("z.json", index_runs(runs_log)).register_artifact(
            "lineage", artifact, run_id="R-lineage", anchor_pointer="/artifact_sha256"
        )


def test_per_user_artifact_source_full_vs_record_mode(evidence, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    from batch_recsys_lab.demo.export_core import source_per_user_artifact

    rel = "data/eval/per_user/R-blend_content_pop_blend.parquet"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    table = pa.table(
        {
            "user_index": pa.array([10, 17], pa.int32()),
            "ndcg@10": pa.array([0.5, 0.25], pa.float64()),
            "top50": pa.array([[1, 2], [3, 4]], pa.list_(pa.int32())),
        }
    )
    pq.write_table(table, path)

    cfg = evidence["cfg"]
    runs = index_runs(cfg["runs_log"])
    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    w = TracedWriter("shoppers.json", runs)
    digest = sha256_file(path)
    src = lambda rp: source_per_user_artifact(  # noqa: E731
        parquet_path=rel, sha256=digest, row_pointer=rp, run_id="R-blend"
    )
    w.put("/s/0/ndcg@10", 0.25, src("user_index=17/ndcg@10"))
    w.put("/s/0/top1", 3, src("user_index=17/top50/0"))
    write_document(w, cfg["out_dir"], manifest)
    write_document(build_receipts(cfg, runs, ["R-blend", "R-repro"]), cfg["out_dir"], manifest)
    manifest.write()

    rep = _verify(evidence, mode="full")
    assert rep.ok, rep.failures
    assert rep.counts["src_per_user_artifact"] == 2

    # record mode does not open the parquet at all
    rep = _verify(evidence, mode="record")
    assert rep.ok, rep.failures
    assert rep.counts["per_user_skipped"] == 2
    path.unlink()
    assert _verify(evidence, mode="record").ok
    assert "SOURCE_MISSING" in _classes(_verify(evidence, mode="full"))


def test_per_user_row_pointer_must_select_one_row(evidence, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    from batch_recsys_lab.demo.export_core import source_per_user_artifact

    rel = "data/eval/per_user/R-blend_content_pop_blend.parquet"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"user_index": [7, 7], "ndcg@10": [0.5, 0.5]}), path)

    cfg = evidence["cfg"]
    runs = index_runs(cfg["runs_log"])
    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    w = TracedWriter("shoppers.json", runs)
    w.put(
        "/s/0/ndcg@10",
        0.5,
        source_per_user_artifact(
            parquet_path=rel, sha256=sha256_file(path), row_pointer="user_index=7/ndcg@10", run_id="R-blend"
        ),
    )
    write_document(w, cfg["out_dir"], manifest)
    write_document(build_receipts(cfg, runs, ["R-blend", "R-repro"]), cfg["out_dir"], manifest)
    manifest.write()
    rep = _verify(evidence, mode="full")
    assert "SOURCE_MISSING" in _classes(rep)
    assert any("matched 2 rows" in m for _, m in rep.failures)


# --- exporter guards ----------------------------------------------------------


def test_crossover_rejects_wrong_split(evidence):
    runs = index_runs(evidence["runs_log"])
    cfg = dict(evidence["cfg"], split="val")
    with pytest.raises(ValueError, match="eval_split"):
        build_crossover(cfg, runs)


def test_crossover_rejects_unknown_run(evidence):
    runs = index_runs(evidence["runs_log"])
    cfg = dict(evidence["cfg"], models=[{"key": "x", "label": "x", "run_id": "R-ghost"}])
    with pytest.raises(ValueError, match="not found"):
        build_crossover(cfg, runs)


def test_receipts_requires_a_reproduce_record(evidence):
    runs = index_runs(evidence["runs_log"])
    runs = {k: v for k, v in runs.items() if k != "R-repro"}
    with pytest.raises(ValueError, match="no kind='reproduce' record"):
        build_receipts(evidence["cfg"], runs, ["R-blend"])


# --- export_policy_grid (T27) --------------------------------------------------

N_STAR_LABELS = ["0", "1", "5", "10", "20", "inf"]


def _policy_grid_cell(variant, label, mult):
    n_star = {"0": 0, "1": 1, "5": 5, "10": 10, "20": 20, "inf": None}[label]

    def block(m):
        return {
            met: {"value": 0.01 * m * mult, "ci_lo": 0.01 * m * mult - 0.001, "ci_hi": 0.01 * m * mult + 0.001}
            for met in METRICS
        }

    return {
        "variant": variant,
        "low": "blend_a30",
        "high": "als_chosen" if variant == "A" else "pop_t12m",
        "n_star": n_star,
        "n_star_label": label,
        "low_share": {"0": 0.0, "1": 0.1, "5": 0.3, "10": 0.6, "20": 0.9, "inf": 1.0}[label],
        "global": block(1.0),
        "per_segment": {seg: {"n_users": 10 * (i + 1), **block(i + 1)} for i, seg in enumerate(SEGMENTS)},
    }


def _policy_grid_record(run_id, val_sha256):
    grid = []
    for variant in ("A", "B"):
        for i, label in enumerate(N_STAR_LABELS):
            grid.append(_policy_grid_cell(variant, label, mult=1.0 + 0.1 * i))
    return {
        "schema_version": 1,
        "kind": "policy_grid",
        "run_id": run_id,
        "run_ts": "2026-08-08T03:06:59+00:00",
        "git_sha": "4" * 40,
        "git_dirty": False,
        "config_path": "configs/policy_grid_test.yaml",
        "config_hash": "sha256:" + "e" * 64,
        "split": "test",
        "n_star_grid": [0, 1, 5, 10, 20, None],
        "source_run_ids": {"blend_a30": "R-blend", "als_chosen": "R-als", "pop_t12m": "R-pop"},
        "identity_checks": {"inf_matches_blend": True, "zero_matches_high": True},
        "val_selection_ref": {"output": "results/policy_select_val.json", "winner": "B/inf", "sha256": val_sha256},
        "grid": grid,
        "seeds": {"bootstrap": 20260805},
        "bootstrap": {"n_resamples": 1000, "seed": 20260805},
    }


def _val_selection_doc():
    return {
        "run_ids": {"blend_a30": "R-blend", "als_chosen": "R-als", "pop_t12m": "R-pop"},
        "objective": "segment_weighted_ndcg10_unweighted_mean",
        "metric": "ndcg@10",
        "grid": [
            {
                "variant": "B",
                "low": "blend_a30",
                "high": "pop_t12m",
                "n_star": None,
                "n_star_label": "inf",
                "segment_means": {seg: 0.01 * (i + 1) for i, seg in enumerate(SEGMENTS)},
                "objective": 0.0123,
            }
        ],
        "winner": {
            "variant": "B",
            "low": "blend_a30",
            "high": "pop_t12m",
            "n_star": None,
            "n_star_label": "inf",
            "segment_means": {seg: 0.01 * (i + 1) for i, seg in enumerate(SEGMENTS)},
            "objective": 0.0123,
        },
        "git_sha": "5" * 40,
    }


@pytest.fixture()
def policy_grid_evidence(evidence, tmp_path):
    """Extends the base ``evidence`` fixture with a policy_grid record + its
    anchored VAL-selection artifact."""
    val_path = tmp_path / "results" / "policy_select_val.json"
    val_path.write_text(json.dumps(_val_selection_doc()))
    val_sha256 = sha256_file(val_path)

    rec = _policy_grid_record("R-grid", val_sha256)
    with open(evidence["runs_log"], "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    cfg = dict(
        evidence["cfg"],
        policy_grid={
            "run_id": "R-grid",
            "variants": {
                "A": {"label": "blend -> ALS", "low": "blend", "high": "als"},
                "B": {"label": "blend -> pop-t12m", "low": "blend", "high": "pop_t12m"},
            },
            "shipped": {"variant": "B", "n_star_label": "inf"},
        },
        artifacts={
            "policy_select_val": {
                "path": str(val_path),
                "run_id": "R-grid",
                "anchor_pointer": "/val_selection_ref/sha256",
            }
        },
    )
    evidence = dict(evidence, cfg=cfg, val_path=val_path)
    return evidence


def test_policy_grid_export_round_trip(policy_grid_evidence):
    evidence = policy_grid_evidence
    cfg = evidence["cfg"]
    runs = index_runs(evidence["runs_log"])
    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    w = build_policy_grid(cfg, runs)
    write_document(w, cfg["out_dir"], manifest)
    manifest.write()

    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    seeds = manifest.run_ids() | {"R-blend", "R-repro"}
    run_ids = closure(seeds, runs)
    write_document(build_receipts(cfg, runs, run_ids), cfg["out_dir"], manifest)
    manifest.write()

    rep = _verify(evidence)
    assert rep.ok, rep.failures

    doc = json.loads((evidence["data_dir"] / "policy_grid.json").read_text())
    assert doc["record_run_id"] == "R-grid"
    assert set(doc["cells"].keys()) == {"A", "B"}
    assert set(doc["cells"]["A"].keys()) == set(N_STAR_LABELS)
    assert doc["cells"]["B"]["inf"]["identity"]["equals_run_id"] == "R-blend"
    assert doc["cells"]["B"]["0"]["identity"]["equals_run_id"] == "R-pop"
    assert doc["cells"]["A"]["0"]["identity"]["equals_run_id"] == "R-als"
    assert len(doc["val_grid"]) == 1
    assert doc["val_grid"][0]["variant"] == "B"
    assert doc["val_winner"]["variant"] == "B"
    assert doc["n_star_selected_on_val"] is None
    assert "R-grid" in json.loads((evidence["data_dir"] / "receipts.json").read_text())["runs"]


def test_policy_grid_export_rejects_wrong_split(policy_grid_evidence):
    evidence = policy_grid_evidence
    cfg = dict(evidence["cfg"], split="val")
    runs = index_runs(evidence["runs_log"])
    with pytest.raises(ValueError, match="split"):
        build_policy_grid(cfg, runs)


def test_policy_grid_export_detects_artifact_drift(policy_grid_evidence):
    evidence = policy_grid_evidence
    cfg = evidence["cfg"]
    runs = index_runs(evidence["runs_log"])
    evidence["val_path"].write_text(json.dumps({**_val_selection_doc(), "objective": "drifted"}))
    with pytest.raises(ValueError, match="drifted from the record"):
        build_policy_grid(cfg, runs)


# --- export_lineage (T30) ------------------------------------------------------


def _lineage_artifact_doc():
    return {
        "schema_version": 1,
        "stages_count": 2,
        "complete": True,
        "footnotes": {
            "runtime_not_persisted_at_build": "Wall clock is null because this stage never wrote a "
            "runtime to any ledger."
        },
        "stages": [
            {
                "stage": "bronze.reviews",
                "layer": "bronze",
                "table": "local.bronze.reviews",
                "rows_in": 1000,
                "rows_out": 1000,
                "bytes": 5000,
                "wall_clock_s": 12.5,
                "wall_clock_source": "data/ingest_summary.jsonl",
                "snapshot_id": 111,
            },
            {
                "stage": "raw_download",
                "layer": "raw",
                "table": "data/raw/",
                "rows_in": None,
                "rows_out": None,
                "bytes": 9000,
                "wall_clock_s": None,
                "wall_clock_source": "runtime_not_persisted_at_build",
                "snapshot_id": None,
            },
        ],
    }


def _ops_record(run_id, table, before, after, scenario):
    return {
        "schema_version": 1,
        "kind": "ops",
        "run_id": run_id,
        "run_ts": "2026-08-07T15:41:16+00:00",
        "git_sha": "6" * 40,
        "git_dirty": False,
        "scenario": scenario,
        "table": table,
        "snapshot_before": before,
        "snapshot_after": after,
        "rows_before": 10,
        "rows_after": 20,
        "wall_clock_s": 1.5,
        "hardware": "arm64 · Darwin",
    }


@pytest.fixture()
def lineage_evidence(evidence, tmp_path):
    """Extends the base ``evidence`` fixture with a lineage record + its
    anchored ``results/lineage.json`` artifact, a second reproduce record, and
    two kind=ops records — the minimum for a lineage/timetravel round trip."""
    lineage_path = tmp_path / "results" / "lineage.json"
    lineage_path.write_text(json.dumps(_lineage_artifact_doc()))
    lineage_sha256 = sha256_file(lineage_path)

    lineage_rec = {
        "schema_version": 1,
        "kind": "lineage",
        "run_id": "R-lineage",
        "run_ts": "2026-08-07T16:09:10+00:00",
        "git_sha": "7" * 40,
        "git_dirty": False,
        "artifact_sha256": lineage_sha256,
        "stages_count": 2,
        "complete": True,
    }
    ops_recs = [
        _ops_record("R-ops1", "local.ops.interactions_monthly", None, 111, "backfill"),
        _ops_record("R-ops2", "local.ops.interactions_monthly", 111, 222, "append"),
    ]
    repro2 = _reproduce_record("R-repro2", "R-blend")

    with open(evidence["runs_log"], "a") as fh:
        for rec in [lineage_rec, *ops_recs, repro2]:
            fh.write(json.dumps(rec) + "\n")

    cfg = dict(
        evidence["cfg"],
        artifacts={
            "lineage": {
                "path": str(lineage_path),
                "run_id": "R-lineage",
                "anchor_pointer": "/artifact_sha256",
            }
        },
    )
    evidence = dict(evidence, cfg=cfg, lineage_path=lineage_path)
    return evidence


def test_lineage_and_timetravel_export_round_trip(lineage_evidence):
    evidence = lineage_evidence
    cfg = evidence["cfg"]
    runs = index_runs(evidence["runs_log"])
    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])

    lineage_w = build_lineage(cfg, runs)
    write_document(lineage_w, cfg["out_dir"], manifest)
    manifest.write()

    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    timetravel_w = build_timetravel(cfg, runs)
    write_document(timetravel_w, cfg["out_dir"], manifest)
    manifest.write()

    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    seeds = manifest.run_ids() | {"R-blend", "R-repro"}
    run_ids = closure(seeds, runs)
    write_document(build_receipts(cfg, runs, run_ids), cfg["out_dir"], manifest)
    manifest.write()

    rep = _verify(evidence)
    assert rep.ok, rep.failures

    lineage_doc = json.loads((evidence["data_dir"] / "lineage.json").read_text())
    assert lineage_doc["record_run_id"] == "R-lineage"
    assert lineage_doc["stages_count"] == 2
    assert lineage_doc["artifact_sha256"] == sha256_file(evidence["lineage_path"])
    assert [s["stage"] for s in lineage_doc["stages"]] == ["bronze.reviews", "raw_download"]
    assert lineage_doc["stages"][1]["wall_clock_s"] is None
    assert "runtime_not_persisted_at_build" in lineage_doc["footnotes"]

    tt_doc = json.loads((evidence["data_dir"] / "timetravel.json").read_text())
    assert tt_doc["pinned"]["run_id"] == "R-blend"
    assert tt_doc["pinned"]["snapshot_ids"] == {"local.gold.interactions_5core": 8184397443787800955}
    assert {r["run_id"] for r in tt_doc["reproduce"]} == {"R-repro", "R-repro2"}
    assert all(r["verdict"] == "byte_exact" for r in tt_doc["reproduce"])
    assert [o["run_id"] for o in tt_doc["ops_chain"]] == ["R-ops1", "R-ops2"]
    assert tt_doc["ops_chain"][0]["snapshot_id_before"] is None
    assert tt_doc["ops_chain"][1]["snapshot_id_after"] == 222
    assert tt_doc["today"]["snapshot_ids"]["local.ops.interactions_monthly"] == 222
    assert tt_doc["today"]["source_run_id"] == "R-ops2"

    receipts_doc = json.loads((evidence["data_dir"] / "receipts.json").read_text())
    for rid in ("R-lineage", "R-ops1", "R-ops2", "R-repro2"):
        assert rid in receipts_doc["runs"], rid


def test_lineage_export_detects_artifact_drift(lineage_evidence):
    evidence = lineage_evidence
    cfg = evidence["cfg"]
    runs = index_runs(evidence["runs_log"])
    evidence["lineage_path"].write_text(json.dumps({**_lineage_artifact_doc(), "complete": False}))
    with pytest.raises(ValueError, match="drifted from the record"):
        build_lineage(cfg, runs)


def test_timetravel_export_requires_reproduce_record(evidence):
    cfg = evidence["cfg"]
    runs = index_runs(evidence["runs_log"])
    ops_rec = _ops_record("R-ops-only", "local.ops.interactions_monthly", None, 1, "backfill")
    with open(evidence["runs_log"], "a") as fh:
        fh.write(json.dumps(ops_rec) + "\n")
    runs = index_runs(evidence["runs_log"])
    # base evidence already carries R-repro targeting R-blend, so this instead
    # checks the ops-records-required guard by pointing headline at a run with
    # no ops records at all.
    cfg2 = dict(cfg, headline_run_id="R-pop")
    with pytest.raises(ValueError, match="reproduce"):
        build_timetravel(cfg2, runs)
