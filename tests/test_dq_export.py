"""DQ exhibit: audit selection, reconciliation guards, the ``kind="dq_export"``
record, and the ``demo/data/dq.json`` projection (Phase 6, T29).

Everything here runs on a synthetic miniature warehouse — no Spark, no
``data/`` — so CI exercises the same code paths the real run does. The Spark
half of :mod:`batch_recsys_lab.demo.dq_export_job` (``collect``) is the only
part not covered: it is a read of five Iceberg tables whose *pure* tail
(``select_audit_run`` → ``build_matrix`` → ``waterfall_block`` →
``headline_counts`` → ``reconciliation_checks``) is assembled below exactly as
``collect`` assembles it.

The round trip mirrors ``tests/test_demo_export.py``: the writer builds the
manifest, the independent verifier re-resolves it, and each negative test breaks
exactly one link.
"""

from __future__ import annotations

import json

import pytest
import yaml

from batch_recsys_lab.demo import dq_export_job as job
from batch_recsys_lab.demo.export_core import (
    TraceManifest,
    index_runs,
    sha256_file,
    write_document,
)
from batch_recsys_lab.demo.export_dq import (
    DqProjectionError,
    build as build_dq,
    select_record,
)
from batch_recsys_lab.demo.export_receipts import build as build_receipts
from batch_recsys_lab.demo.export_receipts import closure
from batch_recsys_lab.demo.verify_traceability import verify

BUILD_RUN = "B-build"
AUDIT_RUN = "A-full"
HEADLINE_RUN = "R-blend"

TABLES = {
    "local.silver.interactions": ("silver_interactions", 940),
    "local.silver.items": ("silver_items", 200),
    "local.gold.interactions_5core": ("gold_interactions_5core", 700),
}


# --- synthetic substrate ------------------------------------------------------


def _waterfall_doc() -> dict:
    def edge(stage_from, stage_to, source, kept, reasons, target_table):
        return {
            "stage_from": stage_from,
            "stage_to": stage_to,
            "source_rows": source,
            "kept_rows": kept,
            "reason_sum": source,
            "target_table": target_table,
            "target_count": kept,
            "sum_ok": True,
            "count_ok": True,
            "reasons": [{"reason": r, "rows": n} for r, n in reasons],
        }

    return {
        "run_id": BUILD_RUN,
        "generated_at": "2026-08-05T15:00:25.583617+00:00",
        "datasets": {
            "reviews": {
                "chain": "raw -> bronze -> silver -> gold",
                "edges": [
                    edge("raw", "bronze", 1000, 1000, [("kept", 1000), ("corrupt", 0)], "local.bronze.reviews"),
                    edge(
                        "bronze",
                        "silver",
                        1000,
                        940,
                        [
                            ("kept", 940),
                            ("quarantine:rating_domain", 2),
                            ("exact_duplicate", 50),
                            ("superseded_by_later_review", 8),
                        ],
                        "local.silver.interactions",
                    ),
                    edge(
                        "silver",
                        "gold",
                        940,
                        700,
                        [("kept", 700), ("kcore_pruned", 240)],
                        "local.gold.interactions_5core",
                    ),
                ],
                "kcore_run_id": BUILD_RUN,
                "kcore_funnel": [
                    {"iteration": 0, "rows": 940, "users": 400, "items": 90, "converged": False, "wall_clock_s": 1.5},
                    {"iteration": 1, "rows": 700, "users": 300, "items": 70, "converged": True, "wall_clock_s": 0.5},
                ],
            },
            "items": {
                "chain": "raw -> bronze -> silver",
                "edges": [
                    edge("raw", "bronze", 200, 200, [("kept", 200), ("corrupt", 0)], "local.bronze.items"),
                    edge(
                        "bronze",
                        "silver",
                        200,
                        200,
                        [("kept", 200), ("exact_duplicate", 0), ("superseded_by_later_review", 0)],
                        "local.silver.items",
                    ),
                ],
            },
        },
    }


def _ledger_rows(waterfall: dict) -> list[dict]:
    return [
        {
            "run_id": BUILD_RUN,
            "dataset": e["dataset"],
            "stage_from": e["stage_from"],
            "stage_to": e["stage_to"],
            "reason": r["reason"],
            "rows": r["rows"],
        }
        for e in job.waterfall_edges(waterfall)
        for r in e["reasons"]
    ]


def _audit_row(run_id, run_ts, table, check_id, kind, *, status="pass", violations=0, metric=None, details=None):
    contract, total = TABLES[table]
    return {
        "run_id": run_id,
        "run_ts": run_ts,
        "table_name": table,
        "contract_name": contract,
        "contract_version": 1,
        "check_id": check_id,
        "check_kind": kind,
        "column": None,
        "status": status,
        "violation_count": violations,
        "total_rows": total,
        "metric_value": metric,
        "details": json.dumps(details or {}),
    }


def _audit_rows() -> list[dict]:
    """One complete audit, one partial run, one duplicated run."""
    ts = "2026-08-06T10:41:12+00:00"
    full = [
        _audit_row(AUDIT_RUN, ts, "local.silver.interactions", "keys_non_null", "not_null"),
        _audit_row(
            AUDIT_RUN, ts, "local.silver.interactions", "item_fk", "orphan_rate",
            status="measured", violations=0, metric=0.0, details={"denominator": 940},
        ),
        _audit_row(AUDIT_RUN, ts, "local.silver.items", "key_non_null", "not_null"),
        _audit_row(
            AUDIT_RUN, ts, "local.silver.items", "brand_unknown_share", "unknown_share",
            status="measured", violations=20, metric=0.1, details={"sentinel": "unknown"},
        ),
        _audit_row(AUDIT_RUN, ts, "local.gold.interactions_5core", "rating_domain", "allowed_values"),
    ]
    partial = [_audit_row("A-partial", "2026-08-07T00:00:00+00:00", "local.silver.items", "key_non_null", "not_null")]
    dupe_ts = "2026-08-08T00:00:00+00:00"
    dupe = [
        _audit_row("A-dupe", dupe_ts, t, "key_non_null", "not_null") for t in TABLES
    ] + [_audit_row("A-dupe", dupe_ts, "local.silver.items", "key_non_null", "not_null")]
    return full + partial + dupe


def _contracts() -> dict[str, dict]:
    return {table: {"name": name, "version": 1} for table, (name, _) in TABLES.items()}


def _assemble_dq_raw(*, waterfall=None, audit_rows=None, quarantine_items_rows=0) -> dict:
    """The pure tail of ``dq_export_job.collect`` — same functions, same order."""
    waterfall = waterfall or _waterfall_doc()
    audit_rows = audit_rows if audit_rows is not None else _audit_rows()
    contracts = _contracts()

    audit_run_id = job.select_audit_run(audit_rows, set(contracts))
    selected = [r for r in audit_rows if r["run_id"] == audit_run_id]
    matrix = job.build_matrix(selected, contracts)
    summary = job.summarize_matrix(matrix)
    funnel = [dict(f) for f in waterfall["datasets"]["reviews"]["kcore_funnel"]]
    wf = job.waterfall_block(waterfall, _ledger_rows(waterfall))

    quarantine = {
        "local.quarantine.interactions": {
            "rows": 2,
            "rows_all_runs": 6,
            "by_primary_reason": [{"reason": "rating_domain", "rows": 2}],
            "by_violation_reason": [{"reason": "rating_domain", "rows": 2}],
            "snapshot_id": 111,
        },
        "local.quarantine.items": {
            "rows": quarantine_items_rows,
            "rows_all_runs": quarantine_items_rows,
            "by_primary_reason": (
                [{"reason": "key_non_null", "rows": quarantine_items_rows}] if quarantine_items_rows else []
            ),
            "by_violation_reason": [],
            "snapshot_id": 222,
        },
    }
    q_total = sum(v["rows"] for v in quarantine.values())
    hc = job.headline_counts(waterfall, q_total)
    recon = job.reconciliation_checks(hc, funnel, matrix)
    rates = job.measured_rates(audit_rows, audit_run_id, {})

    return {
        "schema_version": job.RAW_SCHEMA_VERSION,
        "generated_by": "batch_recsys_lab.demo.dq_export_job",
        "warehouse": "data/warehouse",
        "headline_run_id": HEADLINE_RUN,
        "audit_run_id": audit_run_id,
        "audit_run_ts": min(r["run_ts"] for r in selected),
        "build_run_id": BUILD_RUN,
        "table_snapshots": {"local.dq.dq_results": 333, "local.dq.waterfall": 444},
        "headline_snapshots": {"local.gold.interactions_5core": 8184397443787800955},
        "contracts": contracts,
        "contract_matrix": matrix,
        "contract_summary": summary,
        "quarantine": {
            "build_run_id": BUILD_RUN,
            "total_rows": q_total,
            "by_reason": job._quarantine_by_reason(quarantine, hc),
            "tables": quarantine,
        },
        "kcore_funnel": funnel,
        "waterfall": wf,
        "measured_rates": rates,
        "headline_counts": hc,
        "reconciliation": recon,
    }


def _headline_record() -> dict:
    return {
        "schema_version": 1,
        "kind": "eval",
        "run_id": HEADLINE_RUN,
        "run_ts": "2026-08-07T05:53:33+00:00",
        "git_sha": "0" * 40,
        "git_dirty": False,
        "config_path": "configs/eval_blend_test.yaml",
        "config_hash": "sha256:" + "a" * 64,
        "splits": {"version": 1, "frozen_at": "2026-08-05", "file_hash": "sha256:" + "b" * 64},
        "dataset_manifest_hash": "sha256:" + "c" * 64,
        "iceberg_snapshots": {"local.gold.interactions_5core": 8184397443787800955},
        "contracts": {},
        "protocol": {"eval_split": "test", "catalog_size": 100, "n_users": 30},
        "model": {"name": "content_pop_blend", "params": {}},
        "seeds": {"bootstrap": 20260805, "model": None},
        "metrics": {"global": {"ndcg@10": {"value": 0.005726, "ci_lo": 0.005, "ci_hi": 0.006}}},
        "beyond_accuracy": {},
        "per_user_artifact": "data/eval/per_user/R-blend_blend.parquet",
        "wall_clock_s": 12.5,
        "hardware": "arm64 · Darwin",
    }


def _reproduce_record() -> dict:
    return {
        "schema_version": 1,
        "kind": "reproduce",
        "run_id": "R-repro",
        "run_ts": "2026-08-07T15:38:23+00:00",
        "git_sha": "2" * 40,
        "git_dirty": False,
        "reproduces_run_id": HEADLINE_RUN,
        "verdict": "byte_exact",
        "hardware": "arm64 · Darwin",
    }


@pytest.fixture()
def dq_repo(tmp_path, monkeypatch):
    """A miniature repo rooted at ``tmp_path``, cwd'd into like production.

    Every config path is repo-root-relative exactly as in the real configs, so
    the manifest records portable relative paths and the verifier resolves them
    against ``repo_root=tmp_path``.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "MANIFEST.md").write_text("# synthetic manifest\n")
    (tmp_path / "data" / "build_summary.jsonl").write_text(
        json.dumps({"run_id": BUILD_RUN, "table": "local.silver.interactions", "kept": 940}) + "\n"
    )
    (tmp_path / "data" / "waterfall.json").write_text(json.dumps(_waterfall_doc(), indent=2) + "\n")
    (tmp_path / "data" / "demo_export").mkdir()
    (tmp_path / "data" / "demo_export" / "dq_raw.json").write_text(
        json.dumps(_assemble_dq_raw(), indent=2) + "\n"
    )

    (tmp_path / "configs" / "demo_export.yaml").write_text(
        yaml.safe_dump(
            {
                "runs_log": "results/runs.jsonl",
                "out_dir": "demo/data",
                "manifest": "demo/data/trace_manifest.json",
                "headline_run_id": HEADLINE_RUN,
                "repro_command": "make reproduce-headline",
            }
        )
    )
    (tmp_path / "configs" / "dq_export.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "demo_export_config": "configs/demo_export.yaml",
                "warehouse": "data/warehouse",
                "contracts_dir": "contracts",
                "tables": {"dq_results": "local.dq.dq_results"},
                "waterfall_json": "data/waterfall.json",
                "build_summary": "data/build_summary.jsonl",
                "dq_raw": "data/demo_export/dq_raw.json",
                "published_dir": "results/dq",
                "dataset_manifest": "data/MANIFEST.md",
                "audit_run_id": None,
                "build_run_id": None,
                "record_run_id": None,
            }
        )
    )
    runs_log = tmp_path / "results" / "runs.jsonl"
    with open(runs_log, "w") as fh:
        for rec in (_headline_record(), _reproduce_record()):
            fh.write(json.dumps(rec) + "\n")

    cfg = job.load_config("configs/dq_export.yaml", repo_root=tmp_path)
    return {"root": tmp_path, "cfg": cfg, "runs_log": runs_log}


def _append_dq_record(dq_repo, *, run_id="D-dq") -> dict:
    record = job.build_record(dq_repo["cfg"], run_id=run_id)
    with open(dq_repo["runs_log"], "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def _export(dq_repo) -> TraceManifest:
    cfg = dq_repo["cfg"]
    runs = index_runs(cfg["runs_log"])
    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    write_document(build_dq(cfg, runs), cfg["out_dir"], manifest)
    manifest.write()

    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    run_ids = closure(manifest.run_ids() | {HEADLINE_RUN, "R-repro"}, runs)
    write_document(build_receipts(cfg, runs, run_ids), cfg["out_dir"], manifest)
    manifest.write()
    return manifest


def _verify(dq_repo, mode="record"):
    cfg = dq_repo["cfg"]
    return verify(
        cfg["out_dir"], cfg["manifest"], runs_log=cfg["runs_log"], mode=mode, repo_root=dq_repo["root"]
    )


# --- audit selection ----------------------------------------------------------


def test_select_audit_run_picks_the_latest_complete_run():
    rows = _audit_rows()
    assert job.select_audit_run(rows, set(TABLES)) == AUDIT_RUN


def test_select_audit_run_rejects_a_partial_pin():
    rows = _audit_rows()
    with pytest.raises(job.DqExportError, match="not a complete audit"):
        job.select_audit_run(rows, set(TABLES), "A-partial")


def test_select_audit_run_rejects_a_duplicated_pin():
    rows = _audit_rows()
    with pytest.raises(job.DqExportError, match="duplicate"):
        job.select_audit_run(rows, set(TABLES), "A-dupe")


def test_select_audit_run_aborts_when_nothing_is_complete():
    rows = [r for r in _audit_rows() if r["run_id"] != AUDIT_RUN]
    with pytest.raises(job.DqExportError, match="no complete contract audit"):
        job.select_audit_run(rows, set(TABLES))


def test_build_matrix_rejects_contract_version_drift():
    rows = [r for r in _audit_rows() if r["run_id"] == AUDIT_RUN]
    contracts = _contracts()
    contracts["local.silver.items"]["version"] = 2
    with pytest.raises(job.DqExportError, match="contract YAML on disk declares version 2"):
        job.build_matrix(rows, contracts)


# --- reconciliation -----------------------------------------------------------


def test_headline_counts_reconcile_against_the_waterfall():
    raw = _assemble_dq_raw()
    hc = raw["headline_counts"]
    assert hc["raw_reviews_rows"] == hc["bronze_reviews_rows"] == 1000
    assert (
        hc["bronze_reviews_rows"]
        - hc["exact_duplicate_rows"]
        - hc["superseded_by_later_review_rows"]
        - hc["quarantined_interaction_rows"]
        == hc["silver_interactions_rows"]
    )
    assert hc["silver_interactions_rows"] - hc["kcore_pruned_rows"] == hc["gold_interactions_5core_rows"]
    assert raw["reconciliation"]["all_ok"]


def test_reconciliation_names_a_broken_identity():
    wf = _waterfall_doc()
    # An extra quarantined row the quarantine ledger does not hold.
    edge = wf["datasets"]["reviews"]["edges"][1]
    edge["reasons"][1]["rows"] = 3
    edge["reasons"][0]["rows"] = 939
    edge["kept_rows"] = 939
    edge["target_count"] = 939
    hc = job.headline_counts(wf, quarantine_total=2)
    recon = job.reconciliation_checks(hc, [], {})
    broken = {c["name"] for c in recon["checks"] if not c["ok"]}
    assert not recon["all_ok"]
    assert "quarantine_ledger_equals_waterfall" in broken


def test_waterfall_block_detects_ledger_drift():
    wf = _waterfall_doc()
    rows = _ledger_rows(wf)
    rows[0]["rows"] = 999
    with pytest.raises(job.DqExportError, match="waterfall drift"):
        job.waterfall_block(wf, rows)


def test_waterfall_block_requires_a_ledger_row_per_reason():
    wf = _waterfall_doc()
    rows = [r for r in _ledger_rows(wf) if r["reason"] != "kcore_pruned"]
    with pytest.raises(job.DqExportError, match="no row for"):
        job.waterfall_block(wf, rows)


# --- the dq_export record -----------------------------------------------------


def test_dry_run_record_shape(dq_repo):
    record = job.build_record(dq_repo["cfg"], run_id="D-dq")
    assert record["kind"] == "dq_export"
    for field in ("waterfall_sha256", "build_summary_sha256", "dq_raw_sha256"):
        assert record[field].startswith("sha256:"), field
    root = dq_repo["root"]
    assert record["waterfall_sha256"] == sha256_file(root / "data" / "waterfall.json")
    assert record["build_summary_sha256"] == sha256_file(root / "data" / "build_summary.jsonl")
    assert record["dq_raw_sha256"] == sha256_file(root / "data" / "demo_export" / "dq_raw.json")

    # Both artifacts are published byte-identically under a committed path.
    for key, src in (("waterfall", "data/waterfall.json"), ("dq_raw", "data/demo_export/dq_raw.json")):
        published = root / record["published_artifacts"][key]["path"]
        assert published.read_bytes() == (root / src).read_bytes()
        assert record["published_artifacts"][key]["sha256"] == record[f"{key}_sha256"]

    assert record["audit_run_id"] == AUDIT_RUN
    assert record["build_run_id"] == BUILD_RUN
    assert record["reconciliation"]["all_ok"]
    assert record["headline_counts"]["gold_interactions_5core_rows"] == 700
    assert record["iceberg_snapshots"] == {"local.gold.interactions_5core": 8184397443787800955}
    assert record["table_snapshots"] == {"local.dq.dq_results": 333, "local.dq.waterfall": 444}
    assert record["kcore_iterations"] == 2
    assert record["quarantine_totals"]["total_rows"] == 2


def test_dry_run_appends_nothing(dq_repo, capsys):
    before = dq_repo["runs_log"].read_bytes()
    rc = job.main(["--config", "configs/dq_export.yaml", "--phase", "record", "--dry-run"])
    assert rc == 0
    assert dq_repo["runs_log"].read_bytes() == before
    out = capsys.readouterr().out
    assert "DRY RUN — nothing appended" in out
    printed = json.loads(out[out.index("{") : out.rindex("}") + 1])
    assert printed["kind"] == "dq_export"
    assert printed["dq_raw_sha256"].startswith("sha256:")


def test_record_phase_refuses_both_flags(dq_repo):
    with pytest.raises(SystemExit):
        job.main(["--config", "configs/dq_export.yaml", "--phase", "record", "--dry-run", "--append"])


def test_record_detects_a_waterfall_that_moved_under_it(dq_repo):
    wf = _waterfall_doc()
    wf["datasets"]["reviews"]["edges"][2]["kept_rows"] = 699
    (dq_repo["root"] / "data" / "waterfall.json").write_text(json.dumps(wf, indent=2) + "\n")
    with pytest.raises(job.DqExportError, match="no longer yields the headline counts"):
        job.build_record(dq_repo["cfg"], run_id="D-dq")


def test_find_equivalent_is_content_based(dq_repo):
    record = _append_dq_record(dq_repo)
    later = dict(record, run_id="D-dq-2", run_ts="2026-08-09T00:00:00+00:00", git_sha="f" * 40)
    records = [json.loads(line) for line in dq_repo["runs_log"].read_text().splitlines() if line.strip()]
    assert job.find_equivalent(records, later)["run_id"] == "D-dq"
    different = dict(later, dq_raw_sha256="sha256:" + "0" * 64)
    assert job.find_equivalent(records, different) is None


# --- dq.json projection -------------------------------------------------------


def test_dq_export_round_trip(dq_repo):
    record = _append_dq_record(dq_repo)
    _export(dq_repo)
    rep = _verify(dq_repo)
    assert rep.ok, rep.failures

    doc = json.loads((dq_repo["root"] / "demo" / "data" / "dq.json").read_text())
    assert doc["schema_version"] == 1
    assert doc["record_run_id"] == record["run_id"]
    assert doc["audit_run_id"] == AUDIT_RUN
    assert doc["artifact_sha256"]["waterfall"] == record["waterfall_sha256"]
    assert doc["artifact_sha256"]["dq_raw"] == record["dq_raw_sha256"]

    assert doc["waterfall"]["reconciles"] is True
    assert doc["quarantine"]["total_rows"] == 2
    assert [r["reason"] for r in doc["quarantine"]["by_reason"]] == ["rating_domain"]
    assert doc["contract_matrix"]["local.silver.items"]["brand_unknown_share"]["measured"] == 0.1
    assert doc["contract_matrix"]["local.silver.items"]["brand_unknown_share"]["threshold"] is None
    assert doc["contract_tables"]["local.silver.items"]["total_rows"] == 200
    assert doc["measured_rates"]["unknown_brand_share"]["rate"] == 0.1
    assert [f["interactions"] for f in doc["kcore_funnel"]] == [940, 700]
    assert doc["kcore_funnel"][-1]["converged"] is True
    assert doc["reconciliation"]["all_ok"] is True
    assert doc["iceberg_snapshots"]["local.gold.interactions_5core"] == 8184397443787800955


def test_dq_json_waterfall_byte_matches_data_waterfall(dq_repo):
    _append_dq_record(dq_repo)
    _export(dq_repo)
    doc = json.loads((dq_repo["root"] / "demo" / "data" / "dq.json").read_text())
    wf = json.loads((dq_repo["root"] / "data" / "waterfall.json").read_text())

    edges = [
        (dataset, edge)
        for dataset, block in wf["datasets"].items()
        for edge in block["edges"]
    ]
    assert len(doc["waterfall"]["stages"]) == len(edges)
    for stage, (dataset, edge) in zip(doc["waterfall"]["stages"], edges, strict=True):
        assert stage["dataset"] == dataset
        assert stage["stage_from"] == edge["stage_from"]
        assert stage["stage_to"] == edge["stage_to"]
        assert stage["rows_in"] == edge["source_rows"]
        assert stage["rows_out"] == edge["kept_rows"]
        assert stage["target_count"] == edge["target_count"]
        assert stage["delta"] == edge["source_rows"] - edge["kept_rows"]
        assert [(r["reason"], r["rows"]) for r in stage["reasons"]] == [
            (r["reason"], r["rows"]) for r in edge["reasons"]
        ]

    # …and every one of those leaves is anchored on the record's waterfall_sha256.
    manifest = json.loads((dq_repo["root"] / "demo" / "data" / "trace_manifest.json").read_text())
    anchored = [
        e
        for e in manifest["entries"]
        if e["file"] == "dq.json" and e["pointer"].startswith("/waterfall/stages/")
    ]
    from_waterfall = [e for e in anchored if e["source"]["source_file"].endswith("waterfall.json")]
    assert from_waterfall, "no stage leaf resolves into the waterfall artifact"
    assert all(e["source"]["anchor_pointer"] == "/waterfall_sha256" for e in from_waterfall)


def test_dq_export_detects_dq_raw_waterfall_drift(dq_repo):
    _append_dq_record(dq_repo)
    # Re-publish a dq_raw whose stage counts no longer match the waterfall, and
    # re-anchor it so the export gets past the artifact-hash check and has to be
    # caught by the byte-match assertion instead.
    published = dq_repo["root"] / "results" / "dq" / "dq_raw.json"
    raw = json.loads(published.read_text())
    raw["waterfall"]["stages"][0]["rows_in"] = 1001
    published.write_text(json.dumps(raw, indent=2) + "\n")
    records = [json.loads(line) for line in dq_repo["runs_log"].read_text().splitlines() if line.strip()]
    records[-1]["dq_raw_sha256"] = sha256_file(published)
    records[-1]["published_artifacts"]["dq_raw"]["sha256"] = sha256_file(published)
    with open(dq_repo["runs_log"], "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    cfg = dq_repo["cfg"]
    with pytest.raises(DqProjectionError, match="waterfall drift"):
        build_dq(cfg, index_runs(cfg["runs_log"]))


def test_dq_export_detects_artifact_drift(dq_repo):
    _append_dq_record(dq_repo)
    published = dq_repo["root"] / "results" / "dq" / "waterfall.json"
    published.write_text(published.read_text().replace("1000", "1001", 1))
    cfg = dq_repo["cfg"]
    with pytest.raises(DqProjectionError, match="no kind='dq_export' record anchors"):
        build_dq(cfg, index_runs(cfg["runs_log"]))


def test_select_record_rejects_a_pinned_non_dq_record(dq_repo):
    _append_dq_record(dq_repo)
    cfg = dict(dq_repo["cfg"], record_run_id=HEADLINE_RUN)
    with pytest.raises(DqProjectionError, match="expected 'dq_export'"):
        select_record(index_runs(cfg["runs_log"]), cfg)


def test_select_record_requires_a_record(dq_repo):
    cfg = dq_repo["cfg"]
    with pytest.raises(DqProjectionError, match="no kind='dq_export' record in"):
        select_record(index_runs(cfg["runs_log"]), cfg)


def test_dq_export_is_byte_stable(dq_repo):
    _append_dq_record(dq_repo)
    _export(dq_repo)
    first = (dq_repo["root"] / "demo" / "data" / "dq.json").read_bytes()
    _export(dq_repo)
    assert (dq_repo["root"] / "demo" / "data" / "dq.json").read_bytes() == first


def test_perturbing_a_dq_leaf_fails_verification(dq_repo):
    _append_dq_record(dq_repo)
    _export(dq_repo)
    path = dq_repo["root"] / "demo" / "data" / "dq.json"
    doc = json.loads(path.read_text())
    doc["waterfall"]["stages"][0]["rows_in"] += 1
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    rep = _verify(dq_repo)
    assert not rep.ok
    assert {kind for kind, _ in rep.failures} & {"FILE_HASH", "DOC_MISMATCH"}
