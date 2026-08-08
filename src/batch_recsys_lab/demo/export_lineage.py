"""``demo/data/lineage.json`` + ``demo/data/timetravel.json`` — exhibit 5
(Phase 6, T30; plan §9).

``lineage.json`` is a projection of the committed ``results/lineage.json``
(24-stage table), anchored transitively through the ``kind="lineage"`` record
(``20260807T160910Z-739833b``, which carries the artifact's SHA-256 at
``/artifact_sha256``) — every stage leaf uses the ``results_artifact`` source
kind (``TracedWriter.register_artifact`` / ``copy_from_artifact``).

``timetravel.json`` documents the pinned headline snapshots, both
``kind="reproduce"`` verdicts against the headline run, and the recorded
11-record ``kind="ops"`` snapshot chain (Phase 5's churn exhibit). The ops
table itself (``local.ops.interactions_monthly``) was dropped after Phase 5's
clean-ops step was accepted — the *records* remain the evidence, so this
document cites them, not a live table.

    uv run python -m batch_recsys_lab.demo.export_lineage --config configs/demo_export.yaml

Run BEFORE export_receipts (receipts reads the manifest this exporter writes
into to find the run_id closure it must document).

Schema: docs/demo-data-schemas.md § lineage.json / § timetravel.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from batch_recsys_lab.demo.export_core import (
    TraceManifest,
    TracedWriter,
    index_runs,
    jp,
    load_export_config,
    write_document,
)

LINEAGE_FILE_NAME = "lineage.json"
TIMETRAVEL_FILE_NAME = "timetravel.json"

STAGE_FIELDS = (
    "stage",
    "layer",
    "table",
    "rows_in",
    "rows_out",
    "bytes",
    "wall_clock_s",
    "wall_clock_source",
    "snapshot_id",
)


def build_lineage(cfg: dict, runs: dict[str, dict]) -> TracedWriter:
    art_cfg = cfg["artifacts"]["lineage"]
    run_id = art_cfg["run_id"]
    if run_id not in runs:
        raise ValueError(f"lineage: run_id {run_id!r} not found in {cfg['runs_log']}")
    rec = runs[run_id]
    if rec.get("kind") != "lineage":
        raise ValueError(f"run {run_id}: kind={rec.get('kind')!r}, expected 'lineage'")

    w = TracedWriter(LINEAGE_FILE_NAME, runs, generated_by="batch_recsys_lab.demo.export_lineage")
    w.put_descriptive("/record_run_id", run_id, note="kind=lineage record this exhibit projects")

    art = w.register_artifact(
        "lineage",
        art_cfg["path"],
        run_id=run_id,
        anchor_pointer=art_cfg["anchor_pointer"],
    )
    w.put_descriptive(
        "/artifact_sha256", art["sha256"], note="anchors every stage leaf below (results_artifact source kind)"
    )

    w.copy_from_artifact("/stages_count", "lineage", "/stages_count")
    w.copy_from_artifact("/complete", "lineage", "/complete")

    stages = art["doc"]["stages"]
    w.ensure_list("/stages", len(stages))
    for i in range(len(stages)):
        base = jp("stages", i)
        for field in STAGE_FIELDS:
            w.copy_from_artifact(base + jp(field), "lineage", base + jp(field))

    w.put_descriptive(
        "/footnotes",
        art["doc"]["footnotes"],
        subtree=True,
        note="stage build-time footnotes, verbatim from results/lineage.json (descriptive: explanatory "
        "text, not itself measured evidence)",
    )
    return w


def build_timetravel(cfg: dict, runs: dict[str, dict]) -> TracedWriter:
    headline_run_id = cfg["headline_run_id"]
    if headline_run_id not in runs:
        raise ValueError(f"timetravel: headline_run_id {headline_run_id!r} not found in {cfg['runs_log']}")

    ops_records = [rec for rec in runs.values() if rec.get("kind") == "ops"]
    if not ops_records:
        raise ValueError("timetravel: no kind='ops' records found in the runs log")

    reproduce_records = [
        rec
        for rec in runs.values()
        if rec.get("kind") == "reproduce" and rec.get("reproduces_run_id") == headline_run_id
    ]
    if not reproduce_records:
        raise ValueError(f"timetravel: no kind='reproduce' record targets headline run {headline_run_id!r}")

    w = TracedWriter(TIMETRAVEL_FILE_NAME, runs, generated_by="batch_recsys_lab.demo.export_lineage")

    # --- pinned: the snapshots the headline run was scored against ------------
    w.copy_from_record("/pinned/snapshot_ids", headline_run_id, "/iceberg_snapshots")
    w.put_descriptive("/pinned/run_id", headline_run_id, note="the headline eval run these snapshots were pinned for")
    for k in ("value", "ci_lo", "ci_hi"):
        w.copy_from_record(
            jp("pinned", "headline", "ndcg@10", k), headline_run_id, jp("metrics", "global", "ndcg@10", k)
        )

    # --- today: the ops-chain table's last recorded snapshot ------------------
    last_ops = ops_records[-1]
    table_key = last_ops["table"]
    w.copy_from_record(jp("today", "snapshot_ids", table_key), last_ops["run_id"], "/snapshot_after")
    w.put_descriptive(
        "/today/source_run_id",
        last_ops["run_id"],
        note="last recorded kind=ops record — the ops-chain table's final snapshot before Phase 5's clean-ops step dropped it",
    )

    # --- reproduce: both byte_exact verdicts against the headline, verbatim ---
    w.ensure_list("/reproduce", len(reproduce_records))
    for i, rep in enumerate(reproduce_records):
        w.copy_from_record(jp("reproduce", i), rep["run_id"], "")

    # --- ops_chain: the 11-record snapshot chain -------------------------------
    w.ensure_list("/ops_chain", len(ops_records))
    for i, rec in enumerate(ops_records):
        rid = rec["run_id"]
        base = jp("ops_chain", i)
        w.copy_from_record(base + "/run_id", rid, "/run_id")
        w.copy_from_record(base + "/step", rid, "/scenario")
        w.copy_from_record(base + "/table", rid, "/table")
        w.copy_from_record(base + "/snapshot_id_before", rid, "/snapshot_before")
        w.copy_from_record(base + "/snapshot_id_after", rid, "/snapshot_after")

    w.put_descriptive(
        "/notes/ops_table_dropped",
        "Phase 5's clean-ops step (accepted, see EXPERIMENT_LOG.md) dropped "
        "local.ops.interactions_monthly after the reproduce-headline acceptance check ran a "
        "second time post-churn. The ops_chain above is the recorded evidence for the snapshot "
        "history — the 11 kind=ops records — not a live, re-queryable table.",
        note="descriptive: explains why 'today' cannot be re-resolved against a live table",
    )
    return w


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/demo_export.yaml")
    args = ap.parse_args(argv)

    cfg = load_export_config(args.config)
    runs = index_runs(cfg["runs_log"])
    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])

    lineage_writer = build_lineage(cfg, runs)
    out1 = write_document(lineage_writer, cfg["out_dir"], manifest)
    manifest.write()

    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    timetravel_writer = build_timetravel(cfg, runs)
    out2 = write_document(timetravel_writer, cfg["out_dir"], manifest)
    manifest.drop_missing_files(cfg["out_dir"])
    manifest.write()

    print(
        f"wrote {out1} ({len(lineage_writer.entries)} traced leaves, {len(lineage_writer.descriptive)} descriptive)\n"
        f"wrote {out2} ({len(timetravel_writer.entries)} traced leaves, {len(timetravel_writer.descriptive)} descriptive)\n"
        f"manifest {Path(cfg['manifest'])}"
    )


if __name__ == "__main__":
    main()
