"""``demo/data/dq.json`` — exhibit 4, the data-quality dashboard
(Phase 6, T29; plan §9 item 4).

Pure projection, JVM-free. Every numeric leaf is traced with the
``results_artifact`` source kind, anchored on the ``kind="dq_export"`` record's
two SHA-256s (the ``results/lineage.json`` precedent):

``results/dq/waterfall.json``  (anchor ``/waterfall_sha256``)
    Byte-identical committed copy of ``data/waterfall.json`` — the reconciliation
    ledger itself. Every raw waterfall count on the dashboard (``rows_in``,
    ``rows_out``, per-reason ``rows``, ``target_count``, ``sum_ok`` /
    ``count_ok``) resolves *into this file*, not into a re-derivation of it.
``results/dq/dq_raw.json``     (anchor ``/dq_raw_sha256``)
    The read-only Spark pull (:mod:`batch_recsys_lab.demo.dq_export_job`):
    contract matrix, quarantine counts by reason, k-core funnel, measured rates,
    headline counts, reconciliation identities, and the values *derived* from
    the waterfall (``delta``, ``share_of_rows_in``, the dominant drop reason).

The split matters: the acceptance criterion is "dq.json's waterfall byte-matches
``data/waterfall.json``", and the strongest way to satisfy it is to make the
committed waterfall the literal source of those leaves. :func:`build` then
re-asserts, independently of the Spark job, that ``dq_raw.json``'s flattened
stage list agrees with the waterfall edges count-for-count — so a drifted
``dq_raw.json`` aborts the export instead of shipping two disagreeing numbers.

    uv run python -m batch_recsys_lab.demo.export_dq --config configs/dq_export.yaml

Paths in the config and in the record are repo-root-relative, so this runs from
the repo root (as every other exporter does). Run BEFORE export_receipts —
receipts reads the manifest this exporter writes into to find the run_id closure
it must document.

Schema: docs/demo-data-schemas.md § dq.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from batch_recsys_lab.demo.dq_export_job import RECORD_KIND, load_config
from batch_recsys_lab.demo.export_core import (
    TraceManifest,
    TracedWriter,
    index_runs,
    jp,
    sha256_file,
    write_document,
)

__all__ = ["FILE_NAME", "DqProjectionError", "build", "main", "select_record"]

FILE_NAME = "dq.json"

# published_artifacts key -> the record field carrying that artifact's digest.
# register_artifact anchors on the top-level field because that is the pointer
# docs/demo-data-schemas.md and the record schema name.
ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("waterfall", "waterfall_sha256"),
    ("dq_raw", "dq_raw_sha256"),
)

# Copied verbatim out of dq_raw.json, subtree by subtree, in document order.
RAW_SUBTREES: tuple[str, ...] = (
    "/headline_counts",
    "/contract_summary",
    "/quarantine",
    "/measured_rates",
    "/reconciliation",
)

# Per-stage fields whose value lives in data/waterfall.json itself:
# (dq.json field, waterfall.json edge field).
WATERFALL_EDGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("rows_in", "source_rows"),
    ("rows_out", "kept_rows"),
    ("target_count", "target_count"),
    ("sum_ok", "sum_ok"),
    ("count_ok", "count_ok"),
)

# Per-stage fields the Spark job derived (absent from data/waterfall.json).
WATERFALL_DERIVED_FIELDS: tuple[str, ...] = ("delta", "reason", "matches_ledger")

# k-core funnel: (dq.json field, dq_raw.json field). ``interactions`` is the
# AGREED site-facing name for the ledger's ``rows`` column.
FUNNEL_FIELDS: tuple[tuple[str, str], ...] = (
    ("iteration", "iteration"),
    ("users", "users"),
    ("items", "items"),
    ("interactions", "rows"),
    ("converged", "converged"),
    ("wall_clock_s", "wall_clock_s"),
)

NOTES = {
    "waterfall_source": (
        "rows_in / rows_out / target_count / per-reason rows / sum_ok / count_ok are read out of "
        "results/dq/waterfall.json, the byte-identical committed copy of data/waterfall.json that "
        "the kind=dq_export record's waterfall_sha256 anchors. delta, share_of_rows_in and the "
        "dominant drop reason are derived by the Spark job and come from results/dq/dq_raw.json "
        "(dq_raw_sha256)."
    ),
    "ledger_cross_check": (
        "The Spark job re-read local.dq.waterfall at a pinned snapshot and refused to publish "
        "unless every edge/reason row equalled data/waterfall.json; waterfall.ledger_rows_checked "
        "is how many ledger rows that covered."
    ),
    "measured_rates_sources": (
        "source=contract_ledger rows are recorded dq_results measures; source=dq_export_job rows "
        "are read-only counts this job took because §9 asks for a null-price rate and no contract "
        "check measures one (price_unparseable counts unparseable strings, not absent prices)."
    ),
    "kcore_funnel_naming": (
        "interactions is the ledger's rows column, renamed for the site per the AGREED schema; "
        "iteration 0 is the silver row count and the converged iteration is the published 5-core "
        "count (both asserted in reconciliation.checks)."
    ),
    "audit_selection": (
        "The matrix is one contract-audit run that covers every contract YAML on disk with exactly "
        "one row per (table, check) — partial or duplicated runs are rejected, never merged."
    ),
}


class DqProjectionError(RuntimeError):
    """dq_raw.json and data/waterfall.json disagree, or no record anchors them."""


# --- record selection ---------------------------------------------------------


def _artifact_path(rec: dict, key: str) -> str:
    try:
        return str(rec["published_artifacts"][key]["path"])
    except (KeyError, TypeError) as exc:
        raise DqProjectionError(
            f"record {rec.get('run_id')!r} has no published_artifacts[{key!r}].path"
        ) from exc


def select_record(runs: dict[str, dict], cfg: dict) -> dict:
    """The ``kind="dq_export"`` record this exhibit projects.

    ``record_run_id: null`` selects the LATEST such record whose published
    artifacts still hash to what it recorded — the most recent record that is
    still a valid anchor for the files on disk. A superseded record (its
    artifacts were regenerated) is skipped rather than used; if none matches the
    caller gets the digests side by side instead of a silently wrong exhibit.
    """
    pinned = cfg.get("record_run_id")
    if pinned:
        if pinned not in runs:
            raise DqProjectionError(f"record_run_id {pinned!r} is not in {cfg['runs_log']}")
        rec = runs[pinned]
        if rec.get("kind") != RECORD_KIND:
            raise DqProjectionError(f"run {pinned}: kind={rec.get('kind')!r}, expected {RECORD_KIND!r}")
        return rec

    candidates = [r for r in runs.values() if r.get("kind") == RECORD_KIND]
    if not candidates:
        raise DqProjectionError(
            f"no kind={RECORD_KIND!r} record in {cfg['runs_log']} — run the dq_export_job record "
            "phase (--append) before exporting dq.json"
        )

    problems: list[str] = []
    for rec in sorted(candidates, key=lambda r: (str(r.get("run_ts")), str(r.get("run_id"))), reverse=True):
        mismatch: list[str] = []
        for key, field in ARTIFACTS:
            path = Path(_artifact_path(rec, key))
            if not path.exists():
                mismatch.append(f"{path} is missing")
                continue
            actual = sha256_file(path)
            if actual != rec.get(field):
                mismatch.append(f"{path} is {actual}, record's {field} is {rec.get(field)}")
        if not mismatch:
            return rec
        problems.append(f"  {rec['run_id']}: " + "; ".join(mismatch))
    raise DqProjectionError(
        f"no kind={RECORD_KIND!r} record anchors the artifacts currently on disk:\n"
        + "\n".join(problems)
    )


# --- waterfall cross-check ----------------------------------------------------


def locate_edge(waterfall_doc: dict, stage: dict) -> tuple[str, int]:
    """``(dataset, edge_index)`` of the waterfall.json edge a dq_raw stage describes.

    Re-derived here, not read out of dq_raw.json: this is the independent half of
    the byte-match check. Exactly one edge must match.
    """
    hits = [
        (dataset, i)
        for dataset, block in waterfall_doc["datasets"].items()
        for i, edge in enumerate(block["edges"])
        if dataset == stage["dataset"]
        and edge["stage_from"] == stage["stage_from"]
        and edge["stage_to"] == stage["stage_to"]
    ]
    if len(hits) != 1:
        raise DqProjectionError(
            f"dq_raw stage {stage['stage']!r} matches {len(hits)} edges in the waterfall artifact"
        )
    return hits[0]


def assert_stage_matches_edge(stage: dict, edge: dict) -> None:
    """Every count dq_raw restates must equal the waterfall artifact's own."""
    for ours, theirs in WATERFALL_EDGE_FIELDS:
        if stage[ours] != edge[theirs]:
            raise DqProjectionError(
                f"waterfall drift at {stage['stage']}: dq_raw {ours}={stage[ours]!r} but "
                f"data/waterfall.json {theirs}={edge[theirs]!r}"
            )
    if stage["delta"] != int(edge["source_rows"]) - int(edge["kept_rows"]):
        raise DqProjectionError(
            f"waterfall drift at {stage['stage']}: delta {stage['delta']} != "
            f"{edge['source_rows']} - {edge['kept_rows']}"
        )
    if len(stage["reasons"]) != len(edge["reasons"]):
        raise DqProjectionError(
            f"waterfall drift at {stage['stage']}: dq_raw lists {len(stage['reasons'])} reasons, "
            f"data/waterfall.json lists {len(edge['reasons'])}"
        )
    for k, (ours, theirs) in enumerate(zip(stage["reasons"], edge["reasons"], strict=True)):
        if ours["reason"] != theirs["reason"] or ours["rows"] != theirs["rows"]:
            raise DqProjectionError(
                f"waterfall drift at {stage['stage']} reason {k}: dq_raw "
                f"{ours['reason']}={ours['rows']} vs data/waterfall.json "
                f"{theirs['reason']}={theirs['rows']}"
            )


# --- build --------------------------------------------------------------------


def build(cfg: dict, runs: dict[str, dict]) -> TracedWriter:
    rec = select_record(runs, cfg)
    run_id = rec["run_id"]

    w = TracedWriter(FILE_NAME, runs, generated_by="batch_recsys_lab.demo.export_dq")
    w.put_descriptive("/record_run_id", run_id, note=f"kind={RECORD_KIND} record this exhibit projects")

    arts: dict[str, dict] = {}
    for key, field in ARTIFACTS:
        path = _artifact_path(rec, key)
        arts[key] = w.register_artifact(key, path, run_id=run_id, anchor_pointer="/" + field)
        w.put_descriptive(
            jp("artifact_sha256", key),
            arts[key]["sha256"],
            note=f"record field {field}; anchors every leaf sourced from {path}",
        )
        w.put_descriptive(jp("artifact_path", key), path, note="committed copy the leaves resolve into")
    raw = arts["dq_raw"]["doc"]
    wfd = arts["waterfall"]["doc"]

    # --- provenance identifiers (strings; declared descriptive) ---------------
    for field, note in (
        ("audit_run_id", "the contract-audit run whose matrix is shown"),
        ("audit_run_ts", "that audit's earliest row timestamp"),
        ("build_run_id", "the build run whose waterfall / funnel / quarantine rows reconcile"),
        ("headline_run_id", "the eval run whose pinned snapshots the Spark job guarded against"),
    ):
        w.put_descriptive(jp(field), raw[field], note=note)

    # --- snapshots: straight out of the record (runs_record source kind) ------
    w.copy_from_record("/iceberg_snapshots", run_id, "/iceberg_snapshots")
    w.copy_from_record("/table_snapshots", run_id, "/table_snapshots")

    # --- waterfall: raw counts from the waterfall artifact itself -------------
    stages = raw["waterfall"]["stages"]
    w.ensure_list("/waterfall/stages", len(stages))
    for i, stage in enumerate(stages):
        dataset, j = locate_edge(wfd, stage)
        edge = wfd["datasets"][dataset]["edges"][j]
        assert_stage_matches_edge(stage, edge)

        base = jp("waterfall", "stages", i)
        edge_ptr = jp("datasets", dataset, "edges", j)
        w.put_descriptive(base + "/stage", stage["stage"], note="stage label")
        w.put_descriptive(base + "/dataset", stage["dataset"], note="dataset chain")
        w.put_descriptive(base + "/stage_from", stage["stage_from"], note="edge tail")
        w.put_descriptive(base + "/stage_to", stage["stage_to"], note="edge head")
        w.put_descriptive(base + "/target_table", stage["target_table"], note="published table")
        for ours, theirs in WATERFALL_EDGE_FIELDS:
            w.copy_from_artifact(base + jp(ours), "waterfall", edge_ptr + jp(theirs))
        for field in WATERFALL_DERIVED_FIELDS:
            w.copy_from_artifact(base + jp(field), "dq_raw", jp("waterfall", "stages", i, field))

        w.ensure_list(base + "/reasons", len(stage["reasons"]))
        for k, reason in enumerate(stage["reasons"]):
            rbase = base + jp("reasons", k)
            w.put_descriptive(rbase + "/reason", reason["reason"], note="drop-reason label")
            w.copy_from_artifact(rbase + "/rows", "waterfall", edge_ptr + jp("reasons", k, "rows"))
            w.copy_from_artifact(
                rbase + "/share_of_rows_in",
                "dq_raw",
                jp("waterfall", "stages", i, "reasons", k, "share_of_rows_in"),
            )
    w.copy_from_artifact("/waterfall/reconciles", "dq_raw", "/waterfall/reconciles")
    w.copy_from_artifact("/waterfall/ledger_rows_checked", "dq_raw", "/waterfall/ledger_rows_checked")
    w.put_descriptive("/waterfall/run_id", raw["waterfall"]["run_id"], note="build run the ledger rows carry")

    # --- contract matrix (AGREED shape: table -> check -> {status, measured, threshold})
    matrix = raw["contract_matrix"]
    for table in matrix:
        for check_id, chk in matrix[table]["checks"].items():
            base = jp("contract_matrix", table, check_id)
            src = jp("contract_matrix", table, "checks", check_id)
            w.copy_from_artifact(base + "/status", "dq_raw", src + "/status")
            w.copy_from_artifact(base + "/measured", "dq_raw", src + "/measured")
            w.copy_from_artifact(base + "/violations", "dq_raw", src + "/violations")
            w.put_descriptive(base + "/kind", chk["kind"], note="check kind")
            w.put_descriptive(base + "/column", chk["column"], note="checked column, if single")
            w.put_descriptive(
                base + "/threshold",
                None,
                note="contracts declare no numeric thresholds (pass/fail predicates and measured "
                "shares only) — always null, kept for the AGREED schema shape",
            )
    # Per-table metadata lives in a sibling map so the matrix namespace stays
    # pure check ids (a check named "total_rows" would otherwise collide).
    for table in matrix:
        base = jp("contract_tables", table)
        src = jp("contract_matrix", table)
        w.put_descriptive(base + "/contract_name", matrix[table]["contract_name"], note="contract YAML name")
        w.copy_from_artifact(base + "/contract_version", "dq_raw", src + "/contract_version")
        w.copy_from_artifact(base + "/total_rows", "dq_raw", src + "/total_rows")
        w.copy_from_artifact(base + "/counts", "dq_raw", src + "/counts")

    # --- the remaining dq_raw subtrees, verbatim ------------------------------
    for pointer in RAW_SUBTREES:
        w.copy_from_artifact(pointer, "dq_raw", pointer)

    # --- k-core funnel --------------------------------------------------------
    funnel = raw["kcore_funnel"]
    w.ensure_list("/kcore_funnel", len(funnel))
    for i in range(len(funnel)):
        for ours, theirs in FUNNEL_FIELDS:
            w.copy_from_artifact(jp("kcore_funnel", i, ours), "dq_raw", jp("kcore_funnel", i, theirs))

    w.put_descriptive(
        "/notes",
        NOTES,
        subtree=True,
        note="exhibit copy explaining the provenance split (descriptive text, not evidence)",
    )
    return w


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/dq_export.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    runs = index_runs(cfg["runs_log"])
    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])

    writer = build(cfg, runs)
    out = write_document(writer, cfg["out_dir"], manifest)
    manifest.drop_missing_files(cfg["out_dir"])
    manifest.write()
    print(
        f"wrote {out} ({len(writer.entries)} traced leaves, "
        f"{len(writer.descriptive)} descriptive) · manifest {Path(cfg['manifest'])}"
    )


if __name__ == "__main__":
    main()
