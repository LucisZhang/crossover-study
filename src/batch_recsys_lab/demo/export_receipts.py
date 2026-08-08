"""``demo/data/receipts.json`` — the receipts drawer's provenance cards
(Phase 6, T26; plan §9 "Receipts drawer").

For the CLOSURE of run_ids the trace manifest depends on, copies each record's
provenance block verbatim out of ``results/runs.jsonl``: kind, run_ts, git SHA
+ dirty flag, config path + hash, dataset manifest hash, splits (frozen file
hash), Iceberg snapshot IDs, seeds, model name/params, wall clock, hardware.
Nothing is computed here — a receipt disagreeing with the log is a verification
failure, not a rounding difference.

Closure = run_ids referenced by manifest entries, transitively expanded through
the records themselves: a ``paired_delta`` pulls in the two runs it compares, a
``reproduce`` pulls in the run it reproduces. The headline run additionally
carries every ``reproduce`` verdict against it and the one-line repro command.

Run AFTER the other exporters (it reads the manifest they wrote):

    uv run python -m batch_recsys_lab.demo.export_receipts --config configs/demo_export.yaml

Schema: docs/demo-data-schemas.md § receipts.json.
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

FILE_NAME = "receipts.json"

# (field name in the receipt, JSON pointer into the record). Copied verbatim,
# leaf by leaf; every one is a trace entry.
RECEIPT_FIELDS: tuple[tuple[str, str], ...] = (
    ("kind", "/kind"),
    ("run_ts", "/run_ts"),
    ("git_sha", "/git_sha"),
    ("git_dirty", "/git_dirty"),
    ("config_path", "/config_path"),
    ("config_hash", "/config_hash"),
    ("dataset_manifest_hash", "/dataset_manifest_hash"),
    ("splits", "/splits"),
    ("iceberg_snapshots", "/iceberg_snapshots"),
    ("seeds", "/seeds"),
    ("model", "/model"),
    ("wall_clock_s", "/wall_clock_s"),
    ("hardware", "/hardware"),
)

# Fields an eval record must have for its receipt to be complete; other record
# kinds (paired_delta, reproduce, lineage, ops, ann_receipt) legitimately carry
# a subset, so for them a missing field is recorded, not an error.
REQUIRED_FOR_EVAL = tuple(name for name, _ in RECEIPT_FIELDS)


def closure(seed_run_ids: set[str], runs: dict[str, dict]) -> list[str]:
    """Expand run_ids through the records that reference other records."""
    seen: set[str] = set()
    frontier = set(seed_run_ids)
    while frontier:
        rid = frontier.pop()
        if rid in seen:
            continue
        if rid not in runs:
            raise ValueError(f"run_id {rid!r} referenced by the trace manifest is not in the runs log")
        seen.add(rid)
        rec = runs[rid]
        for nxt in (
            rec.get("a", {}).get("run_id"),
            rec.get("b", {}).get("run_id"),
            rec.get("reproduces_run_id"),
        ):
            if nxt and nxt not in seen:
                frontier.add(nxt)
    return sorted(seen)


def reproduce_records(headline_run_id: str, runs: dict[str, dict]) -> list[dict]:
    """Every ``kind="reproduce"`` record that targets the headline run, in log order."""
    return [
        rec
        for rec in runs.values()
        if rec.get("kind") == "reproduce" and rec.get("reproduces_run_id") == headline_run_id
    ]


def build(cfg: dict, runs: dict[str, dict], run_ids: list[str]) -> TracedWriter:
    headline = cfg["headline_run_id"]
    w = TracedWriter(FILE_NAME, runs, generated_by="batch_recsys_lab.demo.export_receipts")
    w.put_descriptive("/headline_run_id", headline, note="run the demo headlines")
    w.put_descriptive("/run_order", list(run_ids), subtree=True, note="receipt display order")
    w.put_descriptive(
        "/note",
        "Every field below is copied verbatim from the matching results/runs.jsonl "
        "record (append-only). demo/data/trace_manifest.json re-resolves each one.",
        note="drawer footnote",
    )

    for rid in run_ids:
        rec = runs[rid]
        base = jp("runs", rid)
        w.copy_from_record(base + "/run_id", rid, "/run_id")
        missing = []
        for name, ptr in RECEIPT_FIELDS:
            if ptr.strip("/") in rec:
                w.copy_from_record(base + jp(name), rid, ptr)
            else:
                missing.append(name)
        if missing and rec.get("kind") == "eval":
            raise ValueError(f"eval record {rid} is missing receipt fields {missing}")
        if missing:
            w.put_descriptive(
                base + "/fields_absent_in_record",
                missing,
                subtree=True,
                note=f"kind={rec.get('kind')!r} records do not carry these fields",
            )
        if rid == headline:
            reps = reproduce_records(rid, runs)
            if not reps:
                raise ValueError(f"headline run {rid} has no kind='reproduce' record to attach")
            w.ensure_list(base + "/reproduce", len(reps))
            for i, rep in enumerate(reps):
                w.copy_from_record(base + jp("reproduce", i, "run_id"), rep["run_id"], "/run_id")
                w.copy_from_record(base + jp("reproduce", i, "verdict"), rep["run_id"], "/verdict")
            w.put_descriptive(
                base + "/repro_command",
                cfg.get("repro_command", "make reproduce-headline"),
                note="regenerates this record from the pinned Iceberg snapshot",
            )
    return w


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/demo_export.yaml")
    args = ap.parse_args(argv)

    cfg = load_export_config(args.config)
    runs = index_runs(cfg["runs_log"])
    manifest_path = Path(cfg["manifest"])
    if not manifest_path.exists():
        raise SystemExit(
            f"ERROR: {manifest_path} does not exist — run the data exporters "
            "(e.g. export_crossover) before export_receipts."
        )
    manifest = TraceManifest(manifest_path, cfg["runs_log"])
    seeds = {r for r in manifest.run_ids() if r} | {cfg["headline_run_id"]}
    # The reproduce verdicts this exporter is about to attach are themselves
    # traced to their own records, so the closure has to contain them — else
    # receipts.json would cite run_ids it does not document (the verifier's
    # RECEIPTS check catches exactly that).
    seeds |= {rep["run_id"] for rep in reproduce_records(cfg["headline_run_id"], runs)}
    # Provenance-cited run_ids outside the trace manifest's closure: exhibits may
    # cite a record as provenance without tracing numbers to it (the search
    # exhibit's ann_receipt chip). Config-pinned so the citation set stays explicit.
    seeds |= set(cfg.get("extra_run_ids", []))
    run_ids = closure(seeds, runs)

    writer = build(cfg, runs, run_ids)
    out = write_document(writer, cfg["out_dir"], manifest)
    manifest.drop_missing_files(cfg["out_dir"])
    manifest.write()
    print(
        f"wrote {out} ({len(run_ids)} receipts, {len(writer.entries)} traced leaves) "
        f"· manifest {manifest_path}"
    )


if __name__ == "__main__":
    main()
