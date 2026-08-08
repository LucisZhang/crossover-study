"""``demo/data/policy_grid.json`` — exhibit 2's data (Phase 6, T27; plan §9.2).

Projects the single ``kind="policy_grid"`` record (TEST recomposition — pure
arithmetic regrouping of already-committed per-user metrics, no re-scoring, no
refitting) plus its VAL-selection context (``results/policy_select_val.json``,
consumed as a record-anchored results artifact) into the shape the site's n*
slider reads. Every numeric leaf is traced through :class:`TracedWriter`, at
full precision.

    uv run python -m batch_recsys_lab.demo.export_policy_grid --config configs/demo_export.yaml

Run BEFORE export_receipts (receipts reads the manifest this exporter writes
into to find the run_id closure it must document).

Schema: docs/demo-data-schemas.md § policy_grid.json.
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

FILE_NAME = "policy_grid.json"
N_STAR_LABELS = ("0", "1", "5", "10", "20", "inf")


def _grid_index(rec: dict, variant: str, n_star_label: str) -> int:
    for i, cell in enumerate(rec["grid"]):
        if cell["variant"] == variant and cell["n_star_label"] == n_star_label:
            return i
    raise ValueError(f"policy_grid record {rec['run_id']}: no grid cell for {variant}/{n_star_label}")


def build(cfg: dict, runs: dict[str, dict]) -> TracedWriter:
    pg_cfg = cfg["policy_grid"]
    run_id = pg_cfg["run_id"]
    if run_id not in runs:
        raise ValueError(f"policy_grid: run_id {run_id!r} not found in {cfg['runs_log']}")
    rec = runs[run_id]
    if rec.get("kind") != "policy_grid":
        raise ValueError(f"run {run_id}: kind={rec.get('kind')!r}, expected 'policy_grid'")
    split: str = cfg["split"]
    if rec.get("split") != split:
        raise ValueError(f"run {run_id}: split={rec.get('split')!r} but the demo config wants {split!r}")
    segments: list[str] = cfg["segments"]
    metrics: list[str] = cfg["metrics"]

    w = TracedWriter(FILE_NAME, runs, generated_by="batch_recsys_lab.demo.export_policy_grid")
    w.put_descriptive("/record_run_id", run_id, note="kind=policy_grid record this exhibit projects")
    w.put_descriptive("/split", split, note="eval split of the source per-user metrics")
    w.put_descriptive("/segments", segments, subtree=True, note="segment display order")
    w.put_descriptive("/metrics", metrics, subtree=True, note="exported metric keys")

    w.copy_from_record("/n_star_grid", run_id, "/n_star_grid")
    w.put_descriptive("/n_star_labels", list(N_STAR_LABELS), subtree=True, note="slider snap labels, index-aligned with n_star_grid")

    for key, v in pg_cfg["variants"].items():
        base = jp("variants", key)
        w.put_descriptive(base + "/label", v["label"], note="display label")
        w.put_descriptive(base + "/low", v["low"], note="low arm (n_train < n*)")
        w.put_descriptive(base + "/high", v["high"], note="high arm (n_train >= n*)")

    shipped = pg_cfg["shipped"]
    w.put_descriptive("/shipped/variant", shipped["variant"], note="shipped policy — highlighted cell")
    w.put_descriptive("/shipped/n_star_label", shipped["n_star_label"], note="shipped policy — highlighted cell")

    # --- TEST cells: all 12 (2 variants x 6 n* labels) -----------------------
    inf_run_id = rec["source_run_ids"]["blend_a30"]
    zero_run_ids = {"A": rec["source_run_ids"]["als_chosen"], "B": rec["source_run_ids"]["pop_t12m"]}

    for variant in ("A", "B"):
        for label in N_STAR_LABELS:
            idx = _grid_index(rec, variant, label)
            src = jp("grid", idx)
            base = jp("cells", variant, label)
            w.copy_from_record(base + "/n_star", run_id, src + "/n_star")
            w.copy_from_record(base + "/n_star_label", run_id, src + "/n_star_label")
            w.copy_from_record(base + "/low_share", run_id, src + "/low_share")
            for metric in metrics:
                for k in ("value", "ci_lo", "ci_hi"):
                    w.copy_from_record(
                        base + jp("global", metric, k), run_id, src + jp("global", metric, k)
                    )
            for seg in segments:
                w.copy_from_record(base + jp("segments", seg, "n_users"), run_id, src + jp("per_segment", seg, "n_users"))
                for metric in metrics:
                    for k in ("value", "ci_lo", "ci_hi"):
                        w.copy_from_record(
                            base + jp("segments", seg, metric, k),
                            run_id,
                            src + jp("per_segment", seg, metric, k),
                        )
            if label == "inf":
                w.copy_from_record(base + "/identity/equals_run_id", run_id, "/source_run_ids/blend_a30")
                w.copy_from_record(base + "/identity/asserted", run_id, "/identity_checks/inf_matches_blend")
            elif label == "0":
                src_key = "als_chosen" if variant == "A" else "pop_t12m"
                w.copy_from_record(base + "/identity/equals_run_id", run_id, jp("source_run_ids", src_key))
                w.copy_from_record(base + "/identity/asserted", run_id, "/identity_checks/zero_matches_high")

    # --- VAL selection context ------------------------------------------------
    val_art_cfg = cfg["artifacts"]["policy_select_val"]
    if val_art_cfg["run_id"] != run_id:
        raise ValueError(
            f"policy_select_val artifact must be anchored on the policy_grid run {run_id!r}, "
            f"config has {val_art_cfg['run_id']!r}"
        )
    w.register_artifact(
        "policy_select_val",
        val_art_cfg["path"],
        run_id=run_id,
        anchor_pointer=val_art_cfg["anchor_pointer"],
    )
    val_doc_path = val_art_cfg["path"]
    import json as _json

    val_doc = _json.loads(Path(val_doc_path).read_text())
    n_val_cells = len(val_doc["grid"])
    w.ensure_list("/val_grid", n_val_cells)
    for i in range(n_val_cells):
        vbase = jp("val_grid", i)
        vsrc = jp("grid", i)
        w.copy_from_artifact(vbase + "/variant", "policy_select_val", vsrc + "/variant")
        w.copy_from_artifact(vbase + "/n_star", "policy_select_val", vsrc + "/n_star")
        w.copy_from_artifact(vbase + "/n_star_label", "policy_select_val", vsrc + "/n_star_label")
        w.copy_from_artifact(vbase + "/objective", "policy_select_val", vsrc + "/objective")
        w.copy_from_artifact(vbase + "/segment_means", "policy_select_val", vsrc + "/segment_means")

    w.copy_from_artifact("/val_winner", "policy_select_val", "/winner")
    w.put_descriptive("/objective", val_doc["objective"], note="VAL selection objective (results/policy_select_val.json)")
    w.put_descriptive(
        "/n_star_selected_on_val",
        None,
        note="null: the VAL winner is B/inf (blend-everywhere) — no finite n* beat it on VAL",
    )
    w.put_descriptive(
        "/notes/variant_b_cold_collapse",
        "Variant B's n*=0 and n*=1 cells are bit-identical by construction: "
        "cold users (n_train < n*) have empty history, so the blend arm's "
        "content signal collapses to pop-t12m regardless of where n* sits "
        "below the smallest non-zero history depth.",
        note="descriptive: explains an identity visible in the cells above, not itself a measured value",
    )

    w.copy_from_record("/seeds/bootstrap", run_id, "/seeds/bootstrap")
    w.copy_from_record("/n_resamples", run_id, "/bootstrap/n_resamples")
    return w


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/demo_export.yaml")
    args = ap.parse_args(argv)

    cfg = load_export_config(args.config)
    runs = index_runs(cfg["runs_log"])
    writer = build(cfg, runs)
    manifest = TraceManifest(cfg["manifest"], cfg["runs_log"])
    out = write_document(writer, cfg["out_dir"], manifest)
    manifest.drop_missing_files(cfg["out_dir"])
    manifest.write()
    print(
        f"wrote {out} ({len(writer.entries)} traced leaves, "
        f"{len(writer.descriptive)} descriptive) · manifest {Path(cfg['manifest'])}"
    )


if __name__ == "__main__":
    main()
