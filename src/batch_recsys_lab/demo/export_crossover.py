"""``demo/data/crossover.json`` — exhibit 1's data (Phase 6, T26; plan §9.1).

Projects the five canonical TEST eval records (plus the hybrid confirming run,
which is numerically identical to blend and carried as an annotation, not a
sixth line) and the pinned paired-bootstrap deltas into the shape the site's
crossover explorer reads. Every metric leaf is copied verbatim out of a
``results/runs.jsonl`` record through :class:`TracedWriter`, at full precision;
labels and ordering are the only untraced values.

    uv run python -m batch_recsys_lab.demo.export_crossover --config configs/demo_export.yaml

Schema: docs/demo-data-schemas.md § crossover.json.
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
    resolve_pointer,
    write_document,
)

FILE_NAME = "crossover.json"
CI_KEYS = ("value", "ci_lo", "ci_hi")
DELTA_KEYS = ("delta", "ci_lo", "ci_hi", "excludes_zero")


def _check_eval_record(rec: dict, run_id: str, split: str) -> None:
    if rec.get("kind") != "eval":
        raise ValueError(f"run {run_id}: kind={rec.get('kind')!r}, expected 'eval'")
    got = rec.get("protocol", {}).get("eval_split")
    if got != split:
        raise ValueError(f"run {run_id}: eval_split={got!r} but the demo config wants {split!r}")


def _check_metric_block(rec: dict, run_id: str, pointer: str, keys: tuple[str, ...]) -> None:
    try:
        block = resolve_pointer(rec, pointer)
    except KeyError as e:
        raise ValueError(f"run {run_id}: {e}") from e
    missing = [k for k in keys if k not in block]
    if missing:
        raise ValueError(f"run {run_id}: {pointer} lacks {missing}")


def build(cfg: dict, runs: dict[str, dict]) -> TracedWriter:
    split: str = cfg["split"]
    segments: list[str] = cfg["segments"]
    metrics: list[str] = cfg["metrics"]

    w = TracedWriter(FILE_NAME, runs, generated_by="batch_recsys_lab.demo.export_crossover")
    w.put_descriptive("/split", split, note="eval split of every record cited here")
    w.put_descriptive("/segments", segments, subtree=True, note="segment display order")
    w.put_descriptive("/metrics", metrics, subtree=True, note="exported metric keys")
    w.put_descriptive("/model_order", [m["key"] for m in cfg["models"]], subtree=True, note="line order (palette slot order)")
    for key in ("title", "subtitle", "xlabel"):
        if cfg.get(key):
            w.put_descriptive(jp(key), cfg[key], note="chart copy (configs/demo_export.yaml)")
    w.put_descriptive("/headline_run_id", cfg["headline_run_id"], note="run the exhibit headlines")

    # --- models -------------------------------------------------------------
    for m in cfg["models"]:
        key, run_id = m["key"], m["run_id"]
        base = jp("models", key)
        if run_id not in runs:
            raise ValueError(f"model {key!r}: run_id {run_id!r} not found in {cfg['runs_log']}")
        rec = runs[run_id]
        _check_eval_record(rec, run_id, split)

        w.put_descriptive(base + "/key", key, note="stable model key")
        w.put_descriptive(base + "/label", m["label"], note="display label")
        w.put_descriptive(base + "/highlight", bool(m.get("highlight", False)), note="emphasised line")
        w.put_descriptive(base + "/plot", bool(m.get("plot", True)), note="drawn as a chart line")
        if m.get("identical_to"):
            w.put_descriptive(
                base + "/identical_to",
                m["identical_to"],
                note="this run is numerically identical to the named model (see paired_deltas)",
            )
        w.copy_from_record(base + "/run_id", run_id, "/run_id")
        w.copy_from_record(base + "/model_name", run_id, "/model/name")
        w.copy_from_record(base + "/git_sha", run_id, "/git_sha")
        w.copy_from_record(base + "/n_users", run_id, "/protocol/n_users")
        w.copy_from_record(base + "/catalog_size", run_id, "/protocol/catalog_size")

        for metric in metrics:
            src = jp("metrics", "global", metric)
            _check_metric_block(rec, run_id, src, CI_KEYS)
            for k in CI_KEYS:
                w.copy_from_record(base + jp("global", metric, k), run_id, src + jp(k))

        for seg in segments:
            seg_src = jp("metrics", "per_segment", seg)
            try:
                resolve_pointer(rec, seg_src)
            except KeyError as e:
                raise ValueError(f"run {run_id}: segment {seg!r} missing — {e}") from e
            dst = base + jp("segments", seg)
            w.copy_from_record(dst + "/n_users", run_id, seg_src + "/n_users")
            for metric in metrics:
                _check_metric_block(rec, run_id, seg_src + jp(metric), CI_KEYS)
                for k in CI_KEYS:
                    w.copy_from_record(dst + jp(metric, k), run_id, seg_src + jp(metric, k))

    # --- paired deltas -------------------------------------------------------
    pds = cfg.get("paired_deltas") or []
    w.put_descriptive("/paired_delta_order", [p["key"] for p in pds], subtree=True, note="delta display order")
    for pd in pds:
        key, run_id = pd["key"], pd["run_id"]
        base = jp("paired_deltas", key)
        if run_id not in runs:
            raise ValueError(f"paired_delta {key!r}: run_id {run_id!r} not found in {cfg['runs_log']}")
        rec = runs[run_id]
        if rec.get("kind") != "paired_delta":
            raise ValueError(f"paired_delta {key!r}: run {run_id} has kind={rec.get('kind')!r}")

        w.put_descriptive(base + "/key", key, note="stable comparison key")
        w.put_descriptive(base + "/label", pd["label"], note="display label")
        w.copy_from_record(base + "/run_id", run_id, "/run_id")
        w.copy_from_record(base + "/a", run_id, "/a")
        w.copy_from_record(base + "/b", run_id, "/b")
        w.copy_from_record(base + "/n_common_users", run_id, "/n_common_users")
        for metric in metrics:
            src = jp("deltas", "global", metric)
            _check_metric_block(rec, run_id, src, DELTA_KEYS)
            for k in DELTA_KEYS:
                w.copy_from_record(base + jp("global", metric, k), run_id, src + jp(k))
        for seg in segments:
            for metric in metrics:
                src = jp("deltas", "per_segment", seg, metric)
                _check_metric_block(rec, run_id, src, DELTA_KEYS)
                for k in DELTA_KEYS:
                    w.copy_from_record(base + jp("segments", seg, metric, k), run_id, src + jp(k))
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
