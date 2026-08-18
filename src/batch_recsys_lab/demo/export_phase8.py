"""``demo/data/phase8.json`` — Phase 8 exhibit data (plan §8b line 277).

Projects the T8-2 recency-matched TEST arms, the T8-3 exploratory deep depth
buckets and the T8-1/T8-2 regime maps into one demo document, through the
same ``TracedWriter`` every other exporter uses: every metric leaf is copied
verbatim out of a ``results/runs.jsonl`` record at full precision, labels and
ordering are the only untraced values, and ``make demo-verify`` re-resolves
every leaf independently.

    uv run python -m batch_recsys_lab.demo.export_phase8 --config configs/demo_export.yaml

then re-run export_receipts so the receipts closure picks up the new run_ids:

    uv run python -m batch_recsys_lab.demo.export_receipts --config configs/demo_export.yaml
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

FILE_NAME = "phase8.json"

SEGMENTS = ["0", "1-4", "5-9", "10-19", "20+"]
METRICS = ["ndcg@10", "recall@20"]
CI_KEYS = ("value", "ci_lo", "ci_hi")
DELTA_KEYS = ("delta", "ci_lo", "ci_hi", "excludes_zero")

# --- T8-2: recency-matched arms, one preregistered TEST evaluation each -------
ARMS = [
    {
        "key": "itemknn_t12m",
        "label": "item-kNN-t12m (T8-2)",
        "run_id": "20260818T054430Z-109c271",
        "seed_run_ids": [],
    },
    {
        "key": "alsdecay_hl365",
        "label": "ALS-decay hl365 (T8-2)",
        # Primary seed (carries the per-user artifact); the two sibling seeds
        # are exported alongside so the page can show the 3-seed spread.
        "run_id": "20260818T060704Z-109c271",
        "seed_run_ids": [
            "20260818T060704Z-109c271",
            "20260818T051547Z-109c271",
            "20260818T051858Z-109c271",
        ],
    },
]

PAIRED = [
    ("alsdecay_vs_pop_t12m", "ALS-decay hl365 − pop-t12m", "20260818T064002Z-56d871c"),
    ("alsdecay_vs_blend", "ALS-decay hl365 − blend α=0.3", "20260818T064104Z-56d871c"),
    ("itemknn_t12m_vs_pop_t12m", "item-kNN-t12m − pop-t12m", "20260818T064207Z-56d871c"),
    ("itemknn_t12m_vs_blend", "item-kNN-t12m − blend α=0.3", "20260818T064306Z-56d871c"),
]

# --- T8-3: exploratory deep depth buckets (kind="deep_buckets") ---------------
DEEP_RUN = "20260817T100253Z-633d454"
DEEP_LABELS = ["0", "1-4", "5-9", "10-19", "20-49", "50-99", "100+"]
DEEP_ARMS = ["pop_t12m", "als"]

# --- T8-1 / T8-2: regime maps (kind="regime_map") -----------------------------
# T8-1 appended an accidental duplicate (20260817T100112Z-633d454) whose
# results compare equal field-for-field; the EXPERIMENT_LOG entry cites
# 20260817T095926Z-633d454 as primary, so the demo does too.
MAPS = [
    {
        "key": "knn_t12m",
        "label": "item-kNN-t12m − pop-t12m (T8-2)",
        "run_id": "20260818T072256Z-3f3530a",
        "arm_key": "knn_t12m",
        "arm_label": "item-kNN-t12m",
    },
    {
        "key": "alsdecay",
        "label": "ALS-decay hl365 − pop-t12m (T8-2)",
        "run_id": "20260818T072211Z-3f3530a",
        "arm_key": "alsdecay",
        "arm_label": "ALS-decay hl365",
    },
    {
        "key": "als",
        "label": "ALS static − pop-t12m (T8-1 baseline)",
        "run_id": "20260817T095926Z-633d454",
        "arm_key": "als",
        "arm_label": "ALS (static)",
    },
]
GATE_RUN = "20260817T095926Z-633d454"  # T8-1: the measured-churn receipt
AXES = ("support", "recency")

# Verbatim from EXPERIMENT_LOG.md, "Phase 8 T8-2 VERDICT" (2026-08-18),
# "Honest scope note for the routing narrative." (*...* = emphasis in source).
CAVEAT = (
    "The winning cells are defined by properties of the *ground-truth item* "
    "(its TRAIN recency / support), which a serving-time router cannot "
    "observe. The measured crossover is therefore diagnostic — popularity's "
    "blind spot is real and a recency-matched CF arm can exploit it — but "
    "converting it into a routable policy needs a serve-time proxy for "
    "“this user shops the stale catalog”, which is out of T8-2 scope."
)
CAVEAT_SOURCE = (
    "EXPERIMENT_LOG.md — “Phase 8 T8-2 VERDICT — recency-matched arms: "
    "technical crossover, in the stale-item pocket, by the arm nobody "
    "favored” (2026-08-18), “Honest scope note for the routing "
    "narrative.” Quoted verbatim; source emphasis marked *…*."
)
MULTIPLICITY = (
    "Multiplicity disclosure: ~40 cells × 2 arms tested with no correction, "
    "per the preregistered any-cell rule; the clustering and two-metric "
    "agreement argue against pure chance, but the per-cell magnitudes are "
    "small and the affected mass is a minority pocket (each cell 0.7–2.2% of "
    "TEST GT; the cells overlap across axes)."
)


def check_eval(rec, run_id, split="test"):
    if rec.get("kind") != "eval":
        raise ValueError(f"run {run_id}: kind={rec.get('kind')!r}, expected 'eval'")
    got = rec.get("protocol", {}).get("eval_split")
    if got != split:
        raise ValueError(f"run {run_id}: eval_split={got!r}, expected {split!r}")


def build(runs: dict[str, dict]) -> TracedWriter:
    w = TracedWriter(FILE_NAME, runs, generated_by="batch_recsys_lab.demo.export_phase8")
    w.put_descriptive("/split", "test", note="eval split of every record cited here")
    w.put_descriptive("/segments", SEGMENTS, subtree=True, note="frozen segment display order")
    w.put_descriptive("/metrics", METRICS, subtree=True, note="exported metric keys")

    # ---------------- arms (same shape as crossover.json models) -------------
    w.put_descriptive("/arm_order", [a["key"] for a in ARMS], subtree=True, note="line order")
    for arm in ARMS:
        key, run_id = arm["key"], arm["run_id"]
        rec = runs[run_id]
        check_eval(rec, run_id)
        base = jp("arms", key)
        w.put_descriptive(base + "/key", key, note="stable model key")
        w.put_descriptive(base + "/label", arm["label"], note="display label")
        w.put_descriptive(base + "/highlight", False, note="emphasised line")
        w.put_descriptive(base + "/plot", True, note="drawn as a chart line")
        w.put_descriptive(
            base + "/phase8", True, note="Phase 8 recency-matched arm (one preregistered TEST run)"
        )
        w.copy_from_record(base + "/run_id", run_id, "/run_id")
        w.copy_from_record(base + "/model_name", run_id, "/model/name")
        w.copy_from_record(base + "/git_sha", run_id, "/git_sha")
        w.copy_from_record(base + "/n_users", run_id, "/protocol/n_users")
        w.copy_from_record(base + "/catalog_size", run_id, "/protocol/catalog_size")
        if key == "itemknn_t12m":
            w.copy_from_record(base + "/train_window_days", run_id, "/model/params/train_window_days")
        if key == "alsdecay_hl365":
            w.copy_from_record(base + "/half_life_days", run_id, "/model/params/half_life_days")
        for metric in METRICS:
            src = jp("metrics", "global", metric)
            for k in CI_KEYS:
                w.copy_from_record(base + jp("global", metric, k), run_id, src + jp(k))
        for seg in SEGMENTS:
            seg_src = jp("metrics", "per_segment", seg)
            dst = base + jp("segments", seg)
            w.copy_from_record(dst + "/n_users", run_id, seg_src + "/n_users")
            for metric in METRICS:
                for k in CI_KEYS:
                    w.copy_from_record(dst + jp(metric, k), run_id, seg_src + jp(metric, k))
        # 3-seed sibling records: run_id, model seed and TEST global per seed.
        if arm["seed_run_ids"]:
            w.ensure_list(base + "/seeds", len(arm["seed_run_ids"]))
            for i, srid in enumerate(arm["seed_run_ids"]):
                srec = runs[srid]
                check_eval(srec, srid)
                sbase = base + jp("seeds", i)
                w.copy_from_record(sbase + "/run_id", srid, "/run_id")
                w.copy_from_record(sbase + "/model_seed", srid, "/model/params/seed")
                for metric in METRICS:
                    src = jp("metrics", "global", metric)
                    for k in CI_KEYS:
                        w.copy_from_record(sbase + jp("global", metric, k), srid, src + jp(k))

    # ---------------- paired deltas (same shape as crossover.json) -----------
    w.put_descriptive(
        "/paired_delta_order", [k for k, _, _ in PAIRED], subtree=True, note="delta display order"
    )
    for key, label, run_id in PAIRED:
        rec = runs[run_id]
        if rec.get("kind") != "paired_delta":
            raise ValueError(f"paired_delta {key!r}: run {run_id} has kind={rec.get('kind')!r}")
        base = jp("paired_deltas", key)
        w.put_descriptive(base + "/key", key, note="stable comparison key")
        w.put_descriptive(base + "/label", label, note="display label")
        w.copy_from_record(base + "/run_id", run_id, "/run_id")
        w.copy_from_record(base + "/a", run_id, "/a")
        w.copy_from_record(base + "/b", run_id, "/b")
        w.copy_from_record(base + "/n_common_users", run_id, "/n_common_users")
        for metric in METRICS:
            src = jp("deltas", "global", metric)
            for k in DELTA_KEYS:
                w.copy_from_record(base + jp("global", metric, k), run_id, src + jp(k))
        for seg in SEGMENTS:
            for metric in METRICS:
                src = jp("deltas", "per_segment", seg, metric)
                for k in DELTA_KEYS:
                    w.copy_from_record(base + jp("segments", seg, metric, k), run_id, src + jp(k))

    # ---------------- deep buckets (T8-3, exploratory/derived) ---------------
    rec = runs[DEEP_RUN]
    if rec.get("kind") != "deep_buckets":
        raise ValueError(f"{DEEP_RUN}: kind={rec.get('kind')!r}, expected 'deep_buckets'")
    if rec.get("exploratory_derived") is not True:
        raise ValueError(f"{DEEP_RUN}: expected exploratory_derived=true")
    got_labels = [b["bucket"] for b in rec["results"]["buckets"]]
    if got_labels != DEEP_LABELS:
        raise ValueError(f"{DEEP_RUN}: bucket labels {got_labels} != {DEEP_LABELS}")
    db = "/deep_buckets"
    w.put_descriptive(db + "/exploratory", True, note="T8-3 is exploratory/derived, not confirmatory")
    w.put_descriptive(db + "/labels", DEEP_LABELS, subtree=True, note="bucket display order")
    w.put_descriptive(db + "/arm_keys", DEEP_ARMS, subtree=True, note="arms in the deep_buckets record")
    w.put_descriptive(
        db + "/preregistered",
        rec["buckets_spec"]["preregistered"],
        note="copied from the record's buckets_spec (string)",
    )
    w.copy_from_record(db + "/run_id", DEEP_RUN, "/run_id")
    for a in DEEP_ARMS:
        w.copy_from_record(db + jp("source_run_ids", a), DEEP_RUN, jp("source_run_ids", a))
    for i, label in enumerate(DEEP_LABELS):
        src = jp("results", "buckets", i)
        dst = db + jp("buckets", label)
        w.copy_from_record(dst + "/n_users", DEEP_RUN, src + "/n_users")
        w.copy_from_record(dst + "/user_share", DEEP_RUN, src + "/user_share")
        for a in DEEP_ARMS:
            for metric in METRICS:
                for k in CI_KEYS:
                    w.copy_from_record(dst + jp("arms", a, metric, k), DEEP_RUN, src + jp("arms", a, metric, k))
        for metric in METRICS:
            for k in DELTA_KEYS:
                w.copy_from_record(dst + jp("delta", metric, k), DEEP_RUN, src + jp("delta", metric, k))
    w.copy_from_record(db + "/delta_label", DEEP_RUN, "/results/buckets/0/delta_label")

    # ---------------- regime maps (T8-1 measured churn + T8-2 recomposition) --
    rm = "/regime_map"
    w.put_descriptive(rm + "/map_order", [m["key"] for m in MAPS], subtree=True, note="arm switch order")
    w.put_descriptive(rm + "/axes_order", list(AXES), subtree=True, note="cell axis display order")
    w.put_descriptive(rm + "/caveat", CAVEAT, note="verbatim from the T8-2 verdict entry")
    w.put_descriptive(rm + "/caveat_source", CAVEAT_SOURCE, note="where the caveat is quoted from")
    w.put_descriptive(rm + "/multiplicity", MULTIPLICITY, note="verbatim disclosure from the T8-2 verdict entry")

    gate_rec = runs[GATE_RUN]
    if gate_rec.get("kind") != "regime_map":
        raise ValueError(f"{GATE_RUN}: kind={gate_rec.get('kind')!r}, expected 'regime_map'")
    # Axis bucket labels + spec strings, copied from the T8-1 record.
    for axis in AXES:
        labels = resolve_pointer(gate_rec, jp("axes", axis, "labels"))
        w.put_descriptive(rm + jp("axes", axis, "labels"), labels, subtree=True, note="bucket display order")
        for lb in labels:
            w.copy_from_record(rm + jp("axes", axis, "spec", lb), GATE_RUN, jp("axes", axis, "spec", lb))
    # The churn gate — T8-1's missing receipt, now measured.
    g = rm + "/gate"
    w.copy_from_record(g + "/run_id", GATE_RUN, "/run_id")
    w.copy_from_record(g + "/statistic", GATE_RUN, "/results/gate/statistic")
    w.copy_from_record(g + "/wrong_below", GATE_RUN, "/results/gate/wrong_below")
    w.copy_from_record(g + "/supported_at_or_above", GATE_RUN, "/results/gate/supported_at_or_above")
    w.copy_from_record(g + "/measured_share", GATE_RUN, "/results/gate/measured_share")
    w.copy_from_record(g + "/band", GATE_RUN, "/results/gate/band")
    w.copy_from_record(g + "/verdict", GATE_RUN, "/results/gate/verdict")
    hd = rm + "/headline"
    w.copy_from_record(hd + "/n_users", GATE_RUN, "/results/headline/n_users")
    w.copy_from_record(hd + "/gt_interactions_total", GATE_RUN, "/results/headline/gt_interactions_total")
    w.copy_from_record(hd + "/catalog_size", GATE_RUN, "/results/headline/catalog_size")
    for bucket in ("zero", "low", "high"):
        for k in ("n", "share"):
            w.copy_from_record(
                hd + jp("gt_interactions_by_support", bucket, k),
                GATE_RUN,
                jp("results", "headline", "gt_interactions_by_support", bucket, k),
            )
            w.copy_from_record(
                hd + jp("catalog_items_by_support", bucket, k),
                GATE_RUN,
                jp("results", "headline", "catalog_items_by_support", bucket, k),
            )

    for m in MAPS:
        key, run_id, arm_key = m["key"], m["run_id"], m["arm_key"]
        rec = runs[run_id]
        if rec.get("kind") != "regime_map":
            raise ValueError(f"map {key!r}: run {run_id} has kind={rec.get('kind')!r}")
        if rec["delta"]["minuend"] != arm_key or rec["delta"]["subtrahend"] != "pop_t12m":
            raise ValueError(f"map {key!r}: unexpected delta {rec['delta']}")
        base = rm + jp("maps", key)
        w.put_descriptive(base + "/key", key, note="stable map key")
        w.put_descriptive(base + "/label", m["label"], note="display label")
        w.put_descriptive(base + "/arm_key", arm_key, note="challenger arm key inside this record's cells")
        w.put_descriptive(base + "/arm_label", m["arm_label"], note="challenger display label")
        w.copy_from_record(base + "/run_id", run_id, "/run_id")
        w.copy_from_record(base + "/minuend", run_id, "/delta/minuend")
        w.copy_from_record(base + "/subtrahend", run_id, "/delta/subtrahend")
        for a in ("pop_t12m", arm_key):
            w.copy_from_record(base + jp("source_run_ids", a), run_id, jp("source_run_ids", a))
        # Cross-machine recomposition disclosure (T8-2 maps only).
        if "regime_map_input_equivalence" in rec:
            eq = base + "/input_equivalence"
            w.copy_from_record(eq + "/exception_used", run_id, "/regime_map_input_equivalence/exception_used")
            w.copy_from_record(
                eq + "/exception_id", run_id, "/regime_map_input_equivalence/declaration/exception_id"
            )
            w.copy_from_record(eq + "/status", run_id, "/regime_map_input_equivalence/validation/status")
        for axis in AXES:
            cells = rec["results"]["cells"][axis]
            w.ensure_list(base + jp("cells", axis), len(cells))
            for i, cell in enumerate(cells):
                src = jp("results", "cells", axis, i)
                dst = base + jp("cells", axis, i)
                for field in ("segment", "bucket", "n_users", "gt_interactions", "user_share", "gt_share"):
                    w.copy_from_record(dst + jp(field), run_id, src + jp(field))
                for a in ("pop_t12m", arm_key):
                    for metric in METRICS:
                        for k in CI_KEYS:
                            w.copy_from_record(
                                dst + jp("arms", a, metric, k), run_id, src + jp("arms", a, metric, k)
                            )
                for metric in METRICS:
                    for k in DELTA_KEYS:
                        w.copy_from_record(dst + jp("delta", metric, k), run_id, src + jp("delta", metric, k))
    return w


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/demo_export.yaml")
    args = ap.parse_args(argv)

    cfg = load_export_config(args.config)
    runs = index_runs(cfg["runs_log"])
    writer = build(runs)
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
