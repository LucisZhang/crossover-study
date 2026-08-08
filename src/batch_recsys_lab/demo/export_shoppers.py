"""``demo/data/shoppers.json`` — exhibit 3's data (Phase 6, T28; plan §9.2).

Composes three already-produced inputs — the pre-declared 30-user selection
(:mod:`select_shoppers`), the read-only gold pull (:mod:`shopper_history_job`),
and the five one-shot TEST runs' per-user parquets — into the shape the
pick-a-shopper page reads. **No model is re-scored**: every ranking shown is the
first 10 entries of the ``top50`` column the harness already wrote, mapped
through the eval cache's ``item_ids.parquet`` (catalog order).

    uv run python -m batch_recsys_lab.demo.export_shoppers --config configs/shoppers_export.yaml

Trace classes (docs/demo-data-schemas.md § shoppers.json):

*traced to the per-user parquet a record names* — each shopper's ``segment``,
  each arm's ``ndcg@10`` / ``recall@20`` / ``hitrate@10``, and the
  ``catalog_index`` of every displayed recommendation (``user_idx=N/top50/i``);
*traced to the records themselves* — the five ``run_id``s, the pinned Iceberg
  snapshot ids, and the per-segment ``n_users`` / blend ``hitrate@10`` that
  disclose how much the curation over-samples hits;
*descriptive* — titles, prices, categories, timelines, ranks and the item ids
  those catalog indices resolve to. Presence-checked, never value-matched.

``n_train`` is descriptive by necessity, not by convenience: the trace manifest
has exactly three source kinds and none of them can express "``gold.user_stats``
at the pinned snapshot". Its traced proxy is ``segment`` (the parquet column
that *is* ``n_train`` bucketed), and the export asserts, per shopper, that the
gold ``user_stats.n_train``, the eval cache's ``n_train``, the length of the
exported TRAIN timeline, and ``segment_of(n_train)`` all agree — so a wrong
``n_train`` cannot survive the export.

Cold users (``n_train == 0``) get ``cold_collapse: true`` and **no** ``top10``
on ALS and item-kNN: those arms score every catalog item exactly 0 for a user
they never saw, so the stored top50 is the harness's index tie-break, not a
recommendation (``models/als.py``). Their zero metrics are still exported —
that collapse is the finding, not a gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from batch_recsys_lab.demo.export_core import (
    TraceManifest,
    TracedWriter,
    jp,
    resolve_pointer,
    sha256_file,
    source_per_user_artifact,
    write_document,
)
from batch_recsys_lab.demo.select_shoppers import Context, load_config
from batch_recsys_lab.eval.protocol import SEGMENT_LABELS, segment_of

FILE_NAME = "shoppers.json"
TOP_K = 10
PER_USER_METRICS = ("ndcg@10", "recall@20", "hitrate@10")


# --- inputs -------------------------------------------------------------------


class _Arm:
    """One model's per-user artifact, loaded once and indexed by ``user_idx``."""

    def __init__(self, key: str, run_id: str, rel_path: str, abs_path: Path) -> None:
        self.key = key
        self.run_id = run_id
        self.rel_path = rel_path
        self.sha256 = sha256_file(abs_path)
        table = pq.read_table(abs_path)
        self.rows = {int(u): i for i, u in enumerate(table.column("user_idx").to_pylist())}
        self.segment = table.column("segment").to_pylist()
        self.metrics = {
            name: table.column(name).to_pylist()
            for name in (*PER_USER_METRICS, "recall@10")
            if name in table.column_names
        }
        self.top50 = table.column("top50").to_pylist()

    def row(self, user_idx: int) -> int:
        if user_idx not in self.rows:
            raise KeyError(f"{self.key}: user_idx {user_idx} is not in {self.rel_path}")
        return self.rows[user_idx]

    def source(self, user_idx: int, column: str, index: int | None = None) -> dict:
        pointer = f"user_idx={user_idx}/{column}"
        if index is not None:
            pointer += f"/{index}"
        return source_per_user_artifact(
            parquet_path=self.rel_path,
            sha256=self.sha256,
            row_pointer=pointer,
            run_id=self.run_id,
        )


def _load_arms(ctx: Context) -> dict[str, _Arm]:
    return {
        key: _Arm(key, ctx.run_ids[key], ctx.artifact_rel(key), path)
        for key, path in ctx.artifacts.items()
    }


# --- consistency guards -------------------------------------------------------


def _check_shopper(
    *,
    shopper_id: str,
    member: dict,
    raw: dict,
    arms: dict[str, _Arm],
    gt_idx: list[int],
    item_ids: np.ndarray,
    cache_n_train: int,
    cold_models: set[str],
) -> None:
    """Everything the exporter refuses to publish a shopper without."""
    user_idx = member["user_idx"]
    n_train_gold = raw["user_stats"]["n_train"]
    history = raw["history"]
    if not (n_train_gold == cache_n_train == len(history) == member["n_train"]):
        raise AssertionError(
            f"{shopper_id}: n_train disagrees — gold.user_stats={n_train_gold}, "
            f"eval cache={cache_n_train}, exported timeline={len(history)}, "
            f"selection={member['n_train']}"
        )
    expected_segment = str(segment_of(np.array([n_train_gold]))[0])
    for key, arm in arms.items():
        got = arm.segment[arm.row(user_idx)]
        if got != expected_segment:
            raise AssertionError(
                f"{shopper_id}: {key} artifact says segment {got!r}, "
                f"segment_of(n_train={n_train_gold}) is {expected_segment!r}"
            )

    gt_asins = {str(item_ids[i]) for i in gt_idx}
    gold_test = {r["item_id"] for r in raw["test_rows"]}
    if gt_asins != gold_test:
        raise AssertionError(
            f"{shopper_id}: TEST ground truth from the eval cache {sorted(gt_asins)} != the "
            f"TEST-window rows in gold.interactions_5core {sorted(gold_test)}"
        )
    if not gt_asins:
        raise AssertionError(f"{shopper_id}: no TEST ground truth — cannot be an eval user")

    gt_set = set(gt_idx)
    for key, arm in arms.items():
        row = arm.row(user_idx)
        top10 = [int(v) for v in arm.top50[row][:TOP_K]]
        n_hits = sum(1 for i in top10 if i in gt_set)
        if key in cold_models and n_train_gold == 0:
            # suppressed arm: its metrics must be the documented all-zero collapse
            for name in PER_USER_METRICS:
                if arm.metrics[name][row] != 0.0:
                    raise AssertionError(
                        f"{shopper_id}: {key} is cold-collapsed but {name} is "
                        f"{arm.metrics[name][row]!r}, expected 0.0"
                    )
            continue
        hitrate = arm.metrics["hitrate@10"][row]
        if (n_hits > 0) != (hitrate == 1.0):
            raise AssertionError(
                f"{shopper_id}: {key} top-10 has {n_hits} ground-truth item(s) but the recorded "
                f"hitrate@10 is {hitrate!r}"
            )
        # recall@10 = |{g : rank(g) <= 10}| / |GT|  (eval/metrics.py, pinned)
        expected_recall = n_hits / len(gt_idx)
        if abs(arm.metrics["recall@10"][row] - expected_recall) > 1e-12:
            raise AssertionError(
                f"{shopper_id}: {key} top-10 has {n_hits}/{len(gt_idx)} ground-truth items "
                f"(recall@10 {expected_recall}) but the record says "
                f"{arm.metrics['recall@10'][row]}"
            )


# --- build --------------------------------------------------------------------


def build(ctx: Context, selection: dict, raw: dict) -> TracedWriter:
    cfg = ctx.cfg
    cold_models = set(cfg["cold_collapse_models"])
    arms = _load_arms(ctx)
    item_ids = ctx.item_ids()
    gt = ctx.test_gt()
    cache_n_train = ctx.n_train()
    items_meta: dict = raw["items"]
    headline = ctx.headline_run_id

    if raw["shopper_order"] != selection["shopper_order"]:
        raise ValueError("shoppers_raw.json and shopper_selection.json disagree on shopper_order")
    if raw["rule_id"] != selection["rule_id"]:
        raise ValueError("shoppers_raw.json was produced for a different curation rule")

    w = TracedWriter(FILE_NAME, ctx.runs, generated_by="batch_recsys_lab.demo.export_shoppers")
    w.put_descriptive("/segments", list(SEGMENT_LABELS), subtree=True, note="segment display order")
    w.put_descriptive("/models", ctx.model_keys, subtree=True, note="arm display order")
    w.put_descriptive(
        "/model_labels",
        {k: ctx.labels[k] for k in ctx.model_keys},
        subtree=True,
        note="display labels (configs/demo_export.yaml)",
    )
    w.put_descriptive("/seed", selection["seed"], note="curation seed, pre-declared in EXPERIMENT_LOG.md T28")
    w.put_descriptive(
        "/curation_rule",
        {
            "rule_id": selection["rule_id"],
            "declared_in": selection["rule_declared_in"],
            "per_segment": selection["per_segment"],
            "drawn_from_hit_stratum": selection["min_blend_hits"],
            "drawn_from_miss_stratum": selection["per_segment"] - selection["min_blend_hits"],
            "min_test_ground_truth_items": selection["min_test_gt"],
            "note": (
                "Stratified, not random: each segment shows 2 users blend hits in its top 10 and "
                "4 it misses, so both outcomes are visible. blend's real hit rate is 1.5-4% per "
                "segment (see /curation/<segment>/blend_hitrate@10) — the exhibit over-samples "
                "hits by roughly 50x and says so."
            ),
        },
        subtree=True,
        note="the pre-declared curation rule, restated for the viewer",
    )
    w.put_descriptive("/shopper_order", selection["shopper_order"], subtree=True, note="card order")
    w.put_descriptive(
        "/n_train_note",
        (
            "n_train is descriptive: no trace-manifest source kind can express "
            "'gold.user_stats at the pinned snapshot'. Its traced proxy is each shopper's "
            "`segment` (the per-user parquet column that is n_train bucketed); the export "
            "asserts gold.user_stats.n_train == eval-cache n_train == len(timeline) and "
            "segment_of(n_train) == segment before writing."
        ),
        note="trace-class disclosure",
    )

    for key in ctx.model_keys:
        w.copy_from_record(jp("run_ids", key), ctx.run_ids[key], "/run_id")
    w.copy_from_record("/headline_run_id", headline, "/run_id")
    w.copy_from_record("/iceberg_snapshots", headline, "/iceberg_snapshots")
    for table, sid in raw["iceberg_snapshots"].items():
        recorded = resolve_pointer(ctx.runs[headline], jp("iceberg_snapshots", table))
        if int(sid) != int(recorded):
            raise AssertionError(
                f"the Spark pull read {table} at snapshot {sid}, the headline record pins {recorded}"
            )

    # --- curation disclosure: the real per-segment hit rate, traced ----------
    blend_run = ctx.run_ids["blend"]
    for seg in SEGMENT_LABELS:
        base = jp("curation", seg)
        s = selection["by_segment"][seg]
        w.copy_from_record(base + "/eval_users", blend_run, jp("metrics", "per_segment", seg, "n_users"))
        for k in ("value", "ci_lo", "ci_hi"):
            w.copy_from_record(
                base + jp("blend_hitrate@10", k),
                blend_run,
                jp("metrics", "per_segment", seg, "hitrate@10", k),
            )
        w.put_descriptive(
            base + "/drawn",
            {
                "from_hit_stratum": s["blend_hits"],
                "from_miss_stratum": s["blend_misses"],
                "candidate_pool": s["pool_size"],
                "hit_stratum_size": s["hit_stratum_size"],
                "miss_stratum_size": s["miss_stratum_size"],
                "pool_fallback_to_all_users": s["pool_fallback_to_all_users"],
                "attempts": s["attempts"],
            },
            subtree=True,
            note="how the six cards in this segment were drawn",
        )

    # --- shoppers ------------------------------------------------------------
    members = {m["shopper_id"]: m for seg in selection["by_segment"].values() for m in seg["members"]}
    for shopper_id in selection["shopper_order"]:
        member = members[shopper_id]
        user_idx = member["user_idx"]
        raw_shopper = raw["shoppers"][shopper_id]
        gt_idx = gt.get(user_idx, [])
        _check_shopper(
            shopper_id=shopper_id,
            member=member,
            raw=raw_shopper,
            arms=arms,
            gt_idx=gt_idx,
            item_ids=item_ids,
            cache_n_train=int(cache_n_train[user_idx]),
            cold_models=cold_models,
        )
        gt_set = set(gt_idx)
        base = jp("shoppers", shopper_id)
        n_train = raw_shopper["user_stats"]["n_train"]

        w.put_descriptive(base + "/shopper_id", shopper_id, note="HMAC-SHA256(local salt, user_id)[:12]")
        w.put(
            base + "/segment",
            arms["blend"].segment[arms["blend"].row(user_idx)],
            arms["blend"].source(user_idx, "segment"),
        )
        w.put_descriptive(base + "/n_train", n_train, note="see /n_train_note")
        w.put_descriptive(
            base + "/history",
            [_item_entry(r, items_meta) for r in raw_shopper["history"]],
            subtree=True,
            note="TRAIN timeline from gold.interactions_5core at the pinned snapshot (descriptive)",
        )
        w.put_descriptive(
            base + "/test_purchases",
            [_item_entry(r, items_meta) for r in raw_shopper["test_rows"]],
            subtree=True,
            note="TEST-window ground truth, identical to the eval cache's GT set (descriptive)",
        )

        for key in ctx.model_keys:
            arm = arms[key]
            row = arm.row(user_idx)
            dst = base + jp("recommendations", key)
            cold = key in cold_models and n_train == 0
            w.copy_from_record(dst + "/run_id", arm.run_id, "/run_id")
            w.put_descriptive(
                dst + "/cold_collapse",
                cold,
                note=(
                    "no TRAIN history: this arm scores every catalog item 0, so its stored top50 "
                    "is an index tie-break, not a recommendation (models/als.py)"
                ),
            )
            for name in PER_USER_METRICS:
                w.put(dst + jp(name), arm.metrics[name][row], arm.source(user_idx, name))
            if cold:
                continue
            top10 = [int(v) for v in arm.top50[row][:TOP_K]]
            w.put_descriptive(
                dst + "/top10",
                [
                    _rec_entry(rank, idx, item_ids, items_meta, idx in gt_set)
                    for rank, idx in enumerate(top10, start=1)
                ],
                subtree=True,
                note="titles/prices/ranks are descriptive; each catalog_index below is traced",
            )
            for i, idx in enumerate(top10):
                w.put(dst + jp("top10", i, "catalog_index"), idx, arm.source(user_idx, "top50", i))
    return w


def _item_entry(row: dict, items_meta: dict) -> dict:
    meta = items_meta.get(row["item_id"], {})
    return {
        "item_id": row["item_id"],
        "title": meta.get("title"),
        "brand": meta.get("brand_norm"),
        "price_usd": meta.get("price_usd"),
        "main_category": meta.get("main_category"),
        "ts": row["ts"],
        "rating": row["rating"],
    }


def _rec_entry(rank: int, idx: int, item_ids: np.ndarray, items_meta: dict, hit: bool) -> dict:
    item_id = str(item_ids[idx])
    meta = items_meta.get(item_id, {})
    return {
        "rank": rank,
        "item_id": item_id,
        "title": meta.get("title"),
        "brand": meta.get("brand_norm"),
        "price_usd": meta.get("price_usd"),
        "main_category": meta.get("main_category"),
        "hit": hit,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/shoppers_export.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ctx = Context(cfg)
    work = ctx.work_dir
    selection = json.loads((work / "shopper_selection.json").read_text())
    raw_path = work / "shoppers_raw.json"
    if not raw_path.exists():
        raise SystemExit(
            f"{raw_path} is missing — run the read-only Spark pull first:\n"
            "  JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home "
            'PATH="$JAVA_HOME/bin:$PATH" SPARK_LOCAL_IP=127.0.0.1 uv run python -m '
            "batch_recsys_lab.demo.shopper_history_job --config configs/shoppers_export.yaml"
        )
    raw = json.loads(raw_path.read_text())

    writer = build(ctx, selection, raw)
    out_dir = ctx._p(cfg["out_dir"])
    manifest = TraceManifest(ctx._p(cfg["manifest"]), ctx._p(ctx.demo_cfg["runs_log"]))
    out = write_document(writer, out_dir, manifest)
    manifest.drop_missing_files(out_dir)
    manifest.write()

    doc = writer.document
    per_segment: dict[str, int] = {}
    for s in doc["shoppers"].values():
        per_segment[s["segment"]] = per_segment.get(s["segment"], 0) + 1
    print(
        f"wrote {out} ({len(doc['shoppers'])} shoppers, "
        f"{len(writer.entries)} traced leaves, {len(writer.descriptive)} descriptive)"
    )
    print("  segment composition: " + ", ".join(f"{k}={per_segment[k]}" for k in SEGMENT_LABELS))
    cold = [
        (sid, key)
        for sid, s in doc["shoppers"].items()
        for key, rec in s["recommendations"].items()
        if rec["cold_collapse"]
    ]
    print(f"  cold_collapse arms (no top10): {len(cold)} across {len({c[0] for c in cold})} shoppers")


if __name__ == "__main__":
    main()
