"""``demo/data/contrast.json`` — exhibit 1c, the Phase 9 regime contrast
(Amazon Electronics vs ML-32M; plan §8c).

    uv run python -m batch_recsys_lab.demo.export_contrast --config configs/demo_export.yaml

then re-run ``export_receipts`` so the receipts closure picks up the run_ids
this document cites (``make demo-export`` already orders it that way).

Two things about this exhibit are unlike every other one, and both are decided
here rather than papered over.

1. Anchoring ``results/confirmatory_ml32m_test.json``
----------------------------------------------------
The T9-3c confirmatory analysis is a COMMITTED DERIVED artifact that, by
design, ``appends_to_runs_jsonl: false`` — it regroups per-user vectors the
one-shot TEST evals already committed and appends nothing itself. So no
append-only record can carry its SHA-256, and the strong ``results_artifact``
source kind (``TracedWriter.register_artifact``: "the record signed this
digest") is simply unavailable. The options were:

* **rejected** — force a record. Appending a record whose only purpose is to
  attest a hash would rewrite the artifact's own "appends nothing" claim and
  put a demo-driven entry in the scientific log. Provenance must not be
  manufactured to satisfy a checker.
* **rejected** — quote its numbers as ``descriptive``. That is the one thing
  the export design forbids: metric leaves would become untraced display copy.
* **rejected** — re-derive the numbers in the exporter from the per-user
  parquets. That re-runs a preregistered bootstrap in a demo build step; the
  confirmatory result would then have two authorities.
* **chosen** — a third source kind, ``derived_artifact``, that anchors what
  can actually be anchored:
    - the manifest pins the artifact's SHA-256, so post-export drift fails;
    - the artifact must self-declare ``derived: true`` and
      ``appends_to_runs_jsonl: false``, so this weaker kind can never be used
      where the record-anchored one applies;
    - its ``git_sha`` / ``config_hash`` are pinned alongside;
    - **input anchoring**: the artifact's ``families/P/artifact_paths`` must
      equal the ``per_user_artifact`` paths the M*/P* eval RECORDS published.
      That is the real chain — the derived file is pinned to inputs the
      append-only log attests to, and the two run_ids it consumed are pulled
      into the receipts closure like any other citation.

The honest limit, stated so the site can state it: this proves the artifact is
unchanged since export and that it consumed the parquets the log names. It
does not prove the artifact's arithmetic — that is what ``tests/
test_confirmatory_ml32m.py`` and the preregistration are for.

2. The collided run_id ``20260820T221701Z-20d8ff9``
--------------------------------------------------
Two records in the append-only log carry that id (an ALS-decay seed-1 run and
the item-kNN-t12m run — minted in the same second on the same commit). The log
is append-only, so the fix is in RESOLUTION, not in the log: every citation of
it here carries ``record_selector={"/config_path": …}``, and both the writer
and the independent verifier refuse to resolve a collided id without one.
M* is the item-kNN-t12m record, which is also what the confirmatory artifact's
``artifact_paths/m_star`` (``…_item_knn.parquet``) independently pins.

Schema: this module is the definition — docs/demo-data-schemas.md has no
§ contrast.json section yet (owner call; adding one is a docs edit outside
this task).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from batch_recsys_lab.demo.export_core import (
    TraceManifest,
    TracedWriter,
    index_runs_multi,
    jp,
    load_export_config,
    resolve_pointer,
    write_document,
)

FILE_NAME = "contrast.json"

# --- churn gate: the measured statistic on each dataset -----------------------
# T8-1's measured churn on Amazon Electronics (the record the EXPERIMENT_LOG
# cites as primary; a field-identical duplicate exists and is not used) and
# T9-3a's re-run of the same methodology on ML-32M, which also re-states the
# Amazon reference — cross-checked below so the two records cannot drift apart
# behind the exhibit's back.
AMAZON_CHURN_RUN = "20260817T095926Z-633d454"  # kind="regime_map"
ML32M_CHURN_RUN = "20260820T134403Z-e2263d2"  # kind="churn_contrast"

# --- the collided run_id, and how this exhibit resolves it --------------------
ITEMKNN_T12M_RUN = "20260820T221701Z-20d8ff9"
ITEMKNN_T12M_SELECTOR = {"/config_path": "configs/eval_itemknn_t12m_ml32m_test.yaml"}
POP_T12M_RUN = "20260820T221055Z-20d8ff9"

# --- ML-32M one-shot TEST arms (global NDCG@10) -------------------------------
# ALS is the CANONICAL PRIMARY-SEED record (seeds.model = 20260805, the sole
# per-user artifact the §6 seed discipline admits for paired tests); the two
# sibling seeds are stability evidence and are not quoted here. The exported
# ``model_seed`` leaf makes that visible on the page as well as in the receipt.
ARMS = [
    {"key": "pop_t12m", "run_id": POP_T12M_RUN, "selector": None},
    {"key": "itemknn_t12m", "run_id": ITEMKNN_T12M_RUN, "selector": ITEMKNN_T12M_SELECTOR},
    {"key": "als", "run_id": "20260820T222202Z-20d8ff9", "selector": None},
    {"key": "blend_alpha0_1", "run_id": "20260820T221933Z-20d8ff9", "selector": None},
]

# --- T9-3c confirmatory artifact ---------------------------------------------
CONFIRMATORY = "results/confirmatory_ml32m_test.json"
CONF_KEY = "confirmatory_ml32m"
PRIMARY_FAMILY = "/families/P/metrics/ndcg@10/rows"
WINNING_BUCKETS = ["20-49", "50-99", "100+"]
D4_BUCKET = "0"
BUCKET_FIELDS = (
    "label",
    "n_users",
    "delta",
    "ci_lo",
    "ci_hi",
    "p_value_uncorrected",
    "q_value",
    "bh_significant",
    "user_share",
)
VERDICT_FIELDS = (
    "verdict",
    "verdict_code",
    "headline",
    "n_star",
    "crossover_bucket",
    "d4_flag",
    "d4_token",
    "bh_significant_negative_buckets",
)
GUARD_FIELDS = ("definition", "claim_labels", "agreeing_labels", "metric_robust")
SOURCE_RUN_ID_KEYS = (
    "m_star",
    "p_star",
    "random",
    "pop_alltime",
    "itemknn",
    "als",
    "als_decay",
    "content",
    "blend",
    "hybrid",
)

# The derived artifact's inputs, tied to what the append-only records published.
INPUT_ANCHORS = [
    {
        "run_id": ITEMKNN_T12M_RUN,
        "record_selector": ITEMKNN_T12M_SELECTOR,
        "artifact_pointer": "/families/P/artifact_paths/m_star",
        "record_pointer": "/per_user_artifact",
    },
    {
        "run_id": POP_T12M_RUN,
        "artifact_pointer": "/families/P/artifact_paths/p_star",
        "record_pointer": "/per_user_artifact",
    },
]

NOTES = (
    "Exporter-generated (make demo-export). Global arm NDCG@10 and both churn shares are copied "
    "leaf-by-leaf from results/runs.jsonl records (kind=eval / kind=regime_map / "
    "kind=churn_contrast); the depth-bucket deltas, CIs, p/q-values and the D1/D4/recall-guard "
    "outcomes are copied from results/confirmatory_ml32m_test.json, a committed derived T9-3c "
    "analysis of those same one-shot TEST records that appends nothing to the log and is anchored "
    "by SHA-256 plus its per-user-parquet inputs. demo/data/trace_manifest.json re-resolves every "
    "leaf below; `make demo-verify` is the check."
)
AMAZON_SOURCE_NOTE = (
    "results/runs.jsonl (kind=regime_map, T8-1) /results/gate — the same measurement demo/data/"
    "phase8.json shows at /regime_map/gate/measured_share"
)
ML32M_SOURCE_NOTE = "results/runs.jsonl (kind=churn_contrast, T9-3a) /results/gate"

FIGURES = {
    "regime_map": "img/crossover_ml32m_test.svg",
    "deep_buckets": "img/crossover_ml32m_deep_test.svg",
}

CAVEATS = [
    {
        "id": "recall_guard_ns",
        "text": (
            "Top-of-list-only: none of the three winning buckets has the Recall@20 guard "
            "BH-significant in the same direction (§5g); metric_robust is false."
        ),
    },
    {
        "id": "cold_user_loss",
        "text": (
            "Depth-0 (cold) users lose −38.44% NDCG@10 vs pop-t12m, BH-significant "
            "(D4_SIGNIFICANT_NEGATIVES)."
        ),
    },
    {
        "id": "regime_contrast_not_causal",
        "text": (
            "Regime contrast, not causal proof: explicit-rating movie data changes several "
            "variables at once (§8b T8-4)."
        ),
    },
    {
        "id": "ml32m_timestamp_caveat",
        "text": (
            "MovieLens timestamps are rating-ENTRY times on a backfilled catalog (cite Sun et al., "
            "arXiv:2307.09985), not purchase/consumption times."
        ),
    },
]


def _row_index(doc: dict, label: str) -> int:
    rows = resolve_pointer(doc, PRIMARY_FAMILY)
    hits = [i for i, r in enumerate(rows) if r["label"] == label]
    if len(hits) != 1:
        raise ValueError(f"{CONFIRMATORY}: bucket {label!r} appears {len(hits)} times in {PRIMARY_FAMILY}")
    return hits[0]


def _check_churn_records(w: TracedWriter) -> None:
    """The ML-32M churn record re-states the Amazon reference; if its copy ever
    disagrees with the T8-1 record this exhibit quotes, stop rather than pick."""
    amazon = w.record(AMAZON_CHURN_RUN)
    ml32m = w.record(ML32M_CHURN_RUN)
    if amazon.get("kind") != "regime_map":
        raise ValueError(f"{AMAZON_CHURN_RUN}: kind={amazon.get('kind')!r}, expected 'regime_map'")
    if ml32m.get("kind") != "churn_contrast":
        raise ValueError(f"{ML32M_CHURN_RUN}: kind={ml32m.get('kind')!r}, expected 'churn_contrast'")
    ref = resolve_pointer(ml32m, "/results/contrast/reference")
    gate = resolve_pointer(amazon, "/results/gate")
    for field in ("value", "band", "verdict"):
        theirs = ref[field if field != "value" else "value"]
        ours = gate["measured_share" if field == "value" else field]
        if theirs != ours:
            raise ValueError(
                f"churn contrast disagreement on {field!r}: {ML32M_CHURN_RUN} restates "
                f"{theirs!r} but {AMAZON_CHURN_RUN} records {ours!r}"
            )
    if ref["run_id"] != AMAZON_CHURN_RUN:
        raise ValueError(
            f"{ML32M_CHURN_RUN} references Amazon run {ref['run_id']!r}, exhibit pins {AMAZON_CHURN_RUN!r}"
        )
    if resolve_pointer(ml32m, "/results/contrast/statistic") != gate["statistic"]:
        raise ValueError("churn contrast disagreement on the statistic definition")


def build(runs_multi: dict[str, list[dict]]) -> TracedWriter:
    w = TracedWriter(
        FILE_NAME,
        {},
        generated_by="batch_recsys_lab.demo.export_contrast",
        runs_multi=runs_multi,
    )
    w.put_descriptive("/notes", NOTES, note="what this document is and how it is anchored")

    # ---------------- churn gate contrast ------------------------------------
    _check_churn_records(w)
    cc = "/churn_contrast"
    w.copy_from_record(cc + "/statistic", ML32M_CHURN_RUN, "/results/contrast/statistic")
    for key, run_id, note in (
        ("amazon_electronics", AMAZON_CHURN_RUN, AMAZON_SOURCE_NOTE),
        ("ml32m", ML32M_CHURN_RUN, ML32M_SOURCE_NOTE),
    ):
        base = cc + jp(key)
        w.copy_from_record(base + "/value", run_id, "/results/gate/measured_share")
        w.copy_from_record(base + "/run_id", run_id, "/run_id")
        w.put_descriptive(base + "/source", note, note="where this number lives in the log")
        w.copy_from_record(base + "/band", run_id, "/results/gate/band")
        w.copy_from_record(base + "/verdict", run_id, "/results/gate/verdict")
    w.copy_from_record(cc + "/difference", ML32M_CHURN_RUN, "/results/contrast/difference_vs_reference")

    # ---------------- ML-32M global arms -------------------------------------
    for arm in ARMS:
        run_id, sel = arm["run_id"], arm["selector"]
        rec = w.record(run_id, sel)
        if rec.get("kind") != "eval" or rec.get("protocol", {}).get("eval_split") != "test":
            raise ValueError(
                f"arm {arm['key']!r}: run {run_id} is kind={rec.get('kind')!r} / "
                f"split={rec.get('protocol', {}).get('eval_split')!r}, expected eval/test"
            )
        base = jp("ml32m_global_ndcg10", arm["key"])
        w.copy_from_record(base + "/value", run_id, "/metrics/global/ndcg@10/value", selector=sel)
        w.copy_from_record(base + "/run_id", run_id, "/run_id", selector=sel)
        w.copy_from_record(base + "/model_name", run_id, "/model/name", selector=sel)
        # null for the deterministic arms; 20260805 for ALS — the seed label the
        # §6 discipline makes load-bearing.
        w.copy_from_record(base + "/model_seed", run_id, "/seeds/model", selector=sel)

    # ---------------- T9-3c confirmatory artifact ----------------------------
    # Repo-relative on purpose: the manifest records this path and the
    # independent verifier resolves it against the repo root.
    art = w.register_derived_artifact(CONF_KEY, Path(CONFIRMATORY), input_anchors=INPUT_ANCHORS)
    doc = art["doc"]
    for field in VERDICT_FIELDS:
        w.copy_from_artifact(jp("verdict", field), CONF_KEY, jp("verdict", field))

    w.ensure_list("/winning_deep_buckets", len(WINNING_BUCKETS))
    for i, label in enumerate(WINNING_BUCKETS):
        src = PRIMARY_FAMILY + jp(_row_index(doc, label))
        for field in BUCKET_FIELDS:
            w.copy_from_artifact(jp("winning_deep_buckets", i, field), CONF_KEY, src + jp(field))
    d4_src = PRIMARY_FAMILY + jp(_row_index(doc, D4_BUCKET))
    for field in BUCKET_FIELDS:
        w.copy_from_artifact(jp("d4_depth0", field), CONF_KEY, d4_src + jp(field))

    for field in GUARD_FIELDS:
        w.copy_from_artifact(jp("recall_guard", field), CONF_KEY, jp("verdict", "metric_robustness", field))

    # The ten one-shot TEST runs the analysis consumed. Traced against the
    # RECORDS (not the artifact's copy): the entry proves the id resolves to a
    # real record, and pulls a receipts card for it into the closure.
    for key in SOURCE_RUN_ID_KEYS:
        run_id = resolve_pointer(doc, jp("source_run_ids", key))
        sel = ITEMKNN_T12M_SELECTOR if run_id == ITEMKNN_T12M_RUN else None
        w.copy_from_record(jp("source_run_ids", key), run_id, "/run_id", selector=sel)

    cs = "/confirmatory_source"
    w.put_descriptive(cs + "/path", CONFIRMATORY, note="repo-relative path of the derived artifact")
    w.copy_from_artifact(cs + "/generated_ts", CONF_KEY, "/generated_ts")
    w.copy_from_artifact(cs + "/git_sha", CONF_KEY, "/git_sha")
    w.copy_from_artifact(cs + "/config_hash", CONF_KEY, "/config_hash")

    w.put_descriptive("/figures", FIGURES, subtree=True, note="committed SVGs copied under demo/img/")
    w.put_descriptive("/caveats", CAVEATS, subtree=True, note="display copy for the caveat block")
    return w


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/demo_export.yaml")
    args = ap.parse_args(argv)

    cfg = load_export_config(args.config)
    runs_multi = index_runs_multi(cfg["runs_log"])
    writer = build(runs_multi)
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
