# `demo/data/` JSON schemas (Phase 6)

Frozen by **T26** so the site JS (T31+) and the remaining exporters (T27–T30,
T35) can be built in parallel against a fixed contract.

Status of each schema:

| File | Status | Written by | Task |
|---|---|---|---|
| `trace_manifest.json` | **FROZEN** (shipped) | `demo/export_core.py` | T26 |
| `crossover.json` | **FROZEN** (shipped) | `demo/export_crossover.py` | T26 |
| `receipts.json` | **FROZEN** (shipped) | `demo/export_receipts.py` | T26 |
| `policy_grid.json` | **SHIPPED** | `demo/export_policy_grid.py` | T27 |
| `shoppers.json` | **SHIPPED** | `demo/export_shoppers.py` | T28 |
| `dq.json` | **SHIPPED** | `demo/export_dq.py` | T29 |
| `lineage.json` | AGREED (not yet written) | `demo/export_lineage.py` | T30 |
| `timetravel.json` | AGREED (not yet written) | `demo/export_lineage.py` | T30 |
| `search/*` | AGREED (not yet written, not committed) | `demo/export_search.py` | T35 |

AGREED means: the shape below is the contract T31's JS may assume. The owning
task may add fields; it may not rename or remove one without updating this file
and the JS in the same commit.

---

## Rules every document obeys

1. **Projection only.** A number may appear here only if it is byte-identical
   to a value reachable from (a) a `results/runs.jsonl` record, (b) a results
   artifact whose SHA-256 a record carries, or (c) a per-user parquet a record
   names in `per_user_artifact`. Anything else is appended to the log as a
   derived record *first*, then exported.
2. **Full precision.** Values are written exactly as recorded — no rounding,
   ever. Display rounding belongs in `demo/js/fmt.js`.
3. **Every numeric leaf is traced.** Written through `TracedWriter`, which
   emits a `trace_manifest.json` entry for it. The only exceptions are leaves
   explicitly declared *descriptive* (labels, ordering, `schema_version`),
   which the manifest records in its `descriptive` list.
4. **No timestamps in documents.** Re-exporting unchanged evidence is
   byte-stable; only `trace_manifest.json` carries `generated_at`. This lets
   the manifest pin each document's SHA-256.
5. **Segment keys** are always the frozen five, in this order:
   `"0"`, `"1-4"`, `"5-9"`, `"10-19"`, `"20+"`.
6. **Metric keys** are the record's own keys: `ndcg@10`, `recall@20`, …
   (`@` is not escaped; JSON pointers only escape `~` and `/`).

---

## `trace_manifest.json` — FROZEN

```jsonc
{
  "schema_version": 1,
  "generated_at": "2026-08-08T10:53:11.123456+00:00",  // manifest only
  "runs_jsonl_sha256": "sha256:…",   // staleness guard: must equal the log on disk
  "runs_log": "results/runs.jsonl",
  "files": [ { "name": "crossover.json", "sha256": "sha256:…" }, … ],
  "entries": [ … ],       // one per traced leaf, across ALL documents
  "descriptive": [ … ]    // declared non-evidence paths
}
```

### `entries[]`

```jsonc
{
  "file": "crossover.json",
  "pointer": "/models/blend/global/ndcg@10/value",  // RFC 6901, into that file
  "value": 0.005726134272789762,                    // exact copy of the leaf
  "source": { … one of the three kinds below … }
}
```

Source kinds:

```jsonc
// (a) straight out of an append-only record
{ "kind": "runs_record",
  "run_id": "20260807T055333Z-c320c79",
  "source_pointer": "/metrics/global/ndcg@10/value" }

// (b) out of a results artifact whose sha256 a record carries (transitive trace)
{ "kind": "results_artifact",
  "source_file": "results/lineage.json",
  "sha256": "sha256:…",              // must equal both the file and the record
  "pointer": "/stages/3/rows_out",   // pointer INTO the artifact
  "run_id": "20260807T160910Z-739833b",
  "anchor_pointer": "/artifact_sha256" }

// (c) out of the per-user parquet a record names
{ "kind": "per_user_artifact",
  "parquet_path": "data/eval/per_user/<run_id>_<model>.parquet",
  "sha256": "sha256:…",              // computed at export, re-checked in full mode
  "row_pointer": "user_index=17/top50/0",   // <key_col>=<val>[/<col>[/<idx>]]
  "run_id": "20260807T055333Z-c320c79" }    // record whose per_user_artifact == path
```

### `descriptive[]`

```jsonc
{ "file": "crossover.json", "pointer": "/segments", "subtree": true,
  "note": "segment display order" }
```

`subtree: true` declares the whole subtree non-evidence — used only for bulk
descriptive payloads (item titles, timelines). The verifier reports how many
numeric leaves each subtree declaration absorbs.

### Verification

`make demo-verify` (`verify_traceability.py`, shares no code with the writer)
re-resolves every entry and fails on: stale log hash, file-set drift, document
hash drift, an uncovered numeric leaf, an orphan entry, a document/manifest
mismatch, a source mismatch (exact equality — same type, no epsilon), an
artifact whose hash no longer matches its anchoring record, or a cited `run_id`
with no `receipts.json` card. `--mode=record` skips only the per-user parquet
reads (CI has no `data/`).

---

## `crossover.json` — FROZEN (exhibit 1)

```jsonc
{
  "schema_version": 1,
  "generated_by": "batch_recsys_lab.demo.export_crossover",
  "split": "test",
  "segments": ["0","1-4","5-9","10-19","20+"],
  "metrics": ["ndcg@10","recall@20"],
  "model_order": ["blend","pop_t12m","als","item_knn","content","hybrid"],  // palette slot order
  "title": "…", "subtitle": "…", "xlabel": "…",     // descriptive chart copy
  "headline_run_id": "20260807T055333Z-c320c79",
  "paired_delta_order": ["blend_vs_pop_t12m", …],

  "models": {
    "blend": {
      "key": "blend",
      "label": "blend α=0.3 (content+pop)",
      "highlight": true,      // emphasised line
      "plot": true,           // false ⇒ annotation only (hybrid)
      "identical_to": "blend",// present only when the run duplicates another (hybrid)
      "run_id": "…", "model_name": "content_pop_blend", "git_sha": "…",
      "n_users": 228153, "catalog_size": 368228,
      "global":  { "<metric>": { "value": …, "ci_lo": …, "ci_hi": … } },
      "segments": {
        "0": { "n_users": 12866,
               "<metric>": { "value": …, "ci_lo": …, "ci_hi": … } }
      }
    }
  },

  "paired_deltas": {
    "blend_vs_pop_t12m": {
      "key": "blend_vs_pop_t12m",
      "label": "blend α=0.3 − pop-t12m",
      "run_id": "20260807T085819Z-c320c79",         // kind="paired_delta" record
      "a": { "run_id": …, "model": …, "artifact": … },   // verbatim from the record
      "b": { … },
      "n_common_users": 228153,
      "global":   { "<metric>": { "delta": …, "ci_lo": …, "ci_hi": …, "excludes_zero": true } },
      "segments": { "<segment>": { "<metric>": { "delta": …, "ci_lo": …, "ci_hi": …, "excludes_zero": … } } }
    }
  }
}
```

Notes for the JS: `hybrid` is `plot: false` — render it as the "n*=∞ ≡ blend"
annotation, with `paired_deltas.hybrid_vs_blend` (exactly zero in every cell) as
the receipt. Line colours come from `model_order` × the palette in
`eval/crossover_chart.py::SLOTS`, assigned by index, never re-ranked.

---

## `receipts.json` — FROZEN (receipts drawer)

```jsonc
{
  "schema_version": 1,
  "generated_by": "batch_recsys_lab.demo.export_receipts",
  "headline_run_id": "20260807T055333Z-c320c79",
  "run_order": ["20260805T172047Z-035042b", …],   // sorted; card display order
  "note": "…",
  "runs": {
    "<run_id>": {
      "run_id": …, "kind": "eval", "run_ts": …, "git_sha": …, "git_dirty": false,
      "config_path": …, "config_hash": "sha256:…",
      "dataset_manifest_hash": "sha256:…",
      "splits": { "version": 1, "frozen_at": "2026-08-05", "file_hash": "sha256:…" },
      "iceberg_snapshots": { "local.gold.interactions_5core": 8184397443787800955, … },
      "seeds": { "bootstrap": 20260805, "model": null },
      "model": { "name": …, "params": { … verbatim … } },
      "wall_clock_s": 1235.911,
      "hardware": "arm64 · Darwin",

      // headline run only:
      "reproduce": [ { "run_id": "20260807T153823Z-9a9fb4c", "verdict": "byte_exact" }, … ],
      "repro_command": "make reproduce-headline",

      // non-eval kinds (paired_delta, reproduce, …) carry a subset:
      "fields_absent_in_record": ["config_path", …]
    }
  }
}
```

`runs` covers the **closure** of run_ids the trace manifest depends on: every
cited run, plus the runs a `paired_delta` compares, plus the `reproduce` runs
attached to the headline. Every field is a verbatim copy of the record — the
drawer displays, it never computes. Any number the UI shows must key into
`runs[<run_id>]` via the `data-run-id` on the element, which comes from the
manifest entry for that leaf.

---

## `policy_grid.json` — **SHIPPED** (T27, exhibit 2: n\* slider)

Backed by one appended `kind="policy_grid"` record (derived, TEST
recomposition — no re-scoring, no refitting). VAL context comes from
`results/policy_select_val.json`, which is only usable once that record carries
its SHA-256 (see `configs/demo_export.yaml → artifacts.policy_select_val`).

```jsonc
{
  "schema_version": 1,
  "record_run_id": "<kind=policy_grid run_id>",
  "split": "test",
  "segments": ["0","1-4","5-9","10-19","20+"],
  "metrics": ["ndcg@10","recall@20"],
  "n_star_grid": [0, 1, 5, 10, 20, null],        // null = ∞; slider snaps to these only
  "n_star_labels": ["0","1","5","10","20","inf"],
  "variants": {
    "A": { "label": "blend → ALS", "low": "blend", "high": "als" },
    "B": { "label": "blend → pop-t12m", "low": "blend", "high": "pop_t12m" }
  },
  "shipped": { "variant": "B", "n_star_label": "inf" },   // highlighted cell
  "cells": {
    "B": {
      "inf": {
        "n_star": null, "n_star_label": "inf",
        "low_share": 1.0,
        "global":   { "<metric>": { "value": …, "ci_lo": …, "ci_hi": … } },
        "segments": { "<segment>": { "n_users": …, "<metric>": { "value": …, "ci_lo": …, "ci_hi": … } } },
        // OPTIONAL: present ONLY on "0" and "inf" cells (both variants), absent on 1/5/10/20 —
        // A/0 anchors to the ALS record, B/0 to pop-t12m, both "inf" cells to blend;
        // consumers must guard for absence on intermediate positions
        "identity": { "equals_run_id": "20260807T055333Z-c320c79", "asserted": true }
      }
    }
  },
  "val_grid": [ { "variant": "A", "n_star_label": "1", "low": …, "high": …, "objective": …,
                  "segment_means": { "<segment>": … } } ],   // from policy_select_val.json, all 10 cells
  "val_winner": { "variant": "B", "n_star_label": "inf", "objective": …, "segment_means": { … } },
  "objective": "segment_weighted_ndcg10_unweighted_mean",   // VAL selection objective name (descriptive)
  "n_star_selected_on_val": null,      // null ⇒ no finite n* beat blend-everywhere
  "notes": { "variant_b_cold_collapse": "…B's n*=0 and n*=1 cells are bit-identical…" },  // descriptive
  "seeds": { "bootstrap": 20260805 }, "n_resamples": 1000
}
```

## `shoppers.json` — SHIPPED (T28, exhibit 3: pick-a-shopper)

Written by `demo/export_shoppers.py` from the pre-declared 30-user selection
(`demo/select_shoppers.py`) and the read-only gold pull
(`demo/shopper_history_job.py`). Metrics and rankings come from the per-user
parquets (`per_user_artifact` source kind); titles, prices and timelines are
**descriptive** — presence is checked, values are not matched against the log.

```jsonc
{
  "schema_version": 1,
  "generated_by": "batch_recsys_lab.demo.export_shoppers",
  "segments": ["0","1-4","5-9","10-19","20+"],
  "models": ["blend","pop_t12m","als","item_knn","content"],
  "model_labels": { "blend": "blend α=0.3 (content+pop)", … },  // descriptive
  "seed": 20260805,                                  // curation seed (descriptive)
  "curation_rule": { "rule_id": "phase6-t28-v2-stratified", "declared_in": "…",
                     "per_segment": 6, "drawn_from_hit_stratum": 2,
                     "drawn_from_miss_stratum": 4,
                     "min_test_ground_truth_items": 2, "note": "…" },
  "shopper_order": ["a1b2c3d4e5f6", …],             // 30 ids, 6 per segment
  "n_train_note": "…",                               // trace-class disclosure
  "run_ids": { "blend": "…", … },                   // traced; the runs the rankings come from
  "headline_run_id": "20260807T055333Z-c320c79",
  "iceberg_snapshots": { "local.gold.interactions_5core": 8184397443787800955, … },  // traced

  // curation disclosure: the exhibit over-samples blend hits ~50x, and says so
  // with the recorded per-segment hit rate next to the draw counts.
  "curation": {
    "5-9": { "eval_users": 80202,                                   // traced (record)
             "blend_hitrate@10": { "value": …, "ci_lo": …, "ci_hi": … },  // traced (record)
             "drawn": { "from_hit_stratum": 2, "from_miss_stratum": 4,
                        "candidate_pool": …, "hit_stratum_size": …,
                        "miss_stratum_size": …, "pool_fallback_to_all_users": false,
                        "attempts": 1 } }                            // descriptive
  },

  "shoppers": {
    "a1b2c3d4e5f6": {
      "shopper_id": "a1b2c3d4e5f6",                 // hmac_sha256(salt,user_id)[:12]
      "segment": "5-9",                             // TRACED (per-user parquet column)
      "n_train": 7,                                 // descriptive — see below
      "history": [ { "item_id": …, "title": …, "brand": …, "price_usd": …,
                     "main_category": …, "ts": …, "rating": … } ],   // descriptive subtree
      "test_purchases": [ { …same shape… } ],                        // descriptive subtree
      "recommendations": {
        "blend": {
          "run_id": "…",                 // traced (record)
          "cold_collapse": false,        // true ⇒ render "empty by design", no top10 key
          "ndcg@10": …, "recall@20": …, "hitrate@10": …,   // traced (per-user parquet)
          "top10": [ { "rank": 1, "item_id": …, "title": …, "brand": …,
                       "price_usd": …, "main_category": …, "hit": true,
                       "catalog_index": 305546 } ]   // catalog_index TRACED, rest descriptive
        }
      }
    }
  }
}
```

Notes for the JS:

* `top10` is **absent** (not empty) when `cold_collapse` is true — ALS and
  item-kNN for `n_train == 0` users. Render the by-design empty state; their
  zero `ndcg@10` / `recall@20` / `hitrate@10` are real and should still show.
* `catalog_index` is the traced leaf; `item_id` is that index resolved through
  the eval cache's `item_ids.parquet` (catalog order) and is descriptive, as are
  `rank`, titles, brands, prices and categories.
* **`n_train` is descriptive, not traced** (deviation from this file's earlier
  AGREED draft, taken by T28): the manifest has exactly three source kinds and
  none can express "`gold.user_stats` at the pinned snapshot". Its traced proxy
  is `segment` — the per-user parquet column that *is* `n_train` bucketed — and
  the exporter asserts `gold.user_stats.n_train == eval-cache n_train ==
  len(history)` and `segment_of(n_train) == segment` before writing, so a wrong
  `n_train` aborts the export rather than shipping. Same class as the titles.
* Added since the AGREED draft (additive only): `model_labels`,
  `curation_rule`, `curation`, `n_train_note`, `headline_run_id`,
  `iceberg_snapshots`, per-arm `run_id` and `hitrate@10`, per-item `brand` /
  `price_usd` / `main_category`, and `top10[].catalog_index`.

## `dq.json` — SHIPPED (T29, exhibit 4: DQ dashboard)

Anchored by one appended `kind="dq_export"` record carrying the SHA-256 of both
`data/waterfall.json` and the Spark job's `dq_raw.json`. Both are re-published
byte-identically under `results/dq/` (committed — `data/` is gitignored and the
verifier re-reads a `results_artifact` source file in **record** mode too), and
every numeric leaf below uses the `results_artifact` source kind against one of
them. Which one is not arbitrary:

* the **raw waterfall counts** (`rows_in`, `rows_out`, `target_count`,
  `reasons[].rows`, `sum_ok`, `count_ok`) resolve into
  `results/dq/waterfall.json` — i.e. into the committed reconciliation ledger
  itself, so "the dashboard's waterfall byte-matches `data/waterfall.json`" is
  true by construction, not by comparison;
* everything the Spark job **derived** (`delta`, `share_of_rows_in`, the
  dominant `reason`, the contract matrix, quarantine tallies, funnel, measured
  rates, reconciliation identities) resolves into `results/dq/dq_raw.json`.

`export_dq` additionally re-derives the waterfall-edge ↔ stage mapping itself
and aborts if `dq_raw.json` restates any count differently.

```jsonc
{
  "schema_version": 1,
  "generated_by": "batch_recsys_lab.demo.export_dq",
  "record_run_id": "<kind=dq_export run_id>",
  "artifact_sha256": { "waterfall": "sha256:…", "dq_raw": "sha256:…" },   // descriptive
  "artifact_path":   { "waterfall": "results/dq/waterfall.json", "dq_raw": "results/dq/dq_raw.json" },
  "audit_run_id": …, "audit_run_ts": …, "build_run_id": …, "headline_run_id": …,  // descriptive
  "iceberg_snapshots": { "<table>": … },   // traced (record) — headline-pinned tables the job guarded
  "table_snapshots":   { "<table>": … },   // traced (record) — the ledger tables it READ

  "waterfall": {
    "stages": [ { "stage": "reviews:raw->bronze",        // descriptive label
                  "dataset": …, "stage_from": …, "stage_to": …, "target_table": …,   // descriptive
                  "rows_in": …, "rows_out": …, "target_count": …,
                  "sum_ok": true, "count_ok": true,      // ← results/dq/waterfall.json
                  "delta": …, "reason": …, "matches_ledger": true,   // ← dq_raw.json
                  "reasons": [ { "reason": …,            // descriptive label
                                 "rows": …,              // ← results/dq/waterfall.json
                                 "share_of_rows_in": … } ] } ],      // ← dq_raw.json
    "reconciles": true, "ledger_rows_checked": …, "run_id": …
  },

  "contract_matrix": { "<table>": { "<check_id>": {
        "status": "pass" | "measured" | "fail", "measured": … | null, "violations": …,
        "kind": …, "column": … | null,          // descriptive
        "threshold": null } } },                // ALWAYS null — see note
  "contract_tables": { "<table>": { "contract_name": …, "contract_version": …,
                                    "total_rows": …, "counts": { "pass": …, "measured": …, "fail": … } } },

  "headline_counts": { "raw_reviews_rows": …, "bronze_reviews_rows": …,
                       "silver_interactions_rows": …, "exact_duplicate_rows": …,
                       "superseded_by_later_review_rows": …, "quarantined_interaction_rows": …,
                       "quarantine_ledger_rows": …, "kcore_pruned_rows": …,
                       "gold_interactions_5core_rows": …, "raw_items_rows": …, "silver_items_rows": … },
  "contract_summary": { "tables": …, "checks": …, "pass": …, "measured": …, "fail": …,
                        "any_fail": false,
                        "by_kind_status": { "<check_kind>": { "<status>": … } },
                        "failing": [ { "table": …, "check_id": …, "kind": …,
                                       "violations": …, "measured": … } ] },
  "quarantine": { "build_run_id": …, "total_rows": …,
                  "by_reason": [ { "table": …, "reason": …, "rows": …,
                                   "share": …, "share_of_input": … } ],
                  "tables": { "<table>": { "rows": …, "rows_all_runs": …,
                                           "by_primary_reason": [ … ],
                                           "by_violation_reason": [ … ], "snapshot_id": … } } },
  "measured_rates": { "<key>": { "source": "contract_ledger" | "dq_export_job",
                                 "run_id": … | null, "is_selected_audit": …,
                                 "table": …, "check_id": … | null,
                                 "rows": …, "denominator": …, "rate": …, "details": { … } } },
  "reconciliation": { "checks": [ { "name": …, "lhs": …, "rhs": …, "ok": true, "note": … } ],
                      "all_ok": true },
  "kcore_funnel": [ { "iteration": …, "users": …, "items": …, "interactions": …,
                      "converged": …, "wall_clock_s": … } ],
  "notes": { … }        // descriptive subtree: provenance copy for the exhibit
}
```

Notes for the JS (deviations and additions since the AGREED draft, all additive
— no field was renamed or removed):

* **`contract_matrix.<table>.<check>.threshold` is always `null`.** No contract
  YAML declares a numeric threshold: checks are pass/fail predicates
  (`not_null`, `range`, `allowed_values`, `forbidden_values`,
  `no_control_chars`, `no_all_null`, `schema_conformance`) or unbounded
  measures (`unknown_share`, `orphan_rate`). The key is kept for the AGREED
  shape; render "—", not "0".
* **Per-table metadata lives in the sibling `contract_tables` map**, not inside
  `contract_matrix.<table>`, so the matrix namespace holds check ids only and a
  future check named `total_rows` cannot collide with it.
* **`status: "measured"` is not a failure.** In the shipped audit 26 of the 28
  measured checks are `schema_conformance` rows carrying the recorded T8
  nullability downgrade (Spark/Iceberg `createOrReplace` marks every column
  physically nullable; the declared-non-null column is still hard-enforced by
  its `not_null` check). Read `contract_summary.any_fail` for the headline and
  `by_kind_status` for the breakdown.
* **`kcore_funnel[].interactions` is the ledger's `rows` column**, renamed here
  per this file's AGREED draft.
* Added: `generated_by`, `artifact_sha256`, `artifact_path`, `audit_run_id`,
  `audit_run_ts`, `build_run_id`, `headline_run_id`, `iceberg_snapshots`,
  `table_snapshots`, per-stage `dataset` / `stage_from` / `stage_to` /
  `target_table` / `target_count` / `sum_ok` / `count_ok` / `matches_ledger` /
  `reasons[].share_of_rows_in`, `waterfall.ledger_rows_checked`,
  `waterfall.run_id`, `contract_tables`, `headline_counts`,
  `contract_summary`, `measured_rates`, `reconciliation`,
  `kcore_funnel[].converged` / `wall_clock_s`, and `notes`.

## `lineage.json` — AGREED (T30, exhibit 5)

Projection of `results/lineage.json`, anchored by the `kind="lineage"` record
(`20260807T160910Z-739833b`, `/artifact_sha256`) — i.e. every leaf uses the
`results_artifact` source kind.

```jsonc
{
  "schema_version": 1,
  "record_run_id": "20260807T160910Z-739833b",
  "artifact_sha256": "sha256:…",
  "stages_count": 24, "complete": true,
  "stages": [ { "stage": "bronze.reviews", "layer": "bronze", "table": …,
                "rows_in": …, "rows_out": …, "bytes": …, "wall_clock_s": …,
                "wall_clock_source": …, "snapshot_id": …, "footnote": … } ],
  "footnotes": { "<key>": "<text>" }
}
```

## `timetravel.json` — AGREED (T30, exhibit 5 toggle)

```jsonc
{
  "schema_version": 1,
  "pinned": { "snapshot_ids": { "<table>": … }, "run_id": "20260807T055333Z-c320c79" },
  "today":  { "snapshot_ids": { "<table>": … }, "source_run_id": "<ops record>" },
  "reproduce": [ { "run_id": …, "run_ts": …, "verdict": "byte_exact",
                   "reproduces_run_id": "20260807T055333Z-c320c79" } ],
  "ops_chain": [ { "run_id": …, "step": …, "snapshot_id_before": …, "snapshot_id_after": … } ]
}
```

## `search/` — AGREED (T35, exhibit 6; **not committed**)

`embeddings_int8.bin` + `scales_f32.bin` (top-50k items by
`pop_train_end_365`, per-row symmetric int8), `embeddings_meta.json`
(`{schema_version, n_items, dim, quantization, source_embeddings_sha256,
five_core_snapshot_id}`), `items_meta.json` (item_id/title/category, all
descriptive), `example_queries.json` (~12 canned queries with reference top-10
computed by the real Python model — also the in-UI quantization-parity
receipt). Deleting `demo/data/search/` and `demo/vendor/` must leave every
other exhibit green (cut order #2).
