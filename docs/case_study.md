<!--
  Crossover Study — case study, final copy (Phase 6 T37, 2026-08-09).
  Revised 2026-08-18 for the Phase 8 close-out (UPGRADE_PLAN §8b line 277):
  §1/§6/§7 carry the measured regime map and the recency-matched fairness
  test, §6's old caveat paragraph is discharged item by item, and §9/§10
  carry the new verification and non-claims. No prior number was edited;
  where Phase 8 supersedes a framing, the superseding text says so.
  Supersedes docs/case_study_draft.md, which is kept unedited for history.

  Revised 2026-08-19 for Phase 9 T9-1 (UPGRADE_PLAN §8c): headline repinned
  null-first — no history-depth crossover (every segment CI excludes zero),
  41.11% catalog churn as the measured mechanism, blend α=0.3 the one
  effective arm, n*=∞ ("the optimal routing is no routing"). The T8-2
  five-cell result is demoted to a mechanism footnote with the ~80-test
  multiplicity disclosure. No metric value was changed anywhere in this
  revision; the T8-2 verdict is downgraded by a superseding
  EXPERIMENT_LOG.md entry, never an edit, and the T8-4 gate reopening
  (T9-2) is recorded the same way.

  Revised 2026-08-21 for Phase 9 T9-4 (UPGRADE_PLAN §8c): a new cross-dataset
  chapter (§7b) presents catalog churn as the controlling variable, against the
  ML-32M contrast that T9-3 executed — 6.40% churn, crossover at n*=20 on the
  BH-corrected NDCG@10 primary family, not metric-robust, and a catastrophic
  cold-user loss. §6's reopened-gate caveat and §10's "untested hypothesis"
  clause are rewritten to the tested outcome with its qualifications, and §9
  gains the matching verification bullets. Again no prior metric value was
  edited; the Amazon-side null and every Amazon number stand unchanged.
  NUMBERING: the new chapter is §7b, not a renumbered §8. Renumbering would
  invalidate every "§9"/"§10"/"chapter 7"/"chapter 8" cross-reference in this
  document, in EXPERIMENT_LOG.md and in the demo copy, for no reader benefit;
  the b-suffix follows UPGRADE_PLAN's own §8b/§8c convention.

  LINK CONVENTIONS (chosen here; no demo/ code was changed to support them).

  1. Receipt anchors: `demo/index.html#receipt-<run_id>`.
     As of this tree, demo/js/receipts.js has NO hash routing — no `hashchange`
     listener, no `location.hash` read. The drawer opens only through the
     delegated click/keydown handler on `[data-run-id]`. So this anchor is a
     forward convention, deliberately shaped to be adoptable in three lines:
     the fragment payload is exactly the `data-run-id` value the drawer already
     keys on, so `openReceipt(location.hash.slice('#receipt-'.length))` on load
     and on `hashchange` is the whole change, with no markup churn.
     Rejected alternatives: (a) `#sec-receipts` alone — lands the reader in the
     right exhibit but drops the run identity, which is the entire payload;
     (b) `?run=<run_id>` — works on a static host but forces a full reload and
     the drawer's state is in-memory, so it would fight the SPA-less design.
     Until the drawer adopts it, these links land on the page and the reader
     finds the record in the receipts index table under `#sec-receipts`.

  2. Only run_ids that are records in `results/runs.jsonl` are linked. The gold
     build id `20260805T143256Z-7406fc1` is a build-ledger id, not a results-log
     record, so it is cited in prose against its ledger and NOT linked.
     `demo/data/receipts.json` currently exports cards for 28 records; four
     linked run_ids here have no exported card yet — 20260805T165630Z-2dcea79,
     20260805T173018Z-035042b, 20260805T173520Z-035042b, 20260807T040118Z-5e212d7
     — and resolve against `results/runs.jsonl` by run_id.
     The Phase 8 run_ids linked in §6, §7 and §9 (the T8-1 regime map and the
     T8-2 VAL, TEST, paired-delta and regime-map records) are results-log
     records on the same footing; their cards arrive with the demo's Phase 8
     export, and until then they resolve by run_id against the log.
     The ML-32M run_ids in §7b are cited as plain code, NOT linked: the demo's
     receipts index is Amazon-scoped at this tree, so a receipt anchor would
     resolve to nothing. They are records in `results/runs.jsonl` and resolve
     by run_id there. `results/confirmatory_ml32m_test.json` is cited by path
     because it is a committed derived analysis, not a results-log record
     (`appends_to_runs_jsonl: false`); its own `source_run_ids` block names the
     one-shot TEST records every one of its numbers is recomposed from.

  3. Paths (`demo/…`, `results/…`, `configs/…`) are repo-root-relative, matching
     the site's publish root. Rendered from `docs/` on a local viewer they need
     a `../` prefix.

  4. FIGURE hooks below mark the five inline figures approved at Checkpoint 1
     (T32 item 4). They are numbered in document order, and each names both the
     Checkpoint-1 list item it satisfies and the demo exhibit that is its live
     counterpart. Only Figure 1 has a committed render today; the other four are
     hooks for the site build.
-->

# Crossover Study

> **"How much history does a user need before personalization beats popularity?"** Crossover Study ingests 43.9M Amazon Electronics reviews through a contract-checked Spark + Iceberg lakehouse on a single 16GB laptop, evaluates popularity, item-kNN, ALS, and semantic-content retrieval under temporal splits with full-catalog ranking and bootstrap CIs — and measures the answer: **there is no such depth. No history-depth crossover exists in this data: ALS, item-kNN, and pure content lose to recency-weighted popularity at every history depth, with every segment's CI excluding zero — before and after Phase 8 removed the recency asymmetry that could have explained it.** The mechanism is measured, not argued: 41.11% of 2023 TEST purchase mass lands on items with zero or near-zero training support (run `20260817T095926Z-633d454`), capping any TRAIN-frozen model at 65.5% recall before a single modeling choice is made. The one arm that beats popularity is popularity itself, gently re-ranked by a normalized semantic-content blend at α=0.3 (+0.000322 NDCG@10, ≈+6% relative, 1,000-resample paired CIs cleanly separating it from zero). And the fitted routing policy is n\*=∞ — every user routes to the blend: **the optimal routing is no routing.** A five-cell pocket where a recency-matched item-kNN edges popularity survives as a mechanism footnote under ~80 uncorrected tests, not as a headline (§6).

**Reading the labels.** Every claim below carries an evidence class. Three of them are the demo site's own convention: **[measured]** — produced by a recorded run in `results/runs.jsonl`; **[estimated]** — inferred, with the inference stated; **[projected]** — reasoned from code properties, not observed. This lab adds a fourth, **[derived]**, defined in full where it first does work (chapter 7). Where a claim has a receipt, the run_id, record kind, or Make target is named inline and links to that record's card in the [receipts drawer](demo/index.html#sec-receipts).

---

## 1. The question, and the answer the data gave

The plan for this lab predicted a crossover. Popularity should win for users with no history, personalization should take over once a user has accumulated enough signal, and somewhere in between there is a threshold — call it n\* — that a real system could route on. The signature exhibit was going to be NDCG@10 by user-history depth, one line per model, with the crossing point marked.

Half of that prediction held. Recency-weighted popularity does decay with history depth: NDCG@10 falls from 0.007505 in the cold-start segment (`0`) to 0.003711 for users with 20+ TRAIN interactions **[measured, run [`20260805T172047Z-035042b`](demo/index.html#receipt-20260805T172047Z-035042b)]**.

The other half did not happen. Item-kNN loses to trailing-12-month popularity in every segment. ALS — tuned over a ten-point VAL grid, run to the hardware feasibility frontier of a 16GB machine — loses to it in every segment too. Pure semantic-content retrieval loses to it by a factor of six. The ALS deficit *shrinks* monotonically as history deepens, from −0.0026 in the `1-4` bucket to −0.0015 at `20+` **[measured]** — the crossover's direction is real and visible — but it never reaches zero within the depths this dataset contains.

<!-- FIGURE 1 — the signature exhibit. Committed static render, generated strictly
     from results/runs.jsonl by eval/crossover_chart.py via
     `make crossover-chart CONFIG=configs/crossover_test.yaml`:
       results/figures/crossover_test.svg  (preferred; 26.6kB)
       results/figures/crossover_test.png  (raster fallback)
     Five TEST lines with 95% bootstrap CI bands: blend α=0.3, pop-t12m, ALS,
     item-kNN, content. Live, metric-switchable counterpart: demo #sec-crossover
     (charts.js is a port of the same geometry — if one changes, both change).
     Caption must state: full-catalog ranking over 368,228 items; bands are 95%
     user-bootstrap CIs, 1,000 resamples, seed 20260805. -->

![NDCG@10 by user-history depth on TEST — five arms, 95% bootstrap CI bands](results/figures/crossover_test.svg)

*The chart above is the committed static render; the interactive version, with a metric switch and the n\* slider, is the [crossover explorer](demo/index.html#sec-crossover).*

What did win is the least glamorous thing on the list: trailing-12-month popularity, re-ranked by a MiniLM content score at a weight of 0.3. On the frozen TEST split it scores NDCG@10 **0.005726 [0.005524, 0.005948]** against popularity's **0.005404 [0.005209, 0.005635]** **[measured, runs [`20260807T055333Z-c320c79`](demo/index.html#receipt-20260807T055333Z-c320c79) and [`20260805T172047Z-035042b`](demo/index.html#receipt-20260805T172047Z-035042b)]**. The paired delta is **+0.000322 [+0.000200, +0.000449]**, CI excluding zero **[measured, `kind=paired_delta`, 1,000 resamples, seed 20260805]**. That is +5.96% relative. It is a small win, and it is stated as small.

So the honest headline is a null with its mechanism measured and one small win attached. On 2023 Amazon Electronics, with training frozen at 2022-06-30, no personalization arm tested here beats knowing what is currently popular — not globally, and not at any history depth, with every per-segment CI excluding zero, including after Phase 8 removed the recency asymmetry that made the first comparison unfair. Content similarity earns its place only as a gentle re-ranker on top of that. A finer-grained observation arrived last: once the catalog is split by how *learnable* each purchased item was at the training cutoff, five regime-map cells show a recency-matched item-kNN edging popularity with per-cell CIs excluding zero — but those cells come from ~80 uncorrected tests, of which ~4 would be expected significant at α=0.05 by chance alone, so they are read as a mechanism footnote, not as evidence against the null (§6). The machinery below — lakehouse, contracts, protocol — is what makes the null trustworthy enough to publish.

---

## 2. Provenance

**Provenance.** This lab grew out of a five-person course project (CISC3018 Cloud Computing and Big Data Systems, University of Macau, Fall 2025) that built a recommendation bot on a 1M-interaction sample of this dataset; my primary individual contribution there was the presentation material. Everything in this lab — pipeline, contracts, models, evaluation harness, demo — was designed and built from the raw public dataset, solo, with its own verification chain.

Nothing from that project ships here, and none of its code, models, or results is presented as individual work or reproduced in this write-up. Two design notes were used as reading material only: a metadata schema shape and the idea of embedding item text with a small sentence transformer. Both were rebuilt from scratch against the raw files, because the archived sample had no timestamp column — which makes temporal evaluation, the entire premise of this lab, impossible from it. Separately, individually-authored Hadoop/HBase/MapReduce coursework is where the batch-processing fundamentals came from; those reports are not republished here either.

The dataset is Amazon Reviews 2023 (McAuley Lab, UCSD), used under its research-use terms; the raw files are never redistributed. Cite Hou et al., 2024, *Bridging Language and Items for Retrieval and Recommendation*.

---

## 3. A lakehouse on a laptop

The scale claim is the point of the ingestion layer, so it is stated with its ledger attached. Two gzipped JSONL files — 6.47GB of reviews, 1.31GB of metadata, both SHA-256'd into `data/MANIFEST.md` at download time — become Iceberg tables on a local Hadoop catalog: **43,886,944 reviews and 1,610,012 items** in bronze **[measured]**. The published release rounds to 43.9M / 1.61M; the manifest records the observed counts as canonical and the delta against the rounded figures explicitly, rather than asserting a match.

Every row that leaves the pipeline leaves with a reason. Bronze → silver drops 521,520 rows: **2** quarantined for rating-domain violations, **477,968** exact duplicates, **43,550** superseded (user,item) re-reviews where only the latest survives. Silver → gold prunes **27,891,888** rows through iterative 5-core filtering, landing at **15,473,536** modeling interactions **[measured, gold build `20260805T143256Z-7406fc1`, carried on the build-summary ledger and on `results/dq/dq_raw.json`]**. The waterfall's arithmetic is enforced, not printed: for every edge, the sum of reason-rows equals the source count *and* the kept count equals the live Iceberg table count re-read at publish time, with a non-zero exit on drift. Items pass through clean — 1,610,012 in, 1,610,012 out, no quarantine loss.

<!-- FIGURE 2 — reconciliation waterfall (Checkpoint 1 figure list, item 3).
     Raw → bronze → silver → gold, every edge labelled with its drop reasons and
     counts, sourced from results/dq/waterfall.json. Live counterpart, with the
     contract pass/fail matrix, quarantine counts, unknown-brand and null-price
     rates alongside it: demo #sec-dq. Caption must state that each edge's
     reason-rows sum to the source count and the kept count is re-read from the
     live Iceberg table at publish time, non-zero exit on drift. -->

*Live version, with the contract matrix and quarantine ledger beside it: the [data-quality dashboard](demo/index.html#sec-dq).*

The k-core loop converged at **iteration 16**, taking users from 18,286,190 to 1,641,026 and items from 1,609,860 to 368,228 **[measured]**. Nearly all of the reduction happens in the first seven iterations; the last eight shave under 200 rows in total. That 368,228-item survivor set is the full-catalog ranking universe for every metric in this case study. What that filtering is worth was later measured directly: the same popularity baseline, re-scored on the un-cored silver universe, loses ×1.20 of its global NDCG@10 and ×1.28–1.59 within matched history segments — the k-core inflation §10 quantifies **[derived]** from two recorded runs.

Seven YAML contracts (one per silver/gold table) drive a hand-rolled PySpark checker that writes pass/fail and violation counts to a `dq_results` Iceberg table and routes violating rows to quarantine tables with a reason column. Interactions→items orphan rate: **0.0** (0 of 43,365,424) **[measured]**. Unparseable prices: 316 rows, 0.0196%. Unknown-brand share: **4.545%**, against an expectation of roughly 18% carried in from the planning notes — because a Manufacturer-field fallback recovers 23.90% of rows that would otherwise be branded "Unknown" **[measured]**. The expectation was wrong; the measurement is what's published.

Two independent `make data` rebuilds produced **8/8 content-identical tables**, ~27.5 minutes each, on `local[10]` with an ~8g driver on an Apple M4 with 16GB of RAM **[measured]**. Single-node, stated plainly.

**A single-node reality check.** The "you could have just used DuckDB" objection deserves a number, not a shrug. DuckDB 1.5.5 rebuilt both silver tables from the same bronze Iceberg snapshots with the exact recorded waterfall — 43,886,944 in; 2 quarantined; 477,968 exact duplicates; 43,550 superseded; 43,365,424 kept — in **12.7 / 17.0 / 19.4 s** across three runs (interactions; items 1.4–2.2 s), against Spark `local[10]`'s recorded 316–569 s **[measured, single machine, record `20260809T212313Z-a5a9e6e`]**. The scopes differ and the gap is stated with that caveat: the Spark ledger times include the contract audit's 43M-row FK join, `dq_results` writes, and Iceberg commits; the DuckDB port times scan → gate → dedup → keep-latest → parquet write only. Survivor content matched up to 74 rows inside keep-latest tie groups — Spark breaks those ties on `xxhash64`, which DuckDB cannot reproduce; the divergence is bounded, measured, and reported rather than patched. The honest conclusion cuts both ways: at 44M rows on one 16GB laptop, a vectorized single-node engine is decisively the faster tool for this transform, and Spark was not chosen for single-node speed. What Spark bought here is the Iceberg-catalog write path, the contract/quarantine engine integration, and semantics that port to a cluster unchanged — capabilities, not throughput — and the distributed claim itself remains **[projected]**, exactly as §10 states.

---

## 4. The evaluation protocol as the product

The models in this lab are deliberately standard. The evaluation is not, and that is where the engineering went.

**Temporal splits, frozen in a file.** TRAIN ≤ 2022-06-30, VAL 2022-07-01 → 2022-12-31, TEST 2023-01-01 → 2023-10-01 (exclusive), frozen in `configs/splits.yaml` on 2026-08-05 and hashed into every run record. Random splits leak future interactions backward into training and flatter every personalized model; a recommender that will be asked about tomorrow should be evaluated on tomorrow. VAL carries 356,362 users; TEST carries 228,153 **[measured]**.

**TEST is touched once per model.** Before every TEST campaign, a pre-declaration was written and committed to `EXPERIMENT_LOG.md` *first*: the arms, the comparisons, the acceptance gate, and an explicit no-return clause — whatever TEST shows, there is no going back to the VAL grid. Four such pre-declarations exist (Phase 3's T7 for ALS, Phase 4's T15 for content/blend/hybrid, Phase 7's un-cored popularity run, and Phase 8's T8-2 recency-matched arms), each timestamped ahead of its results **[measured, `EXPERIMENT_LOG.md`]**. The T8-2 one is checkable by commit clock rather than by prose: it was committed at 2026-08-17T11:45Z — before the T8-2 model code existed, and ~13 hours before the first T8-2 run record. This is the part of the protocol that makes a negative finding worth reading.

**Full catalog, no sampled negatives.** Every user is scored against all 368,228 gold-catalog items via chunked matmul, with TRAIN-seen items masked out. No exhibit in this lab samples negatives. The ANN index built in Phase 4 (chapter 8) exists for the demo only and is flagged `used_in_eval_metrics: false` on its own record.

**CIs on everything, paired deltas for every comparison.** Headline numbers carry user-bootstrap 95% CIs (1,000 resamples, seed 20260805); every comparative claim is a paired bootstrap over common users on a shared resample matrix, and a difference is only claimed where the CI excludes zero. Metrics reported per run: Recall@10/20/50, NDCG@10/20, MRR, HitRate@10, plus coverage, popularity share, and novelty.

**One config per run, append-only results.** Each run is a single YAML config; each result is one JSON line in `results/runs.jsonl` carrying config hash, git SHA, dirty flag, dataset manifest hash, Iceberg snapshot IDs, seeds, and a per-user artifact path. The log is append-only and committed: a wrong run gets a superseding entry, never an edit. Stochastic arms (ALS) report 3-seed mean±sd on top of the CIs — VAL sd 0.0000568, TEST sd 0.0000197, both roughly 3× smaller than the bootstrap CI half-width, which is the evidence that seed variance is not driving any conclusion **[measured]**.

---

## 5. The ladder of challengers

Each rung is a TEST number with a receipt. All are global NDCG@10 on the frozen TEST split.

| arm | NDCG@10 | run_id |
|---|---:|---|
| random (seed 13) | 0.000012 | [`20260805T165630Z-2dcea79`](demo/index.html#receipt-20260805T165630Z-2dcea79) |
| content only (MiniLM) | 0.000886 | [`20260807T050054Z-c320c79`](demo/index.html#receipt-20260807T050054Z-c320c79) |
| popularity, all-time | 0.000832 | [`20260805T173018Z-035042b`](demo/index.html#receipt-20260805T173018Z-035042b) |
| item-kNN (top_n=50) | 0.000946 | [`20260805T185305Z-adbca99`](demo/index.html#receipt-20260805T185305Z-adbca99) |
| ALS (rank 128, primary seed) | 0.002750 | [`20260806T082441Z-2f2f26d`](demo/index.html#receipt-20260806T082441Z-2f2f26d) |
| popularity per-category, t12m | 0.004683 | [`20260805T173520Z-035042b`](demo/index.html#receipt-20260805T173520Z-035042b) |
| **popularity, trailing 12m** | **0.005404** | [`20260805T172047Z-035042b`](demo/index.html#receipt-20260805T172047Z-035042b) |
| **blend α=0.3 (pop × content)** | **0.005726** | [`20260807T055333Z-c320c79`](demo/index.html#receipt-20260807T055333Z-c320c79) |

**[all measured]**

**Popularity is not one baseline, it is a design space.** All-time popularity scores 0.000832; windowing it to the trailing 12 months takes it to 0.005404 — a paired delta of **+0.0046 NDCG@10 [+0.0044, +0.0048]**, roughly 6× **[measured]**. Catalog churn dominates this category, and a "popularity baseline" that ignores recency is a straw man. That single decision is worth more than every personalization arm tested here.

**Item-kNN failed.** Cosine co-occurrence, neighbor-list grid {50, 100, 200} on VAL: flat to three decimals, so the cheapest point won. On TEST it scores 0.000946 and loses to pop-t12m in *every* segment, including `20+` (0.0006 vs 0.0037), global paired delta −0.0045 [−0.0047, −0.0042] **[measured]**. Cold-start users score exactly zero, as they must — no history, no co-occurrence.

**ALS failed at the hardware frontier, and the frontier is stated.** Ten single-variable VAL experiments (E1–E10: rank 32/64/128, reg over two orders of magnitude, α ∈ {1, 10, 40}, iterations 8/25, binary vs rating confidence) selected rank=128, reg=0.01, α=10, iter=25, binary. Two caveats were pre-registered *before* TEST: VAL quality was still rising in both rank and iterations, so this is the 16GB feasibility frontier and not a converged optimum (peak RSS 9.54GB); and the shallow-beats-deep segment gradient held at all ten grid points. On TEST, ALS − pop-t12m is negative in every segment with every CI excluding zero: −0.0075 at `0`, −0.0026 at `1-4`, −0.0023 at `5-9`, −0.0026 at `10-19`, −0.0015 at `20+` **[measured]**. ALS does beat item-kNN everywhere warm (+0.0018 global) — it is the best classical CF representation tried here, and it still loses.

**Content alone failed badly.** MiniLM `all-MiniLM-L6-v2` embeddings over a `title + brand + category + features` recipe, user profile = mean-pooled L2-normalized TRAIN item vectors, scored exactly against the full catalog. TEST NDCG@10 0.000886, losing to pop-t12m in every segment including the shallow-warm buckets the hypothesis specifically named (`1-4`: −0.0044; `5-9`: −0.0044, both CIs excluding zero) **[measured]**. In segment `0` it is exactly 0.0 by construction: no TRAIN items, no profile.

**The blend won.** A normalized 0.7·popularity + 0.3·content score — alpha chosen on VAL by a rule written down before the grid ran, from {0.1, 0.3, 0.5, 0.7, 0.9}, with an interior peak at 0.3 whose CI does not overlap either neighbor's — beats pop-t12m on TEST globally (+0.000322 [+0.000200, +0.000449]) and in all four warm segments, every CI excluding zero **[measured]**. In segment `0` the delta is exactly +0.000000: the blend degenerates to pure popularity for users with no content profile, which is correct behavior, not a miss. The semantic signal that could not rank anything on its own is still useful as a tie-breaker on top of a signal that can.

**Three arms have nothing to say to a cold-start user, and the demo says so.** ALS, item-kNN and content all collapse to an all-zero score vector when `n_train == 0`, which a naive top-k turns into an index-order tie-break list that looks like a recommendation. The [pick-a-shopper exhibit](demo/index.html#sec-shoppers) renders those three columns as an explicit "no personalized signal — empty by design" panel instead — 18 suppressed arms across its 6 cold-start shoppers. The content arm was emitting the tie-break list until Checkpoint 2 caught it; the fix was to make the failure legible, not to hide it.

---

## 6. The crossover that never came — and the mechanism that explains it

Per-segment TEST NDCG@10, five history-depth buckets (labels are the frozen segment edges, identical to the exhibits') **[measured]**:

| arm | 0 | 1-4 | 5-9 | 10-19 | 20+ |
|---|---:|---:|---:|---:|---:|
| pop-t12m | 0.007505 | 0.005821 | 0.005225 | 0.005107 | 0.003711 |
| blend α=0.3 | 0.007505 | 0.006227 | 0.005504 | 0.005426 | 0.004095 |
| ALS | 0.000000 | 0.003259 | 0.002925 | 0.002541 | 0.002254 |
| content | 0.000000 | 0.001458 | 0.000809 | 0.000492 | 0.000307 |

This is the table Figure 1 plots; the live version with CI bands, a Recall@20 switch and per-cell receipts is the [crossover explorer](demo/index.html#sec-crossover).

Two things are visible. First, popularity's advantage genuinely erodes with depth — it is nearly twice as strong for cold-start users as for the `20+` bucket. Second, at history-depth granularity nothing catches it (the Phase 8 addendum below records a finer-grained pocket, held to footnote strength by multiplicity). The ALS-minus-popularity deficit narrows monotonically (−0.0026 → −0.0015 as depth increases) but the sign never flips, and the CI at `20+` is [−0.0021, −0.0008]: still comfortably below zero **[measured]**.

The mechanism is consistent across all three failed arms, and it is no longer an inference. TEST ground truth is 2023 purchases; TRAIN ends mid-2022. A large share of 2023 interactions land on items that barely existed in the training window, and therefore carry almost no co-occurrence mass and no learned factor structure. A trailing-12-month popularity window partially tracks that churn; a TRAIN-frozen co-occurrence matrix or factor matrix structurally cannot. Content embeddings are churn-agnostic — a new HDMI cable looks like an old HDMI cable — which is precisely why content helps as a re-ranker while failing as a ranker. Through Phase 4 that paragraph was **[derived]** from the pattern of failures. Phase 8 measured it.

### The churn, measured: the catalog-learnability regime map (T8-1)

The first Phase 8 task stratified the *already-recorded* TEST results on a second axis — how learnable each purchased item was at the training cutoff (TRAIN support, and recency of its last TRAIN interaction) — crossed with the frozen history-depth segments. No retraining and no new TEST model scoring: the per-cell numbers are recomposed from the persisted per-user top-50 lists, and the recomposition is anchored by an identity check that reproduces the recorded per-user metric vectors to max |diff| **1.11e-16** **[measured, record [`20260817T095926Z-633d454`](demo/index.html#receipt-20260817T095926Z-633d454), `kind=regime_map`; an accidental second invocation appended `20260817T100112Z-633d454`, whose payload compares equal field-for-field — append-only means both stand]**. The thresholds (support zero / 1–4 / ≥5, anchored to the 5-core *k*; recency ≤90d / 91–365d / >365d, the frozen popularity windows) and the acceptance gate were registered before any cell outcome was computed.

Three numbers carry the section, all from that record:

- **41.1% of TEST ground-truth interaction mass sits on items a TRAIN-frozen model cannot rank** — 34.54% (172,302) on items with *zero* TRAIN support, 6.58% (32,813) on items with 1–4. The preregistered gate read <10% ⇒ churn diagnosis wrong, stop; 10–25% ⇒ partial; **≥25% ⇒ measured and supported**. Measured 0.4111, i.e. 1.6× the support threshold and 4.1× the refutation floor. **[measured]**
- **The attainable-recall ceiling for any TRAIN-frozen factor model is 65.5%** (share of TEST ground truth on items with support ≥1; 58.9% at support ≥5) — an upper bound on Recall@K for *any* K, before a single modeling decision. It is not flat in history depth: 56.8% at segment `0`, peaking at 72.6% for `10-19`, then falling to **43.3% for the deepest users (100+ TRAIN interactions)** — the deepest users buy the newest items hardest, so the arm with the most history to exploit faces the weakest ceiling. **[measured; the 100+ figure sits on T8-3's exploratory deep-bucket axis and inherits its exploratory/derived label]**
- **5.19% of the catalog absorbs 34.5% of TEST purchase mass** — the 19,118 zero-support items, all first seen after the cutoff, a 6.7× over-representation. The mirror image is just as sharp: the 57.2% of the catalog whose last TRAIN interaction is more than a year old draws 2.0%. **[measured]**

Two consequences fall straight out of the map. First, in all ten zero/low-support cells — that 41.1% of ground-truth mass — both pop-t12m and ALS score exactly 0.000000. That is structural, not a null result: a trailing-12-month popularity list contains only recently-supported items, and ALS has no factor vector for an item it never saw. (The identity anchor is what licenses reading those zeros as real rather than as a recomposition bug.) The entire arm contrast in Phase 4 was therefore being fought inside the 58.9% of the mass that both arms could reach. Second, the share is a *rate*, not a fixed offset: the same pipeline run as a VAL rehearsal puts zero+low at 0.2792 against TEST's 0.4111, so unreachable mass grows with distance from the cutoff **[measured, same record]**.

### Phase 8 addendum: the recency-matched rematch (T8-2)

The comparison above had one asymmetry nobody had tested: popularity got a trailing-12-month window while every CF arm was static all-history. Phase 8 re-ran the fight recency-matched, under a preregistration committed before the arms' code existed — two classical arms, one TEST evaluation each, comparators and decision rules fixed in advance.

**ALS-decay.** Implicit ALS with time-decayed confidence, the Phase 3 config held fixed (rank 128, reg 0.01, α 10, iter 25) and *only* the half-life tuned on VAL over a preregistered grid {90, 365, 1460} days. VAL selected **365d**, whose CI overlaps neither neighbor's, and all three half-lives beat the static-ALS VAL baseline — decay genuinely helps ALS (+25% relative on VAL, CIs disjoint) **[measured, run [`20260818T021256Z-6300640`](demo/index.html#receipt-20260818T021256Z-6300640)]**. It still loses. TEST global NDCG@10 is **0.003734 ± 0.000018** across the frozen 3-seed set against pop-t12m's 0.005404 **[measured, runs `20260818T060704Z-109c271` (primary), `20260818T051547Z-109c271`, `20260818T051858Z-109c271`]**; the paired global delta is **−0.001687 [−0.001969, −0.001432]**, and it is significantly negative at depths `0`, `1-4`, `5-9` and `10-19` **[measured, run [`20260818T064002Z-56d871c`](demo/index.html#receipt-20260818T064002Z-56d871c), `kind=paired_delta`]**. At `20+` the NDCG@10 delta turns nominally positive (+0.00036) but its CI straddles zero *and* the Recall@20 robustness guard is significantly negative (−0.0032 [−0.0050, −0.0016]) — under the preregistered two-metric rule that is not a crossover. No regime-map cell is positive-with-CI for ALS-decay either **[measured, run [`20260818T072211Z-3f3530a`](demo/index.html#receipt-20260818T072211Z-3f3530a)]**. **The depth-level null is now robust to the fairness objection by measurement rather than argument** — and the preregistered hypothesis' sharper prediction, that decayed ALS would turn positive on the stale-item cells, failed outright.

**item-kNN-t12m.** Co-occurrence restricted to the same trailing-12-month window, no free parameters. Globally it is the **worst arm in the lab**: TEST NDCG@10 **0.000301**, roughly 18× below pop-t12m, paired delta **−0.005103 [−0.005334, −0.004907]** **[measured, runs [`20260818T054430Z-109c271`](demo/index.html#receipt-20260818T054430Z-109c271) and [`20260818T064207Z-56d871c`](demo/index.html#receipt-20260818T064207Z-56d871c)]**. With a ~6.5-year TRAIN catalog, a 12-month item-side window leaves too few co-occurrences to rank with. And yet, recomposed through the regime map, it beats pop-t12m **with 95% CIs excluding zero on both NDCG@10 and the Recall@20 guard** in five cells **[measured, run [`20260818T072256Z-3f3530a`](demo/index.html#receipt-20260818T072256Z-3f3530a)]**:

| cell (axis, depth, bucket) | Δ NDCG@10 | Δ Recall@20 | share of TEST GT mass |
|---|---:|---:|---:|
| support, `5-9`, low | +0.00045 | +0.00093 | 1.73% |
| support, `20+`, low | +0.00058 | +0.00103 | 0.69% |
| recency, `1-4`, 91–365d | +0.00039 | +0.00102 | 2.16% |
| recency, `5-9`, 91–365d | +0.00069 | +0.00151 | 1.75% |
| recency, `10-19`, 91–365d | +0.00033 | +0.00242 | 1.01% |

That met the preregistered any-cell crossover criterion as written — on exactly the axis the hypothesis named (items popularity's own window under-ranks), by the arm the hypothesis did not favor. But the 2026-08-19 supersession entry in `EXPERIMENT_LOG.md` downgrades what that meeting is worth, and the downgraded verdict is the one this case study reports: **local regime-cell wins under uncorrected multiplicity; the global history-depth crossover null is robust.** Four things hold the cells to footnote strength. *Multiplicity:* roughly 40 cells × 2 arms ≈ 80 uncorrected tests per metric, so at α=0.05 about 4 false positives are expected by chance alone; the clustering (two related regions, three consecutive depth bands) and the agreement of both metrics in every cell are partial evidence against pure chance, not a correction — NDCG@10 and Recall@20 are highly correlated on the same cells, so their agreement is weaker evidence than two independent tests would be. *Magnitude:* the winning arm is the globally weakest model in the lab (TEST NDCG@10 0.000301 vs pop-t12m's 0.005404) and the per-cell deltas sit in the fourth decimal. *Mass:* each cell is 0.7–2.2% of TEST ground-truth mass and the cells overlap across the two axes. *Routability:* the cells are keyed to properties of the *ground-truth item* — its TRAIN support and recency — which a serving-time router cannot observe. What survives is the mechanism reading: pop-t12m's trailing window under-serves stale, thinly-supported items, and in exactly that pocket even a weak personalized arm finds signal — a **measured diagnosis of popularity's blind spot, not a crossover finding and not a routable policy** (§7). The full protocol, preregistration, lineage exception, multiplicity disclosure and the supersession are in `EXPERIMENT_LOG.md` (2026-08-17 / 2026-08-18 / 2026-08-19).

### The caveats this section used to carry, discharged one by one

The pre-Phase-8 version of this chapter closed with a paragraph of hedges, and the Phase 8 plan (§8b) filed two further charges against it — that the churn diagnosis was "derived from the pattern, never directly measured", and that the comparison "was unfair in one axis nobody tested". Every one of them is now either measured or explicitly still standing. The first two entries are §8b's charges; the rest are this chapter's own words:

- *"The churn diagnosis is derived from the pattern, not measured."* — **Now measured.** 41.1% of TEST ground-truth mass on zero/low-support items against a preregistered ≥25% gate, with the 65.5% factor-model ceiling attached (this section, record `20260817T095926Z-633d454`).
- *"Popularity got a recency window and the CF arms did not."* — **Now measured, and discharged as a confound.** Both recency-matched arms were built, VAL-selected and given one TEST run each; the depth-level null survives (T8-2 addendum above).
- *"A shorter train-to-test gap could move it."* — **Direction now measured, magnitude not.** The same regime map on VAL (a 6-month-shorter gap) puts unreachable mass at 27.9% versus TEST's 41.1%, so a shorter gap demonstrably shrinks the structural handicap. How much of the *arm ranking* that would change was not evaluated, and this lab makes no claim about it.
- *"A category with slower catalog turnover could move it."* — **Now tested once, and it does move it.** The external contrast (T8-4, MovieLens-32M) was gated on T8-2 confirming the null and was formally skipped when T8-2 met the any-cell criterion (`EXPERIMENT_LOG.md`, 2026-08-18); the 2026-08-19 supersession downgraded that verdict and reopened the gate (UPGRADE_PLAN §8c). Phase 9 ran it. On a catalog with 6.40% churn against Amazon's 41.11%, the same ladder under the same harness produces a BH-corrected crossover at depth n\*=20 (§7b, record `20260820T134403Z-e2263d2` and `results/confirmatory_ml32m_test.json`). What that licenses is narrow and stated as such: it is one contrast dataset, on which domain, density, feedback and timestamp semantics all differ at once, whose win holds on NDCG@10 but not on the Recall@20 guard, and which loses catastrophically for cold users and on the global average. "The crossover exists only when the catalog holds still" is no longer untested — it is **supported on one measured contrast and not identified as causal by anything in this lab**.
- *"A model with access to post-cutoff signal could move it."* — **Still stands, untested.** Every arm here is TRAIN-frozen, and the 65.5% ceiling is a direct measurement of the size of what post-cutoff access would unlock. No arm in this lab has it.
- *"At the depths its users actually reach, the crossover does not occur."* — **Holds at history-depth granularity, everywhere it was tested.** T8-3 extended the depth axis to 20-49 / 50-99 / 100+ and found no crossover that survives a change of metric; what remains below depth granularity is the five-cell mechanism pocket above, held to footnote strength by the uncorrected-multiplicity disclosure.
- *"This is a statement about this regime, not about collaborative filtering."* — **Still stands, and Phase 9 is the proof of it rather than the exception to it.** One category, one snapshot, one train/test cutoff. Phase 8 replaced guesses about this regime with measurements of it; Phase 9 measured a second regime and got a different answer there (§7b), which is precisely what "a statement about this regime" means. Neither phase extends the Amazon claim beyond Amazon.

The one caveat that turned into an experiment is the catalog-turnover one, and it now has its own chapter: §7b runs this same ladder against a catalog that barely churns and reports what changes — and, more interestingly, what does not.

---

## 7. The routing policy that collapsed to a constant

The plan called for a history-depth router: blend below n\*, a stronger personalized arm above it, with n\* fitted on VAL by maximizing segment-weighted NDCG@10 (the unweighted mean over the five segment means, so each depth bucket counts equally). The rule and the grid were committed before the grid was run.

The VAL grid is 2 variants × 5 thresholds = 10 cells: variant A routes deep users to ALS, variant B routes them to pop-t12m, with n\* ∈ {1, 5, 10, 20, ∞}. Result: **every finite threshold scores worse than n\*=∞**, monotonically, in both variants. Variant A degrades fastest (objective 0.010706 at ∞ down to 0.005412 at n\*=1); variant B's descent is shallower but still strictly downward (0.010706 → 0.009816) **[measured]**. The winner is variant B at n\*=∞, which routes every single user to the blend — mechanically identical to running the blend alone.

That identity was then asserted, not assumed. A confirming VAL run of the hybrid recommender at n\*=null was compared per-user against the blend artifact: 356,362 users, set equality on user IDs, every metric column equal at `rtol=0 atol=0`, zero mismatches in the top-50 list column **[measured, run [`20260807T040118Z-5e212d7`](demo/index.html#receipt-20260807T040118Z-5e212d7)]**. The same identity reappeared on TEST — the hybrid arm's numbers are bit-identical to the blend's at every segment **[measured, run [`20260807T082125Z-c320c79`](demo/index.html#receipt-20260807T082125Z-c320c79)]**.

So the centerpiece finding is a constant function: n\*=∞, **the optimal routing is no routing**. That is a legitimate, decision-relevant result — it spares the segmentation infrastructure a production system would otherwise carry — and the exhibit that makes it legible is the counterfactual: what *would* the TEST metric have been at each threshold?

<!-- FIGURE 3 — policy grid (Checkpoint 1 figure list, item 2), two panels:
     (a) VAL objective by threshold for variants A and B, the 10 cells the n*
         selection actually ran on — the numbers quoted in prose above are its
         endpoints;
     (b) the TEST counterfactual table below, labelled [derived].
     Live counterpart: the n* slider inside demo #sec-crossover, where moving
     the threshold redraws the routed share (dashed underline: browser-side
     arithmetic over traced per-segment n_users) and the recomposed metric.
     Caption must repeat that (a) is the fitting evidence and (b) is a
     recomposition, never a fresh TEST run. -->

| n\* | share routed to blend | variant B global NDCG@10 |
|---:|---:|---:|
| 0 | 0.0% | 0.005404 |
| 5 | 39.4% | 0.005541 |
| 10 | 74.5% | 0.005639 |
| 20 | 91.4% | 0.005693 |
| ∞ | 100% | 0.005726 |

**[derived, record [`20260808T030659Z-43c90c8`](demo/index.html#receipt-20260808T030659Z-43c90c8), `kind=policy_grid`, `derived: true`]**

**[derived]** is this lab's fourth evidence class, an extension of the site's three-label measured / estimated / projected convention: a value **recomposed by arithmetic from values already recorded in `results/runs.jsonl`, with no new measurement** — no re-scoring, no refitting, no fresh consultation of ground truth. It is deliberately not "measured", because a badge-rendering site must never present a recomposition as a fresh TEST measurement.

The label is load-bearing here. This table is not a set of new TEST runs. It is an arithmetic recomposition of the per-user metric vectors already recorded by the one-shot TEST evals named in the record's `source_run_ids` (blend, ALS, pop-t12m), regrouped by the frozen segment edges — the frozen-TEST invariant is untouched, and n\* was selected on VAL only. The grid is monotone in blend coverage: more blend is better everywhere, which is the same statement as "no finite n\* exists," rendered as a slider the reader can move. (Two identical `policy_grid` records exist in the log: the first emit, [`20260808T025446Z-1f1ab07`](demo/index.html#receipt-20260808T025446Z-1f1ab07), carries `git_dirty: true`; the clean-tree re-emit cited above is the one the demo reads. Append-only means the first one stays.)

The `HybridRecommender` machinery survives in the codebase — built, unit-tested, VAL-validated, TEST-confirmed to compose exactly — exercising no non-trivial routing. Shipping it that way is the point: the policy the evidence supports is `content_pop_blend(alpha=0.3)`, unconditionally.

**What Phase 8 changes here: nothing, and that is a finding too.** Phase 8 produced a candidate reason to want a non-trivial router — a pocket of the catalog where popularity is structurally blind and a personalized arm nominally wins (§6; per-cell CIs exclude zero, but under ~80 uncorrected tests that is a mechanism observation, not confirmed signal). It also produced the reason that pocket could not be routed to even if it were confirmed: the cells are defined by the TRAIN support and TRAIN recency of the *item the user is about to buy*, and a serving-time router does not know that item. Routing on the winning cells would require a proxy — some observable, user-side signal that this shopper is working the stale, thinly-supported corner of the catalog — fitted on VAL and given its own preregistered TEST run. That is out of Phase 8's declared scope, and it is stated as an open lead rather than smuggled in as a result.

The n\*=∞ conclusion above is untouched by Phase 8. Neither recency-matched arm was ever routed, nothing was refitted, and the regime-map recomposition is a diagnostic over recorded per-user scores, not a policy fit. On the depth axis the collapse if anything hardens: both new arms lose to pop-t12m at every depth, and the single cell where one is nominally ahead — ALS-decay at `20+`, +0.00036 NDCG@10 — has a CI straddling zero and a significantly *negative* Recall@20. Routing on that would be fitting noise, which is precisely what the pre-committed two-metric rule exists to stop.

---

## 7b. The same ladder on a catalog that holds still

Everything above rests on a mechanism claim: personalization loses here because the catalog moves faster than a TRAIN-frozen model can follow. A claim of that shape has one obvious test — find a catalog that does not move, run the identical ladder against it, and see whether the crossover appears. Phase 8 wrote that test (T8-4), gated it on the null being confirmed, and then skipped it when the five-cell pocket looked like a win. The 2026-08-19 supersession removed the gate; Phase 9 ran the test on **MovieLens-32M**, preregistered in full — hypotheses, VAL selection rules, grids, multiplicity policy, and symmetric verdict rules D1–D5 — at commit `1731cef`, roughly seven hours before the first ML-32M TEST record existed. Both outcomes were pre-committed to publication: "a crossover appears when the catalog holds still" and "popularity dominates even there" were equally shippable before the numbers came in.

**The hinge was measured before any model existed.** The first number produced from ML-32M is the T8-1 churn statistic — the share of TEST ground-truth interactions on items with TRAIN support ≤ 4 — computed by the same imported code paths, on a frozen split (TRAIN ≤ 2022-06-30, VAL 2022-H2, TEST 2023-01-01 → 2023-11-01 exclusive, `configs/splits_ml32m.yaml`, frozen 2026-08-19), before a single model was trained: **zero+low = 6.40%** (zero 5.79% = 45,408 interactions, low 0.61% = 4,785) over 783,896 TEST ground-truth interactions from 8,843 TEST users against a 43,884-item catalog, versus Amazon's **41.11%** — a difference of −0.3471, a 6.4× regime gap **[measured, record `20260820T134403Z-e2263d2`, `kind=churn_contrast`, against `20260817T095926Z-633d454` re-derived from the log at run time]**. The recency axis says the same thing in the other direction: 91.0% of ML-32M's TEST mass lands on items that were active in TRAIN within 90 days of the cutoff, where Amazon's post-cutoff items — 5.19% of its catalog — absorbed 34.5% of TEST mass **[measured, same two records]**.

One piece of that gate is reported rather than smoothed over: the preregistered T8-1 bands are one-sided Amazon language, in which "<10% ⇒ churn diagnosis wrong, stop." The ML-32M run returns `<0.10` and prints exactly that verdict string. On this dataset the sub-10% reading is the *intended antecedent*, not a refutation — it is the catalog-holds-still regime the contrast was designed to find — and the reinterpretation is recorded in the log entry rather than patched into the code (`EXPERIMENT_LOG.md`, 2026-08-20).

### Two regimes, side by side

| | **Amazon Electronics** | **MovieLens-32M** |
|---|---|---|
| catalog churn (TEST GT mass on TRAIN-support ≤ 4) | **41.11%** (`20260817T095926Z-633d454`) | **6.40%** (`20260820T134403Z-e2263d2`) |
| density (5-core mean interactions/user) | 9.4 **[derived: 15,473,536 / 1,641,026, gold build `20260805T143256Z-7406fc1`]** | 159 **[derived: 31,921,467 / 200,948, T9-3a build receipts, `EXPERIMENT_LOG.md` 2026-08-20]** |
| gold catalog / TEST users | 368,228 items / 228,153 users | 43,884 items / 8,843 users |
| global TEST winner | blend α=0.3, NDCG@10 **0.005726** vs pop-t12m 0.005404 (`20260807T055333Z-c320c79`) | blend α=0.1, NDCG@10 **0.25088** vs pop-t12m 0.24751 (`20260820T221933Z-20d8ff9`) |
| best personalized arm, globally | ALS 0.002750 — loses to pop-t12m by ≈2× (`20260806T082441Z-2f2f26d`) | ALS 0.09037 ± 0.00009 across the frozen 3-seed set — loses to pop-t12m by ≈2.7× (primary-seed record `20260820T222202Z-20d8ff9`; stability seeds per the T9-3c verdict entry) |
| crossover on the history-depth axis | **none at any depth**; every personalized arm negative with CIs excluding zero, before and after recency matching | **crossover at n\* = 20**; BH-significant wins at 20-49 / 50-99 / 100+ |
| cold-start users | popularity strongest exactly where personalization is empty | same, harder: Δ −0.3844 at depth 0 |
| fitted routing policy | **n\*=∞ — route everyone to the blend, i.e. no routing** (`20260808T030659Z-43c90c8`) | fitted n\*=100 on VAL, objective 0.089686 against P\* alone at 0.212916 — routing away from popularity still hurts |

Absolute metric levels are not comparable across the two columns — different catalogs, different feedback semantics, different k-core survivors — for the same reason §10 forbids comparing these numbers to published figures. Only the within-dataset contrasts in each column carry meaning.

### The crossover, where it lives

Primary confirmatory family (preregistered): the VAL-selected personalized arm M\* = item-kNN-t12m (top_n 50, 365-day window, Rule S4) minus the VAL-selected popularity comparator P\* = pop-t12m (Rule S6), paired ΔNDCG@10 on TEST, Benjamini–Hochberg at FDR 0.05 across the eight tests in the family **[measured, `results/confirmatory_ml32m_test.json`, recomposed from the one-shot TEST records `20260820T221701Z-20d8ff9` (M\*) and `20260820T221055Z-20d8ff9` (P\*)]**:

| depth bucket | users | ΔNDCG@10 | 95% CI | q | BH verdict | ΔRecall@20 (§5g label only) |
|---|---:|---:|---|---:|---|---:|
| 0 | 3,882 | −0.3844 | [−0.3926, −0.3750] | 0.0053 | **significant loss** | −0.1001 (sig.) |
| 1-4 | 25 | −0.0140 | [−0.1016, +0.0656] | 0.879 | ns | −0.0169 (ns) |
| 5-9 | 19 | −0.0038 | [−0.1390, +0.1254] | 0.949 | ns | −0.0355 (ns) |
| 10-19 | 53 | −0.0439 | [−0.1003, +0.0071] | 0.141 | ns | +0.0020 (ns) |
| 20-49 | 242 | **+0.0333** | [+0.0107, +0.0552] | 0.0120 | **significant win** | +0.0228 (ns) |
| 50-99 | 431 | **+0.0198** | [+0.0034, +0.0381] | 0.0256 | **significant win** | +0.0097 (ns) |
| 100+ | 4,191 | **+0.0112** | [+0.0071, +0.0149] | 0.0053 | **significant win** | −0.0008 (ns, sign flips) |
| global | 8,843 | −0.1619 | [−0.1677, −0.1560] | 0.0053 | **significant loss** | −0.0434 (sig.) |

The shallowest bucket whose win is coherent with every bucket above it is `20-49`, so the preregistered D1 rule reads **n\* = 20**. Two secondary families corroborate the shape without setting n\*: ALS and ALS-decay each independently win the same three deep buckets against P\* and lose bucket 0 and the global test; static item-kNN and the content arm win nowhere **[measured, same file, family S1, BH per arm]**.

That is the whole finding of this chapter, and it is one sentence long: **same ladder, same harness, same protocol, 6.4× less churn — and a real crossover appears at n\* = 20.** On Amazon Electronics the personalized arms never catch popularity at any depth the dataset contains; on a catalog whose items are still there next year, they catch it at twenty interactions and stay ahead. The mechanism the Amazon null diagnosed now has a regime where its absence is measured, not assumed.

### The symmetry nobody predicted

The blend wins globally on **both** datasets. Amazon: 0.7·popularity + 0.3·content beats pop-t12m by +0.000322 [+0.000200, +0.000449], ≈+5.96% relative. ML-32M: α=0.1 (chosen by the same VAL rule from the same grid) beats P\* by **+0.00336 [+0.00268, +0.00403]**, BH-significant globally and in five buckets with zero losing buckets, ≈+1.4% relative **[measured, `results/confirmatory_ml32m_test.json`, family S1 `blend`, run `20260820T221933Z-20d8ff9` vs `20260820T221055Z-20d8ff9`]**. Two catalogs six-fold apart in churn, two independent VAL selections, and the same deployment answer: rank by what is currently popular, re-rank gently by content. What the churn regime changes is not the winner — it is whether a *tail* of deep-history users exists for whom a personalized arm is better, and how big that tail is. The regime flips the tail, not the head.

### What the policy actually is, in each regime

On the high-churn catalog the fitted policy is a constant: n\*=∞, every user to the blend, **the optimal routing is no routing** (§7). On the low-churn catalog a depth-20 gate is defensible in a way it never was on Amazon — the deep buckets hold 4,864 of 8,843 TEST users (55.0%) and the wins there are BH-significant — but the qualifications are not decoration:

- The gate buys **top-of-list ranking quality only**, not recall mass (below).
- It must never touch cold users. Depth 0 is 43.9% of ML-32M's TEST users and routing them to M\* costs −0.3844 NDCG@10 — an order of magnitude larger than every win above it. That single bucket is why the global delta is −0.1619 and why popularity still wins on average on the low-churn catalog too.
- The only routing policy this lab actually *fitted* on ML-32M lands at n\*=100 with VAL objective **0.089686**, against P\* alone at **0.212916** — the preregistered grid mirrors the Amazon grid's shape and therefore contains no pure-popularity cell, and the fitted cell's TEST global NDCG@10 is 0.08032 (`20260820T222041Z-20d8ff9`). Read plainly: even here, routing traffic *away* from popularity loses. The crossover depth n\*=20 and the fitted routing depth n\*=100 are two different quantities and are not interchangeable.

So the decision-relevant sentence is narrower than "stable catalogs justify routing." It is: **on a high-churn catalog, the optimal routing is no routing; on a stable catalog, a depth-20 gate earns its keep in top-of-list quality for the deep-history tail — and nowhere else, and never for cold users.** Anyone deploying against that sentence still owes their own catalog the churn measurement first; it is one cheap counting job and it is the number that decides whether the segmentation infrastructure is worth building at all.

### Four labels this result never travels without

1. **Not metric-robust (§5g).** Recall@20, corrected in its own BH family, confirms **none** of the three winning buckets: the sign agrees at 20-49 (+0.0228) and 50-99 (+0.0097) but neither is BH-significant, and at 100+ the delta is −0.0008 — the sign flips. The preregistered confirmatory criterion is BH-corrected NDCG@10, so D1 stands as written, but the win is a **top-of-list ranking-quality effect, not a recall-mass effect**, and every exhibit that cites the crossover carries this label **[measured, `results/confirmatory_ml32m_test.json`, `metric_robustness.metric_robust: false`]**.
2. **The cold-user loss is part of the finding, not a footnote to it (D4).** Bucket 0 and the global test are BH-significant *losses*. The verdict reports them in the same register as the wins because that is what the preregistration required of a D1 with significant negatives.
3. **Regime contrast, not causal proof.** The two datasets differ in domain, density (159 vs 9.4 mean interactions/user), catalog size, feedback semantics (explicit 0.5–5 ratings vs review-as-positive), and timestamp semantics — simultaneously. Churn is the axis that was *measured* on both sides; it is not the only axis that moved. Nothing here identifies churn as the cause, and no experiment in this lab holds the other axes fixed.
4. **The MovieLens timestamp caveat.** ML-32M timestamps are rating-*entry* times against a backfilled catalog (Sun et al., [arXiv:2307.09985](https://arxiv.org/abs/2307.09985)), so a temporal split cuts rating behavior rather than consumption or release — which mechanically dampens measured churn and is part of why 6.40% is small. The content arm's tag inputs were cutoff-filtered at `train_end` (preregistration §3a), which guards tag-time leakage but does nothing about the backfilled-metadata caveat.

<!-- FIGURE 6 — the both-regime crossover pair (Phase 9 T9-3c; not on the
     Checkpoint-1 list, added by T9-4). Committed static renders:
       results/figures/crossover_ml32m_test.{svg,png}       — 5-segment axis,
         frozen Phase 4 segment edges, drawn for Amazon-comparability only,
         NO BH correction (caption must say so);
       results/figures/crossover_ml32m_deep_test.{svg,png}  — the confirmatory
         deep-bucket deltas with BH markers and n* = 20 annotated.
     Amazon-side counterpart is Figure 1 (results/figures/crossover_test.svg).
     Each regime is drawn against its own VAL-selected popularity reference;
     both resolved to pop-t12m, which is why the two panels are readable side
     by side at all. Caption must repeat: absolute levels are not comparable
     across regimes. -->

![ML-32M: M* − P* by history depth, deep buckets, BH-marked, n*=20 annotated](results/figures/crossover_ml32m_deep_test.svg)

**Reproduction, and the two blemishes on the campaign.** `make reproduce-ml32m` re-derives both pinned records from Iceberg snapshot `3433604384732745693` with the frozen `configs/splits_ml32m.yaml` and `data/MANIFEST_ML32M.md`, compares record fields *and* per-user artifact arrays, and re-runs the confirmatory analysis to assert its verdict block is identical; it ran on the machine of record on 2026-08-21, exit 0, verdict `byte_exact`. By design it appends nothing to `results/runs.jsonl` — a reproduction is a check, not a new result. Two protocol blemishes are disclosed in the verdict entry rather than tidied away: three arm-evaluations carry duplicate record pairs (a dirty-tree guard refused a campaign's first 11 runs, but its ALS tail — whose training step precedes the guard — completed after the tree went clean), with the duplicates byte-identical in every metric and the canonical campaign pinned by run_id in `configs/confirmatory_ml32m_test.yaml`; and one run_id, `20260820T221701Z-20d8ff9`, names two records from same-second launches across the overlapping campaigns, resolved last-match-wins with the reproduce target pinning the intended config path explicitly. No arm was evaluated twice in any information sense: TEST was read once per arm.

---

## 8. Ops receipts

A number is only as good as its ability to survive the pipeline changing underneath it.

**Snapshot pinning and byte-exact reproduction, twice.** `make reproduce-headline` re-extracts the eval cache by Iceberg snapshot ID (time travel), re-scores `configs/eval_blend_test.yaml` after verifying the config's sha256 is unchanged, and compares every deterministic field of the resulting record against the original. First run: `verdict=byte_exact`, empty field diff, per-file cache sha256 equality, per-user parquet arrays identical **[measured, record [`20260807T153823Z-9a9fb4c`](demo/index.html#receipt-20260807T153823Z-9a9fb4c)]**. Then the warehouse was deliberately churned — roughly 40 new snapshots from backfill, appends, a MERGE upsert, fragmentation, compaction, and expiry — and it was run again: `byte_exact` a second time **[measured, record [`20260807T164622Z-3e2c665`](demo/index.html#receipt-20260807T164622Z-3e2c665)]**. The headline number cannot move while the catalog evolves. Demonstrated, not asserted.

<!-- FIGURE 4 — byte-exact reproduce receipts (Checkpoint 1 figure list, item 5).
     Side-by-side of the two reproduce records: verdict, field diff (empty),
     per-file cache sha256 equality, per-user parquet array equality, and the
     pinned snapshot ids — pre-churn vs post-churn.
     Live counterpart: demo #sec-lineage (the time-travel toggle showing the
     same eval pinned to an older snapshot) and the headline card in
     #sec-receipts, which carries its reproduction verdicts and the regenerating
     command. -->

**Ops exhibits, including a measured no-op.** Eleven `kind="ops"` records cover a monthly-partitioned backfill of 43,216,395 rows (source minus a deterministic 11,959-row late-arrival holdout, exact), three incremental monthly appends whose added-record counts equal their source months exactly, and a MERGE upsert that inserted precisely those 11,959 held-back rows and reconciled to 43,365,424 against the full silver slice **[measured]**. The compaction exhibit begins with a failure to find a problem: `rewrite_data_files` on the freshly built table rewrote **0 files**, because monthly-batch ingestion produces exactly one well-formed file per partition. That no-op is published rather than hidden, and the real exhibit is staged on top of it — one partition re-ingested as 30 daily slices (298 → 327 files), compacted back to one (327 → 298), then expired in two stages: `retain_last=2` freed only 3 data files because the retained pre-compaction snapshot still pinned the other 30; `retain_last=1` freed exactly those 30. Retention pins files; compaction alone frees nothing.

**Lineage, sourced from ledgers rather than prose.** `make lineage` emits a 24-stage table — raw download through bronze, silver, the gold funnel, the eval extract cache, the headline eval, both reproduce runs, and all eleven ops records — with every number pulled from a named machine ledger (build summaries, the k-core funnel table, Iceberg snapshot summaries, `runs.jsonl`), and a completeness check that fails the build rather than warning **[measured, record [`20260807T160910Z-739833b`](demo/index.html#receipt-20260807T160910Z-739833b)]**. Building it surfaced a discrepancy worth keeping: a hand-written wall-clock line in `data/MANIFEST.md` contradicts the machine ledger. It was not corrected, because the manifest's sha256 is part of every eval record and of the byte-exact reproduce comparison — editing it would permanently break reproduction. The prose line stands, superseded by a note. Immutability has costs, and this is what one looks like.

<!-- FIGURE 5 — the 24-stage lineage table (Checkpoint 1 figure list, item 4).
     Source: results/lineage.md — per stage: layer, table, rows in/out, bytes,
     wall clock, Iceberg snapshot id, and the machine ledger each field came
     from. Render the ledger-source column; it is the point of the figure.
     Live counterpart: demo #sec-lineage. -->

*The full table, with the snapshot-id chain and the time-travel toggle: the [pipeline lineage panel](demo/index.html#sec-lineage).*

**One more under-expectation, reported as-is.** The demo's ANN index (hnswlib, M=16, ef_construction=200 over 368,228 MiniLM vectors) was receipted against exact brute-force top-10 on 10,000 seeded users: mean overlap **0.9472** at `ef_search=200`, just under the informally expected 0.95 **[measured, record [`20260807T090857Z-97af81f`](demo/index.html#receipt-20260807T090857Z-97af81f)]** — recorded at the value it came in at, with no extra tuning, and never used to produce a single evaluation metric (`used_in_eval_metrics: false`). It powers the [live semantic-search exhibit](demo/index.html#sec-search) and nothing else.

**A curation rule that was arithmetically impossible, aborted rather than relaxed.** The pick-a-shopper exhibit needed 6 users per segment. The pre-declared v1 rule asked for uniform draws re-drawn until at least 2 of 6 had a blend top-10 hit — and it failed loudly, exactly as written, after exhausting all 50 attempts on segment `1-4`. The reason was arithmetic, not luck: blend's recorded per-segment HitRate@10 is 0.0403 / 0.0197 / 0.0147 / 0.0152 / 0.0151, so at p ≈ 0.02 a uniform draw of 6 contains ≥2 hits about 6 times in 1,000 **[measured]**. Raising the attempt cap would have let the seed, not the rule, pick the shoppers. The rule was superseded in the log by a stratified draw — exactly 2 from the hit stratum and 4 from the miss stratum — declared before it ran, and the exhibit discloses the over-sampling next to the real per-segment hit rate, because 2-in-6 where the truth is roughly 1-in-50 is a ~50× distortion that would misrepresent the model if it were hidden. The v1 predicate was written as if hits were common. They are not, and that is the finding the exhibit exists to show.

**Scale caveat, stated once and meant.** Every ops number above is a single-node, single-writer, local-catalog measurement. The code avoids driver-side collection of large tables and is cluster-portable by construction, so distributed behavior is **[projected]**, not measured. That is the only projected claim in this case study.

---

## 9. How this was verified

- **Frozen dataset manifest.** Source URLs, download date (2026-08-05), byte sizes, and locally computed SHA-256s for both raw files in `data/MANIFEST.md`, plus a published-count reconciliation table (observed 43,886,944 / 1,610,012 against the release's rounded 43.9M / 1.61M, delta stated explicitly rather than glossed). **[measured]**
- **Locked environment and stated hardware.** Python 3.12 with a committed `uv.lock` (`uv sync --locked`), `pyspark==4.0.4`, Iceberg runtime `1.11.0`, and a project-local JDK 21 pin applied in the Makefile and CI but never globally. Hardware stated plainly: Apple M4, 10 cores, 16GB RAM, Spark `local[10]` with an ~8g driver. **[measured]**
- **Deterministic build, snapshot-pinned reproduction.** Two independent `make data` rebuilds produced 8/8 content-identical tables; `make reproduce-headline` returned `byte_exact` twice, once before and once after the ops churn. **[measured]**
- **One config per run, append-only log with git SHAs.** 81 records in `results/runs.jsonl` across evals, paired deltas, ANN receipt, ops, lineage, reproduce, the DQ export, the DuckDB bench, the derived policy grid, and Phase 8's regime maps and deep-bucket recomposition — each carrying run_id, git SHA and dirty flag, config path and hash, dataset manifest hash, and Iceberg snapshot IDs. **[measured]**
- **Bootstrap CIs and paired deltas.** Every headline number carries a user-bootstrap 95% CI (1,000 resamples, seed 20260805); every comparison is a paired bootstrap over common users on a shared resample matrix, claimed only where the CI excludes zero. **[measured]**
- **Full-catalog ranking, stated explicitly.** All 368,228 gold-catalog items scored per user with TRAIN-seen items masked; no sampled negatives anywhere in this lab. **[measured]**
- **Metric math unit-tested against an independent reference.** `tests/test_metrics.py` checks hand-computed micro-cases and fuzzes *every* public metric function against a naive full-`argsort` reference implementation over 50 seeded instances, plus edge cases (all-zero score rows, k > catalog, single-item catalog) and hand-verified coverage/novelty/Gini values. **[measured]**
- **Contract engine and quarantine ledger with exact reconciliation.** Seven YAML contracts drive a PySpark checker writing to a `dq_results` Iceberg table with violating rows routed to quarantine; the raw→bronze→silver→gold waterfall reconciles exactly on every edge, enforced in code with a non-zero exit on drift. **[measured]**
- **Embedding artifact versioned, with the machine it ran on.** `sentence-transformers/all-MiniLM-L6-v2` at HF revision `1110a243…`, recipe `v1_title_brand_cat_features` (recipe_hash `1f7878ff82bf`), 368,228 × 384 fp16, sha256 recorded, computed on the same M4 via MPS in 2,115s; alignment re-verified by recomputing the export parquet and item-ID sequence hashes rather than trusting the manifest. **[measured]**
- **Experiment log published, failures included.** `EXPERIMENT_LOG.md` carries dated hypothesis → result → verdict entries including the rejected kNN neighbor-list hypothesis, the ALS negative, the content-alone rejection, the collapsed router, the sub-expectation ANN overlap, the aborted shopper-curation rule, and Phase 8's failed prediction that time-decayed ALS would win the stale-item cells. **[measured]**
- **Phase 8's thresholds, arms and decision rules were registered before the runs, and the gate was executed as written.** The regime-map bands (<10% wrong / 10–25% partial / ≥25% supported), the half-life grid {90, 365, 1460}, the VAL selection rule, the deep-bucket boundaries, and the two-metric crossover criterion were all committed ahead of the outcomes they judge — the T8-2 preregistration at 2026-08-17T11:45Z, before the arms' code existed and ~13h before the first T8-2 record. Both outcomes were pre-committed to ship. The same preregistration made T8-4 (a MovieLens-32M external contrast) conditional on T8-2 confirming the null; T8-2 met the crossover criterion, so the gate closed and T8-4 was formally skipped by the rule as written, with the cost of skipping it recorded (`EXPERIMENT_LOG.md`, 2026-08-18). On 2026-08-19 that verdict was downgraded by a superseding log entry — local regime-cell wins under uncorrected multiplicity; the global history-depth crossover null is robust — and the T8-4 gate was reopened as approved scope with a preregistered multiplicity policy (UPGRADE_PLAN §8c; `EXPERIMENT_LOG.md`, 2026-08-19). Supersession, not edit: every prior entry and every record in `results/runs.jsonl` stands byte-identical. **[measured]**
- **Multiplicity disclosed rather than corrected — and the verdict downgraded accordingly.** The five winning regime-map cells come from roughly 40 cells × 2 arms ≈ 80 uncorrected tests per metric under the preregistered any-cell rule; at α=0.05, about 4 false positives are expected by chance alone. The disclosure is on the claim itself (§6), not buried: each winning cell is 0.7–2.2% of TEST ground-truth mass, the cells overlap across axes, and the supporting argument is coherence — two highly correlated metrics agreeing in every cell, three consecutive depth bands — which is partial evidence against pure chance, not a correction. This arithmetic is why the 2026-08-19 supersession reads the cells as a mechanism footnote rather than a crossover finding. **[measured]**
- **The machine-of-record migration was verified by rebuild, not by assertion.** Phase 8's model runs moved from the 16GB MacBook to a rented 16-vCPU/120GB Linux box. The box was qualified by a full fresh rebuild from the SHA-256-verified raw bytes — every waterfall count and all 17 k-core funnel iterations identical to the 2026-08-05 Mac build (bronze 43,886,944 / 1,610,012; silver 43,365,424; gold 15,473,536 × 1,641,026 × 368,228), contracts green, the same 2 quarantined rows. The one cross-platform difference found is stated rather than smoothed: an x86-OpenBLAS GEMM result differs from Apple Accelerate by 1 float32 ulp, so counts are identical while floats may drift in the last bit — which is exactly why no T8-2 comparison mixes machines, and why the log records that zero T8-2 records existed on the Mac before the move. **[measured]**
- **The one lineage exception is single-use, digest-gated, and published with its digests.** Crossing the Mac-era pop-t12m comparator with the box's rebuilt cache trips the regime map's snapshot-lineage guard, correctly, because the rebuild produced new Iceberg snapshot IDs for verified-identical data. Rather than relaxing the guard, a config-declared equivalence block scoped to exactly one arm, one run_id and one directed snapshot pair is honored only after every other arm matches normally (an unused exception is itself an error), the cell-axis parquet's raw-byte sha256 matches on **both** machines (`72a71aee…`), the committed T8-1 reference record confirms that sha and the comparator's artifact sha, and the transferred comparator artifact re-hashes to the sha the T8-1 record committed. The exception's verbatim declaration and every digest are written into the output records, 13 unit tests cover tampering and misuse, and the live end-to-end receipt is the identity anchor still reproducing the comparator's recorded per-user metrics to **1.11e-16** across the migration **[measured, records [`20260818T072256Z-3f3530a`](demo/index.html#receipt-20260818T072256Z-3f3530a), [`20260818T072211Z-3f3530a`](demo/index.html#receipt-20260818T072211Z-3f3530a)]**.
- **The ML-32M contrast was preregistered before it could see TEST, and the hinge statistic was computed before it could see a model.** The T9-3b preregistration — hypotheses, arms, VAL grids, selection rules S1–S6, the multiplicity policy, and symmetric verdict rules D1–D5 — was committed at `1731cef` on 2026-08-20, about seven hours before the first ML-32M TEST record (`20260820T220854Z-20d8ff9`) and before any ML-32M model artifact existed. Earlier still, the churn statistic that the whole contrast hinges on was produced at the data stage, before any model was trained (`20260820T134403Z-e2263d2`). Ordering is checkable by commit clock and record timestamp, not by prose. **[measured]**
- **Multiplicity corrected this time, not merely disclosed.** The Phase 9 primary family is the per-depth-bucket crossover on the history axis — eight tests, Benjamini–Hochberg at FDR 0.05, deterministic tie-break, with the two-sided bootstrap ASL floored at 2/1001 and reported as a floor rather than as "< 0.001". Secondary arm families are BH-corrected separately, and the five-segment Amazon-comparability axis is labelled comparability-only with no correction and no confirmatory standing. The `metric_robust: false` label on the winning buckets is carried by the verdict itself, not left to the reader. **[measured, `results/confirmatory_ml32m_test.json`]**
- **The ML-32M lane got the same pipeline treatment, not a shortcut.** Its own contracts (`gold_ml32m_*`, 58 checks, overall PASS, zero quarantined rows), its own frozen split file hashed into every record, its own SHA-256 manifest in a separate `data/MANIFEST_ML32M.md` — kept separate precisely because `data/MANIFEST.md`'s whole-file hash is a compared field of the pinned Amazon headline, so appending to it would have flipped `make reproduce-headline` from `byte_exact` to mismatch. The deviation from the plan's letter is recorded in the log with the invariant it protects. **[measured, `EXPERIMENT_LOG.md` 2026-08-20; record `20260820T134403Z-e2263d2`]**
- **A second snapshot-pinned reproduce target, byte-exact on the machine of record.** `make reproduce-ml32m` re-derives both pinned ML-32M records from Iceberg snapshot `3433604384732745693` — record fields *and* per-user artifact arrays — and re-runs the confirmatory analysis to assert an identical verdict block; run 2026-08-21, exit 0, verdict `byte_exact`. It writes no results-log record by design. **[measured]**
- **CI smoke on a bundled fixture.** GitHub Actions pins Java 21, installs from the lockfile, and runs the full pytest suite — including a fixture pipeline test and an end-to-end eval-harness smoke test — over a committed, deterministically sampled ~50k-row bronze fixture (regenerable byte-identically via `make fixture`, since the sampling predicate is a content hash). 376 tests passed at the Phase 6 tree. **[measured]**
- **The demo is static and offline, scanned and then proven.** `make demo-offline-check` scans the assembled tree for external URLs in executable positions: **CLEAN**. It found one real violation on its first run — a marketing URL inside a product title in `items_meta.json` — fixed at the source (the exporter now strips URL substrings from display titles; no evidence value touched). **[measured]**
- **Offline proven at runtime, not just by scan.** The fully assembled site was driven in headless Chrome 151 on a clean profile with DNS black-holed (`--host-resolver-rules="MAP * ~NOTFOUND, EXCLUDE 127.0.0.1"`) and a netlog captured: **27 of 27 page-initiated requests were loopback** with bodies returned, all eight data files loaded, zero placeholder sections; the only non-loopback attempts were Chrome-internal endpoints, every one black-holed with zero bytes transferred. Live semantic search was then activated and loaded **44.6MB of runtime and model weights** (21.6MB ONNX Runtime WASM + 23.0MB quantized MiniLM) plus the int8 payload **from 127.0.0.1 only**, ran inference in-tab, and produced an int8-vs-Python quantization-parity receipt of **overlap@10 = 9/10**, inside the expected 8–10 band (export-side mean 9.83/10). The run earned its keep by finding a defect a static scan cannot see: the vendored `transformers.web.min.js` dynamically imports bare specifiers no browser can resolve without an import map, so activation threw. It was replaced with the fully-bundled build from the same hash-pinned tarball. **[measured]**
- **Every displayed number re-resolved independently of the writer.** `make demo-verify` re-checks **4,617 trace-manifest entries covering 6,756 numeric leaves across 28 run_ids** — coverage, exact match (same type, no epsilon), document hashes, artifact-hash-vs-record agreement, and a staleness guard on the results log's own hash — and every leaf re-resolves exactly. The verifier reads the log itself, not the exporter's state, so a number the exporter got wrong cannot pass. `make demo-verify-record` is the CI mode. **[measured]**
- **The receipts drawer is the mechanism, not a claim about one.** Every number on the site is rendered through `getTraced()`, which carries the `run_id` it was copied from; clicking a dotted number opens that record's card — config path and hash, git SHA and dirty flag, dataset manifest hash, frozen-splits hash, Iceberg snapshot IDs, seeds, model params, wall clock, hardware. Values the browser computes from traced components are drawn **dashed** instead of dotted and name their components' record. An untraced number is impossible by construction: the exporter writes leaves through a `TracedWriter`. See [Receipts](demo/index.html#sec-receipts). **[measured]**
- **Fetched assets are pinned by hash, not by URL.** The two uncommitted asset trees (79.5MB: 31.9MB search payload, 47.6MB vendored runtime and weights) are regenerated by `make demo-assets`, which verifies every fetched file against a SHA-256 recorded in `demo/README.md` — that table is machine-parsed, so a file the script wants but the table does not name is itself a hard failure. Any mismatch, including a truncated download, exits non-zero with no partial install; the correct response is to investigate the drift, never to update the hash. Model pins are commit-pinned, never `main`, and the delivery CDN is irrelevant because the hash gate is what is trusted. **[measured]**

---

## 10. What this does not prove

- **No online evidence.** Every number here is offline. A ranking win of +0.000322 NDCG@10 does not establish a CTR, conversion, or revenue lift, and nothing in this lab is an A/B result.
- **The feedback is missing-not-at-random and popularity-biased.** Users review what they were shown, and what they were shown was already popularity-influenced. No counterfactual correction (IPS, doubly-robust, or otherwise) is applied. A protocol whose winner is a popularity variant is exactly the protocol where this bias matters most, and that tension is not resolved here.
- **Reviews are not purchases.** "One review = one positive interaction" is a modeling assumption. Reviewed items skew toward memorable experiences at both ends; unreviewed purchases are invisible.
- **k-core filtering inflates absolute metrics — measured at ×1.20 globally for the popularity baseline.** The 5-core funnel discards 27.9M of 43.4M silver interactions and flattens the long tail. A pre-declared, one-shot un-cored TEST run of the same pop-t12m protocol — all silver users, full 1,609,860-item catalog, 1,374,880 TEST users — scored NDCG@10 0.004513 vs 0.005404 on the 5-core universe: a **[derived]** protocol-matched ratio of ×1.20 (Recall@20 ×1.08), and ×1.28–1.59 within matched history-depth segments, largest for strict-cold users (records `20260809T160227Z-5c70b7c` vs `20260805T172047Z-035042b`). The ratio compares disjoint universes and is not a paired delta; it is measured for the popularity arm only. Compare arms *within* one protocol; do not compare these absolute numbers to published figures from other protocols.
- **Single category, single snapshot.** Amazon Electronics only, one download whose data ends 2023-09. No claim of generalization to other categories, and no freshness claim beyond that snapshot.
- **Not a serving system.** There is no service, no latency SLA, no throughput target. The ANN latency figures are an artifact receipt for a static demo, not a production benchmark, and the exact-scoring "latency" quoted alongside them is amortized batch throughput, explicitly not comparable as a speedup ratio.
- **Single-node Spark.** Distributed-scale behavior is **[projected]** from cluster-portable code, never measured. There is no cluster in this story.
- **The Iceberg ops exhibits are single-writer, local-catalog scenarios.** No concurrent writers, no commit contention, no catalog service, no object store. They demonstrate the semantics, not production operation.
- **Routing thresholds are fitted to this dataset.** The finding that no finite n\* helps is a finding about Amazon Electronics with a mid-2022 cutoff. It transfers nowhere without re-measurement — as §7b demonstrates by re-measuring: on ML-32M the same fitting machinery returns a different answer (a crossover at depth 20, and a fitted routing depth of 100 that still loses to pure popularity). Two datasets, two thresholds, no transferable constant.
- **The five-cell pocket is a mechanism footnote, not a measured crossover.** Five regime-map cells beat popularity with per-cell CIs excluding zero on both metrics; each is 0.7–2.2% of TEST ground-truth mass with per-cell deltas in the fourth decimal, and the winning arm is globally the weakest in the lab (TEST NDCG@10 0.000301). Roughly 40 cells × 2 arms ≈ 80 tests were run under the preregistered any-cell rule with **no multiple-comparison correction** — at α=0.05, about 4 false positives are expected by chance — and coherence across two highly correlated metrics and adjacent depth bands is partial evidence, not a correction. The cells are keyed to the TRAIN support and recency of the ground-truth item, which no serving-time router can observe. This is a diagnosis of where popularity goes blind, not a crossover finding, and not a policy anyone can deploy from these numbers (superseding verdict: `EXPERIMENT_LOG.md`, 2026-08-19).
- **One external contrast exists now, and it is a contrast, not a causal test.** T8-4 (MovieLens-32M) ran in Phase 9 under a preregistered multiplicity policy, and it produced a BH-corrected crossover at n\*=20 on a catalog with 6.40% churn against Amazon's 41.11% (§7b). Four limits sit on that result and none of them is optional: the two datasets differ in domain, density, catalog size, feedback semantics and timestamp semantics **simultaneously**, so churn is the axis that was measured, not the axis that was isolated — no experiment here holds the others fixed; the win is on BH-corrected NDCG@10 only and the Recall@20 guard confirms **none** of the three winning buckets, so it is a top-of-list ranking effect, not a recall-mass effect; the same arm loses BH-significantly at depth 0 (−0.3844, 43.9% of TEST users) and globally (−0.1619), so popularity still wins on average even on the stable catalog; and MovieLens timestamps are rating-entry times on a backfilled catalog (Sun et al., arXiv:2307.09985), which mechanically dampens measured churn. "The crossover exists only when the catalog holds still" is supported by one measured contrast and proved by none. Every Amazon number remains a single-dataset, single-cutoff measurement, and so does every ML-32M number.
- **The demo's search scores are a capability demo, not evaluation evidence.** The cosine similarities in the live-search exhibit have no held-out interactions, no full-catalog ranking protocol, and no bootstrap CI behind them. They are the only numbers on the site with no results-log record, and they are deliberately not drawn with the traced-number affordance.
- **The win is small.** +5.96% relative NDCG@10 with CI [+0.000200, +0.000449] is a real, cleanly separated effect and a modest one. Nothing here argues that a semantic re-rank transforms a recommender; it argues that it measurably helps, and that measuring it properly is the harder half.
