# Crossover Study

> **DRAFT — pending Checkpoint 2 (assembled demo review); final copy will be revised after demo verification.**

> **"How much history does a user need before personalization beats popularity?"** Crossover Study ingests 43.9M Amazon Electronics reviews through a contract-checked Spark + Iceberg lakehouse on a single 16GB laptop, evaluates popularity, item-kNN, ALS, and semantic-content retrieval under temporal splits with full-catalog ranking and bootstrap CIs — and measures the answer: **no amount.** Pure personalization never beats recency-weighted popularity at any observed depth; the only arm that wins is popularity itself, gently re-ranked by a normalized semantic-content blend — and it wins everywhere, by a small margin (+6% relative NDCG@10) that 1,000-resample paired CIs cleanly separate from zero.

**Reading the labels.** Every claim below carries an evidence class: **[measured]** — produced by a recorded run in `results/runs.jsonl`; **[derived]** — arithmetic recomposition of values already recorded, no new scoring; **[estimated]** — inferred, with the inference stated; **[projected]** — reasoned from code properties, not observed. Where a claim has a receipt, the run_id, record kind, or Make target is named inline; those become links in the receipts drawer.

---

## 1. The question, and the answer the data gave

The plan for this lab predicted a crossover. Popularity should win for users with no history, personalization should take over once a user has accumulated enough signal, and somewhere in between there is a threshold — call it n\* — that a real system could route on. The signature exhibit was going to be NDCG@10 by user-history depth, one line per model, with the crossing point marked.

Half of that prediction held. Recency-weighted popularity does decay with history depth: NDCG@10 falls from 0.007505 in the strict-cold segment to 0.003711 for users with 20+ TRAIN interactions **[measured, run `20260805T172047Z-035042b`]**.

The other half did not happen. Item-kNN loses to trailing-12-month popularity in every segment. ALS — tuned over a ten-point VAL grid, run to the hardware feasibility frontier of a 16GB machine — loses to it in every segment too. Pure semantic-content retrieval loses to it by a factor of six. The ALS deficit *shrinks* monotonically as history deepens, from −0.0026 in the 1–4 bucket to −0.0015 at 20+ **[measured]** — the crossover's direction is real and visible — but it never reaches zero within the depths this dataset contains.

What did win is the least glamorous thing on the list: trailing-12-month popularity, re-ranked by a MiniLM content score at a weight of 0.3. On the frozen TEST split it scores NDCG@10 **0.005726 [0.005524, 0.005948]** against popularity's **0.005404 [0.005209, 0.005635]** **[measured, runs `20260807T055333Z-c320c79` and `20260805T172047Z-035042b`]**. The paired delta is **+0.000322 [+0.000200, +0.000449]**, CI excluding zero **[measured, `kind=paired_delta`, 1,000 resamples, seed 20260805]**. That is +5.96% relative. It is a small win, and it is stated as small.

So the honest headline is a negative finding with a modest positive attached: on 2023 Amazon Electronics, with training frozen at 2022-06-30, no personalization arm tested here beats knowing what is currently popular. Content similarity earns its place only as a gentle re-ranker on top of that. The machinery below — lakehouse, contracts, protocol — is what makes the negative trustworthy enough to publish.

---

## 2. Provenance

**Provenance.** This lab grew out of a five-person course project (CISC3018 Cloud Computing and Big Data Systems, University of Macau, Fall 2025) that built a recommendation bot on a 1M-interaction sample of this dataset; my primary individual contribution there was the presentation material. Everything in this lab — pipeline, contracts, models, evaluation harness, demo — was designed and built from the raw public dataset, solo, with its own verification chain.

Nothing from that project ships here, and none of its code, models, or results is presented as individual work or reproduced in this write-up. Two design notes were used as reading material only: a metadata schema shape and the idea of embedding item text with a small sentence transformer. Both were rebuilt from scratch against the raw files, because the archived sample had no timestamp column — which makes temporal evaluation, the entire premise of this lab, impossible from it. Separately, individually-authored Hadoop/HBase/MapReduce coursework is where the batch-processing fundamentals came from; those reports are not republished here either.

The dataset is Amazon Reviews 2023 (McAuley Lab, UCSD), used under its research-use terms; the raw files are never redistributed. Cite Hou et al., 2024, *Bridging Language and Items for Retrieval and Recommendation*.

---

## 3. A lakehouse on a laptop

The scale claim is the point of the ingestion layer, so it is stated with its ledger attached. Two gzipped JSONL files — 6.47GB of reviews, 1.31GB of metadata, both SHA-256'd into `data/MANIFEST.md` at download time — become Iceberg tables on a local Hadoop catalog: **43,886,944 reviews and 1,610,012 items** in bronze **[measured]**. The published release rounds to 43.9M / 1.61M; the manifest records the observed counts as canonical and the delta against the rounded figures explicitly, rather than asserting a match.

Every row that leaves the pipeline leaves with a reason. Bronze → silver drops 521,520 rows: **2** quarantined for rating-domain violations, **477,968** exact duplicates, **43,550** superseded (user,item) re-reviews where only the latest survives. Silver → gold prunes **27,891,888** rows through iterative 5-core filtering, landing at **15,473,536** modeling interactions **[measured, run `20260805T143256Z-7406fc1`]**. The waterfall's arithmetic is enforced, not printed: for every edge, the sum of reason-rows equals the source count *and* the kept count equals the live Iceberg table count re-read at publish time, with a non-zero exit on drift. Items pass through clean — 1,610,012 in, 1,610,012 out, no quarantine loss.

The k-core loop converged at **iteration 16**, taking users from 18,286,190 to 1,641,026 and items from 1,609,860 to 368,228 **[measured]**. Nearly all of the reduction happens in the first seven iterations; the last eight shave under 200 rows in total. That 368,228-item survivor set is the full-catalog ranking universe for every metric in this case study.

Seven YAML contracts (one per silver/gold table) drive a hand-rolled PySpark checker that writes pass/fail and violation counts to a `dq_results` Iceberg table and routes violating rows to quarantine tables with a reason column. Interactions→items orphan rate: **0.0** (0 of 43,365,424) **[measured]**. Unparseable prices: 316 rows, 0.0196%. Unknown-brand share: **4.545%**, against an expectation of roughly 18% carried in from the planning notes — because a Manufacturer-field fallback recovers 23.90% of rows that would otherwise be branded "Unknown" **[measured]**. The expectation was wrong; the measurement is what's published.

Two independent `make data` rebuilds produced **8/8 content-identical tables**, ~27.5 minutes each, on `local[10]` with an ~8g driver on an Apple M4 with 16GB of RAM **[measured]**. Single-node, stated plainly.

---

## 4. The evaluation protocol as the product

The models in this lab are deliberately standard. The evaluation is not, and that is where the engineering went.

**Temporal splits, frozen in a file.** TRAIN ≤ 2022-06-30, VAL 2022-07-01 → 2022-12-31, TEST 2023-01-01 → 2023-10-01 (exclusive), frozen in `configs/splits.yaml` on 2026-08-05 and hashed into every run record. Random splits leak future interactions backward into training and flatter every personalized model; a recommender that will be asked about tomorrow should be evaluated on tomorrow. VAL carries 356,362 users; TEST carries 228,153 **[measured]**.

**TEST is touched once per model.** Before every TEST campaign, a pre-declaration was written and committed to `docs/engineering-log/EXPERIMENT_LOG.md` *first*: the arms, the comparisons, the acceptance gate, and an explicit no-return clause — whatever TEST shows, there is no going back to the VAL grid. Two such pre-declarations exist (Phase 3's T7 for ALS, Phase 4's T15 for content/blend/hybrid), both timestamped ahead of their results **[measured, `docs/engineering-log/EXPERIMENT_LOG.md`]**. This is the part of the protocol that makes a negative finding worth reading.

**Full catalog, no sampled negatives.** Every user is scored against all 368,228 gold-catalog items via chunked matmul, with TRAIN-seen items masked out. No exhibit in this lab samples negatives. The ANN index built in Phase 4 (chapter 8) exists for the demo only and is flagged `used_in_eval_metrics: false` on its own record.

**CIs on everything, paired deltas for every comparison.** Headline numbers carry user-bootstrap 95% CIs (1,000 resamples, seed 20260805); every comparative claim is a paired bootstrap over common users on a shared resample matrix, and a difference is only claimed where the CI excludes zero. Metrics reported per run: Recall@10/20/50, NDCG@10/20, MRR, HitRate@10, plus coverage, popularity share, and novelty.

**One config per run, append-only results.** Each run is a single YAML config; each result is one JSON line in `results/runs.jsonl` carrying config hash, git SHA, dirty flag, dataset manifest hash, Iceberg snapshot IDs, seeds, and a per-user artifact path. The log is append-only and committed: a wrong run gets a superseding entry, never an edit. Stochastic arms (ALS) report 3-seed mean±sd on top of the CIs — VAL sd 0.0000568, TEST sd 0.0000197, both roughly 3× smaller than the bootstrap CI half-width, which is the evidence that seed variance is not driving any conclusion **[measured]**.

---

## 5. The ladder of challengers

Each rung is a TEST number with a receipt. All are global NDCG@10 on the frozen TEST split.

| arm | NDCG@10 | run_id |
|---|---:|---|
| random (seed 13) | 0.000012 | `20260805T165630Z-2dcea79` |
| content only (MiniLM) | 0.000886 | `20260807T050054Z-c320c79` |
| popularity, all-time | 0.000832 | `20260805T173018Z-035042b` |
| item-kNN (top_n=50) | 0.000946 | `20260805T185305Z-adbca99` |
| ALS (rank 128, primary seed) | 0.002750 | `20260806T082441Z-2f2f26d` |
| popularity per-category, t12m | 0.004683 | `20260805T173520Z-035042b` |
| **popularity, trailing 12m** | **0.005404** | `20260805T172047Z-035042b` |
| **blend α=0.3 (pop × content)** | **0.005726** | `20260807T055333Z-c320c79` |

**[all measured]**

**Popularity is not one baseline, it is a design space.** All-time popularity scores 0.000832; windowing it to the trailing 12 months takes it to 0.005404 — a paired delta of **+0.0046 NDCG@10 [+0.0044, +0.0048]**, roughly 6× **[measured]**. Catalog churn dominates this category, and a "popularity baseline" that ignores recency is a straw man. That single decision is worth more than every personalization arm tested here.

**Item-kNN failed.** Cosine co-occurrence, neighbor-list grid {50, 100, 200} on VAL: flat to three decimals, so the cheapest point won. On TEST it scores 0.000946 and loses to pop-t12m in *every* segment, including 20+ (0.0006 vs 0.0037), global paired delta −0.0045 [−0.0047, −0.0042] **[measured]**. Strict-cold users score exactly zero, as they must — no history, no co-occurrence.

**ALS failed at the hardware frontier, and the frontier is stated.** Ten single-variable VAL experiments (E1–E10: rank 32/64/128, reg over two orders of magnitude, α ∈ {1, 10, 40}, iterations 8/25, binary vs rating confidence) selected rank=128, reg=0.01, α=10, iter=25, binary. Two caveats were pre-registered *before* TEST: VAL quality was still rising in both rank and iterations, so this is the 16GB feasibility frontier and not a converged optimum (peak RSS 9.54GB); and the shallow-beats-deep segment gradient held at all ten grid points. On TEST, ALS − pop-t12m is negative in every segment with every CI excluding zero: −0.0075 cold, −0.0026 at 1–4, −0.0023 at 5–9, −0.0026 at 10–19, −0.0015 at 20+ **[measured]**. ALS does beat item-kNN everywhere warm (+0.0018 global) — it is the best classical CF representation tried here, and it still loses.

**Content alone failed badly.** MiniLM `all-MiniLM-L6-v2` embeddings over a `title + brand + category + features` recipe, user profile = mean-pooled L2-normalized TRAIN item vectors, scored exactly against the full catalog. TEST NDCG@10 0.000886, losing to pop-t12m in every segment including the shallow-warm buckets the hypothesis specifically named (1–4: −0.0044; 5–9: −0.0044, both CIs excluding zero) **[measured]**. Strict-cold is exactly 0.0 by construction: no TRAIN items, no profile.

**The blend won.** A normalized 0.7·popularity + 0.3·content score — alpha chosen on VAL by a rule written down before the grid ran, from {0.1, 0.3, 0.5, 0.7, 0.9}, with an interior peak at 0.3 whose CI does not overlap either neighbor's — beats pop-t12m on TEST globally (+0.000322 [+0.000200, +0.000449]) and in all four warm segments, every CI excluding zero **[measured]**. In the strict-cold segment the delta is exactly +0.000000: the blend degenerates to pure popularity for users with no content profile, which is correct behavior, not a miss. The semantic signal that could not rank anything on its own is still useful as a tie-breaker on top of a signal that can.

---

## 6. The crossover that never came

Per-segment TEST NDCG@10, five history-depth buckets **[measured]**:

| arm | 0 (cold) | 1–4 | 5–9 | 10–19 | 20+ |
|---|---:|---:|---:|---:|---:|
| pop-t12m | 0.007505 | 0.005821 | 0.005225 | 0.005107 | 0.003711 |
| blend α=0.3 | 0.007505 | 0.006227 | 0.005504 | 0.005426 | 0.004095 |
| ALS | 0.000000 | 0.003259 | 0.002925 | 0.002541 | 0.002254 |
| content | 0.000000 | 0.001458 | 0.000809 | 0.000492 | 0.000307 |

Two things are visible. First, popularity's advantage genuinely erodes with depth — it is nearly twice as strong for cold users as for the 20+ bucket. Second, nothing catches it. The ALS-minus-popularity deficit narrows monotonically (−0.0026 → −0.0015 as depth increases) but the sign never flips, and the CI at 20+ is [−0.0021, −0.0008]: still comfortably below zero **[measured]**.

The mechanism is consistent across all three failed arms and was diagnosed the same way each time. TEST ground truth is 2023 purchases; TRAIN ends mid-2022. A large share of 2023 interactions land on items that barely existed in the training window, and therefore carry almost no co-occurrence mass and no learned factor structure. A trailing-12-month popularity window partially tracks that churn; a TRAIN-frozen co-occurrence matrix or factor matrix structurally cannot. Content embeddings are churn-agnostic — a new HDMI cable looks like an old HDMI cable — which is precisely why content helps as a re-ranker while failing as a ranker.

This is a statement about *this* regime, not about collaborative filtering. A shorter train-to-test gap, a category with slower catalog turnover, or a model with access to post-cutoff signal could all move it. What this lab can say is that in the regime it measured, at the depths its users actually reach, the crossover does not occur.

---

## 7. The routing policy that collapsed to a constant

The plan called for a history-depth router: blend below n\*, a stronger personalized arm above it, with n\* fitted on VAL by maximizing segment-weighted NDCG@10 (the unweighted mean over the five segment means, so each depth bucket counts equally). The rule and the grid were committed before the grid was run.

The VAL grid is 2 variants × 5 thresholds = 10 cells: variant A routes deep users to ALS, variant B routes them to pop-t12m, with n\* ∈ {1, 5, 10, 20, ∞}. Result: **every finite threshold scores worse than n\*=∞**, monotonically, in both variants. Variant A degrades fastest (objective 0.010706 at ∞ down to 0.005412 at n\*=1); variant B's descent is shallower but still strictly downward (0.010706 → 0.009816) **[measured]**. The winner is variant B at n\*=∞, which routes every single user to the blend — mechanically identical to running the blend alone.

That identity was then asserted, not assumed. A confirming VAL run of the hybrid recommender at n\*=null was compared per-user against the blend artifact: 356,362 users, set equality on user IDs, every metric column equal at `rtol=0 atol=0`, zero mismatches in the top-50 list column **[measured, run `20260807T040118Z-5e212d7`]**. The same identity reappeared on TEST — the hybrid arm's numbers are bit-identical to the blend's at every segment.

So the centerpiece finding is a constant function. That is a legitimate result, and the exhibit that makes it legible is the counterfactual: what *would* the TEST metric have been at each threshold?

| n\* | share routed to blend | variant B global NDCG@10 |
|---:|---:|---:|
| 0 | 0.0% | 0.005404 |
| 5 | 39.4% | 0.005541 |
| 10 | 74.5% | 0.005639 |
| 20 | 91.4% | 0.005693 |
| ∞ | 100% | 0.005726 |

**[derived, record `20260808T025446Z-1f1ab07`, `kind=policy_grid`, `derived: true`]**

This table is labeled **derived** and the label is load-bearing. It is not a set of new TEST runs. It is an arithmetic recomposition of the per-user metric vectors already recorded by the one-shot TEST evals named in the record's `source_run_ids` (blend, ALS, pop-t12m), regrouped by the frozen segment edges. No re-scoring, no refitting, no fresh consultation of TEST ground truth — the frozen-TEST invariant is untouched, and n\* was selected on VAL only. The grid is monotone in blend coverage: more blend is better everywhere, which is the same statement as "no finite n\* exists," rendered as a slider the reader can move.

The `HybridRecommender` machinery survives in the codebase — built, unit-tested, VAL-validated, TEST-confirmed to compose exactly — exercising no non-trivial routing. Shipping it that way is the point: the policy the evidence supports is `content_pop_blend(alpha=0.3)`, unconditionally.

---

## 8. Ops receipts

A number is only as good as its ability to survive the pipeline changing underneath it.

**Snapshot pinning and byte-exact reproduction, twice.** `make reproduce-headline` re-extracts the eval cache by Iceberg snapshot ID (time travel), re-scores `configs/eval_blend_test.yaml` after verifying the config's sha256 is unchanged, and compares every deterministic field of the resulting record against the original. First run: `verdict=byte_exact`, empty field diff, per-file cache sha256 equality, per-user parquet arrays identical **[measured, record `20260807T153823Z-9a9fb4c`]**. Then the warehouse was deliberately churned — roughly 40 new snapshots from backfill, appends, a MERGE upsert, fragmentation, compaction, and expiry — and it was run again: `byte_exact` a second time **[measured, record `20260807T164622Z-3e2c665`]**. The headline number cannot move while the catalog evolves. Demonstrated, not asserted.

**Ops exhibits, including a measured no-op.** Eleven `kind="ops"` records cover a monthly-partitioned backfill of 43,216,395 rows (source minus a deterministic 11,959-row late-arrival holdout, exact), three incremental monthly appends whose added-record counts equal their source months exactly, and a MERGE upsert that inserted precisely those 11,959 held-back rows and reconciled to 43,365,424 against the full silver slice **[measured]**. The compaction exhibit begins with a failure to find a problem: `rewrite_data_files` on the freshly built table rewrote **0 files**, because monthly-batch ingestion produces exactly one well-formed file per partition. That no-op is published rather than hidden, and the real exhibit is staged on top of it — one partition re-ingested as 30 daily slices (298 → 327 files), compacted back to one (327 → 298), then expired in two stages: `retain_last=2` freed only 3 data files because the retained pre-compaction snapshot still pinned the other 30; `retain_last=1` freed exactly those 30. Retention pins files; compaction alone frees nothing.

**Lineage, sourced from ledgers rather than prose.** `make lineage` emits a 24-stage table — raw download through bronze, silver, the gold funnel, the eval extract cache, the headline eval, both reproduce runs, and all eleven ops records — with every number pulled from a named machine ledger (build summaries, the k-core funnel table, Iceberg snapshot summaries, `runs.jsonl`), and a completeness check that fails the build rather than warning **[measured, record `20260807T160910Z-739833b`]**. Building it surfaced a discrepancy worth keeping: a hand-written wall-clock line in `data/MANIFEST.md` contradicts the machine ledger. It was not corrected, because the manifest's sha256 is part of every eval record and of the byte-exact reproduce comparison — editing it would permanently break reproduction. The prose line stands, superseded by a note. Immutability has costs, and this is what one looks like.

**One more under-expectation, reported as-is.** The demo's ANN index (hnswlib, M=16, ef_construction=200 over 368,228 MiniLM vectors) was receipted against exact brute-force top-10 on 10,000 seeded users: mean overlap **0.9472** at `ef_search=200`, just under the informally expected 0.95 **[measured, record `20260807T090857Z-97af81f`]** — recorded at the value it came in at, with no extra tuning, and never used to produce a single evaluation metric (`used_in_eval_metrics: false`).

**Scale caveat, stated once and meant.** Every ops number above is a single-node, single-writer, local-catalog measurement. The code avoids driver-side collection of large tables and is cluster-portable by construction, so distributed behavior is **[projected]**, not measured. That is the only projected claim in this case study.

---

## 9. How this was verified

- **Frozen dataset manifest.** Source URLs, download date (2026-08-05), byte sizes, and locally computed SHA-256s for both raw files in `data/MANIFEST.md`, plus a published-count reconciliation table (observed 43,886,944 / 1,610,012 against the release's rounded 43.9M / 1.61M, delta stated explicitly rather than glossed). **[measured]**
- **Locked environment and stated hardware.** Python 3.12 with a committed `uv.lock` (`uv sync --locked`), `pyspark==4.0.4`, Iceberg runtime `1.11.0`, and a project-local JDK 21 pin applied in the Makefile and CI but never globally. Hardware stated plainly: Apple M4, 10 cores, 16GB RAM, Spark `local[10]` with an ~8g driver. **[measured]**
- **Deterministic build, snapshot-pinned reproduction.** Two independent `make data` rebuilds produced 8/8 content-identical tables; `make reproduce-headline` returned `byte_exact` twice, once before and once after the ops churn. **[measured]**
- **One config per run, append-only log with git SHAs.** 58 records in `results/runs.jsonl` across evals, paired deltas, ANN receipt, ops, lineage, reproduce, and the derived policy grid — each carrying run_id, git SHA and dirty flag, config path and hash, dataset manifest hash, and Iceberg snapshot IDs. **[measured]**
- **Bootstrap CIs and paired deltas.** Every headline number carries a user-bootstrap 95% CI (1,000 resamples, seed 20260805); every comparison is a paired bootstrap over common users on a shared resample matrix, claimed only where the CI excludes zero. **[measured]**
- **Full-catalog ranking, stated explicitly.** All 368,228 gold-catalog items scored per user with TRAIN-seen items masked; no sampled negatives anywhere in this lab. **[measured]**
- **Metric math unit-tested against an independent reference.** `tests/test_metrics.py` checks hand-computed micro-cases and fuzzes *every* public metric function against a naive full-`argsort` reference implementation over 50 seeded instances, plus edge cases (all-zero score rows, k > catalog, single-item catalog) and hand-verified coverage/novelty/Gini values. **[measured]**
- **Contract engine and quarantine ledger with exact reconciliation.** Seven YAML contracts drive a PySpark checker writing to a `dq_results` Iceberg table with violating rows routed to quarantine; the raw→bronze→silver→gold waterfall reconciles exactly on every edge, enforced in code with a non-zero exit on drift. **[measured]**
- **Embedding artifact versioned, with the machine it ran on.** `sentence-transformers/all-MiniLM-L6-v2` at HF revision `1110a243…`, recipe `v1_title_brand_cat_features` (recipe_hash `1f7878ff82bf`), 368,228 × 384 fp16, sha256 recorded, computed on the same M4 via MPS in 2,115s; alignment re-verified by recomputing the export parquet and item-ID sequence hashes rather than trusting the manifest. **[measured]**
- **Experiment log published, failures included.** `docs/engineering-log/EXPERIMENT_LOG.md` carries dated hypothesis → result → verdict entries including the rejected kNN neighbor-list hypothesis, the ALS negative, the content-alone rejection, the collapsed router, and the sub-expectation ANN overlap. **[measured]**
- **CI smoke on a bundled fixture.** GitHub Actions pins Java 21, installs from the lockfile, and runs the full pytest suite — including a fixture pipeline test and an end-to-end eval-harness smoke test — over a committed, deterministically sampled ~50k-row bronze fixture (regenerable byte-identically via `make fixture`, since the sampling predicate is a content hash). 249 tests passed at the phase-final tree. **[measured]**

---

## 10. What this does not prove

- **No online evidence.** Every number here is offline. A ranking win of +0.000322 NDCG@10 does not establish a CTR, conversion, or revenue lift, and nothing in this lab is an A/B result.
- **The feedback is missing-not-at-random and popularity-biased.** Users review what they were shown, and what they were shown was already popularity-influenced. No counterfactual correction (IPS, doubly-robust, or otherwise) is applied. A protocol whose winner is a popularity variant is exactly the protocol where this bias matters most, and that tension is not resolved here.
- **Reviews are not purchases.** "One review = one positive interaction" is a modeling assumption. Reviewed items skew toward memorable experiences at both ends; unreviewed purchases are invisible.
- **k-core filtering inflates absolute metrics.** The 5-core funnel discards 27.9M of 43.4M silver interactions and flattens the long tail, which raises all absolute numbers. **This inflation has not been quantified** — the planned un-cored popularity comparison is a stretch item and has not been run. Compare arms *within* this protocol; do not compare these absolute numbers to published figures from other protocols.
- **Single category, single snapshot.** Amazon Electronics only, one download whose data ends 2023-09. No claim of generalization to other categories, and no freshness claim beyond that snapshot.
- **Not a serving system.** There is no service, no latency SLA, no throughput target. The ANN latency figures are an artifact receipt for a static demo, not a production benchmark, and the exact-scoring "latency" quoted alongside them is amortized batch throughput, explicitly not comparable as a speedup ratio.
- **Single-node Spark.** Distributed-scale behavior is **[projected]** from cluster-portable code, never measured. There is no cluster in this story.
- **The Iceberg ops exhibits are single-writer, local-catalog scenarios.** No concurrent writers, no commit contention, no catalog service, no object store. They demonstrate the semantics, not production operation.
- **Routing thresholds are fitted to this dataset.** The finding that no finite n\* helps is a finding about Amazon Electronics with a mid-2022 cutoff. It transfers nowhere without re-measurement.
- **The win is small.** +5.96% relative NDCG@10 with CI [+0.000200, +0.000449] is a real, cleanly separated effect and a modest one. Nothing here argues that a semantic re-rank transforms a recommender; it argues that it measurably helps, and that measuring it properly is the harder half.
