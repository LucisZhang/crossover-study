# Experiment Log

Dated entries tracking pipeline runs, acceptance checks, and notable findings for the
batch-recsys-lab silver/gold/contracts pipeline.

---

## Phase 1 acceptance — silver+gold+contracts (2026-08-05)

Full-scale acceptance run (`run_id 20260805T143256Z-7406fc1`, second of two independent
`make data` rebuilds; first build `run_id 20260805T135836Z-7406fc1` same day).

### Reconciliation waterfall — exact on all edges

**Reviews:**

| stage | rows | delta |
|---|---:|---:|
| raw | 43,886,944 | — |
| silver (post-gate) | 43,365,424 | −521,520 (2 quarantined rating_domain, 477,968 exact duplicates, 43,550 superseded) |
| gold 5-core | 15,473,536 | −27,891,888 (k-core filtering) |

**Items:** 1,610,012 clean through (raw → silver, no quarantine loss).

k-core converged at **iteration 16**: users 18,286,190 → 1,641,026; items 1,609,860 →
368,228. Convergence has a long tail — iterations 8–16 shave fewer than 200 rows total
across both sides, i.e. nearly all of the reduction happens in the first ~7 iterations.

### Determinism

8/8 tables content-identical across two independent `make data` rebuilds (~27.5 min
each).

### Contracts

All contracts pass; `run_audit` exit code 0. Querying `local.dq.dq_results` for the
latest run (`20260805T143256Z-7406fc1`, latest row per `(table_name, check_id)`) shows
**zero rows with `status == 'fail'`** — the only `fail` rows in the ledger belong to
earlier/intermediate audit passes on the same run_id and on the prior run_id
(`20260805T135836Z-7406fc1`, `20260805T121746Z-7406fc1`), superseded by later
re-assertions with `status` downgraded to `measured` (T8 nullable-marking note on
Iceberg `createOrReplace`).

### DQ metrics (latest run, `local.dq.dq_results`)

| metric | value |
|---|---|
| orphan_rate (interactions → items FK, `item_fk`) | 0.0 (0 / 43,365,424) |
| brand_unknown_share | 4.545% (73,178 / 1,610,012) |
| price_unparseable | 316 rows, 0.0196% (316 / 1,610,012) |
| rating_domain violations (gate, silver ingest) | 2 (out of 43,886,944 raw rows; matches the 2 quarantined rows in the waterfall above) |
| rating_domain violations (post-gate, silver/gold) | 0 |

**Brand source share** (`brand_source_share` details JSON, `local.silver.items`):

| source | rows | share of 1,610,012 |
|---|---:|---:|
| Brand field | 1,153,897 | 71.67% |
| Manufacturer fallback | 384,785 | 23.90% |
| Neither (none) | 71,330 | 4.43% |

**Measured vs expected unknown-brand share:** UPGRADE_PLAN.md §7 (and the raw-hazards
note in §"Known raw-data hazards") flags ~18% unknown-brand share as the expectation for
this category. Measured `brand_unknown_share` is **4.545%**, well below the ~18%
expectation — the Manufacturer fallback (23.90% of rows) recovers most of the brand
signal that would otherwise show up as unknown, so the *residual* unknown share after
fallback is much lower than the raw/no-fallback expectation.

### Gold table sizes

| table | rows |
|---|---:|
| user_stats | 1,641,026 |
| item_features | 368,228 |
| popularity | 1,223,106 |

### MANIFEST

`data/MANIFEST.md` contains the `## Reconciliation waterfall` section with the funnel
(verified present, not rewritten here).

### Splits

Frozen splits decision: defaults, per `docs/volume_by_month.md`.

## Phase 2 — item-kNN neighbor-truncation selection on VAL (2026-08-06)

**Hypothesis:** larger neighbor lists (`top_n`) improve item-kNN ranking quality;
grid `top_n ∈ {50, 100, 200}` (cosine co-occurrence, shrinkage 0, TRAIN-only fit),
selected by VAL NDCG@10 per the pre-declared rule. TEST untouched during selection.

**Result** (VAL, 356,362 users, full-catalog ranking, run_ids in `results/runs.jsonl`):

| top_n | NDCG@10 | Recall@20 | MRR | wall |
|---:|---:|---:|---:|---:|
| 50 | 0.001680 | 0.004229 | 0.002341 | 527s |
| 100 | 0.001672 | 0.004270 | 0.002341 | 581s |
| 200 | 0.001671 | 0.004314 | 0.002345 | 655s |

**Verdict:** hypothesis rejected — quality is flat in `top_n` (differences ≪ CI width;
95% CI on NDCG@10 is ±0.0001 for all three). Selection metric VAL NDCG@10 picks
**top_n = 50** (also cheapest to build and score). `configs/eval_itemknn_test.yaml`
finalized to top_n 50 for the single TEST run. Note: strict-cold segment (n_train=0)
scores exactly 0 for kNN as expected — no TRAIN history means zero co-occurrence signal.

## Phase 2 acceptance — eval harness + baselines (2026-08-06)

All runs: TRAIN-only knowledge cutoff, full-catalog ranking (368,228 items,
TRAIN-seen excluded), user-bootstrap 95% CIs (1,000 resamples, seed 20260805),
5 history-depth segments. 10 records in `results/runs.jsonl` (5 TEST evals,
3 VAL kNN-grid evals, 2 paired deltas). TEST touched once per model.

**TEST headline (228,153 users):**

| model | Recall@20 | NDCG@10 |
|---|---:|---:|
| random (seed 13) | 0.0001 | 0.0000 |
| popularity all-time | 0.0035 | 0.0008 |
| item-kNN (top_n=50) | 0.0025 | 0.0009 |
| popularity per-category (t12m) | 0.0142 | 0.0047 |
| **popularity trailing-12m** | **0.0178** | **0.0054** |

**Acceptance criterion 3 — windowing delta (paired bootstrap, CI excludes 0 on all
7 metrics):** trailing-12m − all-time = **+0.0046 NDCG@10 [+0.0044, +0.0048]**,
+0.0143 Recall@20 [+0.0139, +0.0148]. A 12-month window is ~6× better than all-time
popularity on Electronics — catalog churn dominates.

**Failed hypothesis (logged, not discarded):** expected item-kNN to beat popularity
in deep-history segments (early crossover signal). It does not — kNN loses to
trailing-12m popularity in EVERY segment (20+: 0.0006 vs 0.0037 NDCG@10; paired
delta −0.0045 NDCG@10 [−0.0047, −0.0042] globally). Plausible mechanism: 2023 TEST
purchases concentrate on items released after train_end with little/no TRAIN
co-occurrence mass, which recency-windowed popularity partially tracks and static
co-occurrence cannot. Popularity NDCG@10 declines with history depth (0.0075 cold →
0.0037 at 20+), so the "popularity stops working for deep users" half of the
crossover thesis IS visible; the "personalization takes over" half now rests on
ALS/content (Phase 3+).

## Phase 3 — implicit ALS grid on VAL (2026-08-06)

**Setup.** Spark MLlib ALS (`implicitPrefs=True`), trained on the eval cache's TRAIN
pairs (14,206,658 pairs, 1,641,026 users × 368,228 items, five_core snapshot
8184397443787800955). Factors persisted as npy artifacts keyed by param hash;
rescoring from an artifact is bit-deterministic (sha256s in each run record) while
Spark retraining is not bit-stable — seed + params recorded, 3-seed sd bounds
stochastic variance. Coordinate-descent single-variable sweep, anchor
rank=64 / reg_param=0.01 / alpha=10 / max_iter=15 / weighting=binary / seed=20260805.
**Pre-declared selection rule:** VAL NDCG@10; candidates within 0.0001 → cheaper
config (smaller rank, fewer iters, binary). TEST untouched throughout.

### E2 — anchor, rank=64 (run_id 20260805T201636Z-f95dbbc)

**Hypothesis:** ALS at moderate rank materially beats item-kNN on VAL (kNN VAL
NDCG@10 = 0.00168) and gives the grid a live anchor.
**Result:** VAL NDCG@10 **0.00382** [0.00367, 0.00396]; Recall@20 0.01050;
wall 896s. Per-segment NDCG@10: 1-4: 0.00446, 5-9: 0.00423, 10-19: 0.00319,
20+: 0.00253, cold: 0 (by construction).
**Verdict:** confirmed — 2.3× kNN on VAL. Note the segment gradient is *inverted*
(shallow-history users score higher than deep-history), the same depth-decline
popularity shows; flagged for the TEST segment analysis.

### E1 — rank=32 (run_id 20260805T203950Z-f95dbbc)

**Hypothesis:** halving rank from the anchor loses ranking quality (capacity-bound
regime, not overfit-bound).
**Result:** VAL NDCG@10 **0.00349** [0.00336, 0.00362]; Recall@20 0.01022; wall 783s.
Segments 1-4: 0.00417, 5-9: 0.00382, 10-19: 0.00290, 20+: 0.00216.
**Verdict:** confirmed — rank 64 leads by 0.00033 (> 0.0001 tie band). Same inverted
segment gradient as E2.

### E3 — rank=128, attempt 1: FAILED (disk), retried

First attempt died in-train: `java.io.IOException: No space left on device` during
shuffle write (~24GB free on disk). Mechanism: ALS solve shuffles ≈ n_pairs × rank
× 4B ≈ 7.3GB/iteration at rank 128, and shuffle files are reclaimed only when
checkpointing truncates lineage — at checkpointInterval=5 that retains ~36GB.
Fix: checkpoint_interval exposed as a training-infra knob (numerically neutral,
excluded from the param hash by design) and set to 2 for this config, bounding
retained shuffle at ~15GB. Partial 0-byte artifact removed. Retry below.

### E3 — rank=128, attempt 2 (run_id 20260805T210841Z-c82a35f)

**Hypothesis:** doubling rank from 64 keeps improving VAL ranking (still
capacity-bound at 64).
**Result:** VAL NDCG@10 **0.00415** [0.00400, 0.00430]; Recall@20 0.01039;
wall 1034s with checkpoint_interval=2 (disk peak stayed under budget, ~22GB free
after). Segments 1-4: 0.00479, 5-9: 0.00458, 10-19: 0.00346, 20+: 0.00298.
**Verdict:** confirmed — rank 128 leads rank 64 by 0.00033 (> 0.0001 tie band).
**Rank axis winner: 128.** Model remains capacity-bound at the largest feasible
rank on this hardware; inverted segment gradient persists.

### E4 — reg_param=0.001 (run_id 20260805T215103Z-c82a35f)

**Hypothesis:** at rank 128 with implicit confidence weighting, the anchor
reg 0.01 over-regularizes; lighter reg improves VAL NDCG@10.
**Result:** VAL NDCG@10 **0.00412** [0.00398, 0.00427]; wall 1045s.
**Verdict:** rejected — 0.00003 below the anchor's 0.00415, inside the 0.0001
tie band. Reg axis insensitive downward.

### E5 — reg_param=0.1 (run_id 20260805T223316Z-c82a35f)

**Hypothesis:** heavier reg helps generalization on the sparse 5-core matrix.
**Result:** VAL NDCG@10 **0.00417** [0.00402, 0.00432]; wall 1034s.
**Verdict:** tie — 0.00002 above the anchor, inside the 0.0001 band. Per the
pre-declared rule the incumbent stands. **Reg axis winner: 0.01 (flat axis;
quality insensitive to reg over two orders of magnitude).**

### E6 — alpha=1.0 (run_id 20260805T231554Z-c82a35f)

**Hypothesis:** with deduped binary interactions the anchor alpha=10 over-weights
observed pairs; alpha=1 rebalances.
**Result:** VAL NDCG@10 **0.00324** [0.00311, 0.00338]; Recall@20 0.00818; wall 877s.
**Verdict:** rejected decisively — 0.00091 below the anchor. Observed-pair
confidence needs to dominate the unobserved prior on this sparsity.

### E7 — alpha=40.0 (run_id 20260805T235633Z-c82a35f)

**Hypothesis:** pushing confidence higher keeps helping (monotone in alpha).
**Result:** VAL NDCG@10 **0.00411** [0.00396, 0.00425]; Recall@20 0.01100
(best recall of the grid); wall 1401s (+35% vs anchor).
**Verdict:** tie on the selection metric — 0.00004 below anchor, inside the 0.0001
band; incumbent stands and is cheaper. **Alpha axis winner: 10.** Noted: alpha=40
trades NDCG-neutral for a real Recall@20 gain — flagged, not selected (selection
rule is NDCG@10, pre-declared).

### E8 — max_iter=8 (run_id 20260806T024118Z-acd1f81)

**Hypothesis:** ALS converges early on this sparsity; 8 iterations suffice.
**Result:** VAL NDCG@10 **0.00380** [0.00366, 0.00394]; Recall@20 0.00946;
train 1919s (checkpoint_interval=1), eval 1174s, peak RSS 9.85GB
(/usr/bin/time -l, covers Spark train + numpy eval).
**Verdict:** rejected — 0.00035 below the iter=15 anchor (> 0.0001 band); not yet
converged at 8. Ops note: per-point background execution with logs/grid_E8.log +
RSS capture adopted after two external SIGTERMs killed chained E8+E9 attempts;
timeout hypothesis examined and not supported (all runs were background; longer
chains completed) — source unidentified, protections retained.

### E9 — max_iter=25 (run_id 20260806T033333Z-acd1f81)

**Hypothesis:** 15 iterations under-converges at rank 128; more iterations help.
**Result:** VAL NDCG@10 **0.0042542** vs anchor 0.0041470 — delta 0.000107,
just outside the 0.0001 tie band. Recall@20 0.01070; train 2696s
(checkpoint_interval=1), peak RSS 9.48GB. Deep segments gain most
(20+: 0.00325 vs 0.00298; 10-19: 0.00364 vs 0.00346).
**Verdict:** confirmed (narrowly) — **iteration axis winner: 25.** Consistent
with capacity/convergence-bound: both larger rank and more iterations keep
paying on this matrix.

### E10 — weighting=rating (run_id 20260806T043802Z-acd1f81)

**Hypothesis:** star-rating magnitude carries preference-confidence signal beyond
mere presence; c = 1 + alpha*r (r in 1..5) beats binary c = 1 + alpha.
**Result:** VAL NDCG@10 **0.0041710** [0.00402, 0.00432] vs binary-E9 0.0042542 —
delta −0.0000832, within the 0.0001 tie band. Recall@20 0.01158 (best of the
whole grid). Train+eval 4137s total, peak RSS 9.27GB.
**Verdict:** tie on the selection metric — the simpler binary weighting stands per
the pre-declared rule. Rating magnitude adds no NDCG@10 signal on 5-core
Electronics (presence is the signal); its Recall@20 edge is flagged alongside
alpha=40's, not selected.

### Grid conclusion — chosen VAL config

**rank=128, reg_param=0.01, alpha=10, max_iter=25, weighting=binary,
seed=20260805** (= E9, run_id 20260806T033333Z-acd1f81): VAL NDCG@10 0.0042542.
10 single-variable entries (E1–E10) above, each with a runs.jsonl run_id.
Quality was capacity/convergence-bound (rank and iterations both paid);
reg flat over two orders of magnitude; confidence variants NDCG-neutral.

### T6 — 3-seed VAL variance check (chosen config)

Two pre-registered caveats carried into the acceptance record:
1. "VAL quality was still rising at rank 128 / iter 25 — this config is the
   hardware-feasibility frontier on 16GB, not a converged optimum"
2. The shallow>deep segment gradient (NDCG@10 monotonically declining from the
   1-4 bucket through 20+) held across all 10 grid points — pre-registered here,
   before TEST, as the pattern to examine in the segment analysis.

Seeds 20260806 / 20260807 on rank=128, reg=0.01, alpha=10, max_iter=25, binary
(seed 20260805 = E9 already recorded). Runs below.

**T6 result — seed variance (VAL, chosen config; 3 seeds):**

| seed | run_id | NDCG@10 | Recall@20 |
|---|---|---|---|
| 20260805 | 20260806T033333Z-acd1f81 | 0.0042542 | 0.0107023 |
| 20260806 | 20260806T055339Z-de2000b | 0.0042440 | 0.0109077 |
| 20260807 | 20260806T070001Z-de2000b | 0.0041511 | 0.0108160 |

NDCG@10 **mean 0.0042164 ± sd 0.0000568** (1.35% of mean);
Recall@20 **mean 0.0108087 ± sd 0.0001029** (0.95% of mean).
Seed sd is ~2.7× smaller than the per-run bootstrap CI half-width (0.000151):
stochastic (init/parallelism) variance is subordinate to user-sampling
uncertainty and does not threaten grid-selection conclusions at the observed
axis deltas (rank: 0.00033; iter: 0.000107 — the iter margin is ~2 sd, thin
but the selection stands per the pre-declared point-estimate rule).
Peak RSS across the two new runs: 9.54 / 8.29 GB; both exit 0, clean tails.

### T7 — TEST protocol pre-declaration (written and committed BEFORE any TEST run)

1. **One-shot TEST.** Whatever TEST shows, there is no returning to the VAL grid
   for another config afterward; the result is published as-is, win or negative.
2. **Seed handling for paired deltas:** the seed **20260805** TEST run is the
   single arm for all per-user paired-bootstrap deltas (vs pop-t12m and vs
   item-kNN). Seeds 20260806 / 20260807 TEST runs are reported only as 3-seed
   mean±sd stability evidence and enter no delta computation.
3. Comparisons: ALS-vs-pop-t12m (acceptance gate: warm segments 5-9 / 10-19 /
   20+, NDCG@10 CI excluding zero) and ALS-vs-item-kNN (secondary). Arms:
   pop-t12m TEST run 20260805T172047Z-035042b; item-kNN TEST run
   20260805T185305Z-adbca99. Bootstrap: 1000 resamples, seed 20260805, same
   resample matrix both arms (eval.compare).

### T7/T8 — TEST results and Phase-3 acceptance (2026-08-06)

**TEST, chosen config (rank=128, reg=0.01, alpha=10, max_iter=25, binary), 3 seeds:**

| seed | run_id | NDCG@10 | Recall@20 |
|---|---|---|---|
| 20260805 (primary) | 20260806T082441Z-2f2f26d | 0.0027501 | 0.0072037 |
| 20260806 | 20260806T084638Z-2f2f26d | 0.0027574 | 0.0070509 |
| 20260807 | 20260806T085913Z-2f2f26d | 0.0027202 | 0.0070388 |

TEST NDCG@10 **mean 0.0027426 ± sd 0.0000197**; Recall@20 **0.0070978 ± 0.0000919**.

**Paired deltas (primary seed arm, 1000 resamples, seed 20260805, n=228,153 common users):**

ALS − pop-t12m, NDCG@10 per segment — the acceptance gate:

| segment | delta | 95% CI | excludes zero |
|---|---|---|---|
| 0 (cold) | −0.00751 | [−0.00830, −0.00675] | yes |
| 1–4 | −0.00256 | [−0.00299, −0.00216] | yes |
| 5–9 | −0.00230 | [−0.00269, −0.00188] | yes |
| 10–19 | −0.00257 | [−0.00313, −0.00199] | yes |
| 20+ | −0.00146 | [−0.00209, −0.00079] | yes |

Global: NDCG@10 −0.0027 [−0.0029, −0.0024]; all seven metrics negative, all CIs
excluding zero.

ALS − item-kNN: positive in every warm segment with CIs excluding zero
(NDCG@10 global +0.0018 [+0.0016, +0.0020]); segment 0 identically zero
(neither model can score strict-cold users).

**Acceptance verdict (negative published, per the pre-declared accept clause):**
ALS does NOT beat trailing-12m popularity on any warm segment; the deficit is
significant everywhere. This extends the Phase-2 kNN finding to the strongest
classical CF model at the hardware-feasibility frontier: on 2023 Electronics
TEST with TRAIN ending 2022-06-30, recency-windowed popularity dominates
learned static co-occurrence structure at every history depth. Mechanism
(consistent with Phase 2): catalog churn — the 2023 ground truth concentrates
on items whose popularity is recent, which the t12m window tracks and
TRAIN-frozen factor structure cannot. Two effects worth the case study's
attention: (1) ALS clearly beats kNN everywhere warm — it is the best CF
representation tried; (2) the ALS-vs-pop deficit *shrinks* monotonically with
history depth (−0.0026 shallow → −0.0015 at 20+) while pop's absolute quality
also declines with depth — the crossover *direction* exists, but it does not
reach zero within observed depths. Personalization's remaining hope in this
lab: content retrieval (Phase 4) and the routing policy over segments.

**Reproducibility:** all runs seeded and recorded; factor artifacts persisted
with sha256 receipts echoed into runs.jsonl records; Spark ALS retraining is
not bit-stable (float reduction order) — reproducibility is claimed via the
persisted artifacts plus recorded seed/params, with 3-seed sd (VAL 0.0000568,
TEST 0.0000197) bounding stochastic variance. Caveats pre-registered in T6
apply: the chosen config is the 16GB hardware-feasibility frontier, not a
converged optimum; the shallow>deep gradient held across all 10 grid points.

### T10 — MiniLM item-embedding Step A (2026-08-06)

Embedded all 368,228 5-core catalog items (T9 export, `five_core_snapshot_id`
8184397443787800955) with `sentence-transformers/all-MiniLM-L6-v2` (HF
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`). Recipe
`v1_title_brand_cat_features`: `title + " " + brand_norm + " " + main_category
+ " " + " ".join(features)` (null/empty parts skipped; `description` and
`categories` excluded from this recipe). Step A re-verified alignment by
recomputing the export parquet's sha256 and the parent_asin-sequence sha256
against both `export_manifest.json` and the eval cache's `item_ids.parquet`
directly (not trusted from the manifest alone) before embedding.

Hardware/runtime (from `minilm_manifest.json`): device `mps` (Apple M4, no
fallback needed — `mps_failure_detail: null`), batch size 256, fp32 compute
cast to fp16 for storage, wall clock 2115.39s (~35.3 min) for 368,228 items,
sentence-transformers 5.6.1 / transformers 5.14.1 / torch 2.13.0.

Artifact: `data/eval/minilm/8184397443787800955/1f7878ff82bf/` —
`embeddings.npy` (368228×384, float16, 270MB, sha256
`260bf265a29083917852895b2fd006d7641d77aa7fb5daeef7e5019694792110`),
`minilm_manifest.json` (recipe_hash `1f7878ff82bff9fb6c23b0aad5597817dd0e0e0ae0c7859abdd0c7a70efe7bc5`,
source export parquet sha256 `a765db1e8c60890b8e5d90b1c0b23d0243a4befbba529807a39c1af9ebba799f`,
item_ids sha256 `3dacae9d50fec59110d452cde587e91b17aa8812d9c86abce351d782cf863e9f`).
Re-running `make embed-items` after completion printed "up to date" and
exited in 0.77s, confirming idempotency.

Sanity spot-check (cosine neighbors, k=5) for query `B00068NUO4` "Arzonb HDMI
to HDMI Cable 6 Feet": all five nearest neighbors (sim 0.62–0.63) are other
HDMI/Mini-HDMI/Micro-HDMI cables (QING CAOQING, Elebase ×3, CableDirect) —
topically coherent. No metric claims made here; this is Step A (embedding
only), evaluation follows in later Phase 4 tasks.

Deviations: none. `torch`/`sentence-transformers`/`hnswlib` were added to a
new `embed` dependency group (`pyproject.toml`); the default install remains
torch-free (`uv sync` without `--group embed` does not pull torch).

### T12 — Content retrieval VAL campaign + alpha grid (2026-08-06) — PRE-DECLARATION

Before any run: seven configs on VAL only (`eval_pop_t12m_val.yaml`,
`eval_content_val.yaml`, `eval_blend_val_a{10,30,50,70,90}.yaml` for
alpha in {0.1, 0.3, 0.5, 0.7, 0.9}), all using recipe_hash `1f7878ff82bf`
(the T10/T11 MiniLM artifact) and pop params matching the pop-t12m reference
(`as_of: train_end`, `window_days: 365`). No TEST config exists or will be
run in this task.

**Hypothesis:** content retrieval beats trailing-12m popularity on shallow
warm segments (1–4, 5–9), where the ALS/kNN classical-CF arms both lost to
pop-t12m every segment (Phase 2/3) — content similarity offers a different
signal (topical/semantic) than co-occurrence or recency, and may close some
of the shallow-segment gap. Strict-cold (segment 0) has no content signal by
construction (`ContentRecommender` collapses cold users to all-zero scores;
`ContentPopBlendRecommender` degenerates to pure popularity for cold rows) —
no improvement over pop is expected or claimed there.

**Alpha selection rule (pre-declared, applied mechanically in Step 3):**
alpha* = argmax global VAL NDCG@10 over the five-point grid {0.1, 0.3, 0.5,
0.7, 0.9}, ties broken toward the smaller alpha. If the winning alpha sits at
a grid edge (0.1 or 0.9) or the NDCG@10-vs-alpha curve is non-flat around the
winner (i.e., neighboring grid points are not both clearly lower), a follow-up
pair of configs refining +/-0.1 around the winner will be created and run,
and this deviation from the five-point grid will be noted honestly in the
results entry.

**Protocol notes:** all seven runs are VAL-only, single-run (deterministic,
artifact-receipted inference — content and content_pop_blend have no
stochastic step, matching the popularity/kNN treatment in Phase 2; the
3-seed rule that applies to ALS is inapplicable here). Bootstrap:
n_resamples=1000, seed=20260805, matching every prior VAL/TEST config in this
repo. Results and the alpha* decision are appended to this file in a
subsequent entry, after the runs.

## T12 — Content retrieval VAL campaign + alpha grid — results (2026-08-07)

All seven runs completed (356,362 VAL users, full-catalog ranking,
368,228-item catalog, bootstrap n=1000 seed=20260805, `recipe_hash
1f7878ff82bf`). The `blend_val_a70` background attempt referenced in the T12
pre-declaration died mid-run once (machine sleep) and was re-run clean; the
run_id below is the completed re-run. No refinement configs were needed (see
alpha-rule trace below) — the five-point grid plus the two fixed reference
arms (pop, content) is the full and final run set for this task.

### Full run table

| run | run_id | global NDCG@10 [95% CI] | global Recall@20 |
|---|---|---|---:|
| pop-t12m | `20260806T113427Z-e056a2a` | 0.010338 [0.010112, 0.010565] | 0.030979 |
| content | `20260806T114617Z-e056a2a` | 0.001104 [0.001031, 0.001178] | 0.003452 |
| blend α=0.1 | `20260806T121032Z-e056a2a` | 0.011000 [0.010758, 0.011232] | 0.031827 |
| blend α=0.3 | `20260806T124525Z-e056a2a` | 0.011515 [0.011258, 0.011761] | 0.032186 |
| blend α=0.5 | `20260806T142905Z-e056a2a` | 0.010566 [0.010321, 0.010805] | 0.028286 |
| blend α=0.7 | `20260807T003340Z-e056a2a` | 0.007702 [0.007477, 0.007914] | 0.019506 |
| blend α=0.9 | `20260807T015331Z-e056a2a` | 0.003554 [0.003400, 0.003683] | 0.008702 |

Grid shape: NDCG@10 rises 0.1→0.3, then falls monotonically 0.3→0.5→0.7→0.9
— a single interior peak at α=0.3, consistent with a signal (content) that
helps at low weight and hurts as it dominates the blend.

### Per-segment NDCG@10 (segments 0 / 1–4 / 5–9 / 10–19 / 20+)

| run | 0 (cold) | 1–4 | 5–9 | 10–19 | 20+ |
|---|---:|---:|---:|---:|---:|
| pop-t12m | 0.010677 | 0.011719 | 0.010530 | 0.009133 | 0.007021 |
| content | 0.0 | 0.001767 | 0.001132 | 0.000495 | 0.000291 |
| blend α=0.1 | 0.010677 | 0.012450 | 0.011285 | 0.009780 | 0.007319 |
| blend α=0.3 | 0.010677 | 0.012825 | 0.011965 | 0.010344 | 0.007718 |
| blend α=0.5 | 0.010677 | 0.011002 | 0.011209 | 0.009917 | 0.007631 |
| blend α=0.7 | 0.010677 | 0.008347 | 0.008200 | 0.006594 | 0.004553 |
| blend α=0.9 | 0.010677 | 0.004246 | 0.003557 | 0.002200 | 0.001339 |

Every blend run's segment-0 (strict cold) NDCG@10 is exactly 0.010677 —
identical, to full float precision, to the pop-t12m segment-0 value at every
alpha from 0.1 to 0.9. This is the expected mechanical degeneration
(`ContentPopBlendRecommender` falls back to pure popularity for users with no
TRAIN history, since `ContentRecommender` returns all-zero content scores for
them) and confirms the blend implementation behaves as designed for cold
users, independent of alpha.

### Alpha* selection — rule application trace

Pre-declared rule (T12 pre-declaration, this file): alpha* = argmax global
VAL NDCG@10 over {0.1, 0.3, 0.5, 0.7, 0.9}, ties toward smaller alpha; if the
winner sits at a grid edge, or the curve is non-flat around the winner
(neighbors not *both* clearly lower), run a +/-0.1 refinement pair.

1. **Argmax:** α=0.3, NDCG@10 = 0.011515 [0.011258, 0.011761] — highest of
   the five points, and not a tie (next-best α=0.1 at 0.011000, gap
   0.000515, far outside any tie band used elsewhere in this repo).
2. **Edge check:** α=0.3 is interior to {0.1, 0.5, 0.7, 0.9} — not 0.1 or
   0.9. Edge clause does not apply.
3. **Non-flat/neighbor check:** "clearly lower" is operationalized here as
   non-overlapping 95% bootstrap CIs (the CI-based standard used throughout
   this repo's VAL/TEST comparisons, e.g. Phase 2/3 tie bands). Neighbor
   α=0.1: CI upper 0.011232 < α=0.3 CI lower 0.011258 — non-overlapping,
   clearly lower. Neighbor α=0.5: CI upper 0.010805 < α=0.3 CI lower
   0.011258 — non-overlapping, clearly lower. Both immediate neighbors are
   clearly below the winner by this standard.
4. **Verdict:** refinement clause does **not** trigger. **alpha* = 0.3**,
   selected on the five-point grid as pre-declared, no +/-0.1 refinement
   configs created or run.

### Verdict vs the pre-declared hypothesis

**Hypothesis (restated):** content retrieval beats trailing-12m popularity
on shallow warm segments (1–4, 5–9), where ALS/kNN both lost to pop-t12m
everywhere (Phase 2/3).

**Content alone vs pop (eyeball, no formal delta — paired CIs come at
TEST):** rejected, and not narrowly. On segment 1–4, content NDCG@10
(0.001767) is roughly 6.6x *below* pop (0.011719); on segment 5–9, content
(0.001132) is roughly 9.3x below pop (0.010530). Pure content retrieval,
as embedded here (MiniLM title/brand/category/features recipe,
`recipe_hash 1f7878ff82bf`), does not come close to matching co-purchase
recency signal on its own in either shallow-warm segment — semantic/topical
similarity to a user's history is a much weaker ranking signal than recent
popularity for this catalog and this recipe.

**Blend(α=0.3) vs pop (eyeball):** the picture changes once content is
blended at a modest weight. Segment 1–4: blend 0.012825 vs pop 0.011719 —
blend visibly higher. Segment 5–9: blend 0.011965 vs pop 0.010530 — blend
visibly higher, and by a larger absolute margin than segment 1–4. Globally,
blend α=0.3's CI [0.011258, 0.011761] sits entirely above pop's CI
[0.010112, 0.010565] — no overlap at all between the two 95% intervals, the
strongest eyeball signal in this campaign. (This is an unpaired CI
comparison and is explicitly not a formal claim of a nonzero delta; the
pre-declared paired-bootstrap protocol for that claim is reserved for
TEST, matching the T7 ALS precedent.)

**Overall verdict:** the *pure* content-retrieval half of the hypothesis is
rejected — content alone is a weak signal here, weaker even than the
already-weak classical-CF arms on a per-segment basis. But the *blended*
half is directionally supported: a small content weight (α=0.3) lifts VAL
NDCG@10 over pop-t12m in exactly the segments named in the hypothesis
(1–4, 5–9), and the improvement is visible well past the CI-overlap
threshold used for alpha selection above. This reframes Phase 4's live
finding — content is useful as a *complement* to recency-popularity, not as
a standalone competitor to it, unlike what was hypothesized. Whether this
survives a formal paired TEST comparison (blend α=0.3 vs pop-t12m, per a T7-
style pre-declared protocol) is the open question carried into the next
task.

**Logged finding — strict-cold content collapse:** as pre-declared, content
and every blend arm score identically to pop on segment 0 (cold users, no
TRAIN history) — 0.010677 NDCG@10 across the board, confirming the by-
construction fallback to pure popularity for that segment. No content or
blend arm can improve over popularity for strict-cold users under this
recommender design; any cold-start gain would have to come from a different
mechanism (e.g. onboarding signal, not history-based content similarity).

## T13 — Routing-policy n* selection rule — pre-declaration (2026-08-07)

VAL-only task (UPGRADE_PLAN.md §6.4/§8, T13). This section is written and
committed *before* `policy.select` is run against the grid below — the rule
is fixed first so the eventual winner cannot be picked to fit the result.

**Objective (owner-approved):** segment-weighted NDCG@10 = the *unweighted*
mean of the five segment mean-NDCG@10 values (segments 0 / 1-4 / 5-9 / 10-19
/ 20+), i.e. each segment counts equally regardless of its user count. This
matches the crossover/segment framing used throughout Phase 2-4 rather than
a user-count-weighted global mean.

**Inputs (all VAL, per `configs/policy_select_val.yaml`):**
- blend α=0.3 (T12 winner): run_id `20260806T124525Z-e056a2a`
- ALS chosen config (Phase 3: rank=128, reg=0.01, alpha=10, max_iter=25,
  binary, seed=20260805): run_id `20260806T033333Z-acd1f81`
- pop-t12m (as_of=train_end, window_days=365): run_id `20260806T113427Z-e056a2a`

**Grid:** 2 variants x 5 n_star values = 10 cells.
- Variant A: low=blend α=0.3, high=ALS chosen config
- Variant B: low=blend α=0.3, high=pop-t12m
- n_star grid: {1, 5, 10, 20, inf}. Routing convention: `n_train[u] < n_star`
  -> scored by `low`; `n_train[u] >= n_star` -> scored by `high`. n_star=inf
  routes every user to `low` (hybrid degenerates to the blend everywhere).
  The grid values are chosen to align exactly with the frozen segment bucket
  edges (0 / 1-4 / 5-9 / 10-19 / 20+), so routing can be read off each
  user's `segment` label with no need to reload raw `n_train` — no bucket
  straddles a grid edge.

**Winner rule (fixed before computation):** argmax objective over the 10
cells. Ties -> prefer variant B (pop-t12m as the warm/high component, since
Phase 3 found ALS loses to pop-t12m in every segment on VAL/TEST — a tie
should not be broken toward the demonstrably weaker warm-arm candidate);
among remaining ties, prefer the n_star closest to infinity (more blend
coverage = simpler policy, consistent with the "hybrid reduces to the
blend" outcome flagged as a legitimate, publishable result in the task
brief).

**Prior expectation (not binding, stated only to be checked against the
result, not to bias it):** T12 found blend α=0.3 beats both pop-t12m and (by
Phase 3's ALS-loses-everywhere finding) ALS in every warm segment on VAL, so
the likely outcome is variant B, n_star=inf (blend everywhere) — "hybrid
reduces to the blend" would be a legitimate, publishable simplification for
T15, not a failure of this task.

Grid results and the applied winner trace are appended below this line
after running `uv run python -m batch_recsys_lab.policy.select`.

### T13 — grid results (2026-08-07)

Ran `uv run python -m batch_recsys_lab.policy.select --config configs/policy_select_val.yaml`.

| variant | n_star | seg 0 | seg 1-4 | seg 5-9 | seg 10-19 | seg 20+ | objective |
|---|---|---:|---:|---:|---:|---:|---:|
| A (blend/ALS) | 1   | 0.010677 | 0.004801 | 0.004695 | 0.003640 | 0.003247 | 0.005412 |
| A (blend/ALS) | 5   | 0.010677 | 0.012825 | 0.004695 | 0.003640 | 0.003247 | 0.007017 |
| A (blend/ALS) | 10  | 0.010677 | 0.012825 | 0.011965 | 0.003640 | 0.003247 | 0.008471 |
| A (blend/ALS) | 20  | 0.010677 | 0.012825 | 0.011965 | 0.010344 | 0.003247 | 0.009812 |
| A (blend/ALS) | inf | 0.010677 | 0.012825 | 0.011965 | 0.010344 | 0.007718 | 0.010706 |
| B (blend/pop) | 1   | 0.010677 | 0.011719 | 0.010530 | 0.009133 | 0.007021 | 0.009816 |
| B (blend/pop) | 5   | 0.010677 | 0.012825 | 0.010530 | 0.009133 | 0.007021 | 0.010037 |
| B (blend/pop) | 10  | 0.010677 | 0.012825 | 0.011965 | 0.009133 | 0.007021 | 0.010324 |
| B (blend/pop) | 20  | 0.010677 | 0.012825 | 0.011965 | 0.010344 | 0.007021 | 0.010566 |
| B (blend/pop) | inf | 0.010677 | 0.012825 | 0.011965 | 0.010344 | 0.007718 | 0.010706 |

Full grid + winner also written to `results/policy_select_val.json` (not
committed as part of the pre-declaration commit — CLAUDE.md's append-only
invariant applies to `results/runs.jsonl`; this JSON is a derived selection
artifact, regenerable by rerunning `policy.select`).

**Rule trace:** argmax objective = 0.010706, achieved by both A/n*=inf and
B/n*=inf (tie — n*=inf makes the `high` component irrelevant, since every
user routes to `low`=blend, so both variants collapse to the identical
composed vector and tie exactly). Winner rule: ties -> prefer variant B ->
**winner = variant B, n_star=inf**.

**Interpretation:** every ALS/pop `high` arm strictly *reduces* the
objective relative to n_star=inf, monotonically as n_star shrinks from inf
toward 1, for both variants (variant A drops fastest — ALS's segment-1-4/
5-9/10-19/20+ NDCG@10 are all far below blend's, consistent with Phase 3's
"ALS loses to pop-t12m everywhere" finding compounding here). Variant B's
descent is shallower (pop-t12m is a much stronger `high` arm than ALS) but
still monotonically worse than n*=inf at every finite grid point. **The
routing policy on VAL reduces to "blend α=0.3 everywhere"** — i.e. no
finite n* improves on running the single blended recommender for every
user, at any history depth. This confirms the "likely outcome" flagged in
the pre-declaration and in the task brief: a legitimate, publishable
simplification for T15 ("the crossover this lab measured for classical CF
never re-opens once a content signal is blended in at modest weight; the
routing policy that would have exploited a crossover has nothing to route
between on VAL").

### T13 — confirming run + composition assertion (2026-08-07)

Ran `caffeinate -is make eval CONFIG=configs/eval_hybrid_val.yaml` (winning
cell: variant B, n_star=null, i.e. `hybrid(low=content_pop_blend(alpha=0.3),
high=popularity(train_end, 365d))` routing every user to `low`). Result:
run_id `20260807T040118Z-5e212d7`, artifact
`data/eval/per_user/20260807T040118Z-5e212d7_hybrid.parquet`, global VAL
ndcg@10=0.0115 [0.0113, 0.0118] — matches the blend α=0.3 VAL run
(`20260806T124525Z-e056a2a`, ndcg@10=0.011515 [0.011258, 0.011761]) within
bootstrap noise as expected for an n*=inf hybrid.

**Composition-vs-actual assertion:** compared the hybrid run's per-user
artifact against the blend α=0.3 artifact directly (inner-joined on
`user_id`, both cover the identical 356,362 VAL users — set equality
holds). Every per-user metric column (`recall@10/20/50`, `ndcg@10/20`,
`mrr`, `hitrate@10`, `novelty@10`) and the `top50` list column match
**exactly**, `rtol=0 atol=0`:

- max abs diff = 0.0 for every metric column (numpy `array_equal` True on
  all 356,362 rows for each)
- `top50` mismatches: 0 / 356,362 users

This confirms `HybridRecommender` with `n_star=null` composes to *exactly*
the low-arm's scores with no numerical drift, end to end through the real
harness (fit -> score_batch -> mask -> rank -> metrics -> artifact write) —
not just at the toy-fixture level tested in `tests/test_policy.py`.

**T13 conclusion:** the VAL-selected routing policy is variant B with
n_star=inf, which is mechanically identical to running
`content_pop_blend(alpha=0.3)` alone. No finite n* improves on the blend at
any measured history depth on VAL; ALS and pop-t12m both lose to the blend
in every warm segment (0/1-4/5-9/10-19/20+), so there is nothing left for a
history-depth router to route between. This is carried into T15 as: "the
production recommender for this lab's headline TEST report is
`content_pop_blend(alpha=0.3)`, unconditionally — the `HybridRecommender`
machinery built here is retained (tested, registered, VAL-validated) but
this task's VAL evidence does not support deploying a non-trivial n*."
