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

## T15 — One-shot TEST protocol — pre-declaration (2026-08-07)

Written and committed BEFORE any TEST run for this task (CLAUDE.md invariant
#1; mirrors the T7 Phase-3 one-shot TEST protocol above).

**Arms (all VAL-fitted, owner-approved, no further tuning on TEST):**

1. Pure content: `content`, recipe_hash `1f7878ff82bf` (T10/T11 MiniLM
   artifact). Config: `configs/eval_content_test.yaml`.
2. Blend: `content_pop_blend`, alpha=0.3 (T12 alpha* winner). Config:
   `configs/eval_blend_test.yaml`.
3. Hybrid: `hybrid`, variant B, n_star=null (T13 winner — routes every user
   to the blend). Config: `configs/eval_hybrid_test.yaml`. Expected, and to
   be asserted post-run, to reproduce arm 2's per-user scores exactly, as
   confirmed on VAL in T13.

**Single-run justification.** Content/blend/hybrid inference is deterministic
over receipted, already-fitted artifacts (MiniLM embeddings, popularity
tables) — there is no stochastic training step, matching the treatment of
popularity and item-kNN in Phase 2 and pre-declared for these three model
families already in the T12 pre-declaration. The 3-seed mean±sd rule
(CLAUDE.md invariant #2) binds stochastic training only (ALS init); it does
not apply here. One run per arm, on TEST, full stop.

**Comparisons (paired bootstrap, 1000 resamples, seed 20260805, `eval.compare`,
same resample matrix within a comparison):**

- content vs pop-t12m TEST (pop-t12m TEST run `20260805T172047Z-035042b`,
  Phase 2). Config: `configs/compare_content_vs_pop_t12m_test.yaml`.
- blend vs pop-t12m TEST. Config: `configs/compare_blend_vs_pop_t12m_test.yaml`.
- hybrid vs pop-t12m TEST. Config: `configs/compare_hybrid_vs_pop_t12m_test.yaml`.
- hybrid vs its best component. Config:
  `configs/compare_hybrid_vs_best_component_test.yaml`. **Best-component
  resolution rule (mechanical, applied after the arms complete):** the
  component with the higher global TEST NDCG@10 among {blend, pop-t12m} is
  "best". T12/T13 found blend beats pop-t12m on VAL in every segment, so
  blend is expected to resolve as best component; if pop-t12m instead wins
  on TEST, this compare config is pointed at the pop-t12m TEST run instead.
  Because hybrid == blend by construction (T13 confirmed exact per-user
  equality on VAL, to be reconfirmed on TEST), if blend resolves as best
  component this comparison is a trivial exact-zero delta — declared anyway
  to close the §8 "hybrid >= best component" acceptance criterion formally,
  not because a nonzero result is expected.

**Config-schema note (deviation from the plan-brief's literal step order):**
`eval.compare` configs require `a.run_id` / `b.run_id` filled in upfront —
there is no run-id-free schema variant. The four compare configs above
therefore cannot be created with real run_ids until after the three TEST
eval runs (Step 2) produce them. This pre-declaration commits only the intent,
the resolution rule, and the metric/bootstrap parameters (mirroring
`compare_itemknn_vs_pop_t12m_test.yaml`'s structure from Phase 3); the compare
config files themselves are created in Step 3, after the arms complete, and
are explicitly out of scope for "no VAL iteration follows TEST" since they
only combine already-frozen TEST run_ids — no new eval run, no model
selection. This is noted here, before any TEST number is seen, precisely so
it cannot be read as post-hoc rationalization.

**Crossover chart.** `configs/crossover_test.yaml` (TEST twin of
`crossover_val.yaml`) is committed with the blend/content run_ids as `"TBD"`
placeholders, filled in Step 3 once those two TEST runs exist. Fixed lines:
pop-t12m `20260805T172047Z-035042b` (Phase 2), item-kNN `20260805T185305Z-adbca99`
(Phase 2, best top_n=50), ALS `20260806T082441Z-2f2f26d` (Phase 3 chosen
config, seed 20260805 — the primary paired-delta arm per the T7 precedent,
matching the VAL chart's seed choice).

**Null-policy contingency (pre-declared now, not decided after seeing TEST).**
If blend fails to beat pop-t12m on TEST (CI does not exclude zero in favor
of blend, globally or in the relevant warm segments), the shipped policy for
this lab's headline recommender is **recency-weighted popularity
(pop-t12m)**, published as the honest outcome with a per-segment
explanation of where and why content/blend fell short — not silently
swapped for a different content configuration or re-tuned on TEST. This
mirrors the Phase 3 ALS outcome: a negative result is a result, logged in
full in `EXPERIMENT_LOG.md`, not discarded (CLAUDE.md plan-execution
discipline).

**No VAL iteration follows TEST.** Whatever the three TEST runs and four
comparisons show, there is no return to the VAL grid (T12 alpha sweep, T13
n* grid) for a different configuration afterward. The result — positive or
negative — is published as-is.

## T15 — One-shot TEST results and Phase 4 verdict (2026-08-07)

All three TEST runs completed (228,153 TEST users, full-catalog ranking,
368,228-item catalog, bootstrap n=1000 seed=20260805). One eval process at a
time, per protocol; the content and blend runs completed in this task's
session, the hybrid run was executed under the same protocol after a
mid-campaign hand-off (no config changed, no VAL iteration occurred).

**TEST results, global + per-segment NDCG@10 (95% bootstrap CI):**

| arm | run_id | global | seg 0 (cold) | seg 1-4 | seg 5-9 | seg 10-19 | seg 20+ |
|---|---|---|---|---|---|---|---|
| pop-t12m | `20260805T172047Z-035042b` | 0.005404 [0.005209, 0.005635] | 0.007505 [0.006715, 0.008247] | 0.005821 [0.005488, 0.006218] | 0.005225 [0.004895, 0.005575] | 0.005107 [0.004569, 0.005651] | 0.003711 [0.003181, 0.004245] |
| content | `20260807T050054Z-c320c79` | 0.000886 [0.000806, 0.000968] | 0.000000 [0.000000, 0.000000] | 0.001458 [0.001293, 0.001658] | 0.000809 [0.000679, 0.000942] | 0.000492 [0.000337, 0.000666] | 0.000307 [0.000162, 0.000477] |
| blend α=0.3 | `20260807T055333Z-c320c79` | 0.005726 [0.005524, 0.005948] | 0.007505 [0.006715, 0.008247] | 0.006227 [0.005889, 0.006622] | 0.005504 [0.005178, 0.005837] | 0.005426 [0.004888, 0.005967] | 0.004095 [0.003563, 0.004623] |
| hybrid (n*=∞) | `20260807T082125Z-c320c79` | 0.005726 [0.005524, 0.005948] | 0.007505 [0.006715, 0.008247] | 0.006227 [0.005889, 0.006622] | 0.005504 [0.005178, 0.005837] | 0.005426 [0.004888, 0.005967] | 0.004095 [0.003563, 0.004623] |

Hybrid's per-arm numbers are bit-identical to blend's at every segment and
globally, reconfirming the T13 composition assertion (n*=∞ routes every
user to the blend) on TEST, not just VAL.

**Paired-bootstrap deltas (1000 resamples, seed 20260805, n=228,153 common
users), NDCG@10, `excludes_zero` flags:**

*content − pop-t12m:*

| segment | delta | 95% CI | excludes zero |
|---|---|---|---|
| global | −0.004518 | [−0.004756, −0.004315] | yes |
| 0 (cold) | −0.007505 | [−0.008304, −0.006754] | yes |
| 1–4 | −0.004363 | [−0.004770, −0.003973] | yes |
| 5–9 | −0.004416 | [−0.004806, −0.004042] | yes |
| 10–19 | −0.004616 | [−0.005158, −0.004100] | yes |
| 20+ | −0.003404 | [−0.003996, −0.002903] | yes |

Pure content loses to pop-t12m everywhere on TEST, badly — including the
cold segment, where `ContentRecommender` collapses to all-zero scores by
construction (no content signal at all for strict-cold users) and the
1–4/5–9 shallow-warm segments the T12 hypothesis specifically targeted. The
§8 cold-segment criterion (content beating pop-t12m in segments 0/1-4/5-9)
**does not hold** for the pure-content arm on TEST — it never held on VAL
either (T12); pure content is not a competitive standalone arm.

*blend − pop-t12m (the Phase 4 acceptance gate):*

| segment | delta | 95% CI | excludes zero |
|---|---|---|---|
| global | +0.000322 | [+0.000200, +0.000449] | yes |
| 0 (cold) | +0.000000 | [+0.000000, +0.000000] | no (exact zero — blend degenerates to pure popularity for cold rows by construction) |
| 1–4 | +0.000407 | [+0.000174, +0.000641] | yes |
| 5–9 | +0.000279 | [+0.000075, +0.000487] | yes |
| 10–19 | +0.000319 | [+0.000023, +0.000583] | yes |
| 20+ | +0.000384 | [+0.000102, +0.000706] | yes |

The blend beats pop-t12m on TEST globally and in every warm segment
(1-4/5-9/10-19/20+), CI excluding zero in every case. Segment 0 shows an
exact zero delta, not a failure to beat pop — `ContentPopBlendRecommender`
degenerates to pure popularity for cold users by construction (no content
profile exists), so a zero delta there is the expected, correct behavior,
not a miss against the §8 cold-segment criterion (which was about *content*
closing the shallow-segment gap, not about the blend improving on cold
users it has no additional signal for). On the shallow-warm segments the
§8 criterion cares about most (1-4, 5-9) the blend does beat pop-t12m with
CI excluding zero.

*hybrid − pop-t12m:* identical to blend − pop-t12m in every row (hybrid ==
blend by construction under n*=∞); not reproduced again here.

*hybrid − best component:*

Best-component resolution (mechanical rule from the pre-declaration):
blend global TEST NDCG@10 = 0.005726 > pop-t12m global TEST NDCG@10 =
0.005404, so **best component = blend**. The comparison resolves to hybrid
vs blend, and — as pre-declared — is the trivial exact-zero case: delta =
+0.000000, CI = [0.000000, 0.000000], `excludes_zero=false` at every
segment and globally. This is not a failure; it is the expected numerical
identity, and it closes the §8 "hybrid ≥ best component" acceptance
criterion (hybrid is tied with, and therefore not worse than, its best
component, by construction).

**Chart:** `results/figures/crossover_test.png` (and `.svg`), rendered by
`make crossover-chart CONFIG=configs/crossover_test.yaml`. Confirmed
legible: blend α=0.3 (highlighted) tracks visibly above pop-t12m at every
history depth from 1-4 through 20+, converging with pop-t12m only at
segment 0 (cold, no content signal); ALS sits well below both, roughly
flat/declining with depth; item-kNN and content cluster together far below
pop-t12m and the blend across the whole depth range — the pattern matches
the VAL chart (T14) with tighter TEST CIs.

**Phase 4 verdict (§6.4-as-amended):**

The blend (`content_pop_blend`, alpha=0.3, VAL-selected in T12) **beats
pop-t12m on TEST**, both globally (delta +0.000322, CI excludes zero) and
in every non-cold segment (1-4, 5-9, 10-19, 20+ — all CIs exclude zero in
favor of the blend). It does not (and cannot, by construction) improve on
pop-t12m in the strict-cold segment, where it degenerates to pure
popularity. This TEST result confirms the VAL finding (T12) on the frozen
split, one shot, no iteration.

The hybrid routing policy (`HybridRecommender`, n*=∞, T13's VAL-selected
variant B) **is ≥ its best component** on TEST — trivially, since it
reduces mechanically to the blend at every user. There is no finite
history-depth threshold at which routing to a different arm (ALS or raw
content) would help: both lose to pop-t12m and to the blend at every
measured depth, on both VAL and now TEST. The "routing policy" that ships
from this lab is therefore not a depth-conditional router in the sense
originally scoped by §6.4 — it is the single global policy
`content_pop_blend(alpha=0.3)`, unconditionally, with the `HybridRecommender`
machinery retained (tested, VAL- and now TEST-confirmed to compose
correctly) but not exercising any non-trivial routing in this lab's
headline result.

The pre-declared **null-policy contingency does not trigger**: blend beat
pop-t12m on TEST, so the shipped policy is the blend, not recency-weighted
popularity alone. Pure content, evaluated as a standalone arm for
completeness, is confirmed a clear loser on TEST (as on VAL) — it
contributes only as a blended signal, never as a recommender in its own
right, and the §8 cold-segment hope that content alone would close the gap
is not supported by TEST evidence either.

**Uncommitted after this task:** `results/runs.jsonl` (3 eval appends + 4
paired_delta appends), `configs/compare_content_vs_pop_t12m_test.yaml`,
`configs/compare_blend_vs_pop_t12m_test.yaml`,
`configs/compare_hybrid_vs_pop_t12m_test.yaml`,
`configs/compare_hybrid_vs_best_component_test.yaml`,
`configs/crossover_test.yaml` (filled run_ids), `results/figures/crossover_test.png`,
`results/figures/crossover_test.svg`, this EXPERIMENT_LOG.md results entry,
and the per-user parquet artifacts under `data/eval/per_user/` for the three
new runs — left uncommitted per the task's Step 3 instruction pending owner
review.

## Phase 4 T16 — ANN index artifact + latency/overlap receipt (2026-08-07)

`run_id 20260807T090857Z-97af81f`, `kind="ann_receipt"` in `results/runs.jsonl`
(one appended record; abridged below). **Demo-facing artifact only** —
`used_in_eval_metrics: false` is set explicitly on the record, and every
`kind="eval"` / `kind="paired_delta"` record in this log (including all Phase
4 headline TEST numbers) was produced by exact full-catalog ranking via
chunked matmul, never by this ANN index (CLAUDE.md invariant #4). This index
is not wired into `eval/harness.py` or any `Recommender` used by an eval
config.

**Artifact:** `data/eval/minilm/8184397443787800955/1f7878ff82bf/ann_index.bin`
(592MB) + `ann_manifest.json`, sibling to the T10 embedding artifact. Built
with `hnswlib` 0.8.0, cosine space, over the L2-normalized fp32 view of the
368,228 x 384 MiniLM item embeddings. Parameters: `M=16`, `ef_construction=200`,
built-index `ef_search=200`. Build wall clock: 39.8s.

**Receipt measurement:** 10,000 users with `n_train > 0` sampled from the
eval cache with fixed seed `20260805` (recorded in the manifest), content
profile computed exactly as `content.py`'s `ContentRecommender` does
(mean-pooled, L2-normalized TRAIN-item embeddings). For each user, exact
top-10 (chunked brute-force cosine matmul, batched over all 10,000 users at
once) vs ANN top-10 at `ef_search` in `{50, 100, 200}`:

| ef_search | mean top-10 overlap | ANN latency median | ANN latency p95 |
|---:|---:|---:|---:|
| 50 | 0.7990 | 0.230 ms | 0.352 ms |
| 100 | 0.8915 | 0.395 ms | 0.566 ms |
| 200 | 0.9472 | 0.670 ms | 0.923 ms |

Overlap at `ef=200` came in at **0.9472**, just under the informally
expected ≥0.95 — reported honestly per the task's instruction, no tuning
beyond the declared `{50, 100, 200}` grid.

Exact latency, amortized from the same batched chunked matmul used for the
overlap ground truth (10,000 queries in 42.7s total): 4.27 ms/query. This is
a *throughput* number from a batch of 10,000, not a single-query call, and is
not directly comparable to the ANN's single-threaded, one-query-at-a-time
latency above as a "speedup ratio" without that caveat (recorded verbatim in
the record's `receipt.exact_amortized_note`).

**Idempotency:** re-running `make ann-index` without deleting the artifact
prints `up to date: ...ann_index.bin` and skips the build (verified); the
skip check compares the built index's manifest against the source embeddings
sha256, `M`, and `ef_construction`.

**Files:** `src/batch_recsys_lab/models/ann_index.py` (new), `Makefile`
(`ann-index` target, `embed` group, no Java/Spark gate), this entry.

**Uncommitted after this task:** `results/runs.jsonl` (1 `ann_receipt`
append), `data/eval/minilm/8184397443787800955/1f7878ff82bf/ann_index.bin`
+ `ann_manifest.json` (gitignored `data/` artifacts, not tracked by git
regardless), and this EXPERIMENT_LOG.md entry — left uncommitted pending
owner review per the task's constraints.

## Phase 5 T19 — reproduce-headline, first run (2026-08-07)

**Hypothesis:** the headline blend(α=0.3) TEST record
`20260807T055333Z-c320c79` is byte-exactly reproducible from its recorded
Iceberg snapshot IDs (time-travel extract) on a clean tree at current HEAD.

**Protocol:** `make reproduce-headline` (T18 plumbing): pinned extract via
`snapshot-id` read option into `data/eval/cache_repro/8184397443787800955/`,
re-score `configs/eval_blend_test.yaml` (config sha256 verified unchanged),
compare all deterministic record fields; runs.jsonl append-only.

**Result:** `verdict=byte_exact` (record `20260807T153823Z-9a9fb4c`,
reproduces `20260807T055333Z-c320c79`). Field-level diff: empty. Receipts:
cache files sha256 == original live cache (strict, per-file); order-normalized
pair digests match; MiniLM artifact hashes match; per-user parquet arrays
identical. The pair-array shuffle-order nondeterminism flagged in T18 did
**not** materialize — reproduction is bitwise, not merely order-normalized.
Eval wall clock 2615s.

**Ops note:** the first invocation was refused mid-run by the TEST
dirty-guard — a parallel agent's git worktree under `.claude/worktrees/`
appeared as an untracked path. The guard behaved correctly; fix was
gitignoring agent worktrees (9a9fb4c). The pinned extract from that aborted
invocation (~90s) was reused by the recording run (`extract=0.0s` in the
record reflects the cache hit, not a free extract).

## Phase 5 T21 — ops backfill + monthly incremental appends (2026-08-07)

`local.ops.interactions_monthly` created from the FULL `silver.interactions`
(43,228,354 rows — the earlier ~18M estimate was wrong; dedup only removed
~0.7M from bronze's 43.89M), partitioned by `months(ts)`, backfilled through
2023-06-30 minus a deterministic late-arrival holdout
(`pmod(xxhash64(user_id, parent_asin, ts), 1000) < 50` over 2023-05/06):
**43,216,395 rows written = source − 11,959 holdout, exact.** 295 data
files, 1.34 GB, 48s. Then three incremental appends — 2023-07: 78,710,
2023-08: 54,737, 2023-09: 3,623 rows — each snapshot's added-records equal
to its month's source count, one file per append, snapshot chain contiguous
(each record's snapshot_before == predecessor's snapshot_after). Four
`kind="ops"` records appended; gold/silver snapshots asserted unchanged in
every step's epilogue; disk ≥43GB throughout.

## Phase 5 T22 — late-data MERGE upsert (2026-08-07)

`MERGE INTO local.ops.interactions_monthly` on `(user_id, parent_asin, ts)`:
**inserted 11,959 = exactly the T21 holdout** (the late arrivals land),
matched-and-updated 4,645 (deterministic 20‰ sample, `rating := 5.0` as
visible UPDATE evidence — a row-count reconciliation claim, not content
equality with silver). Post-merge total 43,365,424 = 43,353,465 + 11,959,
`reconciles_with_source=true` against the full silver slice. Copy-on-write
rewrite confined to the affected 2023-05/06 partitions (net file count
unchanged at 298). 19.1s. One `kind="ops"` record; gold/silver snapshots
unchanged; disk 45GB.

## Phase 5 T23 — compaction before/after + snapshot expiry (2026-08-07)

**Measured no-op first (published, not hidden):** on the freshly built ops
table, `rewrite_data_files` with defaults rewrote **0 files** — monthly-batch
ingestion at this scale yields exactly 1 well-formed file per `months(ts)`
partition (298 partitions, 298 files, verified via the `.files` metadata
table), and bin-packing never combines across partitions. The "small-file
problem" on this table is partition granularity, not intra-partition
fragmentation. The no-op compact/expire records stay in the log.

**Staged exhibit (simulated micro-batch ingestion):** `fragment` step —
2023-06 (93,827 rows) materialized to a durable scratch table, deleted
(partition-aligned), re-appended as 30 daily slices → 30 one-file appends
(298→327 files), totals byte-identical before/after (asserted). Then
`compact`: rewrote exactly those 30 files into 1 (327→298; 3.3MB → 3.0MB for
the partition; rows identical). Then expiry in two teachable stages:
`retain_last=2` deleted 32 manifest lists but only 3 data files — the 30
compaction predecessors stayed pinned by the retained pre-compact snapshot —
and a follow-up `retain_last=1` reclaimed exactly those 30
(`deleted_data_files=30`). Retention pins files; compaction alone frees
nothing.

Six `kind="ops"` records this task (no-op compact, no-op-ish expire,
fragment, compact, expire@2, expire@1). Gold/silver snapshots asserted
unchanged in every epilogue; disk ≥45GB throughout.

## Phase 5 T24 — per-stage lineage table (2026-08-08)

`make lineage` (JVM-free): 24 stages, `complete: true`, zero problems —
raw download → bronze (reviews/items) → silver → gold (5-core funnel,
user_stats, item_features, popularity, item_text) → eval extract cache →
headline eval → reproduce → the full 11-record ops chain (one row per ops
record, semantic labels: `ops.compact[noop]` vs `ops.compact[30->1]`,
`ops.expire[retain=1,deleted=30]`). Every number sourced from a named
machine ledger (ingest/build summaries, kcore funnel table, Iceberg
metadata.json snapshot summaries, runs.jsonl); completeness enforced by a
check that fails the build, not a hope. Committed as `results/lineage.json`
(sha256:2291697090d9…) + human `results/lineage.md` + one `kind="lineage"`
record.

Nullable runtimes are footnoted `runtime_not_persisted_at_build` (gold
feature builds, item_text, extract cache, raw download) — re-running stages
to re-measure was rejected: it would churn live snapshots for zero
evidentiary gain. **Ledger discrepancy found and NOT fixed by design:**
`data/MANIFEST.md`'s prose line "Ingest wall-clock: reviews=509s, items=926s"
contradicts the machine ledger (350.1s / 105.5s) — it was operator-supplied
prose, never measured. The file cannot be corrected: `dataset_manifest_hash`
(sha256 of MANIFEST.md) is part of every eval record and of the byte-exact
reproduce comparison; editing it would break reproduction permanently. The
lineage table uses the machine ledger; the prose line is superseded by this
note.

## Phase 5 T25 — second reproduce-headline, post-churn + phase acceptance (2026-08-08)

`make reproduce-headline` re-run after the warehouse absorbed the full ops
chain (~40 new snapshots on `local.ops.interactions_monthly`: backfill, 3
appends, MERGE upsert, fragmentation's 30 slice-appends + delete, 2
compactions, 3 expiries): **verdict=byte_exact again** (record
`20260807T164622Z-3e2c665` reproduces `20260807T055333Z-c320c79`), empty
field diff, strict cache sha256 match, per-user arrays identical, clean
tree. Time travel against the pinned snapshot IDs means the headline number
cannot move while the catalog evolves — demonstrated, not asserted.

**Phase 5 acceptance:** (1) reproduce-headline succeeds from the pinned
snapshot — two byte_exact records, one before any ops mutation and one
after; (2) ops metrics logged — 11 `kind="ops"` records with before/after
snapshots/files/bytes/rows, exact reconciliations throughout; (3) lineage
table complete — 24 stages, completeness check green, artifacts committed.
Full suite + CI smoke: 249 passed / 0 failed at the phase-final tree.

## Phase 5 — formally accepted; ops table cleaned (2026-08-08)

Owner accepted Phase 5 and confirmed receipts complete. MANIFEST.md prose
wall-clock discrepancy: keep as-is permanently — the T24 superseding note is
the final handling (the file is hash-pinned into the byte-exact reproduce
comparison and must never change). `make clean-ops` executed:
`local.ops.interactions_monthly` dropped (PURGE) and `data/warehouse/ops/`
removed; disk 44→52GB. Post-drop verification: all four gold tables and
`silver.interactions` still at their pinned snapshot IDs (JVM-free check).
The 11 ops records, both reproduce records, and the lineage artifacts remain
the exhibits' receipts. Phase 6 (demo + case study) starts next in a fresh
session.

## Phase 6 T32 — Checkpoint 1: narrative skeleton signed off (2026-08-08)

Owner sign-off on the case-study narrative skeleton, per the mandatory
Phase 6 checkpoint (no case-study prose existed before this entry):

1. **Name:** Batch Recsys Lab, confirmed (alternates Cold-Start Lab and
   Recommender Systems Lab rejected — the measured outcome refuted the
   cold-start-crossover framing; the lane name is outcome-independent).
2. **Pitch:** question-inverting variant approved, superseding §1.2, with
   the owner's amendment that the closing claim carries magnitude inline
   and states the small win as small (house style): "…and it wins
   everywhere, by a small margin (+6% relative NDCG@10) that
   1,000-resample paired CIs cleanly separate from zero." (Check: blend
   0.005726 vs pop-t12m 0.005404 → +0.000322/0.005404 = +5.96% ≈ +6%
   relative, from the recorded TEST runs.)
3. **Chapter outline:** approved as proposed (10 chapters: question →
   provenance → lakehouse → protocol → challenger ladder → no-crossover →
   policy-collapsed-to-constant → ops receipts → §10 "How this was
   verified" → §10 "What this does not prove"), with one optional
   addition taken up: ch. 8 gets half a sentence on the ANN artifact
   receipt (top-10 overlap 0.947 vs the informal 0.95 expectation,
   reported as-is, never used in eval) as another instance of the
   under-expectation-reported-honestly pattern.
4. **Exhibit list:** approved — the six §9 demo exhibits plus case-study
   inline figures (crossover chart matching the committed SVG, policy
   grid table VAL+TEST, reconciliation waterfall, lineage table,
   byte-exact reproduce receipts).

Evidence-class labeling convention confirmed: every claim tagged
measured / estimated / projected; distributed-scale portability is the
only anticipated "projected". Next mandatory stop: Checkpoint 2
(assembled demo verified static/offline + fully traced, presented for
owner review before final case-study copy).

## Phase 6 T28 — shopper curation rule (pre-declared) (2026-08-08)

Written **before** any selection code was run, so the pick cannot be
retrofitted to a pretty result (select-then-look discipline). The exhibit shows
30 real users; the rule below is the whole of how they are chosen, and
`src/batch_recsys_lab/demo/select_shoppers.py` implements exactly it. Any
deviation discovered later gets a superseding entry, never an edit here.

**Universe.** The 228,153 TEST-eval users — i.e. the rows of the blend
per-user artifact
`data/eval/per_user/20260807T055333Z-c320c79_content_pop_blend.parquet`
(equivalently: 5-core users with ≥1 TEST ground-truth item). No other filter.
Segment is the artifact's own `segment` column (the frozen five, from
`eval/protocol.py::segment_of(n_train)`), never recomputed here.

**Size.** 6 users per segment × 5 segments (`0`, `1-4`, `5-9`, `10-19`, `20+`)
= 30.

**Seed.** 20260805 (the lab's standard seed). Attempt counter `a` starts at 0;
sub-seed for attempt `a` is `20260805 + a`. Each segment draws from its own
stream: `numpy.random.default_rng([20260805 + a, segment_ordinal])` with
`segment_ordinal ∈ {0..4}` in the frozen segment order, then
`rng.choice(pool, size=6, replace=False)` where `pool` is the segment's
candidate `user_idx` values sorted ascending (so the draw depends on nothing
but the seed and the frozen universe).

**Preference for users with real TEST evidence.** Within a segment the
candidate pool is the users with **≥2 TEST ground-truth items** (counted from
the pinned eval cache `data/eval/cache/8184397443787800955/test_user_idx.npy`).
If that pool holds fewer than 6 users the segment falls back to its full user
set, and the fallback is recorded in the selection artifact.

**Curation predicate (per segment, checked on the blend arm only).** Of the 6
drawn users, at least **2** must have ≥1 TEST ground-truth item inside blend's
top-10 (`hitrate@10 == 1.0` in the blend artifact) and at least **1** must have
none (`hitrate@10 == 0.0`). Rationale: an exhibit of six misses teaches
nothing, and an exhibit of six hits misrepresents a 0.0057 NDCG@10 model. The
predicate is deliberately blind to the other four arms — no arm is curated to
look good.

**Redraws.** A segment that fails the predicate is re-drawn at `a+1`
(per-segment attempt counter; segments that already passed are frozen and never
re-drawn). Attempt counts per segment are recorded in
`data/demo_export/shopper_selection.json` and reported in this log. Hard cap 50
attempts per segment; on exhaustion the export **aborts loudly** — the rule is
never relaxed to fit.

**Ordering.** Within a segment the 6 selected users are ordered by ascending
`user_idx`; segments appear in the frozen order; `shopper_order` in
`shoppers.json` is that concatenation.

**Re-hash (privacy).** Displayed identity is
`shopper_id = HMAC-SHA256(key=salt, msg=user_id.encode("utf-8")).hexdigest()[:12]`,
with a 32-byte random salt generated on first run into `data/demo_salt.txt`.
`data/` is gitignored, so neither the salt nor the `user_id → shopper_id`
mapping (`data/demo_export/shopper_map.parquet`) is ever committed or
published; the mapping is retained locally only. A truncation collision among
the 30 aborts the export. The re-hash is stable only while the local salt
persists — accepted (the demo JSON is a committed projection; regenerating it
after salt loss changes the displayed ids but nothing else).

**Frozen-TEST safety.** This is selection of *users to display*, not of models,
thresholds or metrics: every number shown is read from the already-recorded
one-shot TEST runs' per-user artifacts. No re-scoring, no refitting, no new
ground-truth consultation beyond the membership tests above, and nothing here
feeds back into any model or policy choice.

## Phase 6 T28 — curation rule v1 FAILED, superseded by a stratified draw (2026-08-08)

**Hypothesis (v1, the rule pre-declared above):** uniform 6-of-segment draws,
re-drawn until ≥2 of the 6 have a blend top-10 hit and ≥1 has none, would
converge within 50 attempts.

**Result: it does not, and the export aborted loudly as the rule requires** —
`RuntimeError: segment '1-4': the pre-declared predicate (>=2 blend-hit, >=1
blend-miss of 6) was not met in 50 attempts. The rule is not relaxed.`
Segment `0` passed at attempt 1; `1-4` exhausted all 50.

**Why (arithmetic, not luck).** blend's recorded per-segment `hitrate@10` — the
share of TEST users with ≥1 ground-truth item inside the top 10 — is
0.0403 / 0.0197 / 0.0147 / 0.0152 / 0.0151 across segments `0` … `20+`. At
p ≈ 0.02, P(≥2 hits in a uniform draw of 6) ≈ 6·10⁻³, so 50 attempts succeed
about a quarter of the time and "just raise the cap" is fishing, not curation:
with enough redraws the *rule* stops constraining the pick and the seed does.
The v1 predicate was written as if hits were common. They are not — that is the
finding the whole exhibit exists to show.

**Amended rule (v2, `rule_id = phase6-t28-v2-stratified`), declared here before
re-running the selection.** Only the sampling step changes; universe, segments,
seed, the ≥2-TEST-GT preference, the ordering and the HMAC re-hash are
unchanged from the entry above.

* Within each segment's candidate pool, partition users into a **hit stratum**
  (`hitrate@10 == 1.0` in the blend artifact) and a **miss stratum**
  (`hitrate@10 == 0.0`).
* Draw **exactly 2 from the hit stratum and 4 from the miss stratum**, without
  replacement, with `default_rng([seed, segment_ordinal, 1])` and
  `default_rng([seed, segment_ordinal, 0])` respectively. Deterministic in one
  pass: the v1 predicate is now satisfied *by construction*, so the redraw
  machinery survives only as the abort path for a stratum too small to fill
  (it does not fire — the smallest hit stratum is 208 users, in `20+`).
* Attempt counts are still recorded, and are 1 for every segment by
  construction.

**Disclosure is the price of stratifying.** Over-sampling hits 2-in-6 where the
truth is ~1-in-50 would misrepresent the model if it were hidden, so
`shoppers.json` carries a `curation` block naming the strata and the draw
counts, and — traced to each model record's
`/metrics/per_segment/<segment>/hitrate@10` — the **real** per-segment hit rate
next to it. The exhibit copy (T33) must state that the six shoppers per segment
are 2 hits + 4 misses by design, not a random sample.

**Discipline preserved.** Between the v1 abort and this amendment the only
things inspected were aggregates: per-segment stratum sizes and the
already-recorded per-segment `hitrate@10`. No individual user's row, ranking or
metric was examined, and the amended rule was fixed before the v2 selection ran
— so which 30 users appear is still decided by the seed, not by how they look.

## Phase 6 T36 — offline/static + traceability verification of the assembled demo (2026-08-09)

Run against the FULLY assembled demo/ (all six exhibits, search payload
31.9MB + vendored model 47.6MB in place via `make demo-assets`).

**Static scan** (`make demo-offline-check`, verify_offline.py): CLEAN.
Zero external URLs in executable positions across demo/ excluding vendor;
README citation anchors reported-allowed; demo/vendor/ exemption printed
with justification (171 URL literals inside library code, never fetched:
allowRemoteModels=false, local wasmPaths — authoritative proof below). One
REAL violation found on the first run and fixed at the source: a marketing
URL inside an items_meta.json product title; export_search now strips URL
substrings from display titles (descriptive class, no evidence value
touched).

**Runtime proof** (headless Chrome 151, clean profile, DNS black-holed via
--host-resolver-rules="MAP * ~NOTFOUND, EXCLUDE 127.0.0.1", netlog
captured):
1. Page load: all 8 data files loaded, 0 placeholder sections, 27
   page-initiated requests ALL loopback with bodies returned; the only
   non-loopback attempts were Chrome-internal Google endpoints, every one
   DNS-black-holed with zero bytes transferred.
2. Live-search activation (temporary same-origin harness, deleted after):
   transformers.min.js + ort wasm (21.6MB) + quantized MiniLM onnx
   (23.0MB) + tokenizer + int8 payload all fetched from 127.0.0.1 only;
   in-tab inference completed; quantization-parity receipt: overlap@10 =
   9/10 vs the recorded Python reference — inside the expected 8-10 band
   (export-side mean 9.83/10).

**Defect found by this proof and fixed:** the vendored
transformers.web.min.js variant dynamically imports bare specifiers
(onnxruntime-common/onnxruntime-web) that no browser can resolve without
an import map — activation failed with a TypeError. Switched the vendored
module to the fully-bundled transformers.min.js from the SAME hash-pinned
tarball (207714c3…, hash re-verified on re-download; integrity unchanged —
per-member extraction from a verified archive). This is exactly the class
of failure a static scan cannot catch and the black-holed runtime run
exists to catch.

**Traceability**: `make demo-verify` (full mode, per-user parquets read):
OK — 4,617 manifest entries, 6,756 numeric leaves, 28 run_ids, every leaf
re-resolves exactly. `make demo-verify-record` (CI mode): OK. Full suite:
376 passed.

**Phase 6 acceptance status after T36:** criterion 1 (fully static/
offline) and criterion 2 (every displayed number traces to a results-log
record via the receipts drawer) demonstrated by the commands above.
Criterion 3 (case study reviewed against §10 checklists) lands with T37
final copy after Checkpoint 2. Next: Checkpoint 2 — demo presented to the
owner for review before final case-study copy.

## Phase 6 — Checkpoint 2 signed off; two exhibit fixes applied (2026-08-09)

Owner reviewed the assembled demo hands-on: offline load clean, zero
console errors, receipts drawer verified against a live record. All four
orchestrator rulings ratified (stratified curation as disclosed; 79.5MB
uncommitted assets with committed 58kB fallback; "traces to a results-log
record" read transitively via record-anchored artifact hashes; the
[derived] evidence label), with the direction that [derived] be defined
inline at first use and noted in demo/README + case study as an extension
of the site's three-label convention (README note added).

Fixes from the review, both applied and re-verified:
1. item-kNN cold panel cited models/als.py — per-arm mechanism notes now
   cite their own modules (als.py / item_knn.py / content.py).
2. Cold-start users' content (MiniLM) column rendered its index tie-break
   top-list as if it were recommendations. content.py documents the same
   cold-start collapse as ALS (all-zero profile → every score exactly 0),
   so the content arm now carries cold_collapse for n_train == 0 and
   renders the same "no personalized signal — empty by design" panel.
   cold_collapse_models = [als, item_knn, content]; 18 suppressed arms
   across the 6 cold shoppers. shoppers.json re-exported (1,980 traced
   leaves), full demo-verify green, 376 tests pass.

T37 final copy proceeds.

## Phase 6 T37 — §10 checklist review (2026-08-09)

Final copy `docs/case_study.md` reviewed item-by-item against UPGRADE_PLAN
§10. The draft (`docs/case_study_draft.md`) is left unedited for history.
Every §10 requirement is enumerated below, verbatim-in-substance, against
the chapter/section that carries it.

### §10 — "How this was verified" (11 required items)

| # | §10 requirement (verbatim-in-substance) | where it lands |
|---|---|---|
| 1 | Frozen dataset manifest: source URL, download date, SHA-256, published-count reconciliation | ch. 9 bullet 1 (`data/MANIFEST.md`, 2026-08-05, both SHA-256s, observed 43,886,944 / 1,610,012 vs rounded 43.9M / 1.61M with the delta stated); restated in ch. 3 ¶1 |
| 2 | uv-locked environment + hardware documented (M4, 16GB, `local[10]`), stated plainly | ch. 9 bullet 2 (`uv.lock`, `uv sync --locked`, pyspark 4.0.4, Iceberg 1.11.0, project-local JDK 21; Apple M4 / 10 cores / 16GB / `local[10]` / ~8g driver); restated in ch. 3 ¶5 |
| 3 | Deterministic `make data` and snapshot-pinned `make reproduce-headline` | ch. 9 bullet 3 (8/8 content-identical tables; `byte_exact` twice); full treatment in ch. 8 ¶1 with both reproduce records |
| 4 | One-config-per-run + append-only JSONL results with git SHAs | ch. 9 bullet 4 (60 records, per-record fields incl. git SHA + dirty flag); mechanism in ch. 4 ¶5 |
| 5 | User-bootstrap CIs on every headline number, paired deltas for every comparison | ch. 9 bullet 5; protocol in ch. 4 ¶4; every comparative claim in ch. 5–7 carries its CI |
| 6 | Full-catalog ranking (no sampled negatives) stated explicitly | ch. 9 bullet 6 (368,228 items, TRAIN-seen masked); ch. 4 ¶3, incl. the `used_in_eval_metrics: false` flag on the ANN artifact |
| 7 | Metric unit tests against a reference implementation | ch. 9 bullet 7 (`tests/test_metrics.py`: naive full-`argsort` reference, 50 seeded instances, edge cases) |
| 8 | Contract engine + quarantine ledger with exact reconciliation | ch. 9 bullet 8; measured detail in ch. 3 ¶2 and ¶4 (7 contracts, `dq_results`, quarantine reasons, non-zero exit on drift) |
| 9 | Embedding-artifact version + where it was computed | ch. 9 bullet 9 (HF revision `1110a243…`, recipe_hash `1f7878ff82bf`, 368,228 × 384 fp16, sha256, M4 via MPS, 2,115s) |
| 10 | `EXPERIMENT_LOG.md` published including failed hypotheses | ch. 9 bullet 10 (kNN neighbor-list, ALS negative, content-alone, collapsed router, sub-expectation ANN overlap, aborted curation rule); the failures themselves are ch. 5–8 |
| 11 | CI smoke eval on the bundled fixture | ch. 9 bullet 11 (Actions pins Java 21, installs from lockfile, fixture pipeline + end-to-end eval smoke over the committed ~50k-row fixture, `make fixture` byte-identical; 376 tests) |

Phase 6 additions beyond the §10 minimum, all in ch. 9: static offline scan
CLEAN (with the one real violation it found and the source fix);
DNS-black-holed runtime proof (27/27 loopback page requests, 44.6MB
model+payload from loopback, parity overlap@10 = 9/10, plus the
bare-specifier defect only the runtime run could catch); `make demo-verify`
re-resolving 4,617 manifest entries / 6,756 numeric leaves / 28 run_ids
independently of the writer; the receipts drawer as the §9.6 mechanism; and
the hash-not-URL fetch discipline with its truncated-download hard-fail.

### §10 — "What this does not prove" (9 required items)

| # | §10 requirement (verbatim-in-substance) | where it lands |
|---|---|---|
| 1 | No online evidence — every metric offline; ranking wins do not establish CTR/conversion/revenue lift | ch. 10 bullet 1 |
| 2 | Feedback is missing-not-at-random and popularity-biased; no counterfactual correction (IPS/DR) | ch. 10 bullet 2 |
| 2a | "the case study's offline-vs-online section is the extended version of this admission" | **Structural deviation, declared:** there is no separate offline-vs-online chapter. The Checkpoint 1 (T32) skeleton the owner signed off has 10 chapters with the admission consolidated into ch. 10 bullets 1–2, which carry it at full strength (incl. the "winner is a popularity variant, so this bias matters most, and that tension is not resolved here" sentence). Content complete; location differs from the plan's parenthetical. |
| 3 | Reviews are not purchases — "interaction = positive" is a modeling assumption | ch. 10 bullet 3 |
| 4 | k-core filtering inflates absolute metrics (measured where feasible, §8 P7) — compare within this protocol | ch. 10 bullet 4. **Declared gap:** the inflation is *not* quantified — the un-cored popularity comparison is a Phase 7 stretch item and has not been run. The bullet says so in bold ("**This inflation has not been quantified**") rather than implying a measurement exists. This is the only permitted gap. |
| 5 | Single category (Electronics), single snapshot (ends 2023-09) — no generalization or freshness claim | ch. 10 bullet 5 |
| 6 | Not a serving system — no latency/SLA/throughput claims beyond the demo's static assets | ch. 10 bullet 6 (incl. the ANN "latency" being an artifact receipt and the exact-scoring figure being amortized batch throughput, not a speedup ratio) |
| 7 | Single-node Spark — distributed-scale behavior projected, not measured | ch. 10 bullet 7; the `[projected]` label is declared once in ch. 8 as the only projected claim in the document |
| 8 | Iceberg ops exhibits are single-writer local-catalog scenarios, not concurrent-writer production evidence | ch. 10 bullet 8 |
| 9 | Routing thresholds fitted to this dataset — no transfer claim | ch. 10 bullet 9 |

Two bullets in ch. 10 go beyond the §10 minimum: the small size of the win
(+5.96% relative, CI [+0.000200, +0.000449]), carried over from the draft;
and a new one stating that the live-search exhibit's cosine similarities are
a capability demonstration with no results-log record behind them, and are
deliberately not drawn with the traced-number affordance — the only numbers
on the site that are not evidence. Added because Phase 6 shipped that
exhibit after the draft was written; strike it if the owner reads it as
scope.

**Verdict: checklist complete; gaps as declared** — one content gap (k-core
inflation unquantified, marked as such in the text, Phase 7 item) and one
structural deviation (no separate offline-vs-online section; the admission
lives in ch. 10 bullets 1–2 per the Checkpoint 1 skeleton).

## Phase 7 stretch 3 — un-cored silver popularity, TEST pre-declaration (2026-08-09)

**Context.** `docs/case_study.md` §10 bullet 4 admits: "This inflation has not
been quantified — the planned un-cored popularity comparison is a stretch item
and has not been run." This entry pre-declares the one-shot TEST run that turns
that admission into a measured number. Machinery (chunked bootstrap, Arrow
extract, tables guard, `*_uncored` gold builds) landed in commit `7afd780`;
`uv run pytest` 437 passed; `make reproduce-headline` byte_exact re-confirmed
post-refactor (record in `results/runs.jsonl`).

**VAL evidence (iteration allowed).** Un-cored VAL run
`20260809T052855Z-9911774` (pop-t12m, as_of=train_end, 1,696,246 users,
catalog 1,609,860, wall 35,034s) vs 5-core VAL `20260806T113427Z-e056a2a`
(356,362 users, catalog 368,228):

| NDCG@10 | 5-core VAL | un-cored VAL | 5-core/un-cored |
|---|---|---|---|
| global | 0.010338 | 0.010229 | **1.011** |
| seg 0 | 0.01068 | 0.01071 | 1.00 |
| seg 1-4 | 0.01172 | 0.01050 | 1.12 |
| seg 5-9 | 0.01053 | 0.00905 | 1.16 |
| seg 10-19 | 0.00913 | 0.00783 | 1.17 |
| seg 20+ | 0.00702 | 0.00558 | 1.26 |

The presumed k-core inflation is real **within matched history segments**
(+12% to +26%, monotone in depth) but vanishes globally: the un-cored
population is 84% history-0/1–4 users (1,425,654 of 1,696,246), where
popularity is relatively strongest, and the mix shift almost exactly cancels
the per-segment inflation. Two opposing effects, near-coincidental
cancellation — the global scalar alone would be misleading, so the per-segment
ratios are the headline of this measurement.

**Pre-declared TEST hypothesis.** On the frozen TEST split
(`configs/eval_pop_t12m_uncored_test.yaml`, committed before scoring), vs the
recorded 5-core TEST arm `20260805T172047Z-035042b` (NDCG@10 0.005404,
228,153 users): (a) global 5-core/un-cored NDCG@10 ratio in 0.9–1.1; (b)
per-segment ratios > 1 for all non-zero history segments, monotone-increasing
in history depth, in the 1.1–1.3 range for 1–4 through 20+; (c) segment-0
ratio ≈ 1.0. The comparison is a **[derived] protocol-matched ratio across
disjoint universes** (different user populations, catalogs 368,228 vs
1,609,860) — NOT a paired delta; no `compare.py` run is planned or valid here.

**Protocol.** One shot, clean tree, standard guards (dirty-tree, stale-cache,
tables cross-check). Whatever TEST shows — including a refuted hypothesis —
is recorded and published; there is no returning to VAL or re-running TEST.

## Phase 7 stretch 3 — un-cored TEST result: hypothesis partially refuted (2026-08-10)

**Result.** One-shot TEST run `20260809T160227Z-5c70b7c`
(`configs/eval_pop_t12m_uncored_test.yaml`, clean tree, all guards passed,
1,374,880 users, catalog 1,609,860, wall 18,654s) vs the recorded 5-core arm
`20260805T172047Z-035042b`:

| NDCG@10 | 5-core TEST | un-cored TEST | ratio (5-core/un-cored) |
|---|---|---|---|
| global | 0.005404 | 0.004513 [CI 0.004429–0.004599] | **1.198** |
| seg 0 | 0.007505 | 0.004733 | **1.586** |
| seg 1-4 | 0.005821 | 0.004390 | 1.326 |
| seg 5-9 | 0.005225 | 0.004068 | 1.284 |
| seg 10-19 | 0.005107 | 0.003981 | 1.283 |
| seg 20+ | 0.003711 | 0.002686 | 1.382 |

Recall@20 global ratio: 1.078 (0.017803 vs 0.016513). All ratios are
**[derived]** protocol-matched ratios across disjoint universes, not paired
deltas.

**Verdict.** Pre-declared hypothesis (2026-08-09 entry) **partially refuted**:
(a) global ratio 1.198 falls outside the declared 0.9–1.1 — refuted; (b) all
non-zero segments > 1 as declared, but not monotone in depth (1.33 → 1.28 →
1.28 → 1.38) and above the declared 1.1–1.3 at both ends — partially refuted;
(c) segment-0 ratio declared ≈ 1.0, measured **1.586** — refuted, and in the
opposite direction of the VAL evidence (VAL seg-0 ratio 1.00).

**Diagnosis.** The VAL-based prediction assumed the compositional cancellation
(shallow-user mix offsetting per-segment inflation) would transfer to TEST. It
did not: on TEST every segment is inflated by coring, strict-cold most of all.
The plausible mechanism is period interaction — TEST (2023) sits 6–15 months
past the frozen `as_of=train_end` popularity window, and un-cored cold users'
ground truth carries the largest share of tail/new items that the 5-core
catalog excludes by construction; in VAL (2022-H2, fresher popularity) that
gap had not yet opened. Mechanism is inference, not measured. The headline
number stands regardless of the diagnosis: **coring inflates the popularity
baseline's TEST NDCG@10 by ×1.20 globally and ×1.28–1.59 per segment.** Per
protocol there is no re-run; the refuted clauses are published as declared.

## Phase 7 stretch 1 — DuckDB single-node reality check (2026-08-10)

**Hypothesis.** DuckDB rebuilds `silver.interactions` ≥2× faster than the
recorded Spark `local[10]` ledger timings, with an exactly reconciling
waterfall.

**Result.** Record `20260809T212313Z-a5a9e6e` (kind="bench", clean tree,
DuckDB 1.5.5, threads=10, memory_limit=12GB, iceberg_scan reader, 3 fresh-
connection runs; outputs under `data/bench/duckdb/`, warehouse version-hints
unchanged):

| build | DuckDB (3 runs) | Spark ledger (`data/build_summary.jsonl`) |
|---|---|---|
| silver.interactions | 12.674 / 17.029 / 19.363 s | 316.413 / 474.250 / 569.081 s |
| silver.items | 1.443 / 2.021 / 2.163 s | 12.553 / 15.645 / 23.609 s |

Waterfall parity hard-asserted and exact: 43,886,944 in = 43,365,424 kept +
2 rating_domain + 477,968 exact_duplicate + 43,550 superseded; items
1,610,012 → 1,610,012 with the four dq_raw measures reproduced to the row
(price_unparseable 316; brand sources 1,153,897 / 384,785 / 71,330).
Content parity: 74 diff rows, within the 260-row keep-latest tie-group bound
(xxhash64 tie-break unreproducible in DuckDB; measured, not patched).

**Verdict.** Hypothesis confirmed, by ~28× at the median rather than 2× —
with the declared scope asymmetry: Spark ledger numbers include the contract
audit (43M-row FK join), dq writes, and Iceberg commits; the DuckDB port
times the transform chain only. Both numbers and the asymmetry are published
in case_study §3.

**Diagnosis.** No surprise mechanism: single-machine columnar scan +
vectorized hash operations vs JVM task scheduling and shuffle machinery at a
scale (44M rows, ~4GB) that fits one node comfortably. The measured point is
published as the §6.5 honesty item: Spark's value here is the Iceberg write
path, contract engine integration, and cluster-portable semantics, not
single-node throughput. Distributed behavior remains [projected].

## Phase 8 T8-1/T8-3 preregistration — regime-map thresholds, gate rule, deep buckets (2026-08-17)

Registered 2026-08-17T09:31Z, BEFORE any per-cell or per-bucket outcome is
computed (plan §8b, approved at commit c3f38a2). All thresholds below are
anchored to constants already frozen elsewhere in the lab, not tuned.

**T8-1 hypothesis — the churn diagnosis, finally measured.** Phases 2–4
*inferred* that 2023 TEST ground truth concentrates on items with little or
no TRAIN mass ("catalog churn"); it was never directly measured
(case_study line 178 concedes this). H-churn: **≥25% of TEST ground-truth
interactions land on items with zero or low (1–4) TRAIN support.**
Pre-declared gate (§8b guard): zero+low share **<10%** ⇒ near-total
TRAIN/TEST catalog overlap, churn diagnosis recorded as WRONG, STOP —
T8-2's design must be revisited before proceeding; **10–25%** ⇒ partial
support, T8-2 proceeds with the measured share disclosed as a caveat;
**≥25%** ⇒ diagnosis converts from derived to measured-and-supported.

**Fixed axes and thresholds:**
- Item TRAIN support: **zero** = 0 TRAIN interactions; **low** = 1–4
  (below the k=5 core degree); **high** = ≥5. Anchored to the 5-core k.
- Recency of last TRAIN interaction: ≤90d / 91–365d / >365d before
  train_end 2022-06-30 (the frozen popularity windows), plus absent-in-TRAIN.
- First-seen: calendar-year of the item's earliest 5-core interaction
  (≤2019 / 2020 / 2021 / 2022-H1 / post-cutoff) — an interaction-based
  proxy for release date, disclosed as such.
- User axis: the frozen 5 history-depth segments (0 / 1-4 / 5-9 / 10-19 / 20+).
- **T8-3 deep buckets: 20-49 / 50-99 / 100+** from `n_train`, fixed here
  before any per-bucket outcome is examined. Labeled exploratory/derived
  (motivated by the observed narrowing), not confirmatory; thin buckets are
  expected and will be disclosed via user counts and CI widths.

**Arms and mechanics (no retraining, no new TEST model scoring).**
pop-t12m TEST `20260805T172047Z-035042b` vs ALS TEST primary seed
(model seed 20260805; run_id resolved from `results/runs.jsonl` and recorded
in the output record). Per-cell/per-bucket metrics are recomposed exactly
from the persisted per-user `top50` lists (Recall@{10,20,50} and NDCG@10
restricted to any GT subset are exact for K ≤ 50). Bootstrap: 1,000
resamples, seed 20260805, within-cell user resampling with per-cell child
seeds (mirroring `segment_cis`). Max-attainable factor-model recall per
cell = share of GT interactions on items with TRAIN support ≥1 (a
TRAIN-frozen factor model cannot score an item it never saw).

## Phase 8 T8-1 — churn regime map: diagnosis MEASURED AND SUPPORTED, gate passed (2026-08-17)

**Result.** Record `20260817T095926Z-633d454` (kind="regime_map", clean tree
at 633d454, wall 41.0s). Disclosure: an accidental second invocation of the
same target appended `20260817T100112Z-633d454`; the two records' `results`
payloads compare equal field-for-field (only run_id/run_ts/wall_clock_s
differ — determinism held), both stand per invariant #3, citing either is
valid. Arms: pop-t12m `20260805T172047Z-035042b` vs ALS primary seed
`20260806T082441Z-2f2f26d`; 228,153 TEST users, 498,906 GT interactions,
catalog 368,228; item stats at snapshot 8184397443787800955 (coverage exact:
0 missing, 0 extra). Identity anchor: single-bucket recomposition reproduces
the recorded per-user metric vectors, max |diff| 1.11e-16.

**(a) The churn receipt.** TEST GT interaction share on TRAIN-support-zero
items **0.3454** (172,302), on low (1–4) **0.0658** (32,813) — combined
**zero+low = 0.4111**; distinct-item shares 0.2018 / 0.1282. Concentration:
5.19% of the catalog (19,118 zero-support items, all post-cutoff first-seen
by construction) absorbs 34.5% of TEST purchase mass, a 6.7×
over-representation; conversely the 57.3% of catalog stale >365d in TRAIN
draws 2.0%. Independently cross-checked via a separate cache-bincount code
path (no shared module): counts identical, 0/368,228 support mismatches
vs the Spark job.

**Gate (preregistered bands <0.10 wrong / 0.10–0.25 partial / ≥0.25
supported): measured 0.4111 ⇒ the churn diagnosis converts from derived to
MEASURED-AND-SUPPORTED. The §8b near-total-overlap guard does not trip;
T8-2 proceeds as designed.** VAL rehearsal (same pipeline, dry-run) put
zero+low at 0.2792 — the unreachable share grows sharply with distance from
the cutoff (2022-H2 → 2023), i.e. churn is a rate, not a fixed offset.

**(b) Per-cell (segment × support, NDCG@10, paired ALS−pop, 1,000
resamples, seed 20260805, per-cell child seeds).** Both arms score exactly
0.000000 in all 10 zero/low cells (41.1% of GT mass) — structural, not
null: a t12m-popularity top-50 contains only recently-supported items, and
ALS has no factor for TRAIN-unseen items (the identity anchor is what
licenses reading these zeros as real). The arm contrast therefore lives
entirely in the 5 high cells: pop-t12m wins all five with CI excluding zero;
the deficit narrows −0.003481 (1–4) → −0.002001 (20+) — endpoints separate,
but mid-segment CIs overlap, so no monotonicity claim. Sharpest cell:
seg 0 × high, ALS exactly 0 vs pop 0.010984 (n=10,148). Recency axis is
near-degenerate with support (absent ≡ zero; ≤90d ≈ high) except: ALS is
slightly positive on 91–365d / >365d cells where pop is structurally 0
(~0.02% of GT mass; no TEST CI excludes zero) — recorded as the pre-T8-2
baseline for the one axis a time-decayed ALS should move.

**(c) Factor-model ceilings (max attainable recall, any K).** Global cap
**0.6546** at support ≥1 (0.5889 at support ≥5). Inverted-U in history
depth: 0.5676 (seg 0) → peak 0.7255 (10–19) → 0.4326 (100+): the deepest
users buy the newest items hardest, so the arm with the most history to
exploit faces the weakest ceiling — an independent, non-modeling reason
deep buckets look bad for ALS, to be stated alongside T8-3's thin-CI caveat.

**Verdict.** H-churn confirmed at 1.6× the support threshold and 4.1× the
refutation floor. The Phase 2–4 mechanism claim is no longer inference: a
TRAIN-frozen model was structurally capped at 65.5% of TEST ground truth
before any modeling question was asked, and the frozen-vs-fresh gap, not
representation quality, is the first-order fact of this regime.

## Phase 8 T8-3 — deep depth buckets (exploratory/derived): no crossover; ≥50 too thin to resolve (2026-08-17)

**Result.** Record `20260817T100253Z-633d454` (kind="deep_buckets",
`exploratory_derived: true`, wall 23.0s). Buckets 20-49 / 50-99 / 100+
fixed from `n_train` in the 2026-08-17 preregistration before any outcome
was examined. Self-check: buckets 0/1-4/5-9/10-19 reproduce the recorded
TEST per-segment means for both arms and both metrics bit-identically
(16/16 comparisons, max |diff| 0.0).

NDCG@10, paired ALS−pop-t12m:

| bucket | users | share | delta | 95% CI | width | ≠0 |
|---|---|---|---|---|---|---|
| 20-49 | 16,528 | 0.0724 | −0.001806 | [−0.002563, −0.001083] | 0.001479 | yes |
| 50-99 | 2,591 | 0.0114 | **+0.000523** | [−0.001005, +0.002184] | 0.003189 | **no** |
| 100+ | 609 | 0.0027 | −0.000414 | [−0.002484, +0.001209] | 0.003693 | **no** |

Recall@20 is negative with CI excluding zero in ALL seven buckets,
including −0.003965 [−0.007471, −0.000394] at 50-99 and −0.004796 at 100+.
CI widths at 50-99/100+ are 3.9–4.5× the 1-4 bucket's — the thin buckets
the preregistration predicted, disclosed as such.

**Verdict.** Through 20-49 the null is significant and still narrowing
(−0.002562 at 1-4 → −0.001806 at 20-49). At 50-99 the study's first
sign flip appears (+0.000523 NDCG@10) but its CI straddles zero and the
same bucket is significantly negative on Recall@20, so the flip does not
survive a change of metric: **not a crossover**. 100+ reverts negative
(ns), consistent with its collapsed factor ceiling (0.4326, see T8-1(c)).
Exploratory/derived; no confirmatory claim is made beyond 20-49.

## Phase 8 T8-2 preregistration — recency-matched arms: hypotheses, half-life grid, selection rule, decision rules (2026-08-17)

Registered 2026-08-17T11:44Z, BEFORE any T8-2 VAL or TEST run and before
any T8-2 model code exists (plan §8b T8-2, approved at c3f38a2; T8-1 gate
passed at 0.4111 ≥ 0.25, so T8-2 proceeds as designed). Grid values below
are anchored to constants already frozen in the lab, not tuned.

**Hypothesis (H-fair).** Phase 4's comparison was unfair in one axis:
popularity got a trailing-12m recency window while every CF arm was static
all-history. H-fair: giving classical CF the same recency treatment moves
it toward pop-t12m — specifically, (i) the per-segment ALS−pop deficit
shrinks vs the recorded static-ALS baseline, and (ii) time-decayed ALS
becomes positive-with-CI on the stale-item regime-map cells (91–365d,
>365d) where T8-1(b) found static ALS faintly positive and pop-t12m
structurally zero. H-fair does NOT predict the zero/low-support cells
move: 41.1% of TEST GT mass is on TRAIN-unseen or sub-core items and no
TRAIN-frozen model, decayed or not, can rank those (T8-1 ceiling 0.6546).
A full crossover therefore requires the high-support cells to flip, which
H-fair treats as possible but not likely; the preregistered expectation is
"deficit narrows, null likely survives."

**Exactly two new arms, both classical (plan text is binding):**
1. **item-kNN-t12m** — co-occurrence/similarity computed ONLY on TRAIN
   interactions in the pop-t12m window: `2021-06-30T23:59:59.999Z < ts ≤
   2022-06-30T23:59:59.999Z` (identical boundary semantics to the frozen
   popularity build: strict lower, inclusive upper). All other kNN
   hyperparameters held at the Phase 3 chosen config (top_n=50,
   shrinkage=0.0, cosine, deterministic tie-break). User profile vectors
   at scoring time remain full TRAIN history — the recency treatment is
   on the item-side co-occurrence, mirroring how pop-t12m's recency is on
   the item side. No tuning; kNN-t12m has zero free parameters.
2. **ALS-decay** — implicit ALS with time-decayed confidence. Existing
   chosen config held fixed: rank=128, reg_param=0.01, alpha=10,
   max_iter=25, seeds {20260805 primary, 20260806, 20260807}. Per
   (user,item) TRAIN pair, age_days = (train_end − latest interaction ts)
   in days, train_end = 2022-06-30T23:59:59.999Z; confidence input
   r = 2^(−age_days / half_life_days), entering Spark ALS as ratingCol
   under implicitPrefs (c = 1 + α·r), α unchanged. At age 0, r = 1 —
   identical to the binary baseline; decay only removes stale confidence.
   **Only the half-life is tuned.**

**Half-life grid (preregistered, 3 values): {90, 365, 1460} days.**
Anchors: 90d and 365d are the lab's frozen popularity/recency windows
(§8b regime-map recency axis ≤90d / 91–365d / >365d; pop-t12m window);
1460d (4y) is a deliberately mild decay — TRAIN spans ~2016→2022-06, so
1460d leaves multi-year-old signal at ≥quarter weight, probing whether
any decay at all beats none. No other half-life values may be evaluated.

**VAL selection rule.** All tuning on VAL only, primary model seed
20260805 only (matching Phase 3 tuning practice). Selected half-life =
argmax global VAL NDCG@10 (the lab's primary metric). Ties within
overlapping 95% CIs → smaller half-life (stronger recency treatment,
cheaper to defend as "recency-matched"). If ALL three half-lives score
below the recorded static-ALS VAL baseline on global NDCG@10, the decay
mechanism itself failed on VAL; the best of the three still goes to TEST
(the confirmatory question is vs pop-t12m, not vs static ALS) with the
VAL regression disclosed. kNN-t12m: no free parameters; one VAL run is
recorded as its selection record (sanity, not selection).

**TEST protocol (frozen-TEST invariant).** Exactly ONE TEST evaluation
per selected arm, no iteration: (a) kNN-t12m — one TEST record
(deterministic, no seed); (b) ALS-decay at the selected half-life —
3-seed TEST records (mean±sd reported; primary seed 20260805 carries the
per-user artifact used for paired deltas and recomposition), mirroring
the Phase 4 ALS TEST protocol. Comparators: pop-t12m TEST
`20260805T172047Z-035042b` and blend TEST `20260807T055333Z-c320c79`.
For each new arm: paired-bootstrap deltas vs both comparators (1,000
resamples, seed 20260805, per-segment child seeds), identical
history-depth segment breakdown (0 / 1-4 / 5-9 / 10-19 / 20+) and CI
machinery as the existing exhibits. Additionally each new arm's per-user
TEST scores are recomposed through the committed T8-1 regime-map
machinery (same cell axes, same seeds) against the recorded baseline
records `20260817T095926Z-633d454` (dup `20260817T100112Z-633d454`
disclosed, untouched).

**Decision rules (both outcomes ship):**
- **Crossover** = at any history-depth segment OR any regime-map cell, a
  new arm beats pop-t12m on TEST NDCG@10 with the paired-delta 95% CI
  excluding zero (Recall@20 checked as the metric-robustness guard, as
  in T8-3). Then the headline answer becomes "personalization needs
  history AND freshness", the crossover chart and routing narrative are
  updated, and T8-4 (ML-32M contrast) is likely skipped per the §8b gate.
- **Null** = no such segment/cell. Then the Phase 4 null is robust to the
  recency asymmetry, the fairness objection is discharged by measurement,
  and T8-4 becomes the capstone (regime contrast on a low-churn catalog).
- Either way: verdict logged here, records in runs.jsonl, case_study/demo
  updated only where existing numbers are directly superseded.

**Mechanics note (declared before implementation).** Timestamps are not
persisted in the eval cache; a one-shot deterministic Spark job will
extend the cache with per-TRAIN-pair age_days at the same pinned Iceberg
snapshot (8184397443787800955), with an exact-alignment assertion against
the existing cached pair arrays (any mismatch aborts). ALS param identity:
half_life_days enters canonical params and the param hash ONLY when
weighting="time_decay", so all existing artifact hashes are unchanged.
kNN-t12m identity: train_window_days=365 recorded in the run record's
model params. Unit tests for the decay formula, window boundary, and
hash back-compat land with the implementation, before any VAL run.

## Phase 8 T8-2 substrate — migration: rented Linux box is the machine of record from T8-2 onward (2026-08-17)

**Decision.** All Phase 8 work from T8-2 (VAL grid onward) runs on a rented
Linux box, not the 16GB MacBook. The box was verified by a full fresh
rebuild — raw bytes → bronze → silver → gold with contracts and the
reconciliation waterfall — before any model run.

**Hardware / OS / toolchain (machine of record).**
- Rented container (AutoDL), Ubuntu 22.04, root. cgroup quota **16 vCPU /
  120GB RAM** (nproc/free show the 128-core host; the quota is
  authoritative). Shared box: an unrelated GPU training job co-tenants;
  Spark is sized to leave it headroom.
- Data disk: 150GB XFS at `/root/autodl-tmp`. Repo clone:
  `/root/autodl-tmp/crossover-study` (bare remote
  `/root/autodl-tmp/crossover-study.git`, named `gpu` on the Mac).
  System disk is ~12GB free — everything big (raw, warehouse, uv cache,
  Spark scratch) lives on the data disk.
- OpenJDK 21.0.11 (apt, `/usr/lib/jvm/java-21-openjdk-amd64`), uv 0.12.5,
  Python 3.12.14 (uv-managed), pyspark 4.0.4 + iceberg-spark-runtime
  1.11.0, numpy 2.5.1, scipy 1.18.0, pyarrow 25.0.0.
- Spark sizing: `local[12]`, driver 32g, `spark.local.dir=
  /root/autodl-tmp/spark-tmp` — via the `RECSYS_SPARK_*` env overrides
  added in commit 327c417 (laptop defaults unchanged). Env lives in
  `/etc/profile.d/crossover-study.sh` on the box.
- The Mac's `uv run pytest` spawn breakage does NOT reproduce on Linux;
  bare `uv run pytest` works there.

**Fresh rebuild, not a warehouse copy.** The Mac warehouse was not
copied (its metadata carries absolute paths; and a fresh rebuild is a
live test of the determinism claim). Raw `.gz` files were transported
from the Mac's copies (UCSD origin per `data/MANIFEST.md`) and SHA-256
verified on the box against the recorded manifest hashes — both OK, so
the input bytes are identical to the recorded download.

**Result: exact reproduction.** `make data` (run
`20260817T154801Z-327c417`, commit 327c417) reconciles exactly: every
count in the waterfall and all 17 k-core funnel iterations
(rows/users/items per iteration) are identical to the Mac build of
2026-08-05 — bronze 43,886,944 reviews / 1,610,012 items, silver
43,365,424 / 1,610,012, gold 5-core 15,473,536 rows × 1,641,026 users ×
368,228 items. Only run ids, timestamps and wall-clocks differ (the box
is ~3× faster per funnel iteration). Contracts all pass; quarantine
ledger reproduced (same 2 rating_domain rows).

One fresh-warehouse ordering wrinkle, recorded honestly: plain
`make data` on an empty warehouse fails at the contract audit because
`contracts/gold_item_text.yaml` grades `gold.item_text`, which is built
by the Phase 4 target `make gold-item-text`, not by `make data` (on the
Mac the table predated the audit). Sequence used: bronze ingest →
`make data` (failed at audit as above; silver/gold builds themselves
green) → `make gold-item-text` → `make data` end-to-end green.

Test suite on the box: 455 passed / 1 failed / 3 skipped, the failure
being a bitwise-equality assertion on two BLAS GEMM shapes that differ
by 1 float32 ulp on x86 OpenBLAS (identical on Apple Accelerate) —
relaxed to rtol/atol 1e-6 in commit a0ad01a (intent was row alignment,
not numeric identity); suite green after. This is exactly the expected
cross-platform float behavior: **counts are identical, floats may drift
in the last ulp** — which is why T8-2 VAL/TEST must not mix machines.

**No cross-machine contamination.** T8-2 VAL/TEST had NOT started on the
Mac: zero T8-2 records in `results/runs.jsonl` (67 records total; none
matching the T8-2 configs — hl90/hl365/hl1460, alsdecay, itemknn_t12m),
verified 2026-08-17 before migration. Every T8-2 run will execute and be
recorded on this box.

## Phase 8 T8-2 VAL grid — half-life selected: 365 days (2026-08-18)

All four VAL runs executed on the machine of record (rented box, commit
6300640, snapshot 7217506217965106727 — the box's own gold lineage; the
eval extract cache and the age-cache artifact were built fresh there,
age manifest sha 1046d7df…, n_train_pairs 14,206,658, mean TRAIN pair
age 1772.6d). Primary model seed 20260805 only, per the preregistration.

**Global VAL NDCG@10 (bootstrap 95% CI):**

| arm | run_id | ndcg@10 | CI |
|---|---|---|---|
| ALS-decay hl90 | 20260818T021040Z-6300640 | 0.0047910 | [0.0046294, 0.0049520] |
| **ALS-decay hl365** | 20260818T021256Z-6300640 | **0.0053348** | [0.0051650, 0.0055127] |
| ALS-decay hl1460 | 20260818T033640Z-6300640 | 0.0046472 | [0.0044870, 0.0048074] |
| static ALS (recorded baseline) | 20260806T033333Z-acd1f81 | 0.0042542 | [0.0041009, 0.0044030] |
| item-kNN-t12m (sanity record) | 20260818T003552Z-6300640 | 0.0005557 | [0.0005054, 0.0006099] |
| pop-t12m (recorded, for scale) | 20260806T113427Z-e056a2a | 0.0103379 | [0.0101124, 0.0105646] |

**Selection (rule applied mechanically):** argmax global VAL NDCG@10 =
**hl365**; its CI overlaps neither hl90's nor hl1460's, so the tie-break
clause is not triggered. No regression disclosure needed: all three
half-lives beat the static-ALS VAL baseline (hl365 by +25% relative,
CIs disjoint) — the decay mechanism helps ALS on VAL. The hump shape
(365 > 90, 365 > 1460) says both too-aggressive and too-mild decay lose
signal; the pop-t12m gap (~2×) remains.

**item-kNN-t12m VAL note (sanity, no selection):** windowing the
co-occurrence matrix to t12m collapses kNN (0.00056 vs pop-t12m
0.01034) — with a ~6.5y-old TRAIN catalog, a 12-month item-side window
leaves too few co-occurrences to rank with. Recorded as-is; the TEST
record will quantify it under the frozen protocol.

**TEST configs:** the preregistered placeholder in
`eval_alsdecay_test_seed{1,2,3}.yaml` (365, the grid middle) happens to
equal the VAL selection; comments updated to record that 365 is now the
selected value. TEST protocol next: exactly one TEST evaluation per arm
(kNN-t12m deterministic; ALS-decay hl365 3 seeds), then regime-map
recomposition. No TEST iteration.

## Phase 8 T8-2 — regime-map lineage exception: one-time, single-arm, digest-gated (2026-08-18)

Declared BEFORE the regime-map runs that use it. The T8-2 recomposition
must cross the recorded Mac-era comparator (pop-t12m TEST
20260805T172047Z-035042b, scored on gold snapshot 8184397443787800955)
with the box cache (snapshot 7217506217965106727). regime_map.py's
lineage guard — correctly — refuses: the machine-of-record migration
rebuilt the warehouse, so the snapshot IDs differ even though the data
was verified identical (see the 2026-08-17 substrate entry).

Resolution (adversarially reviewed by two independent advisors;
narrow-exception design chosen over a general equivalence registry):
a config-declared `regime_map_input_equivalence` block, recognized only
by regime_map.py, scoped to exactly this one arm + run_id + directed
snapshot pair, honored only after ALL of: (a) every other arm matches
the cache normally and an unused exception is itself an error; (b) the
box item_train_stats parquet raw-byte sha256 equals both its own
manifest and the proof pin — and it equals the Mac build byte-for-byte:
sha256 72a71aee1c689dbed1259da720f3c8671f5f7721fae2373f2028c2065734dbbf
on BOTH machines (the cell axes are literally the same bytes); (c) the
committed T8-1 reference record 20260817T095926Z-633d454 confirms the
same parquet sha, the comparator's artifact sha, and a passed identity
check; (d) the two item-stats manifests are equal except created_ts and
snapshot id; (e) the rsync'd comparator artifact hashes to the sha the
T8-1 record committed (3a0419da…, re-verified after transfer). The
output record disclones the verbatim declaration and every digest;
records without the block are shape-identical to before. The existing
identity anchor still recomposes the comparator's recorded metrics
bit-for-bit (recall) / at 1e-12 (ndcg) — a live end-to-end receipt that
artifact, cache and axes agree across the migration. 13 new unit tests
cover tampered digests, unused exceptions, second-arm mismatches, and
schema strictness.

## Phase 8 T8-2 VERDICT — recency-matched arms: technical crossover, in the stale-item pocket, by the arm nobody favored (2026-08-18)

All records on the machine of record; configs, seeds, comparators and the
decision rule exactly as preregistered 2026-08-17T11:44Z. One TEST
evaluation per arm, no iteration.

**ALS-decay hl365 (selected on VAL).** TEST global NDCG@10
0.003734 ± 0.000018 (3 seeds: 20260818T060704Z/051547Z/051858Z-109c271;
primary seed carries the per-user artifact). Paired vs pop-t12m
(20260805T172047Z-035042b): global delta −0.0017 [−0.0020, −0.0014];
significantly below pop at depths 0/1-4/5-9/10-19; at 20+ the NDCG@10
delta is +0.00036 [−0.00045, +0.00115] (ns) and the Recall@20 guard is
significantly NEGATIVE (−0.0032 [−0.0050, −0.0016]) → **no
history-depth crossover**. No regime-map cell is positive-with-CI
(record 20260818T072211Z-3f3530a; stale cells sit at exact zeros or
tiny ns positives). Same story vs the blend. H-fair's prediction (ii)
— that decayed ALS turns positive-with-CI on stale cells — **failed**;
prediction (i) held only in the weak form that decay helps ALS itself
(VAL +25% relative over static ALS, CIs disjoint).

**item-kNN-t12m.** TEST global NDCG@10 0.000301 (20260818T054430Z-109c271)
— ~18× below pop-t12m 0.005404: globally the windowed kNN is the worst
arm in the lab. BUT the regime map (20260818T072256Z-3f3530a) shows
**five cells positive with the 95% CI excluding zero on BOTH NDCG@10
and the Recall@20 guard**:

| cell (axis, depth, bucket) | Δndcg@10 | Δrecall@20 | GT share |
|---|---|---|---|
| support, 5-9, low | +0.00045 | +0.00093 | 1.73% |
| support, 20+, low | +0.00058 | +0.00103 | 0.69% |
| recency, 1-4, 91-365d | +0.00039 | +0.00102 | 2.16% |
| recency, 5-9, 91-365d | +0.00069 | +0.00151 | 1.75% |
| recency, 10-19, 91-365d | +0.00033 | +0.00242 | 1.01% |

**Verdict: the preregistered crossover criterion is MET** — by
item-kNN-t12m, on the exact axis H-fair named (items popularity's own
window under-ranks), though by the arm the hypothesis didn't favor.
T8-1 saw static ALS "faintly positive (ns)" here; recency-matching
turned the effect significant, with the guard metric agreeing in every
cell and coherent adjacency (same two related regions, three
consecutive depth bands). Multiplicity disclosure: ~40 cells × 2 arms
tested with no correction, per the preregistered any-cell rule; the
clustering and two-metric agreement argue against pure chance, but the
per-cell magnitudes are small and the affected mass is a minority
pocket (each cell 0.7–2.2% of TEST GT; the cells overlap across axes).

**Honest scope note for the routing narrative.** The winning cells are
defined by properties of the *ground-truth item* (its TRAIN recency /
support), which a serving-time router cannot observe. The measured
crossover is therefore diagnostic — popularity's blind spot is real and
a recency-matched CF arm can exploit it — but converting it into a
routable policy needs a serve-time proxy for "this user shops the stale
catalog", which is out of T8-2 scope.

**T8-4 gate (§8b): crossover ⇒ T8-4 (ML-32M contrast) likely skipped.**
Owner's call; not started here. Headline framing updates to
"personalization needs history AND freshness — and where freshness is
popularity's weakness, even a weak personalized arm finds signal."

## Phase 8 T8-4 gate — owner confirmation: Phase 8 complete as preregistered; T8-4 (ML-32M contrast) formally SKIPPED (2026-08-18)

**Owner's confirmation.** The owner has confirmed completing Phase 8 per the
preregistered plan. T8-4 — the MovieLens-32M regime contrast — is therefore
**not started**. This entry closes the gate on the record rather than
leaving it silently open; it changes no prior entry and no result.

**The gate language it applies (plan §8b, approved at c3f38a2).** T8-4 is
declared "**CONDITIONAL: execute only if T8-2 confirms the null.** Gate is
explicit on T8-2's outcome," and its acceptance line ends: "If the gate
stays closed, **this task is not started**." The matching branch was written
into the T8-2 preregistration (registered 2026-08-17T11:44Z, committed
832d407 at 2026-08-17T11:45:21Z): "**Crossover** = at any history-depth
segment OR any regime-map cell, a new arm beats pop-t12m on TEST NDCG@10
with the paired-delta 95% CI excluding zero … Then the headline answer
becomes 'personalization needs history AND freshness' … and **T8-4 (ML-32M
contrast) is likely skipped per the §8b gate**"; the "**Null**" branch says
"T8-4 becomes the capstone."

**Why the gate is closed.** T8-2 did not confirm the null. The preregistered
crossover criterion was **met** — item-kNN-t12m beats pop-t12m on TEST
NDCG@10 with the paired 95% CI excluding zero, and with the Recall@20
robustness guard agreeing, in five regime-map cells (verdict entry above,
record `20260818T072256Z-3f3530a`). Crossover ⇒ gate closed ⇒ T8-4 skipped.
The skip is the preregistered branch executing, not a scope cut made after
seeing the result: the rule predates every T8-2 record (commit 832d407 at
2026-08-17T11:45Z; earliest T8-2 record `20260818T003552Z-6300640`, ~13h
later), and it predates the T8-2 model code, which did not exist when the
rule was written.

**What the skip costs, recorded as a cost, not waved off.** T8-4 was the only
planned *external* replication in the lab. Skipping it means:
- The "crossover exists only when the catalog holds still" framing stays
  **untested** — this lab still has one category and one snapshot, and the
  measured crossover pocket has no independent-dataset confirmation.
- The T8-2 cell-level result is a single-dataset finding with uncorrected
  multiplicity (~40 cells × 2 arms, per the preregistered any-cell rule).
  Nothing about the gate closing makes that stronger.
- The MovieLens timestamp caveat (Sun et al., arXiv:2307.09985) and the
  ML-32M manifest/contracts work are moot; no partial ML-32M artifact
  exists in the tree.

These costs are written into `docs/case_study.md` §10 in the same commit as
this entry, so the case study does not inherit a claim the gate withdrew.

**Scope consequence.** Phase 8 delivers T8-1, T8-2, T8-3 and the §8b
follow-on doc work (line 277: `docs/case_study.md` §6–7 replaced derived
churn language with the measured regime map, recorded the T8-2 fairness
outcome, and discharged the old line-178 caveats one by one — same commit).
The §12 cut order already ranked T8-4 first to cut, so this outcome needs no
cut-order edit. Demo work (§8b line 277, crossover explorer) is tracked
separately and is not part of this entry.

## `make reproduce-headline` — headline pin repointed to the machine-of-record run; original record superseded, not edited (2026-08-19)

**What changed.** `configs/headline.yaml`'s `headline_run_id` now points at
`20260818T181443Z-6744efc` instead of the original `20260807T055333Z-c320c79`.
The original record is untouched in `results/runs.jsonl` — this is a
supersession, per the append-only invariant's own language (CLAUDE.md #3:
"a wrong run gets a superseding entry, not a rewrite"; here the prior run
wasn't wrong, it was superseded by the T8-2 machine-of-record migration
(EXPERIMENT_LOG entry above, commit `d4605c4`)).

**Why a repoint was needed.** `make reproduce-headline` reconstructs the
recorded headline eval by Iceberg time travel at the exact snapshot IDs the
recorded run carries (`src/batch_recsys_lab/eval/reproduce.py`). Those IDs
are generated per-write and are not portable across warehouses: the original
pin's snapshot IDs belong to the old Mac warehouse, which no longer exists
on the machine of record (the remote box, verified exact-match rebuild,
commit `d4605c4`). Time travel to those IDs cannot resolve there, so
`make reproduce-headline` could not run to completion on the machine of
record without a new pin.

**How the new pin was minted.** On the remote box, the ORIGINAL headline
config (`configs/eval_blend_test.yaml`, `content_pop_blend` alpha=0.3, TEST,
config_hash unchanged) was re-run end to end — Spark item-text export,
MiniLM re-embedding of the rebuilt warehouse's 368,228-item catalog
(`sentence-transformers/all-MiniLM-L6-v2`, same recipe hash `1f7878ff82bf`,
device=cpu), then the eval — producing one new `kind=eval` record,
`20260818T181443Z-6744efc`, against the rebuilt warehouse's own snapshot
IDs.

**Gate: old record vs new record, both `content_pop_blend` alpha=0.3 TEST.**

| metric | recorded (`c320c79`, Mac/MPS) | new (`6744efc`, remote/CPU) | diff | within recorded 95% CI? |
|---|---|---|---|---|
| recall@10 | 0.010282977196998225 | 0.010282977196998225 | 0 | yes (exact) |
| recall@20 | 0.018436684314898002 | 0.018436684314898002 | 0 | yes (exact) |
| recall@50 | 0.032366429095647416 | 0.032366429095647416 | 0 | yes (exact) |
| hitrate@10 | 0.017944098916078247 | 0.017944098916078247 | 0 | yes (exact) |
| ndcg@10 | 0.005726134272789762 | 0.005726160835093965 | +2.66e-08 | yes (CI width ≈4.2e-4) |
| ndcg@20 | 0.008063604584335545 | 0.008063631146639747 | +2.66e-08 | yes |
| mrr | 0.008668393396127867 | 0.008668475698452858 | +8.23e-08 | yes |

Recall/hitrate metrics are bit-for-bit identical; ndcg/mrr differ in the 8th
significant digit, consistent with float summation order under a different
BLAS/backend (CPU on the remote box vs Apple MPS on the original Mac run) —
the exact kind of variation the module docstring for `reproduce.py`
anticipates, and far inside the recorded bootstrap CIs. The deterministic
record fields (`config_hash`, `splits`, `protocol`, `model`, `seeds`) were
identical except for the (expected) `iceberg_snapshots` and `dataset_manifest_hash`
fields, which differ because they name the rebuilt warehouse.

**`make reproduce-headline` against the new pin.** Green end to end:
verdict `byte_exact`, exit 0. Pinned-cache-manifest match, model artifact
hash match, deterministic-field diff empty, per-user artifact arrays
identical. Wall-clock: pinned extract 32.12s, eval re-run 2004.37s (≈33.7
min). (Two informational checks — strict cache-file sha256 and
order-normalized pair-array digest against the pre-existing *live* cache
directory — reported FAIL; these are not part of the `byte_exact` gate by
design, per `reproduce.py`'s own criterion, and reflect the live cache
having been rebuilt independently just before this run, not a determinism
bug in the pinned reproduction itself.)

**What this does not change.** The original record `20260807T055333Z-c320c79`
remains in `results/runs.jsonl`, byte-identical, and remains authoritative
history for what ran on the Mac warehouse. No prior run, comparison, or
regime-map record that cited it is altered. `docs/case_study.md`'s headline
numbers were already sourced from this record's metrics, which are
unchanged by the repoint (only their provenance pin moved).

Commits: `58f064f` (repoint + disclosure), `cd779b3` (reproduce record,
remote), merged to `main` via `07afd2e`.

## Phase 9 T9-1 — SUPERSESSION: T8-2 verdict downgraded; headline narrative repinned null-first (2026-08-19)

**What this entry supersedes.** The 2026-08-18 entry "Phase 8 T8-2 VERDICT"
above declared "**the preregistered crossover criterion is MET**." That entry
stands byte-identical, as the append-only invariant requires. This entry
downgrades its verdict language: from "crossover MET" to **"local
regime-cell wins under uncorrected multiplicity; the global history-depth
crossover null is robust."** No run record, metric value, or prior log entry
is edited; the downgrade is a reinterpretation of the same recorded numbers,
directed by the owner via UPGRADE_PLAN §8c (approved 2026-08-19).

**Why the downgrade.** The criterion was met as preregistered, but what the
meeting is worth was overstated by the headline framing it licensed:

- ~40 regime-map cells × 2 arms ≈ **80 uncorrected tests per metric**; at
  α=0.05 that predicts **~4 false positives by chance alone**, against 5
  observed winning cells.
- The dual-metric guard (NDCG@10 + Recall@20 both significant) and the
  clustering (two related regions, three consecutive depth bands) are
  **partial evidence, not a correction** — the two metrics are highly
  correlated on the same cells.
- The winning arm is the globally weakest model in the lab (item-kNN-t12m
  TEST NDCG@10 0.000301 vs pop-t12m 0.005404, records
  `20260818T054430Z-109c271` / `20260805T172047Z-035042b`), the per-cell
  deltas sit at +0.0003–0.0007, and each cell is 0.7–2.2% of TEST GT mass
  (record `20260818T072256Z-3f3530a`).

Meanwhile the null it was allowed to overshadow is robust: every arm loses
to pop-t12m at every history-depth segment with CIs excluding zero, before
and after recency matching (records `20260818T064002Z-56d871c`,
`20260818T064207Z-56d871c`); the mechanism is measured (41.11% churn, run
`20260817T095926Z-633d454`); and the fitted policy is n\*=∞ (record
`20260808T030659Z-43c90c8`). What survives of T8-2's cell result is the
mechanism reading — pop-t12m's trailing window under-serves stale,
thinly-supported items, and there even a weak personalized arm finds
signal — now a footnote, not a headline.

**Narrative repin shipped with this entry (same commit).** The headline
across `docs/case_study.md` (blockquote, §1, §6, §7, §9, §10), the new
root `README.md`, and the demo copy (`demo/index.html` exhibit-1 and
exhibit-1b ledes) now leads with the four robust findings: (1) no
history-depth crossover, every segment CI excluding zero; (2) 41.11%
catalog churn as the measured mechanism; (3) blend α=0.3 as the one
effective arm (+0.000322 NDCG@10, ≈+6% relative); (4) n\*=∞ — the optimal
routing is no routing. The five-cell result is demoted to a mechanism
footnote carrying the multiplicity disclosure above. **Zero metric values
changed** in the rewrite; every claim still traces to the run_ids cited
here and in the docs.

## Phase 9 T9-2 — DECISION: the T8-4 gate is REOPENED (2026-08-19)

**What this entry records.** The 2026-08-18 entry "Phase 8 T8-4 gate —
owner confirmation" formally skipped T8-4 (the MovieLens-32M regime
contrast) because T8-2 met the preregistered crossover criterion:
"Crossover ⇒ gate closed ⇒ T8-4 skipped." That skip was gated entirely on
the verdict the T9-1 entry above has now downgraded. With the verdict
downgraded to "local regime-cell wins under uncorrected multiplicity; the
global history-depth crossover null is robust," the gate's premise is
withdrawn, and **the owner reopens T8-4 as approved scope**
(UPGRADE_PLAN §8c, approved 2026-08-19). The skip entry itself stands
unedited; this is a superseding decision record, not a rewrite.

**The governing spec.** The §8b T8-4 spec stands as written, with the §8c
amendment quoted verbatim:

> The §8b T8-4 spec stands as written with **one amendment**: the
> preregistration must now include an explicit multiplicity policy — the
> **primary confirmatory test is the per-depth-segment crossover on the
> history axis** (5–8 tests, mirroring the Phase 4 exhibits); regime-map
> cells are **secondary**, corrected with Benjamini–Hochberg at FDR 0.05
> across all cells per arm.

**Sequencing.** Per §8c, execution is T9-3 (staged: data → preregistered
eval → verdict), and this decision entry is required to be appended before
any ML-32M work begins — it is. No ML-32M artifact exists in the tree as
of this entry. Both T9-3 outcomes remain publishable by design: "crossover
appears when the catalog holds still" and "popularity dominates even
there" are each defensible headlines, framed as regime contrast, not
causal proof, with the MovieLens timestamp caveat per §8b.

## Phase 9 T9-3a — ML-32M data stage COMPLETE; churn contrast MEASURED: 0.0640 vs Amazon 0.4111 (2026-08-20)

**Result.** Record `20260820T134403Z-e2263d2` (kind="churn_contrast", clean
tree at e2263d2, machine of record `x86_64 · Linux`, wall 23.9s), the first
number produced from ML-32M and produced **before any model has seen the
data**, per §8c. The T8-1 statistic — share of eval-split GT interactions on
items with TRAIN support ≤ 4 — re-run on ML-32M TEST via the same imported
bucket/gate code paths (`item_train_stats`, `regime_map`): **zero+low =
0.06403** (zero 0.05793 = 45,408 interactions, low 0.00610 = 4,785) against
Amazon's recorded **0.41113** (`20260817T095926Z-633d454`, re-derived from
`results/runs.jsonl` at run time, not trusted from config). Difference
−0.34710; a 6.4× regime gap. Inputs: 8,843 TEST users, 783,896 GT
interactions, catalog 43,884, catalog-join loss 0/0; 5-core snapshot
3433604384732745693, item_features 8148139012671154899; splits
`configs/splits_ml32m.yaml` (frozen 2026-08-19, TRAIN ≤ 2022-06-30, VAL
2022-H2, TEST 2023-01-01 → 2023-11-01 excl.); dataset manifest
`data/MANIFEST_ML32M.md` (zip + 3 CSVs, SHA-256/size/row-verified in-job).

**Build receipts.** Bronze = raw exactly: ratings 32,000,204, movies 87,585,
tags 2,000,072 (`bronze-verify-ml32m` deltas all 0 vs manifest counts).
5-core keeps 31,921,467 interactions (99.75%), all 200,948 users, 43,884
items. Contract audit: **58 checks, overall PASS, zero quarantined rows**,
all FK orphan measures 0. Density vs Amazon at every step: Amazon's funnel
kept 35% of interactions and 2.4% of users; ML-32M loses almost nothing.

**Two defects caught before the record existed (both fixed, commits
0103877/d0c23a0).** (1) Spark's CSV default backslash-escape silently dropped
movieId 284105 (RFC-4180 doubled-quote title) — bronze movies 87,584/87,585
and exactly one FK orphan; fixed with `escape='"'`, catalog rebuilt, orphan
now 0. The churn statistic was never exposed (all interactions were kept; a
1-rating movie cannot reach the 5-core). (2) A first manifest commit appended
ML-32M hashes to `data/MANIFEST.md`; adversarial review caught that this
file's whole-file hash is a compared field of the pinned Amazon headline
(`eval/reproduce.FIELDS_COMPARED`), which would have flipped
`make reproduce-headline` from byte_exact to mismatch. **Deviation from §8c's
letter, disclosed:** ML-32M SHA-256s live in their own committed
`data/MANIFEST_ML32M.md`; `data/MANIFEST.md` verified byte-identical to
5fabb21 before proceeding. Invariant beats letter.

**Gate wording, interpreted.** The preregistered T8-1 bands are one-sided
Amazon language: on Amazon, <0.10 would have meant "churn diagnosis wrong."
On ML-32M the <0.10 verdict is the desired antecedent: this is the
catalog-holds-still regime. The recency axis says it plainly — 91.0% of
ML-32M TEST GT mass falls on items active in TRAIN within 90 days of the
cutoff (Amazon: the 57.3% of catalog stale >365d drew 2.0% of TEST mass, and
5.19% post-cutoff items absorbed 34.5%).

**Implication for the preregistered hypotheses.** The §8b/T9-2 premise —
"crossover exists only when the catalog holds still" — now has its antecedent
measured on both sides instead of assumed: 0.4111 vs 0.0640. The structural
cap that bound every TRAIN-frozen model on Amazon (34.5% of TEST mass
unreachable) essentially does not exist here (5.8%). T9-3b's eval is
therefore a clean two-outcome test per §8c: a crossover at some depth n* on
ML-32M supports the churn mechanism as the explanation of the Amazon null; a
second null means popularity dominates even where the catalog holds still,
and the mechanism story narrows. Both remain publishable; the framing stays
regime contrast, not causal proof (explicit ratings, rating-entry timestamps
on a backfilled catalog — Sun et al., arXiv:2307.09985).

**Boundary.** T9-3b not started: no model artifact exists for ML-32M, and the
T9-3b preregistration entry (hypotheses, VAL selection rules, grids,
multiplicity policy) must precede the first TEST run. Robustness follow-ups
from the 2026-08-20 review (atomic extract/publish, audit receipts in the
record, make -j ordering, snapshot re-pin on read) are queued as a separate
task, none blocking this record.

## Phase 9 T9-3b preregistration — ML-32M model ladder: hypotheses, arms, VAL grids, selection rules, multiplicity policy, decision rules (2026-08-20)

Registered 2026-08-20, BEFORE any ML-32M model run (VAL or TEST) and before
any ML-32M model artifact, eval cache, or item-text export exists in the
tree (plan §8c T9-3b, approved 2026-08-19; §8b T8-4 spec as amended by the
T9-2 multiplicity policy). The only ML-32M number that exists at
registration time is the T9-3a data-stage churn contrast
(`20260820T134403Z-e2263d2`). Every grid value below is *mirrored* from a
constant already frozen on the Amazon side — none is tuned, and none was
chosen after seeing an ML-32M model score, because none exists.

Governing frozen inputs: `configs/splits_ml32m.yaml` (frozen 2026-08-19 —
TRAIN ≤ 2022-06-30T23:59:59.999Z, VAL 2022-H2 ≤ 2022-12-31T23:59:59.999Z,
TEST 2023-01-01 → 2023-11-01T00:00:00Z exclusive), 5-core snapshot
3433604384732745693 (31,921,467 interactions, 200,948 users, 43,884 items),
`local.gold_ml32m.item_features` snapshot 8148139012671154899, manifest
`data/MANIFEST_ML32M.md`. Every VAL and TEST run pins this lineage in its
run record, and **a lineage or cache-alignment mismatch aborts before any
run record is appended**. All runs execute on the machine of record (the
rented Linux box) — no VAL/TEST result may mix machines (T8-2 substrate
entry: counts are identical cross-platform, floats may drift one ulp).

---

### 1. Hypotheses

**Regime premise (measured, not assumed).** T9-3a measured the antecedent
of the §8b/T9-2 claim on both sides: share of TEST GT interactions on items
with TRAIN support ≤ 4 is **0.06403 on ML-32M** (`20260820T134403Z-e2263d2`)
vs **0.41113 on Amazon Electronics** (`20260817T095926Z-633d454`) — a 6.4×
regime gap. On Amazon, 34.5% of TEST GT mass sat on items that did not
exist in TRAIN at all, a structural cap no TRAIN-frozen model could clear
(T8-1 ceiling 0.6546). On ML-32M that cap is 5.8%, and 91.0% of TEST GT
mass falls on items active in TRAIN within 90 days of the cutoff.

**H0 (null, the incumbent hypothesis).** On ML-32M, the primary confirmatory
personalized arm M\* (Rule S4) does not beat the confirmatory popularity
comparator P\* (Rule S6) at any user-history depth: for every depth bucket
*d*, `Δ(d) = NDCG@10(M*, d) − NDCG@10(P*, d) ≤ 0` after the multiplicity
correction in §5. Under H0 the Amazon result generalizes to a low-churn
catalog, catalog churn is **not** the controlling variable, and the
mechanism story from Phase 8/9 narrows sharply: popularity wins for reasons
that survive a 6.4× change in churn (pop-bias in the evaluation protocol,
full-catalog ranking difficulty, or implicit-positive semantics). That is a
*publishable* outcome and this preregistration treats it as the default,
not the failure case.

**H1 (crossover, directional).** On ML-32M there exists a depth bucket
*b* satisfying the full §7 D1 coherence condition — at least one
BH-significant positive bucket in Family P, and a shallowest bucket *b*
such that **every** bucket at or above *b* has point-estimate `Δ > 0` and no
BH-significant negative bucket sits at or above *b* — so that
**n\* = the lower edge of b**, drawn from the deep-bucket edges
**{0, 1, 5, 10, 20, 50, 100}**. H1 is therefore not "some cell came out
positive"; it is "personalization wins from a definable history depth
onward and keeps winning." Under H1 the churn contrast supports "the
crossover appears when the catalog holds still," and the Amazon null is
explained by the measured 41.11% churn rather than by the model ladder.
An outcome with significant positives that fails the coherence condition is
neither H1 nor H0 confirmed; it is the pre-named D5 (§7).

**Directional expectation is permitted here and declared: H1 is expected
to be favored over H0.** The permission derives from the *data stage*, not
from any model evaluation: the 0.06403 vs 0.41113 contrast and the 91.0%
≤90d-recency mass were computed and committed before any model saw the
data (T9-3a, §8c). No model score on either dataset informs the direction.

**What the direction must NOT do.** The folk prior "MovieLens is
collaborative filtering's home turf" is explicitly barred from every
TEST-touching decision rule below. Concretely: (a) no rule in §3, §4, §5 or
§6 conditions on the outcome direction; (b) the acceptance thresholds, the
BH family definitions, and the verdict language for T9-3c are symmetric —
a significant *negative* Δ is recorded and reported with the same
machinery, at the same α, in the same table, as a significant positive Δ;
(c) no arm gets a second TEST run, a re-tune, or a "sanity re-check"
because its result came out on the unexpected side. If the outcome is a
double null, the entry that reports it is titled and framed as a result,
not as a shortfall.

**What H1 does not claim.** Even a clean ML-32M crossover is a **regime
contrast, not causal proof**. ML-32M differs from Amazon Electronics on
several axes simultaneously (explicit 0.5–5.0 ratings vs implicit review
events; 159 mean interactions/user vs 9.4; 43,884 items vs 368,228; movie
catalog with a long-lived back catalog vs a churning electronics catalog;
rating-entry timestamps on a backfilled catalog). Churn is the axis we
*measured*; it is not the only axis that moved. See §7.

---

### 2. Arms and the complete VAL grid (mirrored from the Amazon ladder)

All VAL runs use `protocol.eval_split: val`, `knowledge_cutoff: train_end`,
`k_list: [10, 20, 50]`, `bootstrap: {n_resamples: 1000, seed: 20260805}`,
tables in the `local.gold_ml32m.*` namespace, `cache_dir:
data/eval/cache_ml32m`. Metrics emitted per run: **Recall@{10,20,50},
NDCG@{10,20}, MRR, HitRate@10**, full-catalog ranking over all 43,884
items with TRAIN-seen items excluded (invariant #4 — no sampled negatives).

| # | arm | VAL grid | # VAL runs |
|---|---|---|---|
| A0 | random | none (seeded floor) | 1 |
| A1 | pop-alltime — **P\* candidate** | `as_of: train_end`, `window_days: null` | 1 |
| A2 | pop-t12m — **P\* candidate** | `as_of: train_end`, `window_days: 365` | 1 |
| A3 | item-kNN (static) — M\* eligible | `top_n ∈ {50, 100, 200}`, `shrinkage: 0.0`, cosine, `block_size: 8192` | 3 |
| A4 | item-kNN-t12m — M\* eligible | `train_window_days: 365`; `top_n` **held** at A3's VAL winner; no free params | 1 (sanity, not selection) |
| A5 | ALS (binary) — M\* eligible | coordinate-descent sweep, below | 10 |
| A6 | ALS-decay — M\* eligible | `half_life_days ∈ {90, 365, 1460}`, all other params at A5's selected config | 3 |
| A7 | content (MiniLM) — M\* eligible | none (recipe frozen in §3) | 1 |
| A8 | blend (content+pop) — **M\* INELIGIBLE** (§4 S4) | `alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`, pop component fixed at `as_of: train_end, window_days: 365` | 5 |
| A9 | hybrid / routing policy | `n*` fitted on VAL over the §4 S5 grid; VAL-only fit | 1 confirming |

**A5, the ALS coordinate-descent sweep (mirrors Phase 3 E1–E10 exactly).**
Anchor: `rank=64, reg_param=0.01, alpha=10.0, max_iter=15,
weighting=binary, seeds.model=20260805`, `implicitPrefs=True`. One variable
moves at a time from the anchor; the axis winner is carried into the next
axis in the Phase 3 order (rank → reg → alpha → iter → weighting):

- rank ∈ {32, **64** (anchor), 128}
- reg_param ∈ {0.001, **0.01** (anchor), 0.1}
- alpha ∈ {1.0, **10.0** (anchor), 40.0}
- max_iter ∈ {8, **15** (anchor), 25}
- weighting ∈ {**binary** (anchor), rating}

1 anchor run + 9 non-anchor points = **10 VAL runs** (rank 2, reg 2, alpha
2, iter 2, weighting 1). No other ALS value may be evaluated on VAL.
`checkpoint_interval` is training infra, excluded from the param hash by
design, and may be set freely for memory/disk reasons (numerically
neutral). **Pre-declared feasibility fallback:** if rank 128 is infeasible
on the box, the frontier is published at rank 64 and the infeasibility is
disclosed — the §12-sanctioned cut, mirroring `eval_als_val_rank128.yaml`.

**`weighting=rating` on ML-32M, disclosed.** The existing code path feeds
the raw star rating as `ratingCol` under `implicitPrefs=True`
(`c = 1 + α·r`). On ML-32M `r ∈ {0.5, 1.0, …, 5.0}`, so the low end is 0.5
rather than Amazon's 1.0. Same code, slightly different confidence range;
no rescaling is applied, because rescaling would be a new tunable.

**A6, the half-life grid {90, 365, 1460} days — mirrored, not re-derived.**
These are the lab's frozen recency constants (T8-2 preregistration: the
regime-map recency axis ≤90d / 91–365d / >365d, and 1460d as the
deliberately mild decay). `age_days = train_end − latest TRAIN interaction
ts`; `r = 2^(−age_days / half_life_days)`; at age 0, `r = 1`, identical to
binary. Only the half-life is tuned.

**A8, the blend.** `alpha · minmax_per_user(content) + (1−alpha) ·
minmax(log1p(pop))`; cold rows degenerate to pure popularity by
construction. α grid {0.1, 0.3, 0.5, 0.7, 0.9} is the Amazon grid verbatim.
The blend's **internal** popularity component stays at `as_of: train_end,
window_days: 365` (the Amazon blend's component) **regardless of which
variant wins P\***, because the blend is a mirrored arm, not a comparator;
changing its internals with P\* would make it a different model from the
Amazon one and break the contrast.

**Popularity is computed from TRAIN only and is NOT refreshed at TEST
time**, for both datasets: `as_of: train_end` means pop-t12m is 6–16 months
stale when scored against ML-32M TEST (2023), exactly as it was against
Amazon TEST. Refreshing popularity on ML-32M would change the protocol
between the two regimes and destroy the contrast. This is a deliberate,
declared handicap applied symmetrically.

**Implicit-positive semantics, inherited and disclosed.** Per invariant #6,
every rating in the 5-core is an implicit positive regardless of star
value; a 0.5-star rating is a positive. A "rating ≥ 4" thresholded variant
is **out of scope and may not be added**: the 5-core, the frozen split, and
the T9-3a churn statistic are all computed over all-ratings, and re-coring
on a threshold would invalidate the very number the contrast hinges on.
This is a real limitation of the contrast and belongs in §8, not in a
post-hoc grid extension.

**Rejected arms and alternatives (recorded so they cannot be added later):**
genre-conditional popularity (the ML analogue of `pop_category`) — not in
the §8c ladder, and adding it would introduce an arm with no Amazon
counterpart; a "rating ≥ 4" positives variant (above); any deep/sequential
recommender (invariant #6, lane discipline); any Amazon↔ML-32M transfer or
joint model (meaningless under disjoint catalogs). **Peer-draft
alternatives considered and rejected during registration** (independent
Codex draft, same task, neither draft seen by the other): tag cap K=20, a
field-labeled embedding text template, an NFKC-based normalization
pipeline, the Phase 4 five-segment axis as the primary family without BH
correction, and a `top_n` grid for item-kNN-t12m. All five were rejected
for the same two reasons: they deviate from §8c's "the existing MiniLM
recipe" mirror requirement (K, template, normalization) or from T8-2
precedent and this phase's own correction mandate (uncorrected primary
family; a zero-free-parameter recency arm acquiring free parameters).

---

### 3. Content arm — recipe `v1_ml32m_title_genres_tags`, fully specified

§8c requires the content arm on **title+genres+tags** via the existing
MiniLM recipe. `local.gold_ml32m.item_features` carries only
`(parent_asin, title, genres)`; tags live in `local.silver_ml32m.tags`
(the T9-3a tags lane: `user_id, parent_asin, tag, ts`, control-char
normalized and trimmed, empty tags already quarantined). The aggregation
from tag *events* to a per-item tag *string* is therefore a new degree of
freedom, and it is frozen here, before any embedding exists, so that the
recipe hash is fully determined by this entry.

**(a) Tag source restriction — leakage guard (mandatory).** Tags are
timestamped user events, unlike Amazon's static product metadata. Only
tags with `ts ≤ 2022-06-30T23:59:59.999Z` (= `train_end`, inclusive; the
same boundary semantics as the frozen splits and the pop-t12m window) may
enter the recipe. A tag written in 2023 describing a 2023-popular film
would inject post-cutoff information into the item representation and
would silently violate invariant #1. This is *stricter* than the Amazon
recipe, and the asymmetry is deliberate and disclosed.

**(b) Normalization.** Per tag row: the silver value (already C0/DEL
stripped, `\n`→space, whitespace-collapsed, trimmed) is lowercased with
Spark `lower()` (simple, locale-independent). Rows that are empty after
normalization are dropped (re-assertion; the silver gate should already
have quarantined them — the count is recorded as a measure).

**(c) Dedup and weight.** `tag_weight(item, tag_norm) = COUNT(DISTINCT
user_id)`. One user applying the same tag to the same movie many times
counts once; two users applying it counts twice. Tag *timestamps* beyond
the §3(a) filter are not used for weighting (no recency weighting on tags
— that would be a new tunable).

**(d) Ranking and cap.** Per `parent_asin`, order by `tag_weight DESC`,
then `tag_norm ASC` (Spark's default UTF-8 binary string ordering) as the
deterministic tie-break; take the **top K = 10**. K is frozen a priori at
10, chosen to keep title+genres+tags inside all-MiniLM-L6-v2's 256-word-piece
window rather than by any quality signal; **no K sweep is permitted** —
sweeping K would tune a text recipe and would multiply recipe hashes.
No minimum-weight filter and no tag-length cap are applied: for thinly
tagged movies the top-10 may include idiosyncratic single-user tags, and
that is accepted and disclosed, in preference to inventing a threshold
constant. Over-long assembled text is truncated by the model's own
`max_seq_length=256`, which is deterministic — the same truncation the
Amazon recipe relies on.

**(e) Join order and coverage.** LEFT JOIN from the 5-core catalog
(`local.gold_ml32m.interactions_5core` distinct `parent_asin`, reordered to
the eval cache's `item_ids` sequence, set-equality asserted first, exactly
as `item_text.export_item_text` does on Amazon) onto
`local.gold_ml32m.item_features` onto the aggregated tag table. A movie
with no in-window tags gets an **empty list**, never a placeholder token.
The share of catalog items with zero in-window tags and the share with
empty `genres` are published to `dq_ml32m.dq_results` as measures **before**
the embedding job runs.

**(f) Genres.** Taken as stored in `gold_ml32m.item_features`
(`array<string>`, already split on `|`, with `(no genres listed)` mapped to
an empty array by the T9-3a silver transform). Order preserved as stored —
deterministic given the SHA-verified source; no re-sorting, mirroring how
the Amazon recipe consumes `features`.

**(g) Text template (exact).** Mirrors `minilm_embed.build_recipe_text`:
parts joined by a single space, null/empty parts skipped, no separator
tokens, no field labels:

```
text(item) = " ".join(
    ([title] if title else [])
    + [g for g in genres if g]
    + [t for t in tags_top10 if t]
)
```

Titles keep their MovieLens year suffix (`"Toy Story (1995)"`) — kept, not
stripped, and disclosed.

**(h) Identity / hash / model artifact.** `RECIPE_ID =
"v1_ml32m_title_genres_tags"`, `MODEL_ID =
"sentence-transformers/all-MiniLM-L6-v2"`, `EMBED_DIM = 384`,
`BATCH_SIZE = 256`, fp16 output, row-aligned to the eval cache `item_ids`.
The embedding uses **the same locally cached model artifact the Amazon
recipe used** — not a fresh download that could resolve to a different
revision. The resolved model revision / weights hash is **recorded in the
embedding artifact manifest** (`minilm_manifest.json`) as provenance; no
specific revision hash is preregistered here, because none has been
verified at registration time, and preregistering an unverified hash would
be a fabricated receipt.

The recipe hash **must** bind the aggregation rule, otherwise this
specification is not enforced by the artifact identity. `recipe_hash()`
gains an optional `extra` mapping that enters the canonical JSON **only
when non-None**, mirroring the established `half_life_days`-enters-the-param-
hash-only-under-`time_decay` pattern — so Amazon's recorded recipe hash
`1f7878ff82bf` is provably unchanged (a back-compat unit test asserts this
before any ML-32M embedding is produced). No expected ML-32M hash value is
preregistered (it is a function of this spec and is recorded when first
computed); the run config's declared `recipe_hash` and the artifact
manifest's `recipe_hash` must match exactly, and a mismatch **aborts**.
For ML-32M:

```
fields = ["title", "genres", "tags_top10"]
joiner = " "
extra  = {
  "tag_source": "local.silver_ml32m.tags",
  "tag_cutoff": "2022-06-30T23:59:59.999Z",   # inclusive
  "tag_norm": "silver_sanitized|lower|trim",
  "tag_weight": "count_distinct_user_id",
  "tag_order": "weight_desc,tag_asc",
  "tag_top_k": 10,
  "genres_source": "local.gold_ml32m.item_features.genres",
  "genres_order": "as_stored",
  "empty_policy": "skip"
}
```

**(i) Scoring.** Unchanged from Amazon: user profile = L2-normalized mean
of the user's TRAIN item embeddings, cosine similarity against the
L2-normalized catalog; cold users collapse to all-zero scores by
construction; the same ANN/exact path as the Amazon content arm.

**(j) §8c fallback — fail-closed. Quoted:** "if metadata proves unusable,
disclose as a limitation rather than substituting."

Metadata is **"unusable" only** for (i) a structural/provenance failure
detected *before* content evaluation — missing required title/genre/tag
columns; non-unique or incomplete catalog alignment; an unavailable or
empty TRAIN-cutoff tag source; a recipe/hash/model-revision mismatch;
non-finite or wrong-shape embeddings; or inability to build a receipted
artifact — or (ii) the outcome-blind coverage threshold fixed here:
**> 50% of the 5-core catalog has an entirely empty assembled text string**
(no title AND no genres AND no in-window tags). Coverage below 100% but
above that line is a disclosure, not a stop.

**Poor VAL quality is explicitly NOT unusable metadata.** A content arm
that simply scores badly on VAL goes to TEST under the frozen protocol like
any other arm.

If a qualifying failure occurs: **omit both the content arm (A7) and its
dependent blend (A8)**, select M\* from the remaining classical arms by the
unchanged Rule S4, and disclose the missing arms and the reason in T9-3c
and in the case study's "What this does not prove." Do **not** silently
substitute plots/synopses, the tag-genome features, a different tag cutoff
or cap, another encoder, collaborative embeddings, or any other
metadata/model recipe.

---

### 4. VAL selection rules (all tuning on VAL only — invariant #1)

Selection metric is **global VAL NDCG@10**, the lab's primary metric, at
the primary model seed **20260805** only (Phase 3 / T8-2 tuning practice).

**Rule S1 — argmax with a CI tie-break (mirrors T8-2).** Within each arm's
grid, the selected config is the argmax of global VAL NDCG@10. If the
argmax's 95% bootstrap CI **overlaps** the incumbent/anchor's CI, the
comparison is a tie and the **cheaper** config wins (smaller rank, fewer
iterations, smaller `top_n`, smaller `alpha`, `binary` over `rating`,
smaller half-life). Applied axis by axis for A5's coordinate sweep, in the
declared axis order.

*Rejected alternative, recorded:* Phase 3's absolute 0.0001 tie band. It is
calibrated to Amazon's ~0.004 NDCG@10 scale; ML-32M's dense-regime scores
are expected to be an order of magnitude larger, where 0.0001 would be a
rounding error and the band would never bind. The CI-overlap rule is
scale-free and is the lab's most recent precedent (T8-2). Consequence
accepted: with 200,948 users, VAL CIs will be narrow and the tie-break may
almost never trigger — i.e. selection will be near-pure argmax.

**Rule S2 — A4 (item-kNN-t12m) has no free parameters.** Its single VAL run
is recorded as a *sanity* record, not a selection record, with `top_n` held
at A3's winner. On Amazon this arm collapsed on VAL (0.00056 vs pop-t12m
0.01034) and still went to TEST under the frozen protocol; the same applies
here — a bad VAL number does not remove an arm.

**Rule S3 — A6 (ALS-decay) regression clause (mirrors T8-2 verbatim in
substance).** If all three half-lives score below the selected static-ALS
(A5) VAL baseline, the decay mechanism failed on VAL; the best of the three
still goes to TEST (the confirmatory question is vs P\*, not vs static
ALS) with the VAL regression disclosed. More generally: if every value in a
family loses on VAL to its comparator or incumbent, the mechanically
selected argmax still advances to TEST and the regression is disclosed —
**failure to improve is not permission to expand or rerun a grid.**

**Rule S4 — the primary confirmatory arm M\* is selected on VAL, blind to
TEST.** Exactly one arm carries the primary confirmatory family in §5:
**M\* = argmax of global VAL NDCG@10 among {A3 item-kNN, A4 item-kNN-t12m,
A5 ALS, A6 ALS-decay, A7 content}** at each arm's selected config.
CI-overlap ties break toward the **simpler** arm in the order
**A3 < A4 < A5 < A6 < A7** (fewest moving parts first).

**A8 (the blend) is excluded from M\* eligibility.** The blend contains the
popularity comparator's own signal by construction (`(1−α)·minmax(log1p(pop))`),
so a blend-beats-popularity sign change would not answer the question this
phase asks — *does personalization beat popularity, and at what history
depth* — it would mostly report that adding popularity to popularity does
not hurt. The blend keeps its full VAL α grid, its one TEST run, and its
place in the secondary families; it is reported as **deployment /
performance evidence** (the Amazon-side "one effective arm" result has the
same status) and **cannot set n\***.
A0 (random) is a floor, A1/A2 are the comparator candidates, and A9 is a
routing wrapper — none is M\*-eligible either.

**Rule S6 — the confirmatory popularity comparator P\* is selected on VAL,
blind to TEST.** **P\* = argmax of global VAL NDCG@10 between A1
(pop-alltime) and A2 (pop-t12m).** CI-overlap tie → the variant with the
**larger CI lower bound**; if still tied, **pop-alltime** by arm-id order.
Rationale: on a low-churn catalog where 91.0% of TEST GT mass sits on
items active within 90 days of the cutoff and the back catalog stays
relevant, **all-time popularity may well be the stronger reference**, and a
crossover claim is only credible against the *stronger* popularity baseline
— beating the weaker of two popularity variants is not a crossover, it is a
comparator choice. This rule is outcome-blind (VAL-only, fixed before any
TEST number exists) and mechanical.

*Disclosed asymmetry:* the Amazon side's committed confirmatory comparator
is **pop-t12m** (Phase 4/8 records). If P\* resolves to pop-alltime on
ML-32M, the two regimes are compared each against **its own VAL-selected
popularity reference**, and the cross-regime chart labels each regime's
reference explicitly rather than forcing a single variant across datasets.
Both popularity variants still receive their own TEST records either way,
so the non-selected variant's numbers are published, not hidden.

**Rule S5 — routing policy (A9), VAL-only fit.** `n*` is fitted on VAL over
the Phase 4 T13 grid (`n_star ∈ {none/∞} ∪ the deep-bucket lower edges
{1, 5, 10, 20, 50, 100}`, low arm = M\*, high arm = P\*), selecting the cell
that maximizes global VAL NDCG@10. **If the fitted `n*` is infinite** (i.e.
routing degenerates to a single arm, as it did on Amazon), no hybrid TEST
run is spent — the hybrid would be numerically identical to that arm, and
identity is asserted on the VAL confirming run instead. **If `n*` is
finite**, exactly one hybrid TEST run is spent.

**Freezing.** After the selections are logged in this file, the selected
config hashes, **M\***, and **P\*** are frozen before any TEST run.

---

### 5. Multiplicity policy

**The §8c/T9-2 amendment, verbatim:**

> The §8b T8-4 spec stands as written with **one amendment**: the
> preregistration must now include an explicit multiplicity policy — the
> **primary confirmatory test is the per-depth-segment crossover on the
> history axis** (5–8 tests, mirroring the Phase 4 exhibits); regime-map
> cells are **secondary**, corrected with Benjamini–Hochberg at FDR 0.05
> across all cells per arm.

**Operationalization.**

**(a) The history axis is the 7 T8-3 deep buckets, plus the global cell —
8 tests.** `DEEP_BUCKET_LABELS = ("0", "1-4", "5-9", "10-19", "20-49",
"50-99", "100+")`, lower edges **{0, 1, 5, 10, 20, 50, 100}**, plus one
all-users test = 8, inside §8c's "5–8 tests" allowance. *Why not the Phase
4 five-segment axis (`0/1-4/5-9/10-19/20+`)?* Because ML-32M's 5-core
averages ~159 interactions per user (31,921,467 / 200,948, T9-3a record)
against Amazon's ~9.4: the Amazon axis would place the overwhelming
majority of ML-32M users in a single `20+` bucket and destroy exactly the
resolution the crossover question needs. This choice is made from a
**data-stage** density fact already committed in T9-3a, with no TEST-side
counting, and is fixed now. The five-segment axis is still computed and
reported as a **secondary** cross-dataset comparability exhibit.
Disclosure of an asymmetry: the deep buckets were labeled *exploratory* on
Amazon (T8-3); here they are preregistered as *confirmatory*, and any
Amazon-side numbers shown beside them in the regime-contrast chart retain
the Amazon exploratory label.

**(b) Family P (primary confirmatory).** **M\* vs P\***, paired delta on
**NDCG@10**, one test per populated deep bucket + one global test (m ≤ 8).
A bucket with **zero TEST users** yields no test and is excluded with its
count disclosed; that exclusion is driven by user counts, not by outcomes.
Benjamini–Hochberg at **FDR 0.05** within Family P.

**(c) Family S1 (secondary, per arm).** Every other arm (including A8 the
blend, and the non-selected popularity variant) **vs P\*** across the same
deep buckets: BH at FDR 0.05 **within that arm's** bucket family.

**(d) Family S2 (secondary, per arm).** Regime-map cells (support axis:
zero / low / high; recency axis: ≤90d / 91–365d / >365d / absent — the
committed `CELL_AXES`), recomposed through the T8-1 machinery, **each arm
vs P\***: BH at FDR 0.05 **across all cells per arm**, exactly as the
amendment reads.

**(e) p-values.** The bootstrap emits CIs, not p-values. The two-sided
achieved-significance-level is computed **from the same 1,000 paired
resampled deltas, same seed, same resample matrix** — a pure post-processing
of resamples already drawn, requiring no additional TEST evaluation:

```
p = min(1.0, 2 * min( (1 + #{D_b <= 0}) / (B+1),
                      (1 + #{D_b >= 0}) / (B+1) ))     # B = 1000
```

Resolution floor is 1/1001 ≈ 0.000999; any p at the floor is reported as
`< 0.001`, not as an exact value.

**(f) Direction and symmetry.** BH runs on two-sided p-values. A *win*
requires `p_BH-significant AND Δ > 0`; a *loss* requires `p_BH-significant
AND Δ < 0`. Both are recorded in the same table with the same emphasis.

**(g) Metric robustness, reported not confirmatory.** Recall@20 is computed
for every test in every family and reported alongside. A crossover claim is
labeled **"metric-robust"** only when Recall@20 agrees in sign and
significance. §8c's own critique — that NDCG@10 and Recall@20 are highly
correlated on the same cells, so the dual-metric guard is only partial
protection — is why the dual-metric agreement is a *label*, not the
confirmatory criterion. The confirmatory criterion is BH-corrected NDCG@10.

**(h) Uncorrected numbers are still published**, clearly marked as
uncorrected, so that this phase's own correction of Phase 8 is auditable
rather than a second unauditable judgment call.

---

### 6. TEST protocol (frozen-TEST invariant #1)

**Exactly ONE TEST evaluation per selected arm. No iteration, no re-runs on
an unexpected result.** TEST is entered only after (i) all VAL runs are
recorded in `results/runs.jsonl`, (ii) the §4 selections — including M\*
and P\* — are appended to this log with their VAL numbers, and (iii) the
content-recipe artifact hashes are recorded.

| arm | TEST records | seeds |
|---|---|---|
| A0 random | 1 | fixed seed recorded |
| A1 pop-alltime | 1 | deterministic |
| A2 pop-t12m | 1 | deterministic |
| A3 item-kNN @ selected `top_n` | 1 | deterministic |
| A4 item-kNN-t12m | 1 | deterministic |
| A5 ALS @ selected config | 3 | 20260805 (primary), 20260806, 20260807 — mean±sd |
| A6 ALS-decay @ selected half-life | 3 | same three seeds — mean±sd |
| A7 content (MiniLM) | 1 | deterministic given the recipe hash |
| A8 blend @ selected α | 1 | deterministic |
| A9 hybrid | 0 or 1 | only if VAL-fitted `n*` is finite (Rule S5) |

Both popularity variants are evaluated on TEST regardless of which one is
P\*; only P\* appears as the comparator in Families P/S1/S2.

**Seed discipline for inference.** **Seed 20260805 is the sole per-user
artifact used for all paired TEST deltas and for every segment-level and
cell-level inference.** Seeds 20260806 and 20260807 are **stability
evidence only** — they contribute the reported mean±sd and nothing else,
and they **never enter a paired CI, a p-value, or a BH family**. This
prevents a second seed from becoming a second look at TEST.

**Deterministic vs stochastic, and the "headline at 3 seeds" requirement.**
Only the ALS family is stochastic (Spark ALS retraining is not bit-stable;
rescoring from a persisted factor artifact is). A3/A4/A7/A8 are
deterministic functions of frozen inputs — their identity is carried by
recorded artifact SHA-256s, not by seed replication, and re-running them
would produce byte-identical scores. **If M\* lands on a deterministic arm,
the §8c "headline config at 3 seeds" requirement is satisfied by the ALS
arms' 3-seed records plus M\*'s artifact-hash determinism, and that
substitution is disclosed in T9-3c** — spending three identical TEST runs
on a deterministic arm would consume TEST budget for no information.

**Infrastructure retries.** A run that dies for infrastructure reasons
(OOM, disk, preemption, host failure) may be retried **only with a
byte-identical config**, and the retry is logged here with its cause. A
**completed** TEST record is never replaced, re-run, or deleted — the log
is append-only and a wrong run gets a superseding entry, not a rewrite
(invariant #3). A lineage or cache-alignment mismatch **aborts before any
run record is appended**, so a mis-pinned run never becomes a record that
has to be superseded.

**Comparisons.** For every arm: paired-bootstrap delta **vs P\***
(1,000 resamples, seed 20260805, **paired within comparison** — the same
resample matrix for both members, per-segment child seeds
`default_rng([base_seed, axis_ordinal, segment_ordinal, bucket_ordinal])`),
over: global, the 7 deep buckets (Family P/S1), the 5 Phase 4 segments
(comparability exhibit), and the regime-map cells (Family S2, recomposed
through the committed T8-1 machinery with its identity-anchor assertion —
single-bucket recomposition must reproduce the arm's own recorded global
metrics, and any mismatch aborts).

**Reproducibility.** Every TEST record carries config hash, git SHA,
`data/MANIFEST_ML32M.md` hash, and the ML-32M Iceberg snapshot ID
(invariant #3), and the ladder is regenerable via **`make reproduce-ml32m`**
pinned to that snapshot (§8c acceptance). `data/MANIFEST.md` is **never**
touched — it is a compared field of the pinned Amazon headline
(`eval/reproduce.FIELDS_COMPARED`), as T9-3a discovered.

---

### 7. Decision rules → T9-3c verdict language (both outcomes ship)

Symmetric by construction: each outcome below has a pre-written headline,
a pre-agreed chart, and the same evidence obligations. All tests below are
**M\* vs P\***, BH-corrected within Family P (§5b).

**D1 — CROSSOVER at n\*.** Requires **both**:
   (i) at least one BH-significant **positive** bucket in Family P; **and**
   (ii) a **coherence** condition: there exists a shallowest bucket *b*
   such that **every** bucket at or above *b* has point-estimate `Δ > 0`,
   and **no** BH-significant **negative** bucket sits at or above *b*.
   Then **n\* = the lower edge of b**, drawn from {0, 1, 5, 10, 20, 50, 100}.
T9-3c reports the crossover with CIs; the crossover chart is rendered under
**both** regimes side by side, each against its own VAL-selected popularity
reference (Rule S6); the headline becomes *"the crossover appears when the
catalog holds still — 6.4× less churn, and personalization finally wins
past n\* interactions of history."* Required companions: the
metric-robustness label (§5g), the routing policy refit under the ML-32M
regime, and an explicit statement that Amazon's null is *consistent with*
the churn mechanism, **not proven** by it (§1, "regime contrast, not causal
proof").

**D2 — GLOBAL-ONLY WIN.** The global test in Family P is BH-significant and
positive but no individual depth bucket is. Verdict: *"personalization wins
on average on the low-churn catalog, but no history-depth threshold is
identified"* — an aggregate win with n\* undetermined; the routing narrative
stays at "no routing," and the aggregate win is reported as such. This
intermediate outcome is named now so it cannot be silently rounded into D1.

**D3 — DOUBLE NULL.** No BH-significant positive test in Family P. Verdict:
*"popularity dominates even where the catalog holds still"* — the Amazon
null replicates across a 6.4× churn gap; catalog churn is **not** the
controlling variable; the mechanism story narrows to explanations that
survive both regimes, and those candidate explanations (protocol popularity
bias under full-catalog ranking, implicit-positive semantics on explicit
ratings, TRAIN-frozen popularity's surprising strength on stale windows) are
listed as open questions, not asserted. The case study's headline stays
null-first, now with an external replication behind it — a **stronger**
claim than the single-dataset null it replaces (§12's standing position:
a null here is an outcome, not a failure).

**D4 — SIGNIFICANT NEGATIVES.** If BH-significant *negative* deltas appear
(personalization measurably worse than popularity at some depth), they are
reported in the same table and the same prose register as wins would be.

**D5 — MIXED under the preregistered crossover definition.** Condition (i)
holds but condition (ii) fails: BH-significant positives are interleaved
with BH-significant negatives, or a bucket above the candidate *b* has a
negative point estimate. Verdict language: *"significant per-depth wins
exist but do not form a crossover under the preregistered definition —
no n\* is claimed."* The positive buckets are reported with their CIs and
BH-adjusted p-values, and the incoherence is shown, not smoothed. **D5 may
not be reported as D1, and may not be reported as D3** — an isolated
positive cell is exactly the Phase 8 failure mode this phase exists to
correct (T9-1), and burying real per-depth wins in a "null" headline would
be the mirror-image error.

**In every branch:** secondary families (S1, S2) are reported as
*secondary*, BH-corrected, and may not be promoted to the headline — that
promotion is precisely the Phase 8 error this phase exists to correct
(T9-1). The blend (A8), being M\*-ineligible, may appear in the headline
only as deployment/performance evidence and never as the crossover arm.
Verdict entry appended to this log; records in `results/runs.jsonl`; case
study and demo updated only where existing numbers are directly superseded.

---

### 8. Caveats, fixed in advance

1. **Regime contrast, not causal proof.** Several variables move at once
   between the datasets (§1). Churn is the axis measured; density, catalog
   size, feedback type, and domain all differ. No causal claim about churn
   is licensed by either outcome.
2. **MovieLens timestamp caveat.** ML-32M timestamps are *rating-entry*
   times, not consumption times, on a catalog that users backfill — a user
   in 2023 may rate a 1994 film. Temporal splits on this dataset therefore
   split *rating behavior*, not *release/consumption*, which mechanically
   dampens catalog churn and is part of why 0.06403 is small (Sun et al.,
   arXiv:2307.09985). Cited in T9-3c and in the site's "What this does not
   prove."
3. **Explicit ratings treated as implicit positives** (§2), including
   0.5-star ratings. Frozen upstream; a thresholded variant is out of scope.
   Implicit binary ALS also intentionally ignores star magnitude except on
   the single `weighting=rating` sweep point.
4. **Popularity is TRAIN-frozen and stale at TEST time** on both datasets by
   design (§2). This handicaps popularity identically in both regimes; it
   does not eliminate the protocol's general pop-bias.
5. **The comparator differs by regime if Rule S6 selects pop-alltime** —
   disclosed in §4 and labeled on every cross-regime exhibit.
6. **The ALS sweep is a preregistered coordinate search** over the Amazon
   values, not an exhaustive joint optimum. Reported regardless of outcome.
7. **Tag coverage** is partial and the top-10 rule may admit single-user
   tags on thinly-tagged movies (§3d); the coverage measures are published
   before the embedding runs.
8. **Directional hypothesis** is declared (§1) and is data-stage-derived; the
   decision rules are outcome-symmetric and were written before any model
   number existed.
9. **Cross-machine float drift**: all ML-32M VAL and TEST runs execute on the
   machine of record; no result mixes machines (T8-2 substrate entry).
10. **Research license**: ML-32M is never redistributed; only hashes,
    aggregate metrics, and receipted derived results are published. Cite
    Harper & Konstan 2015.

---

### 9. Mechanics declared before implementation

- New code required before the first VAL run: ML-32M eval cache extract
  (`data/eval/cache_ml32m/<snapshot>/`), the `gold_ml32m.item_text` build +
  export in eval-cache `item_ids` order, the tag aggregation of §3, the
  `recipe_hash(extra=…)` extension, and the ML-32M-namespaced config set.
  Unit tests land **with** that code and **before** any VAL run: recipe-hash
  back-compat (Amazon `1f7878ff82bf` unchanged), tag-cutoff boundary
  (inclusive at `train_end`), tag ranking determinism including the
  lexicographic tie-break, empty-tag/empty-genre handling, and the bootstrap
  ASL p-value against a hand-computed fixture.
- The ALS `age_days` cache for A6 is built by the same one-shot deterministic
  job used in T8-2, pinned to the ML-32M 5-core snapshot, with the
  exact-alignment assertion against the cached pair arrays (mismatch aborts).
- No ML-32M model artifact, eval cache, or embedding exists as of this
  entry. The first VAL run may not start until this entry is committed.

## Phase 9 T9-3b VAL ladder complete — selections under the preregistered rules; M*, P*, n* frozen (2026-08-21)

All 24 preregistered VAL runs are recorded in `results/runs.jsonl` (run IDs
below), executed on the machine of record, each carrying the ML-32M manifest
hash, the frozen-splits file hash, and the four pinned Iceberg snapshot IDs
(5-core 3433604384732745693). VAL has 6,464 users. Every selection below is
the mechanical application of the committed preregistration's rules to these
numbers — no rule was adjusted after any number was seen.

### VAL results (global NDCG@10, 95% bootstrap CI, seed 20260805)

| arm / config | run_id | NDCG@10 [95% CI] |
|---|---|---|
| A0 random | 20260820T162740Z-c6b4195 | 0.001538 [0.001188, 0.001920] |
| A1 pop-alltime | 20260820T162820Z-c6b4195 | 0.165616 [0.159084, 0.172225] |
| A2 pop-t12m | 20260820T162859Z-c6b4195 | 0.212916 [0.206401, 0.219909] |
| A3 kNN n50 | 20260820T162939Z-c6b4195 | 0.071212 [0.067791, 0.074657] |
| A3 kNN n100 | 20260820T163422Z-c6b4195 | 0.071998 [0.068363, 0.075751] |
| A3 kNN n200 | 20260820T163903Z-c6b4195 | 0.072349 [0.068851, 0.076082] |
| A4 kNN-t12m n50/365d | 20260820T172245Z-7ae2396 | 0.079977 [0.076256, 0.083693] |
| A5 anchor r64/0.01/α10/15/bin | 20260820T164811Z-fba92d0 | 0.074795 [0.071388, 0.078358] |
| A5 rank32 | 20260820T170241Z-fba92d0 | 0.071106 [0.067472, 0.074743] |
| A5 rank128 | 20260820T171654Z-fba92d0 | 0.073781 [0.070515, 0.077358] |
| A5 reg0.001 | 20260820T175544Z-414da2c | 0.074714 [0.071318, 0.078313] |
| A5 reg0.1 | 20260820T181436Z-414da2c | 0.075437 [0.071857, 0.079203] |
| A5 alpha1 | 20260820T185409Z-a5fa9cd | 0.083661 [0.080017, 0.087672] |
| A5 alpha40 | 20260820T190912Z-a5fa9cd | 0.059971 [0.056909, 0.063035] |
| A5 iter8 | 20260820T193549Z-a0bb406 | 0.083117 [0.079468, 0.087160] |
| A5 iter25 | 20260820T194848Z-a0bb406 | 0.083419 [0.079655, 0.087403] |
| A5 rating | 20260820T201649Z-7e70f2c | 0.081366 [0.077813, 0.085259] |
| A6 hl90 | 20260820T204324Z-d89b3b3 | 0.084262 [0.080625, 0.088261] |
| A6 hl365 | 20260820T205745Z-d89b3b3 | 0.084841 [0.081076, 0.088768] |
| A6 hl1460 | 20260820T211139Z-d89b3b3 | 0.084369 [0.080681, 0.088282] |
| A7 content | 20260820T172358Z-7ae2396 | 0.021129 [0.019439, 0.022788] |
| A8 blend α0.1 | 20260820T172449Z-7ae2396 | 0.215801 [0.209313, 0.222756] |
| A8 blend α0.3 | 20260820T172544Z-7ae2396 | 0.214668 [0.208233, 0.221687] |
| A8 blend α0.5 | 20260820T172644Z-7ae2396 | 0.204401 [0.197637, 0.211096] |
| A8 blend α0.7 | 20260820T172744Z-7ae2396 | 0.185652 [0.179347, 0.192428] |
| A8 blend α0.9 | 20260820T172855Z-7ae2396 | 0.162221 [0.155722, 0.169266] |

### Rule S1 applications (axis by axis, argmax with CI tie-break)

- **A3 kNN top_n**: argmax n200 (0.072349); n50 and n100 CIs both overlap the
  argmax's → tie → cheapest → **top_n = 50**.
- **A5 rank**: argmax = anchor rank 64 itself → **64** (no tie question).
- **A5 reg**: argmax reg 0.1 (0.075437); CI overlaps incumbent 0.01's → tie →
  cost-neutral axis, tie retains the incumbent (also the independent peer
  draft's rule, recorded in the prereg's rejected-alternatives note) → **0.01**.
- **A5 alpha**: argmax alpha 1 (0.083661); its CI lower bound 0.080017 clears
  the incumbent alpha 10's upper bound 0.078358 → **CI-clear win → alpha 1.0**.
- **A5 iter**: argmax = incumbent 15 → **15**.
- **A5 weighting**: argmax = incumbent binary (0.083661 vs rating 0.081366) →
  **binary**.
- **A5 selected: rank 64, reg 0.01, alpha 1.0, max_iter 15, binary** —
  VAL 0.083661 [0.080017, 0.087672].
- **A6 half-life**: argmax hl365 (0.084841); hl90's CI overlaps → tie →
  smaller half-life (stronger recency, per the preregistered tie-break) →
  **half_life_days = 90**, VAL 0.084262 [0.080625, 0.088261]. Rule S3
  regression clause does not fire (no decay run fell below selected static A5).
- **A8 blend alpha**: argmax α0.1 (0.215801) → **α = 0.1** (also the smallest
  α in its overlap set — argmax and tie-break agree).

### Rule S4 — M* (primary confirmatory arm)

Argmax among M*-eligible arms {A3, A4, A5, A6, A7} at selected configs is
A6 ALS-decay hl90 (0.084262). Its 95% CI overlaps both A5 (0.083661
[0.080017, 0.087672]) and A4 (0.079977 [0.076256, 0.083693]; 0.083693 >
0.080625) → tie set {A4, A5, A6} → tie breaks to the simplest arm in the
preregistered order A3 < A4 < A5 < A6 < A7 →

**M* = A4, item-kNN-t12m (top_n 50, train_window_days 365), VAL NDCG@10
0.079977 [0.076256, 0.083693], run 20260820T172245Z-7ae2396.**

Noted for honesty: the marginal-CI overlap rule (as preregistered) drove this
tie; paired-bootstrap deltas between the tied arms were not computed and may
well have separated them, but the preregistration fixed marginal-CI overlap
as the criterion and it is applied as written.

### Rule S6 — P* (confirmatory popularity comparator)

pop-t12m 0.212916 [0.206401, 0.219909] vs pop-alltime 0.165616 [0.159084,
0.172225]: argmax pop-t12m, CIs disjoint → **P* = pop-t12m** — the same
reference as the Amazon side, so no comparator asymmetry needs disclosing.
(Encoding note: the code expresses all-time as `window_days: 0`, the
prereg's `window_days: null`; semantics identical, param-hash convention
follows the code as on the Amazon side.)

### Rule S5 — routing policy fit (VAL-only)

Fit per the committed grid (`n_star ∈ {1,5,10,20,50,100,∞}`, low = M*
kNN-t12m, high = P* pop-t12m, objective = global VAL NDCG@10), by exact
n_train routing over the per-user artifacts (results/policy_select_ml32m_val.json,
module extension committed before the fit; Amazon default path regression-
tested unchanged): **fitted n* = 100, objective 0.089686**. Per the committed
rule, a finite n* spends exactly one hybrid TEST run.

Disclosures fixed before TEST: (a) the preregistered grid mirrors T13's shape
and therefore contains **no pure-P\* cell**; the fitted cell's 0.089686 is far
below P* alone (0.212916), so the policy-level story on VAL is already
"routing away from popularity hurts" — the hybrid TEST run reports the fitted
cell because the rule says so, not because it is expected to win. (b) A
data-stage fact surfaced by the fit: **1,859 of 6,464 VAL users (28.8%) have
n_train = 0**, and P*'s NDCG@10 on that cold bucket is **0.487783** — the
largest cell mean anywhere on VAL — vs M*'s 0.055590. Popularity's dominance
on ML-32M VAL is concentrated exactly where personalization is impossible.

### Clarification (reporting only, fixed before TEST)

The prereg §5(e) two-sided ASL floor is 2/1001 ≈ 0.001998, not 1/1001 as its
"report as < 0.001" clause implies; floor-attaining p-values will be reported
as "p = 2/1001 (resolution floor)". Affects reporting format only, not BH at
0.05. (Documented in `asl_p_value`'s docstring and pinned by test.)

### Frozen for TEST

M* = kNN-t12m (n50, 365d) · P* = pop-t12m · A5 = r64/0.01/α1.0/15/binary ·
A6 = A5 + hl90 · A7 = content recipe 9fa4d7d913f1 · A8 = blend α0.1 ·
A9 = hybrid n*=100 (one run) · TEST seeds: ALS-family 20260805/20260806/
20260807; seed 20260805 is the sole paired-inference artifact. One TEST
evaluation per arm; no VAL revisiting.

## Phase 9 T9-3c VERDICT — crossover at depth n* = 20 on ML-32M (D1, preregistered definition); the regime contrast lands (2026-08-21)

**Verdict (Family P, the preregistered primary confirmatory family): D1 —
CROSSOVER at n\* = 20.** On ML-32M TEST (8,843 users, full-catalog ranking
over 43,884 items), the primary confirmatory arm M\* (item-kNN-t12m, Rule S4)
beats the confirmatory popularity comparator P\* (pop-t12m, Rule S6) with
BH-corrected significance (FDR 0.05) in every history-depth bucket from 20
TRAIN interactions upward, and the §7 D1 coherence condition holds with no
classifier caveats. Evidence: `results/confirmatory_ml32m_test.json`
(committed, machine of record), derived solely from the one-shot TEST records.

### Family P (M* − P*, paired ΔNDCG@10, BH FDR 0.05 within family, m = 8)

| bucket | n users | Δ | 95% CI | p (uncorr.) | q | BH |
|---|---|---|---|---|---|---|
| 0 | 3,882 | −0.3844 | [−0.3926, −0.3750] | 2/1001 (floor) | 0.0053 | **sig. LOSS** |
| 1-4 | 25 | −0.0140 | [−0.1016, +0.0656] | 0.769 | 0.879 | ns |
| 5-9 | 19 | −0.0038 | [−0.1390, +0.1254] | 0.949 | 0.949 | ns |
| 10-19 | 53 | −0.0439 | [−0.1003, +0.0071] | 0.106 | 0.141 | ns |
| 20-49 | 242 | **+0.0333** | [+0.0107, +0.0552] | 0.0060 | 0.0120 | **sig. WIN** |
| 50-99 | 431 | **+0.0198** | [+0.0034, +0.0381] | 0.0160 | 0.0256 | **sig. WIN** |
| 100+ | 4,191 | **+0.0112** | [+0.0071, +0.0149] | 2/1001 (floor) | 0.0053 | **sig. WIN** |
| global | 8,843 | −0.1619 | [−0.1677, −0.1560] | 2/1001 (floor) | 0.0053 | **sig. LOSS** |

D1 conditions: (i) BH-significant positive buckets {20-49, 50-99, 100+}; (ii)
shallowest coherent bucket = 20-49, all buckets at/above it point-estimate
positive and BH-significant, no significant negative at/above → n\* = 20.
The **D4 flag is raised and reported in the same register**: bucket 0 and the
global test are BH-significant *losses*. Both facts are the finding: on the
low-churn catalog, personalization wins **from depth 20 onward** and loses
**catastrophically for zero-history users** (43.9% of TEST users, Δ −0.38 —
P\*'s NDCG@10 for cold users is enormous on this catalog), so popularity still
wins *on average* (global Δ −0.16).

### The regime contrast (the §8c headline)

Amazon Electronics (41.11% zero/low-support churn): **no crossover at any
depth** — every personalized arm lost to pop-t12m in every history segment
(T9-1 repin; robust null). ML-32M (6.40% churn, 6.4× less): **crossover at
n\* = 20**. The crossover appears when the catalog holds still. This is a
**regime contrast, not causal proof**: the datasets differ on domain, density
(159 vs 9.4 mean interactions/user), catalog size, feedback semantics, and
timestamp semantics simultaneously; churn is the axis we *measured*
(`20260820T134403Z-e2263d2` vs `20260817T095926Z-633d454`), not the only axis
that moved.

### Mandatory labels and qualifications (preregistered, §5g/§7/§8)

1. **NOT metric-robust (§5g).** Recall@20, corrected in its own BH family,
   confirms **none** of the three winning buckets: sign agrees at 20-49
   (+0.0228) and 50-99 (+0.0097) but neither is BH-significant, and at 100+
   the Recall@20 delta is −0.0008 (sign flips). The confirmatory criterion
   is BH-corrected NDCG@10 (preregistered), so D1 stands, but the win is a
   **top-of-list ranking-quality effect**, not a recall-mass effect, and every
   exhibit citing the crossover must carry this label.
2. **Secondary families corroborate the shape** (S1, BH per arm): ALS
   (static) and ALS-decay each independently win the same three deep buckets
   vs P\* and lose bucket 0/global; item-kNN (static) and content win nowhere.
   The **blend α=0.1** is BH-significantly *above* P\* globally (+0.0034) and
   in five buckets with zero losing buckets — as on Amazon, the blend is the
   deployment arm; it is M\*-ineligible by preregistration and sets no n\*.
3. **MovieLens timestamp caveat.** ML-32M timestamps are rating-entry times
   on a backfilled catalog (Sun et al., arXiv:2307.09985); temporal splits cut
   rating behavior, not consumption/release, which mechanically dampens
   churn — part of why 0.0640 is small. The tag inputs to the content arm
   were cutoff-filtered at train_end (prereg §3a), which guards tag-time
   leakage but not the backfilled-metadata caveat.
4. **The 5-segment axis** (Amazon-comparability exhibit, no BH, comparability
   only) and both charts:
   `results/figures/crossover_ml32m_test.{svg,png}` (5-segment, both-regime
   comparable form) and `results/figures/crossover_ml32m_deep_test.{svg,png}`
   (deep-bucket confirmatory deltas, BH markers, n\* = 20 annotated). The
   Amazon-side chart remains `results/figures/crossover_test.{svg,png}`; each
   regime is drawn against its own VAL-selected popularity reference (both
   resolved to pop-t12m).
5. **Two distinct n\* values exist and are not the same quantity**: the
   crossover depth n\* = 20 (D1, this verdict) and the Rule S5 fitted routing
   depth n\* = 100 (VAL policy fit, objective 0.0897 — far below P\* alone at
   0.2129, over a grid that by preregistered construction contained no
   pure-P\* cell). The hybrid TEST record (0.0803 global) reports the fitted
   cell per the finite-n\* clause; the routing story remains "routing away
   from popularity hurts on average," fully consistent with the depth-20
   crossover living in buckets that hold only ~19% of ML-32M's TEST GT users
   outside the 100+ bucket... precisely: buckets ≥20 hold 4,864 of 8,843
   users (55.0%) but the 0-bucket's −0.38 dwarfs their +0.01..+0.03 wins.

### Protocol disclosures

- **One TEST evaluation per arm held**, with one operational exception fully
  disclosed: the first TEST campaign launch was refused by the dirty-tree
  guard for its first 11 runs (no records written) but its ALS tail — whose
  training step precedes the guard — completed after the tree became clean,
  so three arm-evaluations (ALS seed-20260807, ALS-decay seeds
  20260805/20260806) each carry a duplicate record pair. The duplicates are
  **byte-identical in all metrics** (deterministic rescoring from the same
  persisted factor artifacts, same snapshot, same seeds; git SHAs differ as
  d224772/20d8ff9). Per invariant #3 nothing was deleted; the canonical set
  used by every analysis is the single coherent campaign at 20d8ff9, pinned
  by run_id in `configs/confirmatory_ml32m_test.yaml`. No arm was evaluated
  twice in any information sense — no second look at TEST occurred.
- **Run-ID collision**: `20260820T221701Z-20d8ff9` names both a duplicate
  ALS-decay record and the canonical M\* record (same-second launches across
  the overlapping campaigns). All resolvers used are last-match-wins, which
  lands on the canonical M\* item_knn record; `reproduce_ml32m` pins it
  explicitly via `expected_config_path`.
- **Seed discipline audit passed**: every inference artifact is seed
  20260805 or deterministic (random floor seed 13, declared exemption);
  stability seeds 20260806/20260807 enter only the 3-seed mean±sd
  (ALS 0.09037 ± 0.00009; ALS-decay 0.09158 ± 0.00030 — TEST global
  NDCG@10, tight across seeds).
- **Reproducibility receipt**: `make reproduce-ml32m` re-derives, from
  Iceberg snapshot 3433604384732745693 with the frozen
  `configs/splits_ml32m.yaml` and `data/MANIFEST_ML32M.md`, both pinned
  records byte-exact (record fields AND per-user artifact arrays identical)
  and the confirmatory verdict block identical. Run on the machine of record
  2026-08-21, exit 0, verdict `byte_exact`.

### TEST ladder (global NDCG@10, canonical records)

random 0.00218 · pop-alltime 0.21749 · **pop-t12m (P\*) 0.24751** ·
kNN 0.07912 · **kNN-t12m (M\*) 0.08562** · ALS 0.09037±0.00009 (3 seeds) ·
ALS-decay(hl90) 0.09158±0.00030 (3 seeds) · content 0.02988 ·
**blend α0.1 0.25088** · hybrid(n\*=100) 0.08032.

### What this changes upstream

The Phase 8/9 Amazon null now has a measured contrast case: same ladder, same
harness, same protocol, 6.4× less churn → a real (if top-of-list-only,
deep-history-only) crossover. The case-study framing gains its second
regime; the null-first Amazon headline stands unchanged. Site evidence
sections must carry: the §5g non-robustness label, the D4 cold-user loss,
"regime contrast, not causal proof," and the timestamp caveat. Both outcomes
were preregistered as publishable; this one is D1 — reported with its
qualifications, not despite them.

## Phase 9 T9-4 (narrative half) — cross-dataset synthesis published in the case study; site copy drafted (2026-08-21)

Docs-only; zero new runs, zero new records, no metric value edited anywhere.
`docs/case_study.md` gains **§7b** — churn as the controlling variable: the
side-by-side regime table (churn 41.11% `20260817T095926Z-633d454` vs 6.40%
`20260820T134403Z-e2263d2`; global winner = blend in both regimes; crossover
none vs n*=20; policy n*=∞ vs fitted n*=100), the Family P delta table from
`results/confirmatory_ml32m_test.json`, and the four mandatory labels (§5g
non-robustness, D4 cold-user loss −0.3844 at 43.9% of TEST users, regime
contrast not causal proof, MovieLens rating-entry timestamps per Sun et al.
arXiv:2307.09985). §6's "a category with slower turnover could move it" bullet
and §10's "until that contrast runs… untested hypothesis" clause are rewritten
to the tested outcome with its qualifications; §9 gains four bullets (prereg
`1731cef` ~7h before the first TEST record; BH FDR 0.05 + ASL floor 2/1001;
ML-32M contracts/manifest kept separate to protect `reproduce-headline`;
`make reproduce-ml32m` byte_exact, writes no record by design). Numbering is
§7b, not a renumber, to preserve every §9/§10 cross-reference. New file
`docs/site_copy_phase9.md`: null-first, decision-led portfolio copy (~850
words), every figure carrying its run_id. Deliberately NOT published: the
depth-20 router's implied global delta — arithmetic with no backing record.
Remaining T9-4 scope (demo contrast panel, README) is the demo lane's.

## Phase 9 T9-4 (demo + traceability half) — regime-contrast exhibit, exporter-integrated (2026-08-21)

Demo gains **exhibit 1c · Regime contrast**: side-by-side regime table, the
two both-regime figures (byte copies of `results/figures/crossover_ml32m_test.svg`
and `crossover_ml32m_deep_test.svg` under `demo/img/`), and a caveat block
carrying all four labels (§5g top-of-list-only, D4 −0.3844, regime contrast
not causal proof, timestamp caveat). `demo/data/contrast.json` is generated by
a new exporter (`export_contrast.py`, wired into `make demo-export`), not
hand-assembled; the hand-built draft was verified value-identical and replaced.
`trace_manifest.json` anchoring for `results/confirmatory_ml32m_test.json`
uses a new `derived_artifact` source kind: SHA-256 + git_sha/config_hash
pinned, plus input anchors equating the artifact's `families/P/artifact_paths`
with the per-user parquets the eval records published; the kind requires the
artifact to self-declare `derived: true` / `appends_to_runs_jsonl: false` so
it can never displace record-anchored citation. Run-ID collision
`20260820T221701Z-20d8ff9` (alsdecay seed1 vs item-kNN TEST — previously
resolved by dict-order luck) is now refused without an explicit
`record_selector` on `config_path`; 29 manifest entries carry selectors and
the receipt card discloses the collision. ALS quotes the canonical
primary-seed record (seed 20260805), labeled as such next to a new seed
column. Gates: `verify_traceability --mode=record` and `--mode=full` PASS
(57 run_ids, 9,580 numeric leaves); offline check CLEAN; README gains the
Phase 9 contrast paragraph. Full pytest: 636 passed, 1 pre-existing failure
(`tests/test_reproduce.py` stale headline-pin constant, chipped separately).
Both reproduce targets stand as demonstrated (headline
`20260819T102247Z-08457d5` byte_exact; `reproduce-ml32m` byte_exact
2026-08-21 per T9-3c) — T9-4 touches no compared field (`data/MANIFEST.md`
and all configs untouched).
