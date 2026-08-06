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
