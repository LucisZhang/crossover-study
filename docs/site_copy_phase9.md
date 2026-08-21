<!--
  Portfolio-site case-study copy, Phase 9 T9-4 (docs/engineering-log/UPGRADE_PLAN.md §8c).
  Draft for xiangguozhang.com — updating the site itself is out of repo scope.
  Null-first and decision-led by design: the hook is the build/don't-build
  decision, not the stack. Every figure quoted here carries its run_id; the
  run_ids resolve against results/runs.jsonl, and the ML-32M crossover numbers
  resolve against the committed derived analysis
  results/confirmatory_ml32m_test.json (which names its own source records).
  Voice matches docs/case_study.md: dry, receipt-driven, no triumphalism.
  ~820 words excluding this comment and the footnote block.
-->

# The recommender that told me not to build a recommender

Every recommendation roadmap contains the same unexamined assumption: *once a user has enough history, personalization will beat showing them what is popular.* The roadmap item that follows is expensive — user segmentation, a routing layer, a model to route to, and the monitoring for all three. So the question worth answering first is not "which model wins," it is **"at what history depth does personalization start winning, and do enough of our users ever reach it?"**

I built a 43.9M-interaction batch pipeline and evaluation harness to answer that on Amazon Electronics reviews. The answer was no depth at all — and the most useful thing the project produced was a decision *not* to build the routing layer.

## The null, and why it isn't a modeling failure

Trained on everything through mid-2022 and tested on 2023, recency-weighted popularity scores NDCG@10 **0.005404** (run `20260805T172047Z-035042b`). Item-kNN, ALS tuned across a ten-point grid, and semantic-content retrieval all lose to it — not just globally, but in **every** history-depth segment, with every confidence interval excluding zero. ALS, the strongest of them, reaches 0.002750 (`20260806T082441Z-2f2f26d`): roughly half of popularity's score, from a user's full purchase history.

The reason is not that the models are bad. It is that the catalog moves. **41.11% of 2023 test purchases land on items that had zero or near-zero support in the training window** (`20260817T095926Z-633d454`) — items a train-frozen model has no co-occurrence and no learned factors for. That caps *any* such model at 65.5% recall before a single hyperparameter is chosen. A trailing-12-month popularity list partially tracks the churn; a frozen factor matrix structurally cannot.

Two decisions fall out. First, the fitted routing policy is **n\* = ∞**: every threshold the grid tested scored worse than routing nobody, so the optimal routing is no routing (`20260808T030659Z-43c90c8`). The segmentation infrastructure is unbuilt, on evidence. Second, the one arm that beats popularity is popularity, gently re-ranked by content similarity: **+0.000322 NDCG@10, ≈+6% relative, CI [+0.000200, +0.000449]** (`20260807T055333Z-c320c79`). Small, real, and cheap — which is the whole recommendation.

## Then I tested whether churn was really the reason

A mechanism claim you cannot falsify is a story. So I ran the identical ladder — same harness, same protocol, same metrics — against MovieLens-32M, a catalog that barely turns over, with the hypotheses, selection rules and multiple-testing policy committed to the repo *before* the first test run (commit `1731cef`).

The hinge number was computed before any model was trained: **6.40% churn versus Amazon's 41.11%** (`20260820T134403Z-e2263d2`), a 6.4× gap. And on that catalog the crossover appears. Item-kNN with a trailing window beats popularity from **20 interactions of history upward** — +0.0333 at depth 20-49, +0.0198 at 50-99, +0.0112 at 100+, all significant after Benjamini–Hochberg correction at FDR 0.05.[^1]

Same ladder, 6.4× less churn, a real crossover at n\* = 20. That is the finding, and here is everything wrong with reading it as a victory:

- **It's a ranking-quality effect, not a recall effect.** The Recall@20 robustness check confirms none of the three winning buckets; at 100+ the sign flips.
- **It is catastrophic for cold users.** The same arm loses −0.3844 for zero-history users — 43.9% of test users — so popularity still wins on the global average (−0.1619) even on the stable catalog.
- **The deployment answer never changed.** Popularity re-ranked by content is the global winner on *both* datasets (+0.00336 on MovieLens, `20260820T221933Z-20d8ff9`). Churn flips the deep-history tail, not the head.
- **It is a regime contrast, not causal proof.** Domain, density, catalog size and feedback semantics all differ between the two datasets; churn is the axis I measured, not the axis I isolated. MovieLens timestamps are rating-entry times on a backfilled catalog (Sun et al., arXiv:2307.09985), which itself dampens measured churn.

The transferable sentence is narrow: **measure your catalog's churn before you budget for personalization infrastructure.** It is one counting job, and it predicts whether the threshold you are planning to route on exists at all.

## What you can check without trusting me

- **Re-runnable to the byte.** `make reproduce-headline` re-derives the headline from a pinned Iceberg snapshot and compares every deterministic field: verdict `byte_exact` — including once *after* deliberately churning the warehouse with ~40 snapshots of appends, upserts, compaction and expiry (`20260807T164622Z-3e2c665`), and again against the current pin (`20260819T102247Z-08457d5`). The MovieLens contrast has its own target, `make reproduce-ml32m`: `byte_exact` on record fields *and* per-user arrays, run 2026-08-21.
- **Append-only receipts.** Every result is one line in a committed log (129 records) carrying config hash, git SHA, dataset manifest hash, Iceberg snapshot IDs and seeds. Wrong runs get superseding entries, never edits — including the entry that downgraded one of my own earlier verdicts after a multiplicity review.
- **Preregistration with a clock on it.** Arms, thresholds and decision rules were committed before the runs that judge them: the Phase 8 preregistration at 2026-08-17T11:45Z, before the model code existed; the MovieLens one seven hours before the first test evaluation. Both outcomes were declared publishable in advance, which is the only reason a null is worth reading.
- **No sampled negatives, ever.** Full-catalog ranking against all 368,228 items per user, bootstrap CIs on every headline number, paired bootstraps on every comparison, and a published experiment log that includes the hypotheses that failed.

[^1]: MovieLens crossover figures from the committed confirmatory analysis `results/confirmatory_ml32m_test.json`, recomposed from the one-shot test records `20260820T221701Z-20d8ff9` (item-kNN-t12m) and `20260820T221055Z-20d8ff9` (popularity-t12m); 8,843 test users, full-catalog ranking over 43,884 items, 1,000-resample paired bootstraps, seed 20260805.
