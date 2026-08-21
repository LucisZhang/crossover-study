# crossover-study

Crossover Study — when does personalization beat popularity?

> **"How much history does a user need before personalization beats popularity?"**
> The measured answer: **there is no such depth — no history-depth crossover exists
> in this data.** The optimal routing is no routing.

Crossover Study ingests 43.9M Amazon Electronics reviews through a
contract-checked Spark + Iceberg lakehouse on a single 16GB laptop, evaluates
popularity, item-kNN, ALS, and semantic-content retrieval under a frozen temporal
split with full-catalog ranking metrics and bootstrap CIs — and publishes a null,
with its mechanism measured and every number traceable to an append-only results
log.

## The findings, null-first

1. **No history-depth crossover exists.** ALS, item-kNN, and pure content lose to
   trailing-12-month popularity at every history-depth segment, with every
   segment's 95% CI excluding zero — before and after Phase 8 removed the recency
   asymmetry between the arms (paired-delta records
   `20260818T064002Z-56d871c`, `20260818T064207Z-56d871c`).
2. **The mechanism is catalog churn, measured:** 41.11% of 2023 TEST ground-truth
   purchase mass lands on items with zero or near-zero TRAIN support — items a
   TRAIN-frozen model structurally cannot rank — capping any such model at 65.5%
   recall before a single modeling choice (run `20260817T095926Z-633d454`).
3. **The one effective arm is a blend, and it is small:** trailing-12-month
   popularity re-ranked by a MiniLM content score at α=0.3 gains
   +0.000322 NDCG@10 over popularity alone, ≈+6% relative, paired CI excluding
   zero (runs `20260807T055333Z-c320c79` / `20260818T181443Z-6744efc` vs
   `20260805T172047Z-035042b`).
4. **The fitted routing policy is n\*=∞** — every finite history-depth threshold
   scores worse than routing every user to the blend, so **the optimal routing is
   no routing** (policy grid record `20260808T030659Z-43c90c8`).

A five-cell pocket where a recency-matched item-kNN edges popularity (record
`20260818T072256Z-3f3530a`) survives as a mechanism footnote — popularity's
trailing window under-serves stale items — not as a crossover finding: it comes
from ~80 uncorrected tests, of which ~4 would be expected significant at α=0.05
by chance. See `docs/case_study.md` §6 for the full disclosure and
`docs/engineering-log/EXPERIMENT_LOG.md` (2026-08-19) for the superseding verdict.

**Phase 9 — regime contrast, not a repeal.** The null above is a property of a
high-churn catalog, not of personalization in general: on ML-32M, measured
catalog churn is 6.40% against Amazon Electronics' measured 41.11% (records
`20260820T134403Z-e2263d2`, `20260817T095926Z-633d454`), and in that low-churn
regime the same preregistered test finds a crossover — item-kNN-t12m beats
trailing popularity, 95% CI excluding zero and BH-significant, at n\*=20 and
above (D1_CROSSOVER, confirmatory record `results/confirmatory_ml32m_test.json`,
source runs `20260820T221701Z-20d8ff9` / `20260820T221055Z-20d8ff9`). It is a
regime contrast, not causal proof — MovieLens's explicit ratings change several
variables at once, its timestamps are rating-entry times on a backfilled
catalog, the Recall@20 guard does not corroborate the win, and the same run
shows a significant cold-user loss. See `docs/case_study.md` §8c and
`docs/engineering-log/EXPERIMENT_LOG.md` (2026-08-20/21) for the full disclosure.

## Where the details live

Engineering decisions and phase logs live in [docs/engineering-log/](docs/engineering-log/).
> 工程决策与阶段日志在 docs/engineering-log/。

- `docs/case_study.md` — the full case study, every claim labeled with its
  evidence class and its `results/runs.jsonl` run_id.
- `demo/` — static, offline exhibit site (crossover explorer, regime map,
  data-quality dashboard, receipts drawer); every displayed number opens the
  results-log record it came from.
- `docs/engineering-log/EXPERIMENT_LOG.md` — dated hypothesis → result → verdict entries, failures
  and supersessions included, append-only.
- `results/runs.jsonl` — the append-only results log; each record carries config
  hash, git SHA, dataset manifest hash, Iceberg snapshot IDs, and seeds.
- `docs/engineering-log/UPGRADE_PLAN.md` — the single source of truth for scope and phase order.

## Reproducing

```
uv sync
make data                 # deterministic bronze → silver → gold rebuild, contracts enforced
make eval                 # run eval configs, append to results/runs.jsonl
make reproduce-headline   # re-run the headline eval against the pinned Iceberg snapshot
uv run pytest             # metric math, contract engine, split logic
```

The headline number is pinned to an Iceberg snapshot and has reproduced
`byte_exact` on both the original and a deliberately churned warehouse
(records `20260807T153823Z-9a9fb4c`, `20260807T164622Z-3e2c665`).

## Provenance and license

This lab grew out of a five-person course project (CISC3018, University of
Macau, Fall 2025); the owner's uniquely attributed deliverable there was
presentation material. Nothing from that project ships here — pipeline,
contracts, models, evaluation harness, and demo were built solo from the raw
public dataset with their own verification chain. Dataset: Amazon Reviews 2023
(McAuley Lab, UCSD), research-use terms, never redistributed; cite Hou et al.
2024, *Bridging Language and Items for Retrieval and Recommendation*.
