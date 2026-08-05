# CLAUDE.md — Batch Recsys Lab

> **"How much history does a user need before personalization beats popularity?"** Batch Recsys Lab ingests 43.9M Amazon Electronics reviews through a contract-checked Spark + Iceberg lakehouse on a single 16GB laptop, evaluates popularity, item-kNN, ALS, and semantic-content retrieval under temporal splits with full-catalog ranking metrics and bootstrap CIs — and turns the measured crossover into a routing policy that knows exactly where each model stops working.

`UPGRADE_PLAN.md` is the single source of truth. This file is the working contract for any agent in this repo; when in doubt, re-read the plan section it cites.

## Invariants — never break these

These are the identity of the piece (plan §12 "never cut", §6.5, §11). No task, refactor, or shortcut may violate them:

1. **Frozen TEST split.** Temporal splits only (TRAIN ≤ 2022-06-30, VAL 2022-H2, TEST 2023-01-01 → snapshot end; exact dates frozen in `configs/splits.yaml` once set). All tuning, threshold fitting, and model selection happen on VAL. TEST is touched only for final reported runs — never for iteration.
2. **Seeds.** Every stochastic step (bootstrap resamples, ALS init, fixture sampling) uses a fixed, recorded seed. Headline configs report 3-seed mean±sd. A result that can't be re-produced seed-for-seed doesn't exist.
3. **Append-only results log.** `results/runs.jsonl` is append-only and committed. Never edit, reorder, or delete a record — a wrong run gets a superseding entry, not a rewrite. Every entry carries config hash, git SHA, dataset manifest hash, and Iceberg snapshot ID.
4. **Full-catalog ranking.** Metrics score the full catalog per test user (TRAIN-seen items excluded). No sampled-negatives shortcuts; if any exhibit ever samples negatives, it is labeled as such with the known bias cited.
5. **Provenance language.** The ancestor is a five-person course project; the owner's uniquely attributed deliverable there was presentation material. Never present course code, models, or metrics as individual work; the case study uses the §11 paragraph verbatim in substance. `docs/seed-archive/` is read-only reference, never imported or executed. The OneDrive source directories are never modified, moved, or deleted.
6. **Lane discipline.** This lab owns large-scale **batch processing** and **recommender evaluation**. No model training/fine-tuning (Triage Router Lab owns that lane), no deep recsys in core scope, no production cloud deployment. Ratings are implicit-feedback positives; the raw dataset is never redistributed (research license; cite Hou et al. 2024).

Also load-bearing, from the same list: full-dataset ingestion with an exactly-reconciling waterfall, contracts + quarantine ledger, the popularity baseline, the crossover/segment analysis, the routing policy, and both site evidence sections ("How this was verified" / "What this does not prove").

## Plan-execution discipline

- Work **one task at a time, in §8 phase order** (Phase 0 → 7). Do not start a phase before the previous phase's acceptance criteria are met.
- **Before starting a task, restate its acceptance criteria** from §8 in your own words. That restatement is the definition of done.
- **Done means demonstrably done**: the acceptance criteria are shown met by running the command or check that proves it, not asserted.
- **No invented scope.** If it isn't in the plan, it isn't in the task. Improvements beyond the plan are proposed to the owner, not silently built. Cuts follow the §12 cut order only.
- **Every portfolio number is reproducible by a recorded command.** A metric that will appear in the case study or demo must trace to a `results/runs.jsonl` record and be regenerable via a Make target (ultimately `make reproduce-headline` against a pinned Iceberg snapshot). Failed hypotheses are logged in `EXPERIMENT_LOG.md`, not discarded.

## Repo map (as Phase 0+ builds it)

```
UPGRADE_PLAN.md            # single source of truth — read before any phase
CLAUDE.md                  # this file
docs/seed-archive/         # read-only course-project seeds (schema + MiniLM recipe + index)
src/batch_recsys_lab/
  ingest/                  # HF download, gz-jsonl → bronze Iceberg tables
  contracts/               # contract engine: YAML checks → dq_results + quarantine.*
  features/                # silver/gold builds, k-core funnel, user_stats, popularity
  models/                  # popularity variants, item-kNN, implicit ALS, MiniLM+ANN
  eval/                    # harness: config → JSONL, full-catalog ranking, bootstrap CIs, segments
  policy/                  # history-depth routing: fit n* on VAL, report on TEST
contracts/*.yaml           # one contract per table
configs/*.yaml             # one config per run; splits.yaml is frozen once set
results/runs.jsonl         # append-only, committed
EXPERIMENT_LOG.md          # dated hypothesis → result → verdict, failures included
demo/                      # static site: crossover explorer, pick-a-shopper, DQ dashboard, receipts
data/                      # gitignored; MANIFEST.md (URL, date, SHA-256s) is committed
tests/                     # pytest: metric math vs reference impl, contract engine, split logic
  fixtures/                # bundled ~50k-row sample from bronze (CI substrate)
Makefile                   # exports JAVA_HOME/PATH pin; all run targets
.github/workflows/         # CI smoke: contracts + eval on the bundled fixture
```

## Environment & run commands

- **Python 3.12 + `uv`** (`uv sync`; run everything as `uv run …`). Spark 4.x local mode (`local[10]`, ~8g driver), Iceberg local Hadoop catalog with filesystem warehouse.
- **JDK pin (critical):** host default is Java 25; Spark 4 supports 17/21 only. The Makefile (and CI) must set, project-locally — never globally:
  ```
  JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
  PATH="$JAVA_HOME/bin:$PATH"
  ```
  If Spark fails to start with opaque JVM module/reflection errors, check `java -version` inside the project env first.
- **Disk gate before any download:** ≥35GB free, or relocate `data/` to external storage (plan §5).
- Core targets:
  - `make data` — deterministic bronze → silver → gold rebuild, contracts enforced
  - `make eval` — run the eval config(s), append to `results/runs.jsonl`
  - `make reproduce-headline` — re-run the headline eval against the pinned Iceberg snapshot
  - `uv run pytest` — metric math, contract engine, split logic (CI runs these plus a fixture smoke eval)
