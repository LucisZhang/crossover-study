# UPGRADE PLAN — Batch Recsys Lab

**A contract-checked Spark + Iceberg batch lakehouse that processes the full 43.9M-interaction Amazon Electronics dataset on a 16GB laptop, evaluates four recommender families with temporal splits, full-catalog ranking metrics, and bootstrap confidence intervals — and ships a measured answer to the question most recommender demos dodge: how much history does a user need before personalization actually beats popularity?**

This document is self-contained. It was produced after a first-hand audit of every file across three mounted locations (§3). It is written for a coding agent with **no access to the conversation that produced it**. Follow it as the single source of truth.

> **Framing decision (owner's direction, 2026-08-05):** an earlier draft of this plan made the case study an *audit* of the ancestor course project (its RMSE was worse than a global-mean baseline, etc.). The owner rejected that framing — the portfolio already has several review/audit-flavored pieces. **This plan builds a forward-looking system on its own merits.** Findings from the ancestor project appear below only where they drive engineering decisions (what to reuse, what data realities to guard against); they are NOT case-study content, and no "before/after the old project" exhibits are built.

> **Hard constraint:** Do NOT modify, move, or delete anything in the three source directories below. They are archives living in OneDrive. Build the new project in a **fresh repository outside these folders** (suggested: `~/Projects/batch-recsys-lab`). Source files are referenced read-only, as seeds.

Source directories (read-only):
1. **Archive (working dir):** `/Users/hsiangkuochang/Library/CloudStorage/OneDrive-个人/Learning/大三上(UM)/Cloud Computing and Big Data System/Project/Project Archive`
2. **Project root (knowledge base, report, PPT, screenshots):** `/Users/hsiangkuochang/Library/CloudStorage/OneDrive-个人/Learning/大三上(UM)/Cloud Computing and Big Data System/Project`
3. **Big Data course labs (Hadoop fundamentals, provenance only):** `/Users/hsiangkuochang/Library/CloudStorage/OneDrive-个人/Learning/大三下(BIT)/大数据处理技术/Lab`

---

## 1. Project goal and target narrative

### 1.1 Who this is for

Xiangguo Zhang's portfolio site (https://xiangguozhang.com) is a verification-first "systems portfolio" targeting Data Analytics / Data Engineering / AI Application Engineering roles. Every case study has: hard measured metrics, an interactive demo, a "How this was verified" section, and a "What this does not prove" section with evidence-class labels (measured / estimated / projected). Existing case studies: Release Guardian, RAG Quality Lab, Privacy Preflight, Margin Control Tower, Streaming Reliability Lab (MySQL CDC → Flink → Iceberg with failure injection), Credit Policy Lab. A separate plan (Triage Router Lab, NLP coursework seed) owns the **model-training/fine-tuning** gap.

Gaps THIS project must own: **(a) no large-scale BATCH processing** (streaming exists, batch does not), **(b) no recommender systems**. Explicitly NOT this project's job: model training/fine-tuning (owned by Triage Router Lab) and production cloud deployment (deliberately out of scope portfolio-wide; demos are static).

### 1.2 The one-sentence pitch

> **"How much history does a user need before personalization beats popularity?"** Batch Recsys Lab ingests 43.9M Amazon Electronics reviews through a contract-checked Spark + Iceberg lakehouse on a single 16GB laptop, evaluates popularity, item-kNN, ALS, and semantic-content retrieval under temporal splits with full-catalog ranking metrics and bootstrap CIs — and turns the measured crossover into a routing policy that knows exactly where each model stops working.
>
> *[Amendment 2026-08-06: the "measured crossover" claim is superseded by the Phase 3 measured outcome (see §6.4) — CF never crosses popularity at observed depths. Final pitch wording deferred to Phase 6.]*

### 1.3 Why this makes a hiring manager stop

1. **It answers a question with a measurement, not a vibe.** Every ALS tutorial claims collaborative filtering "works"; almost none say *for whom*. The signature exhibit — NDCG@10 by user-history depth, one line per model, CI bands, with the fitted routing threshold marked — is the chart every recsys interview circles back to, and here it's backed by a frozen results log.
2. **It fills the batch gap as the deliberate mirror of the streaming lab.** Streaming Reliability Lab is CDC → Flink → **Iceberg**; this is raw files → Spark → **Iceberg**: same table format, opposite ingestion mode, shared lakehouse vocabulary (snapshots, schema enforcement, compaction, time travel). Together they read as "this candidate owns both halves of the modern data platform."
3. **Recommender evaluation done right is rare at any seniority.** Temporal splits instead of random, full-catalog ranking instead of RMSE, popularity baselines that most published ALS demos silently lose to, per-segment analysis, and an explicit offline-vs-online-gap discussion.
4. **Data contracts with teeth.** The raw dataset is genuinely messy (sentinel prices, dup reviews, orphaned foreign keys, inconsistent brands); every dropped row lands in a quarantine ledger and a reconciliation waterfall whose sums must balance exactly. This is the data-engineering half of the signal.
5. **Honest scale.** 43.9M reviews / 1.61M products end-to-end on a 16GB laptop is a real single-node big-data story, told without cluster cosplay: local-mode stated plainly, code cluster-portable by construction.

### 1.4 Naming

Primary: **Batch Recsys Lab**. Alternates: *Cold-Start Lab* (names the signature finding), *Recommender Systems Lab*. Keep the site's "\<Domain\> \<Noun\> Lab" pattern; avoid "Quality"/"Reliability"/"Policy" (taken). Owner picks the final name; the repo can be renamed cheaply before publication.

---

## 2. Relationship to the ancestor course project

The seed is a five-person course project (CISC3018, University of Macau, Fall 2025): an ALS + semantic-search "recommendation bot" on a 1M-interaction sample of this same dataset. What matters from it for THIS plan:

1. **Its data artifacts cannot be the substrate.** The archived `final_demo_data.csv` (1M rows) has **no timestamp column** (the ETL dropped it — verified), so temporal evaluation is impossible from it; the sample is the *first 1M lines* of the raw file (order-biased); and 1M rows fills no batch-scale gap. **Fresh ingestion of the full raw dataset is mandatory regardless of framing.**
2. **Its design artifacts are worth seeding from.** The metadata StructType schema and `details`-map extraction, and the MiniLM item-text recipe, are sound starting points (§3.4).
3. **Its failure modes inform the contract layer.** Sentinel prices (−1.0) leaking to output, a silent `except: continue` swallowing rows uncounted, an unexplained 1.6M→271k catalog shrinkage from a blind inner join, a 100%-null column shipped to the final table. These become *engineering motivations* for specific contract checks (§7) — cited in code comments and this plan, not dramatized in the case study.
4. **It was team work.** The case study needs one honest provenance paragraph (§11) and must not claim course artifacts as individual work. Everything in the new lab is rebuilt solo from raw public data.

That is the ancestor's entire role. No reproduction of its results, no pinned-legacy-Spark environment, no before/after exhibits.

---

## 3. Ground-truth asset inventory (first-hand audit, 2026-08-02)

### 3.1 Location 1 — Project Archive (working dir)

Total ~454MB. All paths relative to `.../Project/Project Archive/`.

**`1 spilt json , clean and to parquet and ETL/`**
- `split jsonl.py`: splits a hardcoded-path JSONL into 75,000-line chunks. Trivial; discard.
- `jsonl_to_parquet.ipynb`: PySpark metadata ETL — explicit StructType schema for Amazon metadata, extracts brand/manufacturer/model/dimensions/weight from the `details` map, builds `features_text`/`description_text`/`categories_text`, fills nulls (price → −1.0 sentinel), writes snappy Parquet. Its output line "Total products loaded: 1,610,012" exactly matches the published Amazon Reviews **2023** Electronics item count — this pins the dataset version to 2023 (the course report ambiguously cites both 2018 and 2023; ignore the 2018 citation).
- `ETL Pipeline.py`: pandas reader keeping only `user_id, parent_asin, rating, title`; **drops `timestamp`**; stops at first 1,000,000 valid lines; silent `except: continue`; writes `final_demo_data.csv`.
- `ETL_logs.txt`: 1,000,000 interactions, 185,242 users, 271,211 products.
- `final_demo_data.csv` (218MB): 9 columns; no timestamp; `image_url` 100% null; `price` −1.0 sentinel; `brand` "Unknown" ≈ 18%; embedded newlines (1,000,016 physical lines for 1M rows); rating mean 4.2158, σ ≈ 1.291.

**`2 model training and cli bot - meta only/Model Training and Bot - Metadata only.ipynb`**
- SentenceTransformer `all-MiniLM-L6-v2` on Colab T4; filters 1.61M products to top 1M by a quality score; embeds in 3,907 batches; ranking = `0.92·cosine + 0.08·popularity` (unnormalized popularity — can dominate); CLI + placeholder Telegram scaffold. The output pickle is **not in the archive** — this path is not runnable from archived files.

**`3 ALS (meta+review) bot - .../`**
- `index.py`: `implicit` 0.7.2 ALS fed explicit ratings directly as confidence (a conflation the new lab fixes by modeling implicit feedback properly); pickled package present (150MB).
- `index_pyspark.py`: Spark 4.0.1 MLlib ALS, randomSplit 80/20, rank=50 — the course's RMSE path.
- `als_bot.py`, hybrid-router notebook (regex on user-ID shape → ALS path, else hardcoded-keyword title match), Colab notebooks.
- `saved_als_model/` (76MB), `saved_indexer_model/`: loadable under Spark 4.0.1, but useless to the new system (trained on the timestamp-less biased sample).

No `.git`, no tests, no requirements/lockfile, no CI anywhere.

### 3.2 Location 2 — Project root

- `PROJECT_Hybrid_Amazon_Recommendation_Bot_KNOWLEDGE_BASE.md` (1,477 lines): extraction-generated knowledge base, spot-checked against source during the audit — **accurate**. Treat as a reliable secondary index into the archive.
- Report PDF/DOCX (33 pp), PPT (15 slides), full page screenshots. Needed only for the provenance paragraph (team-contribution facts, §11); not build inputs.

### 3.3 Location 3 — Big Data course Lab folder

Hadoop 2.10.2 (local + pseudo-distributed), HDFS Java API, HBase 2.4.17, hand-written MapReduce (global sort, self-join) — on ARM64 Ubuntu under UTM, with screenshot proof. **No standalone source files** — all code is embedded in report DOCX/PDFs. Use: one provenance sentence in the case study ("Hadoop-ecosystem fundamentals — pseudo-distributed HDFS, HBase, hand-authored MapReduce — reports on request"). Do not republish; do not extend.

### 3.4 Reuse vs discard

| Asset (under `Project Archive/` unless noted) | Decision | How |
|---|---|---|
| `1 .../jsonl_to_parquet.ipynb` — metadata StructType schema + `details`-map extraction + text-field assembly | **REUSE (schema seed)** | Port the field list and extraction logic into the new bronze/silver items schema. Fix: no sentinel fills (NULL stays NULL), case-insensitive `details` key matching, keep all units. |
| `2 .../Model Training and Bot - Metadata only.ipynb` — MiniLM item-text recipe | **REUSE (design seed)** | The title+brand+category+features text recipe is sound. Rebuild as a script with a persisted, versioned index and *normalized* score blending (the 0.92/0.08 unnormalized blend is a known defect). |
| `PROJECT_..._KNOWLEDGE_BASE.md` (project root) | REUSE (reference) | Reliable index for the implementing agent; not a public artifact. |
| `1 .../final_demo_data.csv` | DISCARD from build | No timestamps, biased sample, dirty schema. At most: a one-off sanity cross-check that the fresh ingest's Electronics rating distribution is in the same ballpark. Never a pipeline input or fixture. |
| `3 .../saved_als_model/`, `saved_indexer_model/`, `als_model_package.pkl` | DISCARD | Trained on the disqualified sample; nothing to salvage. |
| `ETL Pipeline.py`, `split jsonl.py`, `als_bot.py`, `index.py`, `index_pyspark.py`, hybrid notebook | DISCARD | Superseded end-to-end. Their defects inform §7 contract motivations (code-comment level, not case-study exhibits). |
| Report/PPT/screenshots | DISCARD from build | Source only for the §11 provenance facts. |
| Lab folder (all) | DISCARD; one provenance line | §3.3. |
| Raw data | **Not in any archive** — downloaded fresh (§5). |

---

## 4. Target architecture and rationale

### 4.1 System overview

```
HF: McAuley-Lab/Amazon-Reviews-2023 (Electronics)
  raw_review_Electronics (43.9M rows, jsonl.gz)      raw_meta_Electronics (1.61M rows, jsonl.gz)
        │  SHA-256 frozen snapshot, download manifest        │
        ▼                                                    ▼
  Spark 4.x batch jobs ──────────────────────────────────────┐
        ▼                                                    ▼
  ICEBERG lakehouse (local Hadoop catalog; same table format as Streaming Reliability Lab)
    bronze.reviews   (typed, all scalar fields + title; `text`/`images` projected out — documented)
    bronze.items     (full metadata schema, corrected from the course StructType seed)
    silver.interactions  (contracts enforced: dedup'd, sentinel-free, timestamped, FK-checked)
    silver.items         (normalized brand/price/category; quarantine tables + DQ metrics table)
    gold.interactions_5core (modeling table, documented funnel)   gold.item_features
    gold.user_stats (history depth, tenure)                       gold.popularity (time-windowed)
        │
        ├─▶ Models: popularity / popularity-per-category / item-kNN / implicit-ALS (tuned)
        │           / MiniLM content retrieval (ANN)
        ├─▶ Eval harness: temporal splits, full-catalog ranking, user-bootstrap CIs,
        │           per-segment (cold→heavy), append-only JSONL results log
        ├─▶ Routing policy: history-depth threshold fitted on VAL, measured on TEST
        └─▶ Static demo (precomputed JSON + in-browser MiniLM query embedding) + case study
```

### 4.2 Layer-by-layer rationale

- **Spark 4.x (PySpark), local mode.** The dominant batch keyword in DE job descriptions and the only mainstream engine with first-class Iceberg writes from Python. Local-mode honesty is part of the story: state plainly that this ran `local[10]` on one machine, and that the code is cluster-portable by construction (no `collect()` on large tables, no driver-side loops over data). The "you could've used DuckDB" objection is met head-on with an optional measured single-node comparison (§6.5) rather than ignored.
- **Iceberg (local Hadoop catalog, filesystem warehouse).** The deliberate mirror of the streaming lab: schema enforcement (the contract layer's teeth), snapshot IDs pinned in every eval manifest (reproducibility receipt), time travel for the "re-run the eval on the exact bytes" demo, incremental monthly appends + a late-data upsert exhibit, compaction before/after file-count metrics. Cut-line: plain partitioned Parquet + a manifest file, losing the ops exhibits (§12).
- **Models — deliberately standard, evaluated unusually well.** `implicit` ALS on proper implicit signal (interaction = positive; confidence weighting as a tuned variant); item-kNN co-occurrence as the classic strong baseline; time-windowed popularity (and per-category); MiniLM (`all-MiniLM-L6-v2`) item-text embeddings + ANN (hnswlib or FAISS) for content retrieval — the upgraded descendant of the course's semantic notebook, now versioned, persisted, and score-normalized. **No deep recsys, no model training/fine-tuning** in core scope (lane discipline: that gap belongs to Triage Router Lab).
- **Engineering stack:** Python 3.12 + `uv` lock; repo layout `src/batch_recsys_lab/` (ingest, contracts, features, models, eval, policy) · `contracts/*.yaml` · `configs/*.yaml` (one per run) · `results/runs.jsonl` (append-only, committed) · `EXPERIMENT_LOG.md` (dated hypothesis→result→verdict entries, including failed ones) · `demo/` (static site) · `data/` (gitignored). `make data`, `make eval`, `make reproduce-headline` targets. pytest for metric math (validated against a reference implementation on toy fixtures), contract engine, and split logic; GitHub Actions smoke job on a bundled ~50k-row fixture sampled from the fresh ingest.

---

## 5. Dataset scale decision

**Decision: the FULL Amazon Reviews 2023 Electronics category — 43.9M reviews, 1.61M items — through the lakehouse; a documented k-core subset for modeling.**

- **Why full:** the batch-scale gap is the point. "43.9M interactions processed end-to-end on a laptop, with a reconciliation ledger" is the headline scale claim; anything sampled undercuts it.
- **Why k-core for modeling:** ALS on ~18M raw users × 16GB RAM is hostile, and one-review users are untrainable signal for CF anyway. Build `gold.interactions_5core` (iteratively filter users ≥5 and items ≥5 interactions) with the funnel published (raw → deduped → 5-core row/user/item counts). **Honesty requirement:** k-core filtering flattens the long tail and inflates metrics — say so in the case study, and report the popularity baseline on both the 5-core and the un-cored silver table so the inflation is itself measured.
- **Source & freeze:** Hugging Face `McAuley-Lab/Amazon-Reviews-2023`, files `raw_review_Electronics` and `raw_meta_Electronics` (jsonl.gz). Record URL, download date, file SHA-256s in `data/MANIFEST.md`; verify bronze row counts against the published 43.9M/1.61M. Raw reviews carry millisecond timestamps — the field this whole evaluation design depends on.
- **License:** released by McAuley Lab (UCSD) for research use; **do not redistribute the raw dataset**. Demo may ship small derived excerpts (titles, ASINs, aggregate stats) with citation (Hou et al., 2024, *Bridging Language and Items for Retrieval and Recommendation*). Re-hash user IDs for any demo-visible data.
- **Hardware budget (audited on the target machine: Apple M4, 10 cores, 16GB RAM, ~32GB free disk at plan time).** Tight; manage explicitly: keep raw as compressed gz (~10–15GB); bronze projects out `text`/`images` from reviews (documented — the lab never uses review text; item text comes from metadata), keeping the lakehouse ≈ 5–8GB; embeddings (1M–1.6M × 384 fp16) ≈ 1.2–2.5GB; peak working set ≈ 25–30GB. **Gate in Phase 0: verify ≥ 35GB free or relocate `data/` to an external SSD / rented VM before downloading.** MiniLM embedding of ~1M items: hours on M4 MPS, or a Colab T4; record wherever it ran.
- **Fallback if disk/RAM is truly blocked:** a mid-size 2023 category (e.g. `Office_Products` or `Video_Games`) with the SAME pipeline. Last resort, documented as such — Electronics is preferred for its scale and category richness.

---

## 6. Evaluation design

### 6.1 Splits (temporal, global)

Interactions span 1996 → 2023-09 (dataset snapshot). On `gold.interactions_5core`:
- **TRAIN:** ≤ 2022-06-30. **VAL:** 2022-07-01 → 2022-12-31 (all tuning, threshold fitting, model selection). **TEST:** 2023-01-01 → snapshot end (frozen; touched only for final reported runs).
- Users appearing only in TEST (no TRAIN history) form the **strict cold-start segment** — evaluable by popularity/content only; this is a feature of the design, not a nuisance.
- Exact boundary dates may shift ±1 quarter after inspecting volume-by-month; whatever is chosen is frozen in `configs/splits.yaml` and recorded in every result.
- The case study explains *why* temporal (random splits leak future interactions into training and flatter every personalized model) — as methodology, in its own voice, without referencing any prior project.

### 6.2 Protocol and metrics

- **Task:** implicit-feedback top-K recommendation. A review = one positive interaction (state this modeling assumption and its limits). Ratings enter only as an optional confidence-weighting variant.
- **Ranking:** score the **full catalog** per test user (factor-matrix × item-matrix chunked matmul — no sampled-negatives shortcuts; if any exhibit ever samples negatives, label it and cite the known bias), excluding the user's TRAIN-seen items.
- **Metrics:** Recall@10/20/50, NDCG@10/20, MRR, HitRate@10 (headline: **Recall@20 and NDCG@10**); catalog coverage@10, popularity share of recommendations (vs catalog Gini), novelty (mean −log₂ item popularity). All headline numbers with **user-bootstrap 95% CIs** (1,000 resamples, fixed seed); paired bootstrap deltas for every comparison claim; a difference is claimed only where the CI excludes zero.
- **Segments:** TRAIN-history depth buckets {0 (strict cold), 1–4 (only on un-cored silver eval), 5–9, 10–19, 20+}. Every model reported per segment. The **crossover chart** (NDCG@10 by segment: popularity vs item-kNN vs ALS vs content vs hybrid) is the signature exhibit and the case study's front door.

### 6.3 Models, baselines, tuning

Random floor; global popularity (time-windowed: trailing-12-month, not all-time — measure both, the delta is itself interesting); popularity-per-category; item-kNN (cosine co-occurrence); implicit ALS (grid over factors {32,64,128} × regularization × α confidence × iterations, tuned on VAL only, every run logged to `EXPERIMENT_LOG.md`; 3 seeds on the chosen config, report mean±sd); MiniLM content retrieval (query = user's TRAIN item-text centroid for warm users / item-to-item for cold contexts); hybrid policy (§6.4).

### 6.4 The routing policy (the centerpiece finding)

Fit on VAL: recommend a content/popularity blend for users with TRAIN history < n*, ALS above; choose n* (and blend weights) by maximizing segment-weighted NDCG@10. Report on TEST: hybrid vs every component, overall and per segment. The question Phase 4 inherits (recalibrated after the Phase 3 measured outcome below): *can content retrieval beat pop-t12m on the cold/shallow segments where CF cannot compete?* If nothing beats popularity anywhere, the shipped policy is recency-weighted popularity itself, published as such — the routing exhibit then shows *why* the null policy wins, per segment, with CIs. If the hybrid fails to dominate, diagnose and publish that — the site's credibility model rewards it.

> **Measured outcome (2026-08-06, Phase 3):** ALS (rank 128, best of a 10-entry
> single-variable VAL grid, 3 seeds) loses to pop-t12m on **every** warm segment
> with CIs excluding zero; the deficit shrinks monotonically with history depth
> (−0.0026 shallow → −0.0015 at 20+) but never crosses zero at observed depths.
> ALS beats item-kNN in every warm segment — the best classical CF tried, and it
> still loses to recency-weighted popularity. See EXPERIMENT_LOG.md Phase 3 and
> the paired_delta records in results/runs.jsonl.

### 6.5 Honesty rules and optional reality check

Single-variable experiment discipline; failed hypotheses logged; TEST frozen; no headline without a CI. **Offline-vs-online gap stated in the case study body** (not buried): offline ranking wins do not establish click/revenue lift; missing-not-at-random feedback, popularity feedback loops, and no counterfactual correction are all acknowledged; nothing here is an A/B result. Optional stretch: the **single-node reality check** — rebuild silver in DuckDB, publish both runtimes and an honest paragraph on when Spark is and isn't the right tool.

---

## 7. Data contracts and quality checks

Lightweight hand-rolled engine (no heavy framework): `contracts/*.yaml` per table (columns, types, nullability, value rules, referential rules) + a PySpark checker job emitting pass/fail + violation counts to a `dq_results` Iceberg table; violating rows → `quarantine.*` tables with a `violation_reason` column. CI runs the engine on the bundled fixture. Motivations below are engineering-level (several are failure modes observed first-hand in the ancestor pipeline; they live in code comments, not case-study drama):

| Contract check | Motivation |
|---|---|
| `price` NULL allowed, negative and sentinel values forbidden | Missing-price sentinels (−1.0) propagate into ranking math and UI ("$-1.00") if not stopped at the boundary |
| `rating` ∈ {1.0..5.0}; `timestamp` ∈ [1996-01-01, snapshot date]; keys non-null | Malformed rows must be counted and quarantined, never silently skipped |
| Dedup: (user, item) multi-reviews → keep latest, log count; exact-dup rows dropped, counted | Duplicate interactions inflate both popularity and CF confidence |
| Join integrity: interactions→items orphan rate measured and ledgered | Blind inner joins silently discard rows; orphan rate is a metric, not an accident |
| Row-count reconciliation waterfall: raw → bronze → silver → gold, every drop with a reason; **sums must reconcile exactly** | The single strongest "this pipeline is trustworthy" artifact |
| No all-null columns in published tables | Dead columns signal an unowned schema |
| Text hygiene: embedded newlines/control chars normalized; Parquet-only publication (no CSV) | Free-text fields with embedded newlines corrupt row-oriented formats |
| Brand normalization: case-folded, "Unknown" share tracked as a DQ metric | Brand facets are demo-visible; ~18% unknown-brand share in this category must be measured, not discovered |
| Ingestion completeness: bronze counts == published dataset counts (43.9M / 1.61M) | Guards against partial downloads and truncated reads |
| Schema enforcement at write (Iceberg) + contract version stamped into every eval manifest | Contracts without enforcement are documentation |

DQ metrics (violation rates, quarantine counts, waterfall) feed the demo's data-quality dashboard (§9).

---

## 8. Phased work plan with acceptance criteria

**Phase 0 — Repo, environment, frozen ingestion (2–3 days).** New repo, uv lock, **JDK pinned to 21 project-locally** (see below), CI skeleton, disk gate (≥35GB free or external `data/`). Download both Electronics files; record SHA-256 + date in `data/MANIFEST.md`; Spark job gz-jsonl → `bronze.reviews` (text/images projected out, documented) + `bronze.items` (schema seeded from the course StructType, corrected); bundled 50k-row fixture sampled from bronze.
✅ *Accept:* bronze row counts match published 43.9M / 1.61M (recorded delta if the snapshot differs); manifest complete; `java -version` under the project env reports 21; fixture-based CI smoke green.

> **JDK pin (verified on this machine, 2026-08-05).** Spark 4.x supports **Java 17/21 only**. This machine's default `java` is **25.0.2** (`~/.zprofile` puts Homebrew's unversioned `openjdk` first on PATH) — running Spark under it risks JVM module-access failures whose error messages do not point at the version. `openjdk@21` (21.0.12) is installed but keg-only, so `/usr/libexec/java_home` does not see it. **Do not change the global default.** Pin project-locally — both variables, since PySpark reads `JAVA_HOME` while other tools resolve `java` via PATH:
> ```
> JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
> PATH="$JAVA_HOME/bin:$PATH"
> ```
> Put these in the project's `Makefile` variables (and `.envrc` if direnv is used) so every `make` target inherits them, set the same version in the CI workflow, and record the rationale in the repo's environment docs. The uv lock does not cover the JVM — this pin is the only thing that does.

**Phase 1 — Silver + gold + contracts (3–4 days).** Contract YAMLs + checker; silver.interactions/items with quarantine; reconciliation waterfall; k-core gold table with funnel; user_stats, item_features, windowed popularity.
✅ *Accept:* all contracts pass or violations quarantined+ledgered; waterfall sums exactly; funnel counts published; `make data` rebuilds silver+gold deterministically from bronze.

**Phase 2 — Eval harness + baselines (3–4 days).** Harness (config→JSONL, full-catalog ranking, bootstrap CIs, segments) with pytest vs reference metrics; random/popularity/per-category/item-kNN evaluated on TEST.
✅ *Accept:* metric tests green; baseline numbers with CIs in results log; time-windowed vs all-time popularity delta measured.

**Phase 3 — ALS done right (3–5 days).** implicit ALS grid on VAL (logged), confidence-weighting variant, 3-seed final config, TEST evaluation per segment.
✅ *Accept:* ALS beats popularity on warm segments with CI excluding zero (if not, that IS the finding — publish it); tuning log ≥10 single-variable entries; seed variance reported.

**Phase 4 — Content retrieval + routing policy (3–4 days).** MiniLM item-text embeddings (versioned artifact + runtime/hardware recorded), ANN index, content evaluation (cold + warm), hybrid n*/blend fitted on VAL, TEST report.
✅ *Accept:* cold-segment content-vs-popularity delta with CI; crossover chart rendered from results log; hybrid ≥ best component overall or honest diagnosis published.

**Phase 5 — Lakehouse ops exhibits (2–3 days).** Monthly-partition incremental append; late-data upsert scenario; snapshot-pinned eval (`make reproduce-headline` re-runs the headline eval against a recorded Iceberg snapshot ID); compaction before/after; per-stage runtime/bytes lineage table.
✅ *Accept:* reproduce-headline succeeds from the pinned snapshot; ops metrics logged; lineage table complete.

**Phase 6 — Demo + case study (4–6 days).** Static demo per §9; case-study page with verification sections (§10) and provenance paragraph (§11).
✅ *Accept:* demo fully static/offline; every displayed number traces to a results-log record (receipts drawer); case study reviewed against §10 checklists.

**Phase 7 — Stretch (cut freely, in order):** single-node DuckDB reality check → ANN-vs-exact retrieval latency/recall trade-off exhibit → un-cored silver evaluation of popularity (k-core-inflation measurement; promote into core if cheap) → item2vec or SASRec as a single extra frontier point (only if everything else shipped; do not let it become a training project).

Total realistic effort: **~3–3.5 weeks part-time.**

---

## 9. Interactive demo spec

Static, self-contained, no server; precomputed JSON from the results log; consistent with the site's existing demo language.

1. **Crossover explorer (the front door).** NDCG@10 (switchable to Recall@20) by user-history segment, one line per model, CI bands; slider showing where the routing threshold n* sits and how TEST metrics would move if it were placed elsewhere (precomputed grid).
2. **"Pick a shopper."** ~30 curated real users (IDs re-hashed) across history-depth segments. Shows their TRAIN history timeline, then side-by-side top-10 from popularity / item-kNN / ALS / content / hybrid, with held-out TEST purchases badged as hits, and that user's per-model NDCG@10. Cold-start users included deliberately — ALS's empty answer for them is part of the exhibit.
3. **Live semantic search.** Query box → in-browser MiniLM query embedding (transformers.js, quantized ~25MB, lazy-loaded with a size warning) → similarity against a precomputed int8 embedding payload for the ~50k most popular items (≈20MB) → results with price/brand facets. Fallback: precomputed example queries. Rhymes with Privacy Preflight's browser-local inference story.
4. **Data-quality dashboard.** Reconciliation waterfall (raw→gold with reasons), contract pass/fail matrix, quarantine counts, "Unknown"-brand and null-price rates.
5. **Pipeline lineage panel.** Bronze→silver→gold DAG with per-stage rows in/out, bytes, runtime, and the Iceberg snapshot ID chain; a "time travel" toggle showing the same eval pinned to an older snapshot.
6. **Receipts drawer.** Every number links to its results-log record (run config hash, git SHA, dataset manifest hash, Iceberg snapshot ID).

---

## 10. Site evidence sections

**"How this was verified" must include:** frozen dataset manifest (source URL, download date, SHA-256, published-count reconciliation); uv-locked environment + hardware documented (M4, 16GB, local[10] — stated plainly); deterministic `make data` and snapshot-pinned `make reproduce-headline`; one-config-per-run + append-only JSONL results with git SHAs; user-bootstrap CIs on every headline number and paired deltas for every comparison; full-catalog ranking (no sampled negatives) stated explicitly; metric unit tests against a reference implementation; contract engine + quarantine ledger with exact reconciliation; embedding-artifact version + where it was computed; `EXPERIMENT_LOG.md` published including failed hypotheses; CI smoke eval on the bundled fixture.

**"What this does not prove" must include:** **no online evidence** — every metric is offline; ranking wins do not establish CTR/conversion/revenue lift; feedback here is missing-not-at-random and popularity-biased, with no counterfactual correction (the case study's offline-vs-online section is the extended version of this admission); reviews ≠ purchases — "interaction = positive" is a modeling assumption; k-core filtering inflates absolute metrics (measured where feasible, §8 P7) — compare models *within* this protocol, not absolute numbers across papers; single category (Electronics), single snapshot (ends 2023-09) — no generalization or freshness claim; not a serving system — no latency/SLA/throughput claims beyond the demo's static assets; single-node Spark — distributed-scale behavior is **projected** from cluster-portable code, not measured; Iceberg ops exhibits are single-writer local-catalog scenarios, not concurrent-writer production evidence; routing thresholds are fitted to this dataset — no transfer claim.

---

## 11. Provenance and team-contribution disclosure (required)

The idea descends from a five-person course project (CISC3018, University of Macau, Fall 2025). Per its report's contribution section (pp.30–32), the only deliverable uniquely attributed to Zhang Xiangguo is the poster/presentation material, with coding attributed primarily to two other members. The case study therefore must not claim any course artifact as individual work — and doesn't need to, since nothing from it ships. Use one short paragraph (adapt tone, keep substance):

> **Provenance.** This lab grew out of a five-person course project (CISC3018 Cloud Computing and Big Data Systems, University of Macau, Fall 2025) that built a recommendation bot on a 1M-interaction sample of this dataset; my primary individual contribution there was the presentation material. Everything in this lab — pipeline, contracts, models, evaluation harness, demo — was designed and built from the raw public dataset, solo, with its own verification chain.

Rules: never present the course's code, models, or metrics as personal accomplishments; do not reproduce or cite its results in the case study at all (the framing decision in the header removed that chapter). The Big Data lab reports (§3.3) are individually authored and may be cited as personal fundamentals in one sentence, but not republished.

---

## 12. Risks and cut-lines

| Risk | Mitigation |
|---|---|
| **Disk (~32GB free at plan time)** | Phase-0 hard gate: ≥35GB free, or `data/` on external SSD, or a rented VM for Phases 0–1 (results/artifacts sync back; document where each stage ran). Never store uncompressed raw; drop review `text` at bronze. |
| 16GB RAM vs ALS on full user set | Model on 5-core gold table; chunked full-catalog scoring; Spark `local[10]` with ~8g driver and disk spill. If factors=128 won't fit, report the frontier at 64 and say why. |
| gz-JSONL is non-splittable (single-task read) | One-time bronze conversion tolerates it (streaming decode); everything downstream reads Iceberg/Parquet. Record the conversion runtime in the lineage exhibit, not as a hidden cost. |
| ALS fails to beat popularity even on warm segments | Publishable — with the crossover chart it becomes "when is CF worth it on sparse review data?", a stronger conversation than a rigged win. Diagnose (sparsity, k-core level, confidence weighting) in the log. |
| Embedding 1M+ items too slow locally | Embed only items appearing in gold + demo catalog (≈300–500k) first; full 1.6M as stretch. Or a Colab T4; record hardware either way. |
| MiniLM/transformers.js payload too heavy for the demo | Reduce to 25–50k items int8, or precomputed-queries fallback (kept as the documented degradation path in §9.3). |
| Dataset snapshot drift (HF file updates) | SHA-256 manifest; counts reconciled at Phase 0; if the published files changed, record the new counts and proceed — the manifest, not the paper's numbers, is ground truth. |
| OneDrive interference | The new repo lives outside OneDrive (`~/Projects/`); source archives are read-only inputs. |
| Iceberg local-catalog friction with Spark 4 | Pin a known-good `iceberg-spark-runtime` version at Phase 0; if blocked >1 day, take the cut-line (partitioned Parquet + manifest) and drop only the ops exhibits, not the lakehouse tables. |
| **JVM version drift (host default is Java 25; Spark 4 supports 17/21)** | Project-local `JAVA_HOME`/`PATH` pin to `openjdk@21` in the Makefile + CI, never a global change (Phase 0 note). Symptom if it slips: JVM module-access/reflection errors that name neither Spark nor the Java version. Re-check with `java -version` inside the project env whenever Spark fails to start. |

**Cut order if scope must shrink (cut from the top):** 1) Phase 7 stretch items → 2) live in-browser semantic search (precomputed queries instead) → 3) late-data upsert + compaction exhibits (keep snapshot-pinned repro) → 4) item-kNN baseline (keep popularity variants) → 5) confidence-weighting variant → 6) Iceberg → partitioned Parquet (last infrastructure resort). **Never cut:** full-dataset ingestion with the reconciliation waterfall, contracts + quarantine, temporal splits, full-catalog ranking metrics with CIs, the popularity baseline, the crossover/segment analysis, the routing policy, the two verification sections, and the provenance paragraph — they are the identity of the piece.

---

## Appendix A — Key verified facts (for the implementing agent)

- **Dataset:** Amazon Reviews 2023 (McAuley Lab, HF `McAuley-Lab/Amazon-Reviews-2023`), Electronics: **43.9M reviews, 1,610,012 items**, millisecond timestamps present in raw reviews. Version pinned to 2023 by the archive's metadata ETL output count (1,610,012 — exact match). Research-use license; no raw redistribution; cite Hou et al. 2024.
- **Archived sample (for the sanity cross-check only):** 1,000,000 interactions; 185,242 users; 271,211 items; rating mean 4.2158, σ ≈ 1.291; ratings 1★ 89,031 / 2★ 47,705 / 3★ 71,612 / 4★ 141,739 / 5★ 649,913. No timestamp column. Source: `Project Archive/1 .../{final_demo_data.csv, ETL_logs.txt}`.
- **Reusable design seeds:** metadata StructType schema + `details`-map extraction (`1 .../jsonl_to_parquet.ipynb`); MiniLM item-text recipe, `all-MiniLM-L6-v2`, T4-batched (`2 .../Model Training and Bot - Metadata only.ipynb` — fix the unnormalized 0.92/0.08 blend).
- **Known raw-data hazards to contract against:** missing prices (the course filled them with −1.0 sentinels — don't), duplicate (user,item) reviews, metadata↔review join orphans, ~18% unknown brand share, free text with embedded newlines.
- **Contribution facts (report pp.30–32):** five-member team; Zhang Xiangguo's uniquely attributed deliverable = poster/presentation; coding attributed primarily to two other members. Drives §11's required language.
- **Big Data labs (BIT course, individually authored):** Hadoop 2.10.2 pseudo-distributed, HDFS Java API, HBase 2.4.17, hand-written MapReduce — reports only, no standalone code. One provenance sentence max.
- **Target machine at plan time:** Apple M4, 10 cores, 16GB RAM, ~32GB free disk (drives §5 budget and §12 disk gate).
