# JDK pin (critical — see docs/ENVIRONMENT.md).
# Spark 4.x supports Java 17/21 only; host default java is 25.
# ?= so CI's setup-java env can override.
JAVA_HOME ?= /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)
# Force Spark's driver to bind loopback: this host cannot resolve its own hostname
# for the SparkContext bind (harmless where hostname resolution already works).
export SPARK_LOCAL_IP := 127.0.0.1
# Prevents macOS sleep from killing long Spark runs; expands empty where
# caffeinate is absent (e.g. Linux CI), so recipes degrade gracefully.
CAFFEINATE := $(if $(shell command -v caffeinate 2>/dev/null),caffeinate -dims,)

.PHONY: java-check disk-gate test smoke download manifest ingest-reviews ingest-items bronze-verify fixture silver-items silver-interactions silver gold-core gold-features gold gold-uncored gold-item-text item-text-export contracts-audit waterfall data data-hash data-verify eval-extract eval-extract-uncored eval eval-baselines als-train eval-als compare embed-items crossover-chart ann-index bench-duckdb reproduce-headline ops-backfill ops-append ops-upsert ops-fragment ops-compact ops-expire ops-all clean-ops lineage lineage-check demo-export demo-verify demo-verify-record demo-serve demo-grid demo-shoppers demo-dq demo-assets demo-offline-check

java-check:
	@v=$$(java -version 2>&1); echo "$$v" | grep -q '"21\.' || (echo "ERROR: java -version does not report 21.x under project env (JAVA_HOME=$(JAVA_HOME)). Spark 4.x requires Java 17/21." && exit 1); echo "$$v"

disk-gate:
	@uv run python -c "import shutil, sys; free = shutil.disk_usage('.').free / (1024**3); \
	sys.exit(0) if free >= 35 else (_ for _ in ()).throw(SystemExit('ERROR: only %.1fGB free; need >=35GB. Relocate data/ to external storage per UPGRADE_PLAN.md §5.' % free))"

test:
	uv run pytest

smoke: java-check test

download:
	uv run python -m batch_recsys_lab.ingest.download fetch

manifest:
	uv run python -m batch_recsys_lab.ingest.download manifest

ingest-reviews:
	uv run python -m batch_recsys_lab.ingest.bronze --table reviews

ingest-items:
	uv run python -m batch_recsys_lab.ingest.bronze --table items

bronze-verify:
	uv run python -m batch_recsys_lab.ingest.reconcile

fixture:
	uv run python -m batch_recsys_lab.ingest.make_fixture

# Silver builds (Phase 1, T3). Items first: interactions' FK measure reads silver.items.
silver-items:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.silver --table items

silver-interactions:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.silver --table interactions

silver: silver-items silver-interactions

# Gold builds (Phase 1, T6/T7). gold-core = iterative 5-core prune; gold-features
# = user_stats + item_features + popularity projections off the 5-core table.
gold-core:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.kcore

gold-features:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.gold

gold: gold-core gold-features

# Un-cored gold projections (Phase 7 stretch item 3). Writes ONLY *_uncored
# tables (+ one append to dq.dq_results from the join-loss measure); the
# frozen 5-core tables and their snapshots are never touched.
gold-uncored: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.gold \
	  --five-core-table local.silver.interactions \
	  --user-stats-table local.gold.user_stats_uncored \
	  --item-features-table local.gold.item_features_uncored \
	  --popularity-table local.gold.popularity_uncored

# item_text (Phase 4, T9). Builds local.gold.item_text (5-core catalog x
# item_features x bronze.items text fields), then runs the full contract audit
# (the engine has no per-table invocation) so gold_item_text.yaml is graded.
gold-item-text: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.item_text --mode build
	$(CAFFEINATE) uv run python -m batch_recsys_lab.contracts.run_audit

# JVM-free export (Phase 4, T9): reorders local.gold.item_text to the eval
# cache's item_ids order for the live 5-core snapshot and writes
# data/eval/text/<snapshot>/item_text.parquet + export_manifest.json. Requires
# `make eval-extract` to have already built the cache for the live snapshot.
item-text-export: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.item_text --mode export

# Contract audit (Phase 1, T8). Runs every contracts/*.yaml against its published
# table, appends dq_results, stamps contract.name/version as Iceberg TBLPROPERTIES,
# prints a table×check matrix, and exits non-zero on any status=='fail'.
contracts-audit:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.contracts.run_audit

# Reconciliation waterfall (Phase 1, T4b). Asserts raw→bronze→silver→gold sums
# exactly against live Iceberg counts; publishes MANIFEST section + waterfall.json.
waterfall:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.waterfall

# Deterministic rebuild (Phase 1, T8; §8 acceptance #4). ONE run id per `make data`
# invocation ties every step's dq_results/waterfall/funnel rows together: generated
# once here (UTC ts + git short sha) unless RECSYS_RUN_ID is already set in the env,
# and exported to every step (all CLIs honor RECSYS_RUN_ID). Target-specific export
# so it inherits into the prerequisite recipes (silver, gold, …) but never leaks
# into `make test`. Flow: java-check → silver (items→interactions) → gold (core →
# features) → contracts-audit → waterfall.
data: export RECSYS_RUN_ID := $(if $(RECSYS_RUN_ID),$(RECSYS_RUN_ID),$(shell date -u +%Y%m%dT%H%M%SZ)-$(shell git rev-parse --short HEAD 2>/dev/null))
data: java-check silver gold contracts-audit waterfall
	@echo "make data complete · RECSYS_RUN_ID=$(RECSYS_RUN_ID)"

# Determinism verification (Phase 1, T8). T9 usage:
#   make data        # build #1
#   make data-hash   # record data/table_hashes.json from build #1
#   make data        # build #2 (fresh run id; rebuilds silver+gold from bronze)
#   make data-verify # recompute current warehouse hashes, diff vs the recorded
#                    # file, exit non-zero on ANY drift → proves content-identical.
# data-hash WRITES data/table_hashes.json; data-verify COMPARES the live warehouse
# against that existing file (it does not overwrite it).
data-hash:
	uv run python -m batch_recsys_lab.features.verify_determinism --out data/table_hashes.json

data-verify:
	uv run python -m batch_recsys_lab.features.verify_determinism --compare data/table_hashes.json

# Eval cache extract (Phase 2, T1). Spark->numpy/scipy cache, snapshot-keyed and
# idempotent: skips (exit 0) if the cache for the live 5-core snapshot already
# exists. Step A of the two-process eval design (see eval/extract.py docstring).
eval-extract: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.extract

# Un-cored eval cache: keyed by the silver.interactions snapshot id, in its
# own root so cache resolution stays unambiguous per universe.
eval-extract-uncored: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.extract \
	  --out data/eval/cache_uncored --max-result-size 4g \
	  --five-core-table local.silver.interactions \
	  --user-stats-table local.gold.user_stats_uncored \
	  --item-features-table local.gold.item_features_uncored \
	  --popularity-table local.gold.popularity_uncored

# Eval scoring (Phase 2, T5). Step B: pure numpy/scipy, no Spark. Scores one
# config's model over the snapshot-keyed cache and APPENDS one record to
# results/runs.jsonl (append-only, invariant #3). RECSYS_RUN_ID follows the same
# convention as `make data` (env override, else UTC ts + git short sha) so the
# generated run_id is stable across a single invocation and echoed into the record.
#   make eval CONFIG=configs/eval_pop_t12m_test.yaml
eval: export RECSYS_RUN_ID := $(if $(RECSYS_RUN_ID),$(RECSYS_RUN_ID),$(shell date -u +%Y%m%dT%H%M%SZ)-$(shell git rev-parse --short HEAD 2>/dev/null))
eval:
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/eval_*.yaml"; exit 1; }
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.run_eval --config $(CONFIG)

# The four TEST baselines in sequence (Phase 2, T7): random, trailing-12m pop,
# all-time pop, per-category pop. One RECSYS_RUN_ID ties the batch together.
eval-baselines: export RECSYS_RUN_ID := $(if $(RECSYS_RUN_ID),$(RECSYS_RUN_ID),$(shell date -u +%Y%m%dT%H%M%SZ)-$(shell git rev-parse --short HEAD 2>/dev/null))
eval-baselines:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.run_eval --config configs/eval_random_test.yaml
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.run_eval --config configs/eval_pop_t12m_test.yaml
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.run_eval --config configs/eval_pop_alltime_test.yaml
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.run_eval --config configs/eval_pop_category_test.yaml

# ALS Step A (Phase 2, T2/T3): train Spark MLlib ALS once and persist factor
# artifacts under data/eval/als/<snapshot>/<param_hash>/. Requires JVM (java-check).
# Idempotent: skips (exit 0) if a matching artifact already exists.
#   make als-train CONFIG=configs/eval_als_val_rank64.yaml
als-train: java-check
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/eval_als_*.yaml"; exit 1; }
	$(CAFFEINATE) uv run python -m batch_recsys_lab.models.als_train --config $(CONFIG)

# ALS train + score under ONE RECSYS_RUN_ID (Phase 2, T2/T3): Step A persists the
# factors, then Step B (pure numpy, no Spark) scores the SAME config and appends
# one record to results/runs.jsonl. Same run_id convention as `eval`.
#   make eval-als CONFIG=configs/eval_als_val_rank64.yaml
eval-als: export RECSYS_RUN_ID := $(if $(RECSYS_RUN_ID),$(RECSYS_RUN_ID),$(shell date -u +%Y%m%dT%H%M%SZ)-$(shell git rev-parse --short HEAD 2>/dev/null))
eval-als: java-check
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/eval_als_*.yaml"; exit 1; }
	$(CAFFEINATE) uv run python -m batch_recsys_lab.models.als_train --config $(CONFIG)
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.run_eval --config $(CONFIG)

# Paired-bootstrap delta between two eval runs (Phase 2, T5). Pure numpy.
#   make compare CONFIG=configs/compare_als_vs_itemknn_val.yaml
compare:
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/compare_*.yaml"; exit 1; }
	uv run python -m batch_recsys_lab.eval.compare --config $(CONFIG)

# Crossover chart (Phase 4, T14). Renders per-segment NDCG@10 lines + 95% CI
# bands STRICTLY from results/runs.jsonl (no Spark, no JVM, no model code) to
# results/figures/<stem>.svg/.png. Regenerable from the log alone.
#   make crossover-chart CONFIG=configs/crossover_val.yaml
crossover-chart:
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/crossover_*.yaml"; exit 1; }
	uv run python -m batch_recsys_lab.eval.crossover_chart --config $(CONFIG)

# MiniLM item-embedding Step A (Phase 4, T10). JVM-free: reads the T9 export
# (data/eval/text/<snapshot>/item_text.parquet), re-verifies alignment, and
# writes data/eval/minilm/<snapshot>/<recipe_hash_short>/embeddings.npy +
# minilm_manifest.json. Runs under the torch-carrying `embed` dependency
# group (default install stays torch-free). Idempotent: skips if a matching
# artifact already exists.
embed-items:
	$(CAFFEINATE) uv run --group embed python -m batch_recsys_lab.models.minilm_embed

# ANN index artifact + latency/overlap receipt (Phase 4, T16). Demo-facing
# ONLY — never used in eval metrics (CLAUDE.md invariant #4: all eval records
# use exact full-catalog chunked matmul). Builds an hnswlib cosine index over
# the T10 embeddings (data/eval/minilm/<snapshot>/<recipe_hash>/ann_index.bin
# + ann_manifest.json), measures ANN-vs-exact top-10 overlap and latency over
# a fixed 10k-user sample, and appends one kind="ann_receipt" record to
# results/runs.jsonl. Idempotent build; `--measure` (always passed here) runs
# the receipt regardless. No JVM/Spark gate needed.
ann-index:
	$(CAFFEINATE) uv run --group embed python -m batch_recsys_lab.models.ann_index --measure

# DuckDB single-node reality check (Phase 7 stretch item 1). JVM-free; reads
# bronze via the DuckDB iceberg extension at the CURRENT snapshot, writes only
# under data/bench/duckdb/, never the warehouse. 3 timed runs; --append adds
# one kind="bench" record from a clean tree (use BENCH_FLAGS=--dry-run to
# inspect first). Spark reference timings are quoted from
# data/build_summary.jsonl inside the record, never re-run.
BENCH_FLAGS ?= --append
bench-duckdb:
	$(CAFFEINATE) uv run --group bench python -m batch_recsys_lab.bench.duckdb_silver --runs 3 --content-parity $(BENCH_FLAGS)

# Snapshot-pinned reproduction of the recorded headline eval (Phase 5, T18).
# Reads configs/headline.yaml -> the pinned run_id, rebuilds the eval cache by
# Iceberg TIME TRAVEL at that record's snapshot IDs (never the live tables),
# re-runs the ORIGINAL config, diffs the candidate record against the recorded
# one field by field, and appends one kind="reproduce" record. Refuses a dirty
# tree; exits non-zero unless the verdict is byte_exact. Needs the JVM for the
# extract step only.
reproduce-headline: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.reproduce

# --- Lakehouse ops exhibits (Phase 5, T20) -----------------------------------
# Every step runs against local.ops.interactions_monthly ONLY: a disposable copy
# of silver.interactions partitioned by months(ts). The published
# bronze/silver/gold/dq/quarantine tables are never written, compacted or
# expired — enforced by ops.maintenance.require_ops_table and re-asserted
# JVM-free after every step (the runner exits non-zero if any protected snapshot
# moved, or if free disk drops below 8GB). Each step appends ONE kind="ops"
# record to results/runs.jsonl.

ops-backfill: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.ops.run_scenario --step backfill

#   make ops-append MONTH=2023-07
ops-append: java-check
	@test -n "$(MONTH)" || { echo "ERROR: set MONTH=YYYY-MM"; exit 1; }
	$(CAFFEINATE) uv run python -m batch_recsys_lab.ops.run_scenario --step append --month $(MONTH)

ops-upsert: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.ops.run_scenario --step upsert

# Stage the compaction exhibit: the backfill writes one file per months(ts)
# partition, so rewrite_data_files had nothing to bin-pack (measured no-op).
# `ops-fragment` deletes one month and re-appends the SAME rows one calendar day
# at a time (one small file each) — simulated micro-batch ingestion. Row counts
# are asserted unchanged; the month is staged in a durable scratch table first.
#   make ops-fragment MONTH=2023-06
ops-fragment: java-check
	@test -n "$(MONTH)" || { echo "ERROR: set MONTH=YYYY-MM"; exit 1; }
	$(CAFFEINATE) uv run python -m batch_recsys_lab.ops.run_scenario --step fragment --month $(MONTH)

ops-compact: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.ops.run_scenario --step compact

ops-expire: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.ops.run_scenario --step expire

# Full scenario in order. ONE RECSYS_RUN_ID ties all seven records together
# (same convention as `make data`); it is exported into each sub-make.
ops-all: export RECSYS_RUN_ID := $(if $(RECSYS_RUN_ID),$(RECSYS_RUN_ID),$(shell date -u +%Y%m%dT%H%M%SZ)-$(shell git rev-parse --short HEAD 2>/dev/null))
ops-all: java-check
	$(MAKE) ops-backfill
	$(MAKE) ops-append MONTH=2023-07
	$(MAKE) ops-append MONTH=2023-08
	$(MAKE) ops-append MONTH=2023-09
	$(MAKE) ops-upsert
	$(MAKE) ops-compact
	$(MAKE) ops-expire
	@echo "make ops-all complete · RECSYS_RUN_ID=$(RECSYS_RUN_ID)"

# --- Lineage table (Phase 5, T24) --------------------------------------------
# Per-stage rows/bytes/wall-clock, assembled STRICTLY by reading back artifacts
# earlier runs already committed: data/MANIFEST.md, data/ingest_summary.jsonl,
# data/build_summary.jsonl, Iceberg snapshot summaries (JVM-free — no java-check,
# no Spark, no caffeinate: this reads metadata JSON and a ~34-row parquet), and
# results/runs.jsonl. Nothing is re-measured or re-run; a runtime that was never
# persisted stays null and is footnoted. Exits non-zero, naming what is missing,
# if any expected stage cannot be assembled, and a partial table is never written.
#
# The ops chain contributes ONE ROW PER kind="ops" RECORD, in log order — the
# compaction exhibit deliberately runs compact/expire more than once and includes
# a measured no-op, so repeats are labelled from their own record data
# (ops.compact[noop] vs ops.compact[30->1], ops.expire[retain=2,deleted=3]) and
# never collapsed. --expect-ops is a FLOOR (default: backfill, append x3, upsert,
# fragment, compact, expire); extra records are enumerated, never an error.
#
# `lineage` writes results/lineage.json + results/lineage.md AND appends one
# kind="lineage" record, so the committed artifact and the record that attests to
# its sha256 land in one deliberate invocation. `lineage-check` grades
# completeness and writes nothing.
lineage:
	uv run python -m batch_recsys_lab.ops.lineage --append-record

lineage-check:
	uv run python -m batch_recsys_lab.ops.lineage --check-only

# Teardown: DROP TABLE ... PURGE + remove data/warehouse/ops. Appends no record.
# The directory removal is guarded in code: it only ever deletes a directory
# whose parent is exactly the warehouse root.
clean-ops: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.ops.run_scenario --step clean

# --- Static demo (Phase 6) ----------------------------------------------------
# Pure projection of committed evidence: these targets read results/runs.jsonl
# (never write it — invariant #3) plus record-anchored artifacts, and write
# demo/data/*.json. No Spark, no JVM, no model code — the JAVA_HOME pin above is
# irrelevant here and no java-check prerequisite is declared. Deterministic and
# idempotent: documents carry no timestamp, so re-exporting unchanged evidence
# leaves them byte-identical (only trace_manifest.json's generated_at moves).
#
# Exporter order matters: receipts must run LAST — it reads the trace manifest
# the other exporters wrote to find the run_id closure it has to document.
# Later Phase 6 tasks insert their exporters before it (T27 policy_grid, T28
# shoppers, T29 dq, T30 lineage/timetravel, T35 search).
demo-export:
	uv run python -m batch_recsys_lab.demo.export_crossover --config configs/demo_export.yaml
	uv run python -m batch_recsys_lab.demo.export_policy_grid --config configs/demo_export.yaml
	uv run python -m batch_recsys_lab.demo.export_lineage --config configs/demo_export.yaml
	uv run python -m batch_recsys_lab.demo.export_dq --config configs/dq_export.yaml
	uv run python -m batch_recsys_lab.demo.export_shoppers --config configs/shoppers_export.yaml
	uv run python -m batch_recsys_lab.demo.export_receipts --config configs/demo_export.yaml

# Shopper pipeline prerequisites (T28): deterministic selection, then the Spark
# read-only history job (snapshot-guarded against the headline record's pins).
# Spark lives here, NOT in demo-export — demo-export must stay JVM-free.
demo-shoppers:
	uv run python -m batch_recsys_lab.demo.select_shoppers --config configs/shoppers_export.yaml
	$(CAFFEINATE) uv run python -m batch_recsys_lab.demo.shopper_history_job --config configs/shoppers_export.yaml

# DQ exhibit prerequisite (T29): read-only Spark pull of the contract ledger,
# the k-core funnel, the reconciliation ledger and both quarantine tables ->
# data/demo_export/dq_raw.json. Writes nothing to the warehouse; snapshot-guards
# the headline-pinned tables before the JVM starts, then pins every read to the
# snapshot id it captured. Spark lives here, NOT in demo-export.
#
# The kind="dq_export" record is appended in a SEPARATE, JVM-free step so its
# git_sha names the commit that produced dq_raw.json:
#   uv run python -m batch_recsys_lab.demo.dq_export_job \
#       --config configs/dq_export.yaml --phase record --dry-run   # prints, appends nothing
#   uv run python -m batch_recsys_lab.demo.dq_export_job \
#       --config configs/dq_export.yaml --phase record --append    # from a clean tree
demo-dq: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.demo.dq_export_job --config configs/dq_export.yaml --phase collect

# Independent re-resolution of every exported number (shares no code with the
# writing path). Exits non-zero on any coverage, exact-match, artifact-hash or
# staleness failure. `demo-verify-record` is the CI mode: same checks minus the
# per-user parquet reads (those artifacts are gitignored, CI does not have them).
demo-verify:
	uv run python -m batch_recsys_lab.demo.verify_traceability --data-dir demo/data --mode=full

demo-verify-record:
	uv run python -m batch_recsys_lab.demo.verify_traceability --data-dir demo/data --mode=record

# Serve the static site on loopback. ES modules need an HTTP origin (file://
# blocks them); this is the only "server" the demo ever needs.
demo-serve:
	uv run python -m http.server 8000 --bind 127.0.0.1 -d demo

# n* TEST grid recomposition (T27): appends one kind="policy_grid" record from
# the already-committed per-user metrics named in configs/policy_grid_test.yaml
# (no re-scoring, no refitting). Run before `demo-export` picks up a new record.
demo-grid:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.policy.grid_test --config configs/policy_grid_test.yaml

# Search-exhibit assets (T35): int8 payload from the pinned embeddings artifact,
# then the SHA-256-verified model download. 79.5MB total, never committed
# (UPGRADE_PLAN §12 cut order #2) — this one command rebuilds both from scratch.
# JVM-free like demo-export; the Spark-produced slice it consumes
# (data/demo_export/search_items_raw.parquet) comes from demo-shoppers.
# The download verifies every byte against the SHA-256s recorded in
# demo/README.md and exits non-zero with no partial install on any mismatch.
# To move a pin deliberately, bootstrap the table first:
#   uv run python -m batch_recsys_lab.demo.fetch_search_assets --record-hashes
demo-assets:
	uv run python -m batch_recsys_lab.demo.export_search --config configs/search_export.yaml
	uv run python -m batch_recsys_lab.demo.fetch_search_assets --config configs/search_export.yaml

# External-URL scanner over the assembled demo/ (T36). demo/vendor/ is
# report-only (documented exemption, see verify_offline.py docstring);
# everything else must be clean of URLs in an executable position. Exits
# non-zero on any violation.
demo-offline-check:
	uv run python -m batch_recsys_lab.demo.verify_offline --demo-dir demo
