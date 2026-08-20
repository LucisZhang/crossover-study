# JDK pin (critical — see docs/ENVIRONMENT.md).
# Spark 4.x supports Java 17/21 only; host default java is 25.
# ?= so an already-exported JAVA_HOME (CI's setup-java, or a Linux host's
# /usr/lib/jvm/java-21-openjdk-amd64) overrides; the homebrew path below is
# the macOS laptop fallback.
JAVA_HOME ?= /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)
# Spark host sizing (spark_session.py reads these; empty = shipped defaults
# local[10] / 8g, unchanged for the laptop). Override per host, e.g. the rented
# 16-vCPU box: make data RECSYS_SPARK_MASTER='local[12]' RECSYS_SPARK_DRIVER_MEMORY=32g
# RECSYS_SPARK_LOCAL_DIR must point at the data disk on hosts whose /tmp is
# small (Spark shuffle spill lands there otherwise).
RECSYS_SPARK_MASTER ?=
RECSYS_SPARK_DRIVER_MEMORY ?=
RECSYS_SPARK_LOCAL_DIR ?=
export RECSYS_SPARK_MASTER
export RECSYS_SPARK_DRIVER_MEMORY
export RECSYS_SPARK_LOCAL_DIR
# Force Spark's driver to bind loopback: this host cannot resolve its own hostname
# for the SparkContext bind (harmless where hostname resolution already works).
export SPARK_LOCAL_IP := 127.0.0.1
# Prevents macOS sleep from killing long Spark runs; expands empty where
# caffeinate is absent (e.g. Linux CI), so recipes degrade gracefully.
CAFFEINATE := $(if $(shell command -v caffeinate 2>/dev/null),caffeinate -dims,)

.PHONY: java-check disk-gate test smoke download manifest ingest-reviews ingest-items bronze-verify fixture silver-items silver-interactions silver gold-core gold-features gold gold-uncored gold-item-text item-text-export contracts-audit waterfall data data-hash data-verify eval-extract extract-age eval-extract-uncored eval eval-baselines als-train eval-als item-train-stats regime-map deep-buckets download-ml32m extract-ml32m manifest-ml32m ingest-ml32m-ratings ingest-ml32m-movies ingest-ml32m-tags ingest-ml32m bronze-verify-ml32m silver-ml32m-items silver-ml32m-interactions silver-ml32m-tags silver-ml32m gold-ml32m-core gold-ml32m-features gold-ml32m contracts-audit-ml32m item-train-stats-ml32m data-ml32m churn-ml32m eval-extract-ml32m gold-ml32m-item-text item-text-export-ml32m embed-items-ml32m eval-ml32m compare embed-items crossover-chart ann-index bench-duckdb reproduce-headline ops-backfill ops-append ops-upsert ops-fragment ops-compact ops-expire ops-all clean-ops lineage lineage-check demo-export demo-export-phase8 demo-verify demo-verify-record demo-serve demo-grid demo-shoppers demo-dq demo-assets demo-offline-check

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

# One-shot ADDITIVE cache extension (Phase 8, T8-2): per-TRAIN-pair age in
# fractional days at train_end, written as train_age_days.npy next to the
# existing train_*_idx.npy. Reads local.gold.interactions_5core by Iceberg TIME
# TRAVEL at the snapshot the cache dir is keyed by, asserts the recomputed
# (user, item) multiset is EXACTLY the cached one, and aborts without writing on
# any mismatch. Touches no warehouse table and no results/runs.jsonl. Idempotent
# (AGE_FLAGS=--force rebuilds).
#   make extract-age
extract-age: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.extract_age $(AGE_FLAGS)

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

# --- Phase 8: Crossover Study Part II (§8b) -----------------------------------

# T8-1 Step A: per-item TRAIN support / last-TRAIN recency / first-seen date over
# local.gold.interactions_5core. ONE Spark aggregation; TRAIN columns use
# ts <= train_end only (leak-free), first_seen spans all splits and is used ONLY
# to date items (disclosed proxy). Writes data/eval/item_train_stats/<snapshot>/
# and nothing else — never the warehouse, never results/runs.jsonl. Idempotent:
# skips when the manifest already matches the live 5-core snapshot (--force to
# rebuild).
item-train-stats: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.item_train_stats $(ITEM_STATS_FLAGS)

# T8-1 Step B: the regime map — user history depth x item learnability at the
# train cutoff. JVM-free recomposition of the per-user top-50 lists already
# committed by the runs named in the config, crossed with the item-train-stats
# parquet. Appends one kind="regime_map" record; --dry-run prints everything and
# appends nothing (TEST runs additionally refuse a dirty tree).
#   make regime-map CONFIG=configs/regime_map_test.yaml
#   make regime-map CONFIG=configs/regime_map_val.yaml REGIME_FLAGS=--dry-run
regime-map:
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/regime_map_*.yaml"; exit 1; }
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.regime_map --config $(CONFIG) $(REGIME_FLAGS)

# T8-3: deeper history-depth buckets (20-49 / 50-99 / 100+), EXPLORATORY/derived.
# Regroups the persisted per-user metric values only (top-50 unused); the four
# buckets that coincide with frozen segments are asserted equal to the recorded
# per-segment means before anything is emitted. Appends one kind="deep_buckets"
# record; --dry-run appends nothing.
#   make deep-buckets CONFIG=configs/deep_buckets_test.yaml
deep-buckets:
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/deep_buckets_*.yaml"; exit 1; }
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.deep_buckets --config $(CONFIG) $(DEEP_FLAGS)

# --- Phase 9: ML-32M regime contrast, data stage (§8c, T9-3a) -----------------
# Wholly ADDITIVE to the Amazon lane: separate raw dir (data/raw/ml32m), separate
# Iceberg namespaces (bronze_ml32m / silver_ml32m / gold_ml32m / quarantine_ml32m
# / dq_ml32m), separate contracts dir (contracts/ml32m), separate frozen split
# (configs/splits_ml32m.yaml). No Amazon target, table or record is touched, so
# `make reproduce-headline` is unaffected.

# Download + hash ML-32M (~239MB zip). The §5 disk gate runs first, as for any
# download. The archive's CRCs are verified (there is no published checksum) and
# the three ingested CSVs (ratings/movies/tags) are extracted with their headers
# checked. The manifest step writes **data/MANIFEST_ML32M.md** — its own committed
# file. It must NEVER write data/MANIFEST.md: run records hash the whole manifest
# file and `make reproduce-headline` compares that hash, so ML-32M content in the
# Amazon manifest breaks the pinned headline's byte_exact verdict.
download-ml32m: disk-gate
	uv run python -m batch_recsys_lab.ingest.download_ml32m fetch
	uv run python -m batch_recsys_lab.ingest.download_ml32m manifest

# Re-extract the CSVs from an ml-32m.zip that is already on disk (CRCs verified,
# headers checked). Use this instead of `download-ml32m` when the member list
# grows — e.g. adding tags.csv must not trigger a 239MB re-download.
extract-ml32m:
	uv run python -m batch_recsys_lab.ingest.download_ml32m extract

manifest-ml32m:
	uv run python -m batch_recsys_lab.ingest.download_ml32m manifest

# Bronze (csv -> local.bronze_ml32m.*). Same PERMISSIVE corrupt-record accounting
# as the Amazon lane; the CSV header is verified before the read because the
# declared schema is applied positionally.
ingest-ml32m-ratings:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.ingest.bronze_ml32m --table ratings

ingest-ml32m-movies:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.ingest.bronze_ml32m --table movies

ingest-ml32m-tags:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.ingest.bronze_ml32m --table tags

ingest-ml32m: ingest-ml32m-movies ingest-ml32m-ratings ingest-ml32m-tags

# Bronze reconciliation, ML-32M (mirrors `bronze-verify`, but EXACT). Every
# bronze_ml32m row count must equal the data-row count recorded in
# data/MANIFEST_ML32M.md; any delta exits non-zero. This is the check that catches
# a silent CSV parse loss — the first real ingest landed 87,584 of 87,585 movies.
bronze-verify-ml32m:
	uv run python -m batch_recsys_lab.ingest.reconcile_ml32m

# Silver. Items first: the interactions AND tags contracts' item_fk orphan
# measures read local.silver_ml32m.items.
silver-ml32m-items:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.silver_ml32m --table items

silver-ml32m-interactions:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.silver_ml32m --table interactions

# Tags: silver only (no gold projection). T9-3b's content arm is title+genres+tags
# and builds its own item_text from this table.
silver-ml32m-tags:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.silver_ml32m --table tags

silver-ml32m: silver-ml32m-items silver-ml32m-interactions silver-ml32m-tags

# Gold. core = the shared iterative 5-core prune (k=5, mirror design) with the
# ML-32M projection; features = user_stats + item_features + popularity off the
# 5-core table, using configs/splits_ml32m.yaml.
gold-ml32m-core:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.gold_ml32m --stage core

gold-ml32m-features:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.gold_ml32m --stage features

gold-ml32m: gold-ml32m-core gold-ml32m-features

# Contract audit for the ML-32M tables ONLY: --contracts-dir keeps the glob off
# contracts/*.yaml (the Amazon set), and the results land in a separate ledger so
# the published DQ dashboard's totals are untouched. GOTCHA (same shape as the
# fresh-warehouse wrinkle in EXPERIMENT_LOG 2026-08-17): every contract in the
# directory is graded, so all six ML-32M tables must exist before this runs — a
# missing table is a hard AuditError, not a skip — including
# local.silver_ml32m.tags, whose contract lives in the same directory.
contracts-audit-ml32m:
	$(CAFFEINATE) uv run python -m batch_recsys_lab.contracts.run_audit \
	  --contracts-dir contracts/ml32m --dq-table local.dq_ml32m.dq_results

# Per-item TRAIN support / last-TRAIN recency / first-seen for ML-32M (the T8-1
# item axis, same module). Snapshot-keyed under its own root; idempotent
# (ITEM_STATS_FLAGS=--force rebuilds).
item-train-stats-ml32m: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.item_train_stats \
	  --five-core-table local.gold_ml32m.interactions_5core \
	  --splits-path configs/splits_ml32m.yaml \
	  --out data/eval/item_train_stats_ml32m $(ITEM_STATS_FLAGS)

# Full ML-32M data stage: bronze -> BRONZE RECONCILIATION -> silver -> gold ->
# contracts -> item axis. ONE run id ties every step's dq_results / funnel /
# build-summary rows together, same convention as `make data`. Unlike `make data`
# this DOES include bronze: the ML-32M ingest is minutes, not hours. The
# reconciliation sits between ingest and silver on purpose — a bronze table that
# does not match the hashed bytes must never become a silver table.
data-ml32m: export RECSYS_RUN_ID := $(if $(RECSYS_RUN_ID),$(RECSYS_RUN_ID),$(shell date -u +%Y%m%dT%H%M%SZ)-$(shell git rev-parse --short HEAD 2>/dev/null))
data-ml32m: java-check ingest-ml32m bronze-verify-ml32m silver-ml32m gold-ml32m gold-ml32m-item-text contracts-audit-ml32m item-train-stats-ml32m
	@echo "make data-ml32m complete · RECSYS_RUN_ID=$(RECSYS_RUN_ID)"

# T9-3a hinge: the pre-model churn statistic and its contrast against the recorded
# Amazon 0.4111 (which is re-derived from results/runs.jsonl, not trusted as a
# literal). Appends one kind="churn_contrast" record; --dry-run appends nothing.
# It is a TEST-window number, so the dirty-tree guard applies — commit first.
#   make churn-ml32m
#   make churn-ml32m CHURN_FLAGS=--dry-run
churn-ml32m: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.churn_contrast \
	  --config $(if $(CONFIG),$(CONFIG),configs/churn_contrast_ml32m.yaml) $(CHURN_FLAGS)

# --- Phase 9: ML-32M model ladder, VAL stage (§8c, T9-3b) ---------------------
# Mirrors eval-extract / eval exactly, in the gold_ml32m namespace and its own
# cache root, keeping the Amazon lane (eval-extract/eval) wholly untouched.
# NOTE: `extract-ml32m` (above) already names the raw-CSV re-extract step from
# download-ml32m; this eval-cache build target is deliberately named
# eval-extract-ml32m to avoid colliding with it.

# Eval cache extract for ML-32M (Step A, mirrors eval-extract): Spark ->
# numpy/scipy cache under data/eval/cache_ml32m, snapshot-keyed and idempotent,
# split-labeled via configs/splits_ml32m.yaml (not the Amazon splits.yaml).
#   make eval-extract-ml32m
eval-extract-ml32m: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.eval.extract \
	  --out data/eval/cache_ml32m \
	  --five-core-table local.gold_ml32m.interactions_5core \
	  --user-stats-table local.gold_ml32m.user_stats \
	  --item-features-table local.gold_ml32m.item_features \
	  --popularity-table local.gold_ml32m.popularity \
	  --splits-path configs/splits_ml32m.yaml

# ML-32M item_text (Phase 9, T9-3b §3): the 5-core catalog LEFT JOINed to
# item_features (title/genres) and to the TRAIN-cutoff tag aggregation over
# silver_ml32m.tags (ts <= configs/splits_ml32m.yaml train_end, inclusive;
# COUNT(DISTINCT user_id) weight; weight DESC, tag ASC; top 10). The contract
# audit follows so contracts/ml32m/gold_ml32m_item_text.yaml is graded and the
# §3(e) coverage measures land in dq_ml32m.dq_results BEFORE any embedding.
gold-ml32m-item-text: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.item_text_ml32m --mode build
	$(CAFFEINATE) uv run python -m batch_recsys_lab.contracts.run_audit \
	  --contracts-dir contracts/ml32m --dq-table local.dq_ml32m.dq_results

# JVM-free ML-32M export (T9-3b §3e): reorders local.gold_ml32m.item_text to the
# ML-32M eval cache's item_ids order and writes
# data/eval/text_ml32m/<snapshot>/item_text.parquet + export_manifest.json.
# Requires `make eval-extract-ml32m` for the live 5-core snapshot first.
item-text-export-ml32m: java-check
	$(CAFFEINATE) uv run python -m batch_recsys_lab.features.item_text_ml32m --mode export

# MiniLM Step A on the ML-32M lane (recipe v1_ml32m_title_genres_tags). Same
# module, same locally cached model artifact, ML-32M roots
# (data/eval/text_ml32m -> data/eval/minilm_ml32m). Idempotent.
embed-items-ml32m:
	$(CAFFEINATE) uv run --group embed python -m batch_recsys_lab.models.minilm_embed \
	  --recipe ml32m

# Eval scoring for one ML-32M config (Step B, mirrors eval): pure numpy/scipy,
# no Spark. Scores the config's model over data/eval/cache_ml32m and APPENDS one
# record to results/runs.jsonl. run_eval reads splits_path/manifest_path off the
# config itself (configs/splits_ml32m.yaml, data/MANIFEST_ML32M.md) — see
# run_eval.py's config-carried-path precedence — so no extra flags are needed
# here.
#   make eval-ml32m CONFIG=configs/eval_pop_t12m_ml32m_val.yaml
eval-ml32m: export RECSYS_RUN_ID := $(if $(RECSYS_RUN_ID),$(RECSYS_RUN_ID),$(shell date -u +%Y%m%dT%H%M%SZ)-$(shell git rev-parse --short HEAD 2>/dev/null))
eval-ml32m:
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/eval_*_ml32m_*.yaml"; exit 1; }
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
	uv run python -m batch_recsys_lab.demo.export_phase8 --config configs/demo_export.yaml
	uv run python -m batch_recsys_lab.demo.export_dq --config configs/dq_export.yaml
	uv run python -m batch_recsys_lab.demo.export_shoppers --config configs/shoppers_export.yaml
	uv run python -m batch_recsys_lab.demo.export_receipts --config configs/demo_export.yaml

demo-export-phase8:
	uv run python -m batch_recsys_lab.demo.export_phase8 --config configs/demo_export.yaml

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
