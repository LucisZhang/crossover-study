# JDK pin (critical — see docs/ENVIRONMENT.md).
# Spark 4.x supports Java 17/21 only; host default java is 25.
# ?= so CI's setup-java env can override.
JAVA_HOME ?= /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)
# Force Spark's driver to bind loopback: this host cannot resolve its own hostname
# for the SparkContext bind (harmless where hostname resolution already works).
export SPARK_LOCAL_IP := 127.0.0.1

.PHONY: java-check disk-gate test smoke download manifest ingest-reviews ingest-items bronze-verify fixture silver-items silver-interactions silver gold-core gold-features gold contracts-audit waterfall data data-hash data-verify eval-extract eval eval-baselines als-train eval-als compare

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
	uv run python -m batch_recsys_lab.features.silver --table items

silver-interactions:
	uv run python -m batch_recsys_lab.features.silver --table interactions

silver: silver-items silver-interactions

# Gold builds (Phase 1, T6/T7). gold-core = iterative 5-core prune; gold-features
# = user_stats + item_features + popularity projections off the 5-core table.
gold-core:
	uv run python -m batch_recsys_lab.features.kcore

gold-features:
	uv run python -m batch_recsys_lab.features.gold

gold: gold-core gold-features

# Contract audit (Phase 1, T8). Runs every contracts/*.yaml against its published
# table, appends dq_results, stamps contract.name/version as Iceberg TBLPROPERTIES,
# prints a table×check matrix, and exits non-zero on any status=='fail'.
contracts-audit:
	uv run python -m batch_recsys_lab.contracts.run_audit

# Reconciliation waterfall (Phase 1, T4b). Asserts raw→bronze→silver→gold sums
# exactly against live Iceberg counts; publishes MANIFEST section + waterfall.json.
waterfall:
	uv run python -m batch_recsys_lab.features.waterfall

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
	uv run python -m batch_recsys_lab.eval.extract

# Eval scoring (Phase 2, T5). Step B: pure numpy/scipy, no Spark. Scores one
# config's model over the snapshot-keyed cache and APPENDS one record to
# results/runs.jsonl (append-only, invariant #3). RECSYS_RUN_ID follows the same
# convention as `make data` (env override, else UTC ts + git short sha) so the
# generated run_id is stable across a single invocation and echoed into the record.
#   make eval CONFIG=configs/eval_pop_t12m_test.yaml
eval: export RECSYS_RUN_ID := $(if $(RECSYS_RUN_ID),$(RECSYS_RUN_ID),$(shell date -u +%Y%m%dT%H%M%SZ)-$(shell git rev-parse --short HEAD 2>/dev/null))
eval:
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/eval_*.yaml"; exit 1; }
	uv run python -m batch_recsys_lab.eval.run_eval --config $(CONFIG)

# The four TEST baselines in sequence (Phase 2, T7): random, trailing-12m pop,
# all-time pop, per-category pop. One RECSYS_RUN_ID ties the batch together.
eval-baselines: export RECSYS_RUN_ID := $(if $(RECSYS_RUN_ID),$(RECSYS_RUN_ID),$(shell date -u +%Y%m%dT%H%M%SZ)-$(shell git rev-parse --short HEAD 2>/dev/null))
eval-baselines:
	uv run python -m batch_recsys_lab.eval.run_eval --config configs/eval_random_test.yaml
	uv run python -m batch_recsys_lab.eval.run_eval --config configs/eval_pop_t12m_test.yaml
	uv run python -m batch_recsys_lab.eval.run_eval --config configs/eval_pop_alltime_test.yaml
	uv run python -m batch_recsys_lab.eval.run_eval --config configs/eval_pop_category_test.yaml

# ALS Step A (Phase 2, T2/T3): train Spark MLlib ALS once and persist factor
# artifacts under data/eval/als/<snapshot>/<param_hash>/. Requires JVM (java-check).
# Idempotent: skips (exit 0) if a matching artifact already exists.
#   make als-train CONFIG=configs/eval_als_val_rank64.yaml
als-train: java-check
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/eval_als_*.yaml"; exit 1; }
	uv run python -m batch_recsys_lab.models.als_train --config $(CONFIG)

# ALS train + score under ONE RECSYS_RUN_ID (Phase 2, T2/T3): Step A persists the
# factors, then Step B (pure numpy, no Spark) scores the SAME config and appends
# one record to results/runs.jsonl. Same run_id convention as `eval`.
#   make eval-als CONFIG=configs/eval_als_val_rank64.yaml
eval-als: export RECSYS_RUN_ID := $(if $(RECSYS_RUN_ID),$(RECSYS_RUN_ID),$(shell date -u +%Y%m%dT%H%M%SZ)-$(shell git rev-parse --short HEAD 2>/dev/null))
eval-als: java-check
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/eval_als_*.yaml"; exit 1; }
	uv run python -m batch_recsys_lab.models.als_train --config $(CONFIG)
	uv run python -m batch_recsys_lab.eval.run_eval --config $(CONFIG)

# Paired-bootstrap delta between two eval runs (Phase 2, T5). Pure numpy.
#   make compare CONFIG=configs/compare_als_vs_itemknn_val.yaml
compare:
	@test -n "$(CONFIG)" || { echo "ERROR: set CONFIG=configs/compare_*.yaml"; exit 1; }
	uv run python -m batch_recsys_lab.eval.compare --config $(CONFIG)
