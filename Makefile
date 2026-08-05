# JDK pin (critical — see docs/ENVIRONMENT.md).
# Spark 4.x supports Java 17/21 only; host default java is 25.
# ?= so CI's setup-java env can override.
JAVA_HOME ?= /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)

.PHONY: java-check disk-gate test smoke download manifest ingest-reviews ingest-items bronze-verify fixture data

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

data:
	@echo "not implemented yet (Phase 0 T3+)"
