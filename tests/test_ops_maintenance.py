"""Compaction + snapshot expiry, and the ops-namespace guard (Phase 5, T20).

The mutating half runs once, in a module-scoped ``lifecycle`` fixture that
records the state at each stage; the test functions then assert on those
recordings. Sequencing matters here (expiry is only meaningful *after*
compaction has orphaned files) and test-order dependence would be fragile.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from datetime import datetime
from pathlib import Path

import pytest

from batch_recsys_lab.ops import maintenance
from batch_recsys_lab.ops.snapshot_metrics import files_stats, table_metadata_for
from conftest import warehouse_of

pytestmark = pytest.mark.spark

TABLE = "local.ops.maintenance_under_test"
DDL = "user_id string, parent_asin string, ts timestamp, rating double"
N_APPENDS = 20


def _file_paths(spark, table) -> set[Path]:
    rows = spark.sql(f"SELECT file_path FROM {table}.files").collect()
    return {
        Path(p[len("file:") :] if p.startswith("file:") else p)
        for p in (r["file_path"] for r in rows)
    }


def _checksum(spark, table) -> list:
    rows = spark.sql(
        f"SELECT xxhash64(user_id, parent_asin, ts, rating) AS h FROM {table}"
    ).collect()
    return sorted(int(r["h"]) for r in rows)


@pytest.fixture(scope="module")
def lifecycle(spark):
    warehouse = warehouse_of(spark)
    maintenance.ensure_ops_namespace(spark)
    spark.sql(f"DROP TABLE IF EXISTS {TABLE} PURGE")
    spark.sql(f"CREATE TABLE {TABLE} ({DDL}) USING iceberg PARTITIONED BY (months(ts))")

    # Deliberate fragmentation: 20 separate single-row appends.
    for i in range(N_APPENDS):
        spark.createDataFrame(
            [(f"u{i}", f"p{i}", datetime(2023, 7, 1 + i), 1.0 + (i % 4))], DDL
        ).writeTo(TABLE).append()

    state = {
        "files_before": files_stats(spark, TABLE),
        "paths_before": _file_paths(spark, TABLE),
        "rows_before": spark.table(TABLE).count(),
        "checksum_before": _checksum(spark, TABLE),
    }

    # min-input-files=2 forces bin-packing of the small fixture partition
    # (Iceberg's default of 5 leaves lightly fragmented groups alone by design).
    state["compact"] = maintenance.compact(spark, TABLE, options={"min-input-files": "2"})
    state["files_after"] = files_stats(spark, TABLE)
    state["rows_after"] = spark.table(TABLE).count()
    state["checksum_after"] = _checksum(spark, TABLE)
    state["paths_after"] = _file_paths(spark, TABLE)

    # One more append so BOTH snapshots retained by expire(retain_last=2) are
    # post-compaction — otherwise the newest-but-one snapshot would still
    # reference the pre-compaction files and expiry would have nothing to delete.
    spark.createDataFrame(
        [("u-late", "p-late", datetime(2023, 7, 28), 4.0)], DDL
    ).writeTo(TABLE).append()

    state["orphaned"] = state["paths_before"] - _file_paths(spark, TABLE)
    state["meta_before_expire"] = table_metadata_for(warehouse, TABLE)
    state["orphans_on_disk_before_expire"] = {
        p: p.exists() for p in state["orphaned"]
    }
    state["expire"] = maintenance.expire(spark, TABLE, retain_last=2)
    state["meta_after_expire"] = table_metadata_for(warehouse, TABLE)
    state["live_paths_after_expire"] = _file_paths(spark, TABLE)
    return state


def test_compact_reduces_file_count(lifecycle):
    assert lifecycle["files_before"]["file_count"] == N_APPENDS
    assert lifecycle["files_after"]["file_count"] < lifecycle["files_before"]["file_count"]
    assert lifecycle["compact"]["rewritten_files"] >= 2
    assert lifecycle["compact"]["added_files"] >= 1
    assert lifecycle["compact"]["failed_files"] == 0
    assert lifecycle["compact"]["procedure"] == "rewrite_data_files"


def test_compact_preserves_rows_and_content(lifecycle):
    assert lifecycle["rows_after"] == lifecycle["rows_before"] == N_APPENDS
    assert lifecycle["checksum_after"] == lifecycle["checksum_before"]


def test_expire_drops_snapshots_to_retain_last(lifecycle):
    assert lifecycle["meta_before_expire"]["snapshot_count"] > 2
    assert lifecycle["meta_after_expire"]["snapshot_count"] == 2
    # Expiry never moves the current snapshot.
    assert (
        lifecycle["meta_after_expire"]["current_snapshot_id"]
        == lifecycle["meta_before_expire"]["current_snapshot_id"]
    )


def test_expire_actually_deletes_orphaned_files(lifecycle):
    orphaned = lifecycle["orphaned"]
    assert orphaned, "compaction should have left unreferenced files behind"
    assert all(lifecycle["orphans_on_disk_before_expire"].values())
    assert lifecycle["expire"]["deleted_data_files"] >= len(orphaned)
    assert all(not p.exists() for p in orphaned), "expired files must be gone from disk"
    # ...and the live files were not collateral damage.
    assert all(p.exists() for p in lifecycle["live_paths_after_expire"])


# --- the hard guard -----------------------------------------------------------

FORBIDDEN = (
    "local.gold.x",
    "local.gold.interactions_5core",
    "local.silver.interactions",
    "local.bronze.reviews",
    "local.dq.dq_results",
    "local.quarantine.interactions",
    "ops.interactions_monthly",  # missing catalog
    "local.opsx.table",          # prefix look-alike
    "local.ops",                 # namespace, not a table
    "local.ops.",                # empty leaf
    "local.ops.a.b",             # nested
)


@pytest.mark.parametrize("table", FORBIDDEN)
def test_guard_refuses_non_ops_tables_without_touching_spark(table):
    """``spark=None`` proves the guard is the first statement: any Spark use
    would raise AttributeError, not ValueError."""
    with pytest.raises(ValueError):
        maintenance.compact(None, table)
    with pytest.raises(ValueError):
        maintenance.expire(None, table)
    with pytest.raises(ValueError):
        maintenance.drop_ops_table(None, table)
    with pytest.raises(ValueError):
        maintenance.require_ops_table(table)


def test_guard_accepts_ops_tables():
    assert (
        maintenance.require_ops_table("local.ops.interactions_monthly")
        == "local.ops.interactions_monthly"
    )


def test_ensure_namespace_refuses_non_ops_namespace():
    with pytest.raises(ValueError, match="Refusing to create namespace"):
        maintenance.ensure_ops_namespace(None, "local.gold")
