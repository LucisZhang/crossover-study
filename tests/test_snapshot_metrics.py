"""JVM-free snapshot metadata reader + Spark-side file stats (Phase 5, T20)."""

from __future__ import annotations

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from datetime import datetime
from pathlib import Path

import pytest

from batch_recsys_lab.eval.runlog import iceberg_snapshot_id
from batch_recsys_lab.ops import maintenance
from batch_recsys_lab.ops.snapshot_metrics import (
    current_snapshot_ids,
    disk_avail_gb,
    files_stats,
    metadata_dir,
    parse_df_avail_gb,
    snapshot_summary,
    split_table_name,
    table_metadata,
    table_metadata_for,
)
from conftest import warehouse_of

TABLE = "local.ops.snapmetrics_under_test"
DDL = "user_id string, parent_asin string, ts timestamp, rating double"


def _rows(n, start=0):
    return [
        (f"u{i}", f"p{i}", datetime(2023, 7, 1 + (i % 28)), 1.0 + (i % 4))
        for i in range(start, start + n)
    ]


# --- pure-python parts (no Spark) --------------------------------------------


def test_split_table_name():
    assert split_table_name("local.ops.x") == ("local", "ops", "x")
    assert split_table_name("local.a.b.c") == ("local", "a.b", "c")
    with pytest.raises(ValueError):
        split_table_name("local.x")


def test_metadata_dir_drops_the_catalog_name():
    assert metadata_dir("/w", "ops", "x") == Path("/w/ops/x/metadata")


def test_table_metadata_on_a_missing_table_is_not_an_error(tmp_path):
    meta = table_metadata(tmp_path, "ops", "nope")
    assert meta["exists"] is False
    assert meta["current_snapshot_id"] is None
    assert meta["snapshots"] == []
    assert current_snapshot_ids(tmp_path, ["local.ops.nope"]) == {"local.ops.nope": None}


def test_parse_df_avail_gb():
    out = (
        "Filesystem 1G-blocks Used Avail Capacity iused ifree %iused Mounted on\n"
        "/dev/disk3s5     926   700   211    77%  1.2M  2.2G   35%   /System/Volumes/Data\n"
    )
    assert parse_df_avail_gb(out) == 211.0
    assert parse_df_avail_gb("") is None
    assert parse_df_avail_gb("only a header\n") is None


def test_disk_avail_gb_is_positive(tmp_path):
    assert disk_avail_gb(tmp_path) > 0


# --- Spark-backed parts -------------------------------------------------------


@pytest.mark.spark
def test_table_metadata_tracks_appends_and_replaces(spark):
    warehouse = warehouse_of(spark)
    maintenance.ensure_ops_namespace(spark)
    spark.sql(f"DROP TABLE IF EXISTS {TABLE} PURGE")

    spark.createDataFrame(_rows(10), DDL).writeTo(TABLE).create()
    meta1 = table_metadata_for(warehouse, TABLE)
    assert meta1["exists"] is True
    assert meta1["snapshot_count"] == 1
    assert meta1["snapshots"][0]["operation"] == "append"
    assert meta1["snapshots"][0]["added_records"] == 10
    assert meta1["snapshots"][0]["total_records"] == 10
    # Same fact as the Phase-2 helper this generalises.
    assert meta1["current_snapshot_id"] == iceberg_snapshot_id(warehouse, TABLE)

    spark.createDataFrame(_rows(5, start=100), DDL).writeTo(TABLE).append()
    meta2 = table_metadata_for(warehouse, TABLE)
    assert meta2["snapshot_count"] == 2
    assert meta2["current_snapshot_id"] != meta1["current_snapshot_id"]
    newest = snapshot_summary(warehouse, TABLE, meta2["current_snapshot_id"])
    assert newest["operation"] == "append"
    assert newest["added_records"] == 5
    assert newest["total_records"] == 15
    assert newest["parent_snapshot_id"] == meta1["current_snapshot_id"]
    # Snapshot ids are unique and the chain is coherent.
    ids = [s["snapshot_id"] for s in meta2["snapshots"]]
    assert len(set(ids)) == len(ids)

    spark.createDataFrame(_rows(3, start=200), DDL).writeTo(TABLE).createOrReplace()
    meta3 = table_metadata_for(warehouse, TABLE)
    assert meta3["snapshot_count"] == 3
    replaced = snapshot_summary(warehouse, TABLE, meta3["current_snapshot_id"])
    assert replaced["operation"] in {"overwrite", "replace"}
    assert replaced["total_records"] == 3
    assert spark.table(TABLE).count() == replaced["total_records"]

    # Unknown snapshot id -> empty, not an exception.
    assert snapshot_summary(warehouse, TABLE, -1) == {}


@pytest.mark.spark
def test_files_stats_matches_the_files_on_disk(spark):
    warehouse = warehouse_of(spark)
    table = "local.ops.filestats_under_test"
    maintenance.ensure_ops_namespace(spark)
    spark.sql(f"DROP TABLE IF EXISTS {table} PURGE")
    spark.sql(f"CREATE TABLE {table} ({DDL}) USING iceberg PARTITIONED BY (months(ts))")

    assert files_stats(spark, "local.ops.does_not_exist") == {
        "file_count": 0,
        "total_bytes": 0,
        "avg_file_mb": 0.0,
    }

    for i in range(4):
        spark.createDataFrame(
            [(f"u{i}", f"p{i}", datetime(2023, 8, 1 + i), 3.0)], DDL
        ).writeTo(table).append()

    stats = files_stats(spark, table)
    data_dir = Path(warehouse) / "ops" / table.split(".")[-1] / "data"
    on_disk = sorted(data_dir.rglob("*.parquet"))
    assert stats["file_count"] == len(on_disk) == 4
    assert stats["total_bytes"] == sum(p.stat().st_size for p in on_disk)
    assert stats["avg_file_mb"] == round(stats["total_bytes"] / 4 / (1024 * 1024), 4)
