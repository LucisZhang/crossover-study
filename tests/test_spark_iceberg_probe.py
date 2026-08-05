"""End-to-end probe: Spark 4 + Iceberg local catalog round-trip.

Proves the stack works (create table, insert, read back, snapshot metadata,
drop) before we invest in ingestion. Marked ``spark`` so it can be excluded
from lightweight runs. First execution downloads the Iceberg runtime jar from
Maven into the default Ivy cache (~/.ivy2); subsequent runs are offline-fast.
"""

from __future__ import annotations

import pytest

from batch_recsys_lab.spark_session import get_spark

pytestmark = pytest.mark.spark


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    warehouse = tmp_path_factory.mktemp("warehouse")
    session = get_spark(
        app_name="spark-iceberg-probe",
        warehouse=warehouse,
        master="local[2]",
        driver_memory="2g",
    )
    yield session
    session.stop()


def test_iceberg_roundtrip(spark):
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.probe")
    spark.sql("DROP TABLE IF EXISTS local.probe.t")
    spark.sql(
        "CREATE TABLE local.probe.t (id BIGINT, name STRING) USING iceberg"
    )
    try:
        spark.sql(
            "INSERT INTO local.probe.t VALUES "
            "(1, 'alpha'), (2, 'beta'), (3, 'gamma')"
        )

        rows = spark.sql(
            "SELECT id, name FROM local.probe.t ORDER BY id"
        ).collect()
        assert [(r["id"], r["name"]) for r in rows] == [
            (1, "alpha"),
            (2, "beta"),
            (3, "gamma"),
        ]
        assert spark.table("local.probe.t").count() == 3

        snapshots = spark.sql(
            "SELECT snapshot_id FROM local.probe.t.snapshots"
        ).collect()
        assert len(snapshots) >= 1
    finally:
        spark.sql("DROP TABLE IF EXISTS local.probe.t")
