"""Shared pytest fixtures.

A session-scoped Spark session against a tmp Iceberg warehouse — the pattern
``tests/test_bronze_ingest.py`` uses inline (that file can migrate here later).

Note (see ``spark_session.get_spark`` docstring): the first session started in
a process wins its master/warehouse; later ``get_spark`` calls return it. This
session-scoped fixture is intended to be the process's single Spark session.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from batch_recsys_lab.spark_session import get_spark


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    warehouse = tmp_path_factory.mktemp("warehouse")
    session = get_spark(
        app_name="contracts-test",
        warehouse=warehouse,
        master="local[2]",
        driver_memory="2g",
    )
    yield session
    # Do not stop(): the JVM is torn down at process exit; other spark-marked
    # tests in the process may share this session.


def warehouse_of(spark) -> str:
    """The tmp warehouse root behind the session fixture (JVM-free readers need
    the path, and the fixture does not expose it)."""
    return spark.conf.get("spark.sql.catalog.local.warehouse")


# --- Phase 5 (T20) ops fixtures ----------------------------------------------

# Same column list/types as local.silver.interactions (features/silver.py
# SILVER_INTERACTION_COLS), under a throwaway namespace so no test can be
# confused with — or accidentally exercise — the real silver table.
OPS_SOURCE_TABLE = "local.opssrc.interactions"
OPS_SOURCE_DDL = (
    "user_id string, parent_asin string, asin string, rating double, "
    "ts timestamp, helpful_vote long, verified_purchase boolean"
)
OPS_SOURCE_ROWS = 540


def _ops_source_rows():
    """540 deterministic interactions spread over 2023-01-01 .. 2023-09-27.

    Two rows per day, and ``ts`` is a bijection of the row index, so every
    ``(user_id, parent_asin, ts)`` identity triple is unique (as it is in silver
    after the D2 dedup). No row carries rating 5.0 — the upsert scenario sets
    exactly that value, so its effect is measurable.
    """
    base = datetime(2023, 1, 1)
    rows = []
    for i in range(OPS_SOURCE_ROWS):
        ts = base + timedelta(days=i // 2, minutes=37 * (i % 2))
        rows.append(
            (
                f"u{i % 30}",
                f"p{(i * 7) % 19}",
                f"a{(i * 7) % 19}",
                1.0 + float(i % 4),
                ts,
                i % 5,
                bool(i % 3),
            )
        )
    return rows


@pytest.fixture(scope="session")
def ops_source(spark):
    """Session-scoped read-only stand-in for ``local.silver.interactions``."""
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.opssrc")
    spark.createDataFrame(_ops_source_rows(), OPS_SOURCE_DDL).writeTo(
        OPS_SOURCE_TABLE
    ).createOrReplace()
    return OPS_SOURCE_TABLE
