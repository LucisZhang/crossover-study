"""Shared pytest fixtures.

A session-scoped Spark session against a tmp Iceberg warehouse — the pattern
``tests/test_bronze_ingest.py`` uses inline (that file can migrate here later).

Note (see ``spark_session.get_spark`` docstring): the first session started in
a process wins its master/warehouse; later ``get_spark`` calls return it. This
session-scoped fixture is intended to be the process's single Spark session.
"""

from __future__ import annotations

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
