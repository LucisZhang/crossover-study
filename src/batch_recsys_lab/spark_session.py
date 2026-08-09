"""Spark 4 + Iceberg local session builder.

Single entry point for constructing the SparkSession used across the lab.
Configures the Iceberg Spark runtime (fetched via ``spark.jars.packages``,
not vendored) and a filesystem-backed ``local`` Hadoop catalog.

Note on repeated calls: ``SparkSession.builder.getOrCreate()`` returns the
*existing* active session if one is already running in the process. Config
that must be fixed before the JVM launches — notably ``spark.driver.memory``
and ``spark.master`` — is only honored on the first call that actually starts
the JVM. Calling ``get_spark`` a second time in the same process (e.g. with a
different ``driver_memory``) returns the original session with the original
settings; the new arguments are silently ignored. Tests that need distinct
Spark configs must run in separate processes.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession

DEFAULT_APP_NAME = "batch-recsys-lab"
ICEBERG_RUNTIME_PACKAGE = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.11.0"
ICEBERG_EXTENSIONS = "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"


def get_spark(
    app_name: str = DEFAULT_APP_NAME,
    warehouse: str | Path = "data/warehouse",
    master: str = "local[10]",
    driver_memory: str = "8g",
    extra_conf: dict[str, str] | None = None,
) -> SparkSession:
    """Build (or return the existing) Spark session wired for Iceberg.

    Parameters
    ----------
    app_name:
        Spark application name.
    warehouse:
        Filesystem path for the ``local`` Iceberg Hadoop catalog warehouse.
        Resolved to an absolute path so the session is independent of the
        process working directory.
    master:
        Spark master URL (e.g. ``local[10]`` for production, ``local[2]``
        for tests).
    driver_memory:
        Driver JVM heap (e.g. ``8g``). Must be set before JVM launch; honored
        only on the first call that starts the JVM in this process.
    extra_conf:
        Additional ``spark.*`` settings applied to the builder *before*
        ``getOrCreate()``. This is the hook for conf that cannot be set at
        runtime via ``spark.conf.set`` — notably ``spark.driver.maxResultSize``,
        which the driver reads when the JVM starts. Applied last, so it can
        override any of the defaults above. Same caveat as ``driver_memory``:
        ignored if this call returns an already-running session.

    Returns
    -------
    SparkSession
        A session with the ``local`` Iceberg catalog registered and the
        session timezone pinned to UTC.
    """
    warehouse_path = Path(warehouse).resolve()

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.jars.packages", ICEBERG_RUNTIME_PACKAGE)
        .config("spark.sql.extensions", ICEBERG_EXTENSIONS)
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.local.type", "hadoop")
        .config("spark.sql.catalog.local.warehouse", str(warehouse_path))
        .config("spark.sql.session.timeZone", "UTC")
    )

    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    return builder.getOrCreate()
