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

import os
from pathlib import Path

from pyspark.sql import SparkSession

DEFAULT_APP_NAME = "batch-recsys-lab"
DEFAULT_MASTER = "local[10]"
DEFAULT_DRIVER_MEMORY = "8g"
ICEBERG_RUNTIME_PACKAGE = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.11.0"
ICEBERG_EXTENSIONS = "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"


def _env_override(env_key: str, value: str, shipped_default: str) -> str:
    """Environment-overridable sizing default.

    The env var wins only when the caller passed the shipped default (which is
    what every CLI's argparse default resolves to when the flag is omitted).
    An explicitly different value — e.g. tests' ``local[2]`` or a CLI flag —
    always wins over the environment. Empty env values count as unset.
    """
    if value == shipped_default:
        return os.environ.get(env_key) or value
    return value


def get_spark(
    app_name: str = DEFAULT_APP_NAME,
    warehouse: str | Path = "data/warehouse",
    master: str = DEFAULT_MASTER,
    driver_memory: str = DEFAULT_DRIVER_MEMORY,
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
        for tests). When left at the shipped default, the
        ``RECSYS_SPARK_MASTER`` env var (if set) overrides it.
    driver_memory:
        Driver JVM heap (e.g. ``8g``). Must be set before JVM launch; honored
        only on the first call that starts the JVM in this process. When left
        at the shipped default, ``RECSYS_SPARK_DRIVER_MEMORY`` (if set)
        overrides it.
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

    # Host-sizing overrides (see docstring): honored only when the caller used
    # the shipped default, so tests' local[2] etc. are never overridden.
    master = _env_override("RECSYS_SPARK_MASTER", master, DEFAULT_MASTER)
    driver_memory = _env_override(
        "RECSYS_SPARK_DRIVER_MEMORY", driver_memory, DEFAULT_DRIVER_MEMORY
    )

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

    # Scratch/shuffle-spill directory. Defaults to Spark's own default (/tmp);
    # on hosts whose /tmp sits on a small system disk (e.g. the rented Linux
    # box), point this at the data disk. Pre-JVM conf: same first-call caveat
    # as driver_memory. Empty env value counts as unset.
    spark_local_dir = os.environ.get("RECSYS_SPARK_LOCAL_DIR")
    if spark_local_dir:
        builder = builder.config("spark.local.dir", spark_local_dir)

    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    return builder.getOrCreate()
