"""Iceberg table maintenance: compaction + snapshot expiry (Phase 5, T20).

Both operations are file-destructive — ``rewrite_data_files`` replaces the data
files of a table, ``expire_snapshots`` *deletes* the files no retained snapshot
references. Run against ``local.gold.*`` they would silently invalidate every
snapshot ID recorded in ``results/runs.jsonl``. Hence :func:`require_ops_table`
is the unconditional first statement of every mutator here (and of every mutator
elsewhere in ``batch_recsys_lab.ops``): the fully-qualified table name must start
with ``local.ops.``, or the call raises ``ValueError`` before Spark is touched.

Procedure signatures verified against Iceberg 1.11.0 + Spark 4.0.4 on this
machine (2026-08-07):

* ``CALL local.system.rewrite_data_files(table => 'local.ops.x'[, options => map(...)])``
  -> ``(rewritten_data_files_count, added_data_files_count, rewritten_bytes_count,
  failed_data_files_count, removed_delete_files_count)``
* ``CALL local.system.expire_snapshots(table => 'local.ops.x',
  older_than => TIMESTAMP '...', retain_last => N)``
  -> ``(deleted_data_files_count, deleted_position_delete_files_count,
  deleted_equality_delete_files_count, deleted_manifest_files_count,
  deleted_manifest_lists_count, deleted_statistics_files_count)``

The ``table =>`` argument is passed **catalog-qualified** (``local.ops.x``). The
namespace-only form (``ops.x``) also resolves, but only after Spark 4 fails to
load ``ops`` as a catalog and Iceberg falls back — which dumps a
``CatalogNotFoundException`` stack trace into the log on every call.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from pyspark.sql import SparkSession

OPS_NAMESPACE = "local.ops"
OPS_TABLE_PREFIX = OPS_NAMESPACE + "."

# Tables whose snapshot IDs are cited by recorded runs — never mutated by this
# package. Asserted unmoved after every scenario step (see run_scenario).
PROTECTED_TABLES = (
    "local.gold.interactions_5core",
    "local.gold.user_stats",
    "local.gold.item_features",
    "local.gold.popularity",
    "local.silver.interactions",
)


def require_ops_table(table: str) -> str:
    """Return ``table`` if it lives in ``local.ops``, else raise ``ValueError``.

    The single enforcement point for the package's absolute rule. Must be the
    first statement of any function that writes, rewrites or deletes files.
    """
    name = str(table).strip()
    if not name.startswith(OPS_TABLE_PREFIX):
        raise ValueError(
            f"Refusing to mutate {table!r}: batch_recsys_lab.ops may only write "
            f"tables under {OPS_NAMESPACE!r}. The published bronze/silver/gold/dq/"
            "quarantine tables back recorded results and are never written, "
            "compacted or expired by this package."
        )
    leaf = name[len(OPS_TABLE_PREFIX) :]
    if not leaf or "." in leaf or "/" in leaf:
        raise ValueError(
            f"Refusing to mutate {table!r}: expected exactly "
            f"{OPS_NAMESPACE}.<table>, got a nested or empty name."
        )
    return name


def _catalog_of(table: str) -> str:
    return table.split(".", 1)[0]


def ensure_ops_namespace(spark: SparkSession, namespace: str = OPS_NAMESPACE) -> None:
    """``CREATE NAMESPACE IF NOT EXISTS`` for the ops namespace only."""
    if namespace != OPS_NAMESPACE and not namespace.startswith(OPS_TABLE_PREFIX):
        raise ValueError(f"Refusing to create namespace {namespace!r} outside {OPS_NAMESPACE!r}")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")


def table_exists(spark: SparkSession, table: str) -> bool:
    """True if the catalog can resolve ``table`` (no exception leaks out)."""
    try:
        return spark.catalog.tableExists(table)
    except Exception:  # noqa: BLE001 - a missing namespace also means "no table"
        return False


def compact(
    spark: SparkSession,
    table: str,
    options: dict[str, str] | None = None,
) -> dict:
    """``rewrite_data_files`` on an ops table; returns the procedure's metrics.

    ``options`` is forwarded verbatim as the procedure's ``options`` map (e.g.
    ``{"min-input-files": "2"}`` to force bin-packing of a small fixture table;
    the Iceberg default is 5, so a lightly fragmented partition is otherwise
    left alone by design).
    """
    require_ops_table(table)
    start = time.perf_counter()
    args = [f"table => '{table}'"]
    if options:
        pairs = ", ".join(f"'{k}', '{v}'" for k, v in sorted(options.items()))
        args.append(f"options => map({pairs})")
    sql = f"CALL {_catalog_of(table)}.system.rewrite_data_files({', '.join(args)})"
    row = spark.sql(sql).collect()[0]
    return {
        "procedure": "rewrite_data_files",
        "call": sql,
        "rewritten_files": int(row["rewritten_data_files_count"]),
        "added_files": int(row["added_data_files_count"]),
        "rewritten_bytes": int(row["rewritten_bytes_count"]),
        "failed_files": int(row["failed_data_files_count"]),
        "removed_delete_files": int(row["removed_delete_files_count"]),
        "options": dict(options or {}),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }


def expire(
    spark: SparkSession,
    table: str,
    retain_last: int = 2,
    older_than: datetime | None = None,
) -> dict:
    """``expire_snapshots`` on an ops table; returns the procedure's metrics.

    ``older_than`` defaults to *now* (UTC), so every snapshot beyond the newest
    ``retain_last`` is expired and its unreferenced files are deleted from disk.
    """
    require_ops_table(table)
    start = time.perf_counter()
    cutoff = older_than or datetime.now(timezone.utc)
    literal = cutoff.strftime("%Y-%m-%d %H:%M:%S.%f")
    sql = (
        f"CALL {_catalog_of(table)}.system.expire_snapshots("
        f"table => '{table}', older_than => TIMESTAMP '{literal}', "
        f"retain_last => {int(retain_last)})"
    )
    row = spark.sql(sql).collect()[0]
    return {
        "procedure": "expire_snapshots",
        "call": sql,
        "retain_last": int(retain_last),
        "older_than": literal,
        "deleted_data_files": int(row["deleted_data_files_count"]),
        "deleted_position_delete_files": int(row["deleted_position_delete_files_count"]),
        "deleted_equality_delete_files": int(row["deleted_equality_delete_files_count"]),
        "deleted_manifest_files": int(row["deleted_manifest_files_count"]),
        "deleted_manifest_lists": int(row["deleted_manifest_lists_count"]),
        "deleted_statistics_files": int(row["deleted_statistics_files_count"]),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }


def drop_ops_table(spark: SparkSession, table: str) -> dict:
    """``DROP TABLE ... PURGE`` an ops table (the ``clean-ops`` teardown)."""
    require_ops_table(table)
    existed = table_exists(spark, table)
    spark.sql(f"DROP TABLE IF EXISTS {table} PURGE")
    return {"table": table, "existed": existed, "dropped": existed}
