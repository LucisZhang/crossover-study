"""Iceberg snapshot / file metrics (Phase 5, T20).

:func:`table_metadata` is the generalisation of
``eval.runlog.iceberg_snapshot_id``: it reads the Hadoop-catalog metadata JSON
directly, so the protected-table guard in ``run_scenario`` can assert that
``local.gold.*`` and ``local.silver.interactions`` did not move **without
starting a JVM** — the guard must be trustworthy even when Spark is the thing
that misbehaved.

:func:`files_stats` is the one Spark-side reader here: file counts and bytes come
from the ``<table>.files`` metadata table, which is the same source the
compaction before/after exhibit is graded on.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from pyspark.sql import SparkSession

# Summary keys promoted out of the free-form snapshot summary map.
_SUMMARY_INT_KEYS = (
    "added-records",
    "deleted-records",
    "total-records",
    "added-data-files",
    "deleted-data-files",
    "total-data-files",
    "added-files-size",
    "removed-files-size",
    "total-files-size",
)


def split_table_name(full_name: str) -> tuple[str, str, str]:
    """``"local.ops.x"`` -> ``("local", "ops", "x")`` (multi-level namespaces
    keep their dots in the middle element)."""
    parts = str(full_name).split(".")
    if len(parts) < 3:
        raise ValueError(
            f"expected a catalog-qualified name <catalog>.<namespace>.<table>, got {full_name!r}"
        )
    return parts[0], ".".join(parts[1:-1]), parts[-1]


def metadata_dir(warehouse: str | Path, namespace: str, table: str) -> Path:
    """Physical metadata dir of a Hadoop-catalog table (catalog name is the
    warehouse root and never appears in the path)."""
    return Path(warehouse).joinpath(*namespace.split("."), table, "metadata")


def table_metadata(warehouse: str | Path, namespace: str, table: str) -> dict:
    """Current snapshot ID + the full snapshot list, JVM-free.

    Returns ``{"exists": False, ...}`` rather than raising when the table has no
    metadata dir — a missing table is a legitimate before-state for the backfill
    step, and it must not be confused with a *moved* table by the guard.
    """
    meta_dir = metadata_dir(warehouse, namespace, table)
    hint = meta_dir / "version-hint.text"
    if not hint.exists():
        return {
            "exists": False,
            "metadata_dir": str(meta_dir),
            "current_snapshot_id": None,
            "metadata_version": None,
            "snapshot_count": 0,
            "snapshots": [],
            "partition_spec": [],
        }

    version = int(hint.read_text().strip())
    meta = json.loads((meta_dir / f"v{version}.metadata.json").read_text())
    current = meta.get("current-snapshot-id")

    snapshots = []
    for snap in meta.get("snapshots", []):
        summary = dict(snap.get("summary", {}))
        entry = {
            "snapshot_id": int(snap["snapshot-id"]),
            "parent_snapshot_id": (
                int(snap["parent-snapshot-id"])
                if snap.get("parent-snapshot-id") is not None
                else None
            ),
            "sequence_number": snap.get("sequence-number"),
            "timestamp_ms": snap.get("timestamp-ms"),
            "operation": summary.get("operation"),
            "summary": summary,
        }
        for key in _SUMMARY_INT_KEYS:
            raw = summary.get(key)
            entry[key.replace("-", "_")] = int(raw) if raw is not None else None
        snapshots.append(entry)

    spec_id = meta.get("default-spec-id", 0)
    spec = next(
        (s for s in meta.get("partition-specs", []) if s.get("spec-id") == spec_id),
        {"fields": []},
    )

    return {
        "exists": True,
        "metadata_dir": str(meta_dir),
        "table_uuid": meta.get("table-uuid"),
        "format_version": meta.get("format-version"),
        "metadata_version": version,
        "current_snapshot_id": int(current) if current is not None else None,
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "partition_spec": [
            {"name": f.get("name"), "transform": f.get("transform")}
            for f in spec.get("fields", [])
        ],
    }


def table_metadata_for(warehouse: str | Path, full_name: str) -> dict:
    """:func:`table_metadata` addressed by catalog-qualified name."""
    _, namespace, table = split_table_name(full_name)
    return table_metadata(warehouse, namespace, table)


def current_snapshot_ids(warehouse: str | Path, tables) -> dict[str, int | None]:
    """``{table: current snapshot id or None}`` — the protected-table guard's
    before/after fingerprint. JVM-free by construction."""
    return {
        t: table_metadata_for(warehouse, t)["current_snapshot_id"] for t in tables
    }


def snapshot_summary(warehouse: str | Path, full_name: str, snapshot_id: int) -> dict:
    """The parsed entry for one snapshot id (``{}`` if absent)."""
    for snap in table_metadata_for(warehouse, full_name)["snapshots"]:
        if snap["snapshot_id"] == int(snapshot_id):
            return snap
    return {}


# --- Spark-side file stats ----------------------------------------------------


def files_stats(spark: SparkSession, table: str) -> dict:
    """``{file_count, total_bytes, avg_file_mb}`` from ``<table>.files``.

    Zeros (not an exception) when the table does not exist yet.
    """
    try:
        exists = spark.catalog.tableExists(table)
    except Exception:  # noqa: BLE001
        exists = False
    if not exists:
        return {"file_count": 0, "total_bytes": 0, "avg_file_mb": 0.0}

    row = spark.sql(
        "SELECT count(*) AS n, coalesce(sum(file_size_in_bytes), 0) AS b "
        f"FROM {table}.files"
    ).collect()[0]
    n, b = int(row["n"]), int(row["b"])
    return {
        "file_count": n,
        "total_bytes": b,
        "avg_file_mb": round(b / n / (1024 * 1024), 4) if n else 0.0,
    }


# --- disk -------------------------------------------------------------------


def parse_df_avail_gb(df_output: str) -> float | None:
    """Avail column (GB) from ``df -g`` output; ``None`` if unparseable.

    macOS ``df -g`` reports 1G blocks; the data line's 4th field is Avail.
    """
    lines = [ln for ln in df_output.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    fields = lines[-1].split()
    for idx in (3,):
        if len(fields) > idx:
            try:
                return float(fields[idx])
            except ValueError:
                continue
    return None


def disk_avail_gb(path: str | Path = ".") -> float:
    """Available GB at ``path`` via ``df -g``, falling back to ``shutil``.

    The fallback matters: GNU ``df`` has no ``-g``, so CI (Linux) always takes
    it. Both report the same quantity (bytes available to this user / 1024^3).
    """
    try:
        proc = subprocess.run(
            ["df", "-g", str(path)], capture_output=True, text=True, timeout=20
        )
        if proc.returncode == 0:
            parsed = parse_df_avail_gb(proc.stdout)
            if parsed is not None:
                return round(parsed, 2)
    except Exception:  # noqa: BLE001
        pass
    return round(shutil.disk_usage(str(path)).free / (1024**3), 2)
