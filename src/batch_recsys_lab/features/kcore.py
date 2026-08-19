"""Iterative k-core pruning: silver.interactions -> gold.interactions_5core (T6).

The k-core of the bipartite user x item interaction graph is the maximal edge
set in which every user has >= k distinct interactions AND every item has >= k
distinct interactions. Because a single pass that drops under-degree users can
push a previously-safe item below k (and vice versa), pruning is *iterative*:
recompute both degrees, drop rows failing either bound, repeat until a pass
changes nothing.

Why this terminates, and why the fixed point is exact
-----------------------------------------------------
Each iteration is a pure filter: it only ever removes rows, never adds. Row
count is therefore monotonically non-increasing and bounded below by 0, so the
sequence must stabilize. The filter predicate (both degrees >= k) is evaluated
on the *current* graph each pass, so when a pass leaves the row count unchanged
every surviving edge already satisfies both bounds on the surviving graph — that
is precisely the k-core. Hence "row count unchanged" is a sound and complete
convergence test; no separate node-level check is needed. MAX_ITERS is a guard
against pathological non-termination (which cannot happen for a monotone filter)
and against silently shipping a non-converged table — hitting it is a hard error.

Lineage truncation
-------------------
Each iteration's output is ``localCheckpoint(eager=True)``-ed. Without this the
logical plan would grow by two joins per iteration and, over dozens of passes,
the Catalyst plan (and re-computation cost on every ``count``) explodes. Local
checkpoint materializes the frame to executor-local storage and cuts the RDD
lineage, so each pass starts from a flat, already-computed base. We use *local*
(not reliable ``checkpoint``) because we never need cross-app recovery — the job
either finishes in one app or fails — so paying for HDFS-style replication would
be wasted I/O.

Determinism (D9): the whole job is pure filtering — no seeds, no ordering-
dependent tie-breaks, no nondeterministic expressions — so it is content-
identical across runs by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from batch_recsys_lab.spark_session import get_spark

# Columns carried into the gold 5-core table. `user_id`/`parent_asin` are the
# graph edge; the rest are payload preserved verbatim from silver (D1: asin is
# provenance only, parent_asin is item identity).
PROJECTION = (
    "user_id",
    "parent_asin",
    "ts",
    "rating",
    "asin",
    "helpful_vote",
    "verified_purchase",
)

DEFAULT_SOURCE_TABLE = "local.silver.interactions"
DEFAULT_TARGET_TABLE = "local.gold.interactions_5core"
DEFAULT_FUNNEL_TABLE = "local.dq.kcore_funnel"
DEFAULT_K = 5
MAX_ITERS = 50

FUNNEL_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), False),
        StructField("iteration", IntegerType(), False),
        StructField("rows", LongType(), False),
        StructField("users", LongType(), False),
        StructField("items", LongType(), False),
        StructField("converged", BooleanType(), False),
        StructField("wall_clock_s", DoubleType(), False),
    ]
)


# --------------------------------------------------------------------------- #
# run_id resolution — same precedence the contracts engine uses (arg > env >
# generated UTC-ts + git-sha), reimplemented locally to avoid depending on a
# private helper across module boundaries.
# --------------------------------------------------------------------------- #
def _git_short_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover - git absent / not a repo
        pass
    return None


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    env = os.environ.get("RECSYS_RUN_ID")
    if env:
        return env
    now = datetime.now(timezone.utc)
    sha = _git_short_sha()
    return now.strftime("%Y%m%dT%H%M%SZ") + (f"-{sha}" if sha else "")


# --------------------------------------------------------------------------- #
# Core algorithm.
# --------------------------------------------------------------------------- #
def _stats(df: DataFrame) -> tuple[int, int, int]:
    """(rows, distinct users, distinct items) in a single aggregation pass."""
    r = df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("user_id").alias("users"),
        F.countDistinct("parent_asin").alias("items"),
    ).first()
    return int(r["rows"]), int(r["users"]), int(r["items"])


def _prune_once(df: DataFrame, k: int, projection: tuple[str, ...] = PROJECTION) -> DataFrame:
    """One k-core pass: keep edges whose user AND item degree (in the current
    graph) are both >= k. Degrees are counts of rows, i.e. distinct interactions
    after dedup — silver guarantees one row per (user_id, parent_asin) (D2)."""
    user_deg = df.groupBy("user_id").count().withColumnRenamed("count", "_user_n")
    item_deg = df.groupBy("parent_asin").count().withColumnRenamed("count", "_item_n")
    return (
        df.join(user_deg, "user_id")
        .join(item_deg, "parent_asin")
        .where((F.col("_user_n") >= k) & (F.col("_item_n") >= k))
        .select(*projection)
    )


def run_kcore(
    spark: SparkSession,
    source_table: str,
    target_table: str,
    funnel_table: str,
    k: int = DEFAULT_K,
    max_iters: int = MAX_ITERS,
    run_id: str | None = None,
    projection: tuple[str, ...] = PROJECTION,
) -> list[dict]:
    """Compute the k-core of ``source_table`` and write it to ``target_table``.

    Appends the per-iteration funnel (iteration 0 = input snapshot) to
    ``funnel_table`` and returns it as a list of dicts. On convergence the final
    frame is written ``createOrReplace`` to ``target_table``. If ``max_iters`` is
    exhausted without convergence, the funnel gathered so far is still persisted
    (so the failure is inspectable) and a ``RuntimeError`` is raised — a non-
    converged table is never shipped.

    ``projection`` is the carried column set; it defaults to the Amazon
    :data:`PROJECTION`. The algorithm itself only needs ``user_id``/``parent_asin``
    (the graph edge), so a dataset with a different payload (ML-32M carries no
    ``asin``/``helpful_vote``/``verified_purchase``) passes its own tuple rather
    than forking this module. Amazon callers are byte-for-byte unaffected.
    """
    rid = _resolve_run_id(run_id)

    df = spark.table(source_table).select(*projection)

    funnel: list[dict] = []

    # Iteration 0: input snapshot. Checkpoint to flatten the Iceberg-scan lineage
    # into a materialized base the loop can build on.
    t0 = time.perf_counter()
    df = df.localCheckpoint(eager=True)
    rows, users, items = _stats(df)
    funnel.append(
        {
            "run_id": rid,
            "iteration": 0,
            "rows": rows,
            "users": users,
            "items": items,
            "converged": False,
            "wall_clock_s": round(time.perf_counter() - t0, 3),
        }
    )
    prev_rows = rows

    converged = False
    for i in range(1, max_iters + 1):
        t = time.perf_counter()
        df = _prune_once(df, k, projection).localCheckpoint(eager=True)
        rows, users, items = _stats(df)
        converged = rows == prev_rows  # monotone filter => unchanged means fixed point
        funnel.append(
            {
                "run_id": rid,
                "iteration": i,
                "rows": rows,
                "users": users,
                "items": items,
                "converged": converged,
                "wall_clock_s": round(time.perf_counter() - t, 3),
            }
        )
        if converged:
            break
        prev_rows = rows

    if not converged:
        # Persist what we have so the stall is diagnosable, then fail hard.
        _write_funnel(spark, funnel_table, funnel)
        raise RuntimeError(
            f"k-core did not converge within max_iters={max_iters} "
            f"(k={k}, source={source_table}); last rows={prev_rows}. "
            "Funnel written for inspection."
        )

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {target_table.rsplit('.', 1)[0]}")
    df.writeTo(target_table).createOrReplace()

    _write_funnel(spark, funnel_table, funnel)
    return funnel


def _write_funnel(spark: SparkSession, funnel_table: str, funnel: list[dict]) -> None:
    """Append funnel rows to the (append-only) Iceberg funnel table, creating
    the namespace/table on first write."""
    if not funnel:
        return
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {funnel_table.rsplit('.', 1)[0]}")
    rows = [
        (
            r["run_id"],
            int(r["iteration"]),
            int(r["rows"]),
            int(r["users"]),
            int(r["items"]),
            bool(r["converged"]),
            float(r["wall_clock_s"]),
        )
        for r in funnel
    ]
    sdf = spark.createDataFrame(rows, FUNNEL_SCHEMA)
    if spark.catalog.tableExists(funnel_table):
        sdf.writeTo(funnel_table).append()
    else:
        sdf.writeTo(funnel_table).create()


def build_summary(funnel: list[dict], source_table: str, target_table: str, k: int) -> dict:
    """Waterfall-facing summary derived from the funnel head (input snapshot) and
    tail (converged state). ``kcore_pruned`` is the k-core waterfall edge (D3)."""
    first, last = funnel[0], funnel[-1]
    return {
        "source_table": source_table,
        "target_table": target_table,
        "run_id": first["run_id"],
        "k": k,
        "input_rows": first["rows"],
        "output_rows": last["rows"],
        "kcore_pruned": first["rows"] - last["rows"],
        "input_users": first["users"],
        "input_items": first["items"],
        "output_users": last["users"],
        "output_items": last["items"],
        "iterations": last["iteration"],
        "converged": last["converged"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.features.kcore")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--target-table", default=DEFAULT_TARGET_TABLE)
    parser.add_argument("--funnel-table", default=DEFAULT_FUNNEL_TABLE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    spark = get_spark(
        app_name="gold-kcore",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        funnel = run_kcore(
            spark,
            source_table=args.source_table,
            target_table=args.target_table,
            funnel_table=args.funnel_table,
            k=args.k,
            run_id=args.run_id,
        )
    finally:
        spark.stop()

    summary = build_summary(funnel, args.source_table, args.target_table, args.k)
    # Summary JSON MUST be the last stdout line so the waterfall builder can parse it.
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
