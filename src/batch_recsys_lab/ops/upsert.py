"""Late-data MERGE upsert on the ops table (Phase 5, T20).

The batch that arrives "late" has two deterministically-selected halves:

* **inserts** — exactly the rows the backfill withheld
  (:func:`~batch_recsys_lab.ops.monthly.holdout_predicate`). They are absent
  from the table, so they land via ``WHEN NOT MATCHED THEN INSERT *``.
* **updates** — a sample of rows that DID land in the same two months, replayed
  with ``rating`` set to 5.0. They match on the identity key, so they land via
  ``WHEN MATCHED THEN UPDATE SET *``.

The two halves cannot collide, for two independent reasons: the update sample is
drawn *only* from rows the holdout predicate rejected, and it uses a different
hash — ``xxhash64(parent_asin, user_id, ts)`` (argument order swapped relative to
the holdout's ``xxhash64(user_id, parent_asin, ts)``), which is a different hash
value, not a shifted one.

Idempotence: both halves are pure functions of row identity and the update sets a
constant, so re-running the identical MERGE changes neither the row count nor any
value. That is asserted by ``tests/test_ops_upsert.py``.

Reconciliation caveat (deliberate, recorded): after the merge the ops table's
backfill window has exactly the same ROW COUNT as ``silver.interactions``, but
the updated sample's ``rating`` values intentionally diverge — that divergence is
the visible evidence that ``UPDATE SET *`` actually fired. Content equality is
therefore not claimed; row-count reconciliation is.
"""

from __future__ import annotations

import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from batch_recsys_lab.ops.maintenance import require_ops_table, table_exists
from batch_recsys_lab.ops.monthly import (
    BACKFILL_END,
    HASH_MODULUS,
    HOLDOUT_PERMILLE,
    IDENTITY_COLS,
    LATE_WINDOW_START,
    OPS_MONTHLY,
    SILVER_INTERACTIONS,
    backfill_predicate,
    holdout_predicate,
    late_window_predicate,
)

# Share of already-landed late-window rows replayed as updates, per thousand.
UPDATE_PERMILLE = 20

RATING_COL = "rating"
UPDATED_RATING = 5.0
UPDATE_MUTATION = f"{RATING_COL} := {UPDATED_RATING}"


def update_predicate(
    holdout_permille: int = HOLDOUT_PERMILLE,
    update_permille: int = UPDATE_PERMILLE,
    late_window_start: str = LATE_WINDOW_START,
    backfill_end: str = BACKFILL_END,
) -> str:
    """Already-landed late-window rows selected for replay as updates."""
    hold = holdout_predicate(holdout_permille, late_window_start, backfill_end)
    swapped = ", ".join([IDENTITY_COLS[1], IDENTITY_COLS[0], IDENTITY_COLS[2]])
    return (
        f"NOT ({hold}) AND ({late_window_predicate(late_window_start, backfill_end)}) "
        f"AND pmod(xxhash64({swapped}), {HASH_MODULUS}) < {int(update_permille)}"
    )


def merge_condition(target: str = "t", source: str = "s") -> str:
    return " AND ".join(f"{target}.{c} = {source}.{c}" for c in IDENTITY_COLS)


def late_data_merge(
    spark: SparkSession,
    table: str = OPS_MONTHLY,
    source: str = SILVER_INTERACTIONS,
    holdout_permille: int = HOLDOUT_PERMILLE,
    update_permille: int = UPDATE_PERMILLE,
    late_window_start: str = LATE_WINDOW_START,
    backfill_end: str = BACKFILL_END,
) -> dict:
    """Build the late batch and MERGE it into the ops table.

    Returns matched/inserted counts taken from the pre-computed batch halves,
    cross-checked against the post-merge row counts.
    """
    require_ops_table(table)
    start = time.perf_counter()
    if not table_exists(spark, table):
        raise RuntimeError(
            f"{table} does not exist — run the backfill step before the upsert."
        )

    cols = spark.table(table).columns
    src = spark.table(source)

    hold_pred = holdout_predicate(holdout_permille, late_window_start, backfill_end)
    upd_pred = update_predicate(
        holdout_permille, update_permille, late_window_start, backfill_end
    )
    window_pred = backfill_predicate(backfill_end)

    inserts = src.where(hold_pred).select(*cols)
    updates = (
        src.where(upd_pred)
        .withColumn(RATING_COL, F.lit(UPDATED_RATING).cast(dict(src.dtypes)[RATING_COL]))
        .select(*cols)
    )
    expected_inserted = inserts.count()
    expected_matched = updates.count()

    batch = inserts.unionByName(updates)
    view = "late_batch_ops_t20"
    batch.createOrReplaceTempView(view)

    rows_before = spark.table(table).count()
    merge_sql = (
        f"MERGE INTO {table} t USING {view} s ON {merge_condition()} "
        f"WHEN MATCHED THEN UPDATE SET * "
        f"WHEN NOT MATCHED THEN INSERT *"
    )
    spark.sql(merge_sql)
    spark.catalog.dropTempView(view)

    rows_after = spark.table(table).count()
    table_window_rows = spark.table(table).where(window_pred).count()
    source_window_rows = src.where(window_pred).count()

    return {
        "source": source,
        "table": table,
        "holdout_permille": int(holdout_permille),
        "update_permille": int(update_permille),
        "hash_modulus": HASH_MODULUS,
        "identity_cols": list(IDENTITY_COLS),
        "insert_predicate": hold_pred,
        "update_predicate": upd_pred,
        "update_mutation": UPDATE_MUTATION,
        "merge_condition": merge_condition(),
        "merge_sql": merge_sql,
        "batch_rows": int(expected_inserted + expected_matched),
        "matched_updated": int(expected_matched),
        "inserted": int(expected_inserted),
        "rows_before": int(rows_before),
        "post_merge_total": int(rows_after),
        "insert_count_reconciles": bool(rows_after - rows_before == expected_inserted),
        "backfill_window_rows": int(table_window_rows),
        "source_backfill_window_rows": int(source_window_rows),
        "reconciles_with_source": bool(table_window_rows == source_window_rows),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }
