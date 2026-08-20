"""ML-32M silver builds (Phase 9, T9-3a; UPGRADE_PLAN.md §8c).

The regime-contrast dataset goes through the SAME trusted-layer discipline as the
Amazon lane — typed projection, contract gate → ``quarantine_ml32m.*``, every count
ledgered in ``dq_ml32m.dq_results``, deterministic two-stage dedup (D2), and one
exactly-reconciling waterfall line per build
(``input_rows == kept + Σquarantined + exact_duplicate + superseded``, asserted in
code). The gate/audit/dedup/accounting helpers are imported from
``features/silver.py`` rather than re-implemented; only the transforms and the
column set are ML-32M's own.

Ordering matters, exactly as in the Amazon lane: ``build_items`` MUST run before
``build_interactions`` and ``build_tags`` — both of those contracts' ``item_fk``
orphan-rate measures join against ``local.silver_ml32m.items``.

Two dataset-specific notes:

* **``parent_asin`` holds ``str(movieId)``.** That column name is the lab-wide item
  identity, keyed on by k-core, user_stats, popularity, item_train_stats and the
  eval extract. Normalizing ML-32M onto it costs one line here; renaming it would
  mean threading an item-column parameter through eight shared modules for
  cosmetics. The contract YAMLs carry the same note.
* **Dedup is expected to be a no-op.** MovieLens stores at most one rating per
  (user, movie), so ``exact_duplicate`` and ``superseded_by_later_review`` should
  both be 0. That expectation is *verified, not assumed*: both counts are written
  to the build summary AND published to ``dq_ml32m.dq_results`` as the
  ``interaction_pair_uniqueness`` measure, so a violation shows up in the ledger
  instead of being silently collapsed.

Ratings are kept as a provenance column only — this lab treats every interaction as
an implicit-feedback positive (CLAUDE.md invariant #6), on ML-32M as on Amazon.

``tags`` is landed here too (``build_tags``) because §8c T9-3b's content arm is
"title+genres+tags", so the DATA stage must produce it — the model stage may not
reach back into bronze for un-gated text. It is a silver table only: no gold
projection, because T9-3b builds its own ``item_text`` from it. Its natural key is
(user, item, TAG), not (user, item), so ``keep_latest`` has no meaning for it and
is not applied; exact 4-tuple duplicates are still removed and counted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from batch_recsys_lab.contracts import audit, gate, load_contract, write_dq_results
from batch_recsys_lab.contracts.engine import DqResult, _resolve_run_id
from batch_recsys_lab.features.silver import (
    _assert_conservation,
    _fail_action_failures,
    _gate_count_results,
    _normalize_control_chars,
    _primary_reason_breakdown,
    _write_summary,
    drop_exact_duplicates,
    keep_latest,
)
from batch_recsys_lab.spark_session import get_spark

# --- Locations ---------------------------------------------------------------
CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts" / "ml32m"
ITEMS_CONTRACT = CONTRACTS_DIR / "silver_ml32m_items.yaml"
INTERACTIONS_CONTRACT = CONTRACTS_DIR / "silver_ml32m_interactions.yaml"
TAGS_CONTRACT = CONTRACTS_DIR / "silver_ml32m_tags.yaml"

BRONZE_MOVIES = "local.bronze_ml32m.movies"
BRONZE_RATINGS = "local.bronze_ml32m.ratings"
BRONZE_TAGS = "local.bronze_ml32m.tags"
SILVER_ITEMS = "local.silver_ml32m.items"
SILVER_INTERACTIONS = "local.silver_ml32m.interactions"
SILVER_TAGS = "local.silver_ml32m.tags"
QUARANTINE_ITEMS = "local.quarantine_ml32m.items"
QUARANTINE_INTERACTIONS = "local.quarantine_ml32m.interactions"
QUARANTINE_TAGS = "local.quarantine_ml32m.tags"

# Separate DQ ledger: the Amazon ledger backs the published DQ dashboard, and a
# second dataset's rows in it would silently change that exhibit's totals.
DQ_TABLE = "local.dq_ml32m.dq_results"

BUILD_SUMMARY_LOG = "data/build_summary_ml32m.jsonl"
RUN_ID_COL = "run_id"

# Silver column projections (declared order == contract column order).
SILVER_ITEM_COLS = ["parent_asin", "title", "genres"]
SILVER_INTERACTION_COLS = ["user_id", "parent_asin", "rating", "ts"]
SILVER_TAG_COLS = ["user_id", "parent_asin", "tag", "ts"]

# movies.csv marks a genre-less film with this literal token rather than an empty
# field; silver turns it into an empty array (normalization is silver's job).
NO_GENRES_TOKEN = "(no genres listed)"
GENRE_SEPARATOR = r"\|"


# --- Items transform ---------------------------------------------------------


def transform_items(bronze: DataFrame) -> DataFrame:
    """Typed silver-items projection from ``bronze_ml32m.movies``.

    Returns the three silver columns plus one internal measure helper
    (``_genres_missing`` bool) — callers drop the helper before :func:`gate`.
    """
    title = _normalize_control_chars(F.col("title"))
    genres_raw = _normalize_control_chars(F.col("genres"))
    genres_missing = genres_raw.isNull() | (genres_raw == F.lit(NO_GENRES_TOKEN)) | (
        genres_raw == F.lit("")
    )
    genres = F.when(
        genres_missing, F.array().cast("array<string>")
    ).otherwise(F.split(genres_raw, GENRE_SEPARATOR))

    return bronze.select(
        F.col("movieId").cast("string").alias("parent_asin"),
        title.alias("title"),
        genres.alias("genres"),
        genres_missing.alias("_genres_missing"),
    )


# --- Interactions transform --------------------------------------------------


def transform_interactions(bronze: DataFrame) -> DataFrame:
    """Typed silver-interactions projection from ``bronze_ml32m.ratings``.

    ``ts = timestamp_seconds(timestamp)`` — ML-32M publishes epoch SECONDS (the
    Amazon lane publishes millis and uses ``timestamp_millis``); getting this wrong
    would put every rating in 1970, which the contract's ``ts_range`` would catch.
    Keys are cast to string: ``userId``/``movieId`` are integral in the source, and
    the lab's user/item identities are strings everywhere.
    """
    return bronze.select(
        F.col("userId").cast("string").alias("user_id"),
        F.col("movieId").cast("string").alias("parent_asin"),
        F.col("rating"),
        F.timestamp_seconds(F.col("timestamp")).alias("ts"),
    )


# --- Tags transform ------------------------------------------------------------


def transform_tags(bronze: DataFrame) -> DataFrame:
    """Typed silver-tags projection from ``bronze_ml32m.tags``.

    Same key/timestamp treatment as :func:`transform_interactions` (string keys,
    epoch-SECONDS → timestamp). The tag itself is control-char normalized and
    trimmed, and a tag that is empty after trimming becomes NULL so the contract's
    ``keys_non_null`` quarantines it with a reason — an empty string would
    otherwise survive as a real-looking token in the T9-3b text corpus.

    Returns the four silver columns plus one internal measure helper
    (``_tag_missing`` bool) — callers drop it before :func:`gate`.
    """
    normalized = F.trim(_normalize_control_chars(F.col("tag")))
    tag = F.when(normalized == F.lit(""), F.lit(None).cast("string")).otherwise(normalized)
    return bronze.select(
        F.col("userId").cast("string").alias("user_id"),
        F.col("movieId").cast("string").alias("parent_asin"),
        tag.alias("tag"),
        F.timestamp_seconds(F.col("timestamp")).alias("ts"),
        tag.isNull().alias("_tag_missing"),
    )


# --- Build: items ------------------------------------------------------------


def build_items(
    spark: SparkSession,
    run_id: str | None = None,
    bronze_table: str = BRONZE_MOVIES,
    silver_table: str = SILVER_ITEMS,
    quarantine_table: str = QUARANTINE_ITEMS,
    summary_path: str = BUILD_SUMMARY_LOG,
    write_summary: bool = True,
    dq_table: str = DQ_TABLE,
) -> dict:
    """Build ``local.silver_ml32m.items`` (+ quarantine + dq_results + summary line)."""
    start = time.perf_counter()
    rid, rts = _resolve_run_id(run_id)
    contract = load_contract(ITEMS_CONTRACT)

    projected = transform_items(spark.table(bronze_table))
    projected = projected.localCheckpoint(eager=True)  # materialize once

    measure = projected.agg(
        F.count(F.lit(1)).alias("total"),
        F.sum(F.col("_genres_missing").cast("long")).alias("genres_missing"),
    ).collect()[0]
    total = int(measure["total"] or 0)

    silver_in = projected.select(*SILVER_ITEM_COLS)
    gate_res = gate(silver_in, contract)

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {silver_table.rsplit('.', 1)[0]}")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {quarantine_table.rsplit('.', 1)[0]}")
    gate_res.kept_df.writeTo(silver_table).createOrReplace()
    gate_res.quarantine_df.withColumn(RUN_ID_COL, F.lit(rid)).writeTo(
        quarantine_table
    ).createOrReplace()

    quarantined = _primary_reason_breakdown(gate_res.quarantine_df)
    kept = gate_res.total_rows - gate_res.quarantined_rows

    results = audit(spark, contract, silver_table, run_id=rid)
    results += _gate_count_results(gate_res, contract, silver_table, total, rid, rts)
    results.append(
        DqResult(
            run_id=rid, run_ts=rts, table_name=silver_table,
            contract_name=contract.name, contract_version=contract.version,
            check_id="genres_missing_share", check_kind="measure", column="genres",
            status="measured", violation_count=int(measure["genres_missing"] or 0),
            total_rows=total,
            metric_value=((measure["genres_missing"] or 0) / total if total else 0.0),
            details=json.dumps({"token": NO_GENRES_TOKEN, "normalized_to": "empty array"}),
        )
    )
    write_dq_results(spark, results, dq_table)

    failures = _fail_action_failures(results, contract)
    if failures:
        raise RuntimeError(f"{silver_table}: fail-action audit checks failed: {failures}")

    summary = {
        "table": "ml32m_items",
        "run_id": rid,
        "input_rows": total,
        "kept": kept,
        "quarantined": quarantined,
        "exact_duplicate": 0,
        "superseded_by_later_review": 0,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }
    _assert_conservation(summary)
    if write_summary:
        _write_summary(summary, summary_path)
    return summary


# --- Build: interactions -----------------------------------------------------


def build_interactions(
    spark: SparkSession,
    run_id: str | None = None,
    bronze_table: str = BRONZE_RATINGS,
    silver_table: str = SILVER_INTERACTIONS,
    quarantine_table: str = QUARANTINE_INTERACTIONS,
    summary_path: str = BUILD_SUMMARY_LOG,
    write_summary: bool = True,
    dq_table: str = DQ_TABLE,
) -> dict:
    """Build ``local.silver_ml32m.interactions`` (gate → exact-dup → keep-latest, D2)."""
    start = time.perf_counter()
    rid, rts = _resolve_run_id(run_id)
    contract = load_contract(INTERACTIONS_CONTRACT)

    projected = transform_interactions(spark.table(bronze_table))
    projected = projected.localCheckpoint(eager=True)  # 32M-row frame: materialize once

    gate_res = gate(projected, contract)
    total = gate_res.total_rows
    quarantined_rows = gate_res.quarantined_rows
    kept_n = total - quarantined_rows

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {silver_table.rsplit('.', 1)[0]}")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {quarantine_table.rsplit('.', 1)[0]}")
    gate_res.quarantine_df.withColumn(RUN_ID_COL, F.lit(rid)).writeTo(
        quarantine_table
    ).createOrReplace()
    quarantined = _primary_reason_breakdown(gate_res.quarantine_df)

    # Deterministic dedup (D2), expected to be a no-op on ML-32M — see module docstring.
    deduped = drop_exact_duplicates(
        gate_res.kept_df, SILVER_INTERACTION_COLS
    ).localCheckpoint(eager=True)
    deduped_n = deduped.count()
    exact_duplicate = kept_n - deduped_n

    final = keep_latest(deduped, SILVER_INTERACTION_COLS).localCheckpoint(eager=True)
    final_n = final.count()
    superseded = deduped_n - final_n

    final.writeTo(silver_table).createOrReplace()

    results = audit(spark, contract, silver_table, run_id=rid)
    results += _gate_count_results(gate_res, contract, silver_table, total, rid, rts)
    results.append(
        DqResult(
            run_id=rid, run_ts=rts, table_name=silver_table,
            contract_name=contract.name, contract_version=contract.version,
            check_id="interaction_pair_uniqueness", check_kind="measure",
            column="parent_asin", status="measured",
            violation_count=int(exact_duplicate + superseded),
            total_rows=int(kept_n),
            metric_value=((exact_duplicate + superseded) / kept_n if kept_n else 0.0),
            details=json.dumps(
                {
                    "expectation": "MovieLens stores at most one rating per (user, movie): 0",
                    "exact_duplicate": int(exact_duplicate),
                    "superseded_by_later_review": int(superseded),
                }
            ),
        )
    )
    write_dq_results(spark, results, dq_table)

    failures = _fail_action_failures(results, contract)
    if failures:
        raise RuntimeError(f"{silver_table}: fail-action audit checks failed: {failures}")

    summary = {
        "table": "ml32m_interactions",
        "run_id": rid,
        "input_rows": total,
        "kept": final_n,
        "quarantined": quarantined,
        "exact_duplicate": exact_duplicate,
        "superseded_by_later_review": superseded,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }
    _assert_conservation(summary)
    if write_summary:
        _write_summary(summary, summary_path)
    return summary


# --- Build: tags ---------------------------------------------------------------


def build_tags(
    spark: SparkSession,
    run_id: str | None = None,
    bronze_table: str = BRONZE_TAGS,
    silver_table: str = SILVER_TAGS,
    quarantine_table: str = QUARANTINE_TAGS,
    summary_path: str = BUILD_SUMMARY_LOG,
    write_summary: bool = True,
    dq_table: str = DQ_TABLE,
) -> dict:
    """Build ``local.silver_ml32m.tags`` (+ quarantine + dq_results + summary line).

    Gate → exact-dup, with no ``keep_latest`` stage: a (user, item) pair may carry
    many distinct tags, so "latest wins" would silently destroy data. Only exact
    duplicates of the full (user, item, tag, ts) tuple are collapsed, and the count
    is reported — expected 0, verified rather than assumed.
    """
    start = time.perf_counter()
    rid, rts = _resolve_run_id(run_id)
    contract = load_contract(TAGS_CONTRACT)

    projected = transform_tags(spark.table(bronze_table)).localCheckpoint(eager=True)

    measure = projected.agg(
        F.count(F.lit(1)).alias("total"),
        F.sum(F.col("_tag_missing").cast("long")).alias("tag_missing"),
    ).collect()[0]
    total = int(measure["total"] or 0)

    gate_res = gate(projected.select(*SILVER_TAG_COLS), contract)
    quarantined_rows = gate_res.quarantined_rows
    kept_n = gate_res.total_rows - quarantined_rows

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {silver_table.rsplit('.', 1)[0]}")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {quarantine_table.rsplit('.', 1)[0]}")
    gate_res.quarantine_df.withColumn(RUN_ID_COL, F.lit(rid)).writeTo(
        quarantine_table
    ).createOrReplace()
    quarantined = _primary_reason_breakdown(gate_res.quarantine_df)

    deduped = drop_exact_duplicates(gate_res.kept_df, SILVER_TAG_COLS).localCheckpoint(
        eager=True
    )
    final_n = deduped.count()
    exact_duplicate = kept_n - final_n
    deduped.writeTo(silver_table).createOrReplace()

    results = audit(spark, contract, silver_table, run_id=rid)
    results += _gate_count_results(gate_res, contract, silver_table, total, rid, rts)
    results.append(
        DqResult(
            run_id=rid, run_ts=rts, table_name=silver_table,
            contract_name=contract.name, contract_version=contract.version,
            check_id="tag_missing_share", check_kind="measure", column="tag",
            status="measured", violation_count=int(measure["tag_missing"] or 0),
            total_rows=total,
            metric_value=((measure["tag_missing"] or 0) / total if total else 0.0),
            details=json.dumps(
                {
                    "definition": "bronze tag NULL, or empty after control-char "
                    "normalization + trim",
                    "disposition": "quarantined by keys_non_null (never written as '')",
                }
            ),
        )
    )
    write_dq_results(spark, results, dq_table)

    failures = _fail_action_failures(results, contract)
    if failures:
        raise RuntimeError(f"{silver_table}: fail-action audit checks failed: {failures}")

    summary = {
        "table": "ml32m_tags",
        "run_id": rid,
        "input_rows": total,
        "kept": final_n,
        "quarantined": quarantined,
        "exact_duplicate": exact_duplicate,
        # No supersede semantics for tags: the natural key includes the tag text.
        "superseded_by_later_review": 0,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }
    _assert_conservation(summary)
    if write_summary:
        _write_summary(summary, summary_path)
    return summary


# --- CLI ---------------------------------------------------------------------

_BUILDERS = {"items": build_items, "interactions": build_interactions, "tags": build_tags}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.features.silver_ml32m")
    parser.add_argument("--table", required=True, choices=sorted(_BUILDERS))
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--run-id", default=None)
    # Bronze source override (tests point this at a fixture-loaded table).
    parser.add_argument("--bronze-table", default=None)
    args = parser.parse_args(argv)

    spark = get_spark(
        app_name=f"silver-build-ml32m-{args.table}",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    build = _BUILDERS[args.table]
    kwargs: dict = {"run_id": args.run_id}
    if args.bronze_table:
        kwargs["bronze_table"] = args.bronze_table
    try:
        summary = build(spark, **kwargs)
    finally:
        spark.stop()

    # Summary JSON MUST be the last stdout line (bronze.py convention).
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
