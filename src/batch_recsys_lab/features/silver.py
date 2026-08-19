"""Silver builds (Phase 1, T3; UPGRADE_PLAN.md §8, D1/D2/D5/D6/D7/D8).

Silver is the *trusted* layer. Each build reads a bronze table, applies the
documented typed projection (§8 "Projections"), routes contract violators to
``quarantine.*`` via the T1 contract engine (``gate``), ledgers every count in
``dq.dq_results`` (``audit`` + gate/measure rows), and — for interactions —
performs the deterministic two-stage dedup (D2). Every build appends one
exactly-reconciling line to ``data/build_summary.jsonl`` (the T4 waterfall feed):
``input_rows == kept + Σquarantined + exact_duplicate + superseded`` is asserted
in code before the line is written.

Ordering matters: ``build_items`` MUST run before ``build_interactions`` — the
interactions contract's ``item_fk`` orphan-rate measure joins against
``local.silver.items``.

Performance (43.9M reviews on 16GB, §5): no ``collect()`` on full tables (only
one-row aggregates and the tiny per-primary_reason breakdown are collected);
``gate`` is a single pass; quarantine writes are violation-only; the post-gate
frame is ``localCheckpoint``-ed between the gate → dedup stages so the 44M-row
frame is materialized once and not recomputed per downstream action.
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
from pyspark.sql.window import Window

from batch_recsys_lab.contracts import audit, gate, load_contract, write_dq_results
from batch_recsys_lab.contracts.checks import CONTROL_CHAR_REGEX
from batch_recsys_lab.contracts.engine import DqResult, _resolve_run_id
from batch_recsys_lab.spark_session import get_spark

# --- Locations ---------------------------------------------------------------
CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts"
ITEMS_CONTRACT = CONTRACTS_DIR / "silver_items.yaml"
INTERACTIONS_CONTRACT = CONTRACTS_DIR / "silver_interactions.yaml"

BRONZE_ITEMS = "local.bronze.items"
BRONZE_REVIEWS = "local.bronze.reviews"
SILVER_ITEMS = "local.silver.items"
SILVER_INTERACTIONS = "local.silver.interactions"
QUARANTINE_ITEMS = "local.quarantine.items"
QUARANTINE_INTERACTIONS = "local.quarantine.interactions"

BUILD_SUMMARY_LOG = "data/build_summary.jsonl"
RUN_ID_COL = "run_id"

# Silver column projections (declared order == contract column order).
SILVER_ITEM_COLS = [
    "parent_asin",
    "title",
    "main_category",
    "categories",
    "store",
    "average_rating",
    "rating_number",
    "price_usd",
    "brand_norm",
]
SILVER_INTERACTION_COLS = [
    "user_id",
    "parent_asin",
    "asin",
    "rating",
    "ts",
    "helpful_vote",
    "verified_purchase",
]

# Parsed price must be a bare number, optionally $-prefixed (trimmed). Anything
# else (ranges like "12.99 - 19.99", prose like "see price in cart", empty) → NULL.
_PRICE_REGEX = r"^\$?\d+(\.\d+)?$"


# --- Text hygiene ------------------------------------------------------------


def _normalize_control_chars(col: F.Column) -> F.Column:
    """Replace each control char ([\\x00-\\x1F\\x7F]) with a single space, then trim.

    Uses the engine's exact control-char class so the transform inverts precisely
    what the ``no_control_chars`` audit check flags (else the audit fails the build).
    """
    return F.trim(F.regexp_replace(col, CONTROL_CHAR_REGEX, " "))


def _blank_to_null(col: F.Column) -> F.Column:
    """NULL if the trimmed value is empty, else the original value."""
    return F.when(F.trim(col) == F.lit(""), F.lit(None).cast("string")).otherwise(col)


# --- Items transform ---------------------------------------------------------


def transform_items(bronze: DataFrame) -> DataFrame:
    """Typed silver-items projection from ``bronze.items`` (D1/D7 + brand rules).

    Returns the nine silver columns plus two internal measure helpers
    (``_price_unparseable`` bool, ``_brand_source`` string) — callers drop the
    helpers before :func:`gate`.
    """
    price_raw = F.trim(F.col("price"))
    price_parseable = price_raw.rlike(_PRICE_REGEX)
    price_usd = F.when(
        price_parseable, F.regexp_replace(price_raw, r"^\$", "").cast("double")
    ).otherwise(F.lit(None).cast("double"))
    # Unparseable = a non-empty price string that did not match the numeric form.
    price_unparseable = (
        price_raw.isNotNull() & (price_raw != F.lit("")) & ~price_parseable
    )

    # Brand: details['Brand'] with details['Manufacturer'] fallback; blanks → NULL.
    brand_b = _blank_to_null(F.col("details")["Brand"])
    brand_m = _blank_to_null(F.col("details")["Manufacturer"])
    brand_raw = F.coalesce(brand_b, brand_m)
    brand_source = (
        F.when(brand_b.isNotNull(), F.lit("Brand"))
        .when(brand_m.isNotNull(), F.lit("Manufacturer"))
        .otherwise(F.lit("none"))
    )
    brand_clean = F.lower(_normalize_control_chars(brand_raw))
    brand_norm = F.when(
        brand_clean.isNull() | (brand_clean == F.lit("")), F.lit("unknown")
    ).otherwise(brand_clean)

    return bronze.select(
        F.col("parent_asin"),
        _normalize_control_chars(F.col("title")).alias("title"),
        F.col("main_category"),
        F.col("categories"),
        F.col("store"),
        F.col("average_rating"),
        F.col("rating_number"),
        price_usd.alias("price_usd"),
        brand_norm.alias("brand_norm"),
        price_unparseable.alias("_price_unparseable"),
        brand_source.alias("_brand_source"),
    )


# --- Interactions transform + dedup (D2) -------------------------------------


def transform_interactions(bronze: DataFrame) -> DataFrame:
    """Typed silver-interactions projection from ``bronze.reviews``.

    ``ts`` = ``timestamp_millis(timestamp)``; review ``title`` is dropped (§8
    projection: the lab never uses review text).
    """
    return bronze.select(
        F.col("user_id"),
        F.col("parent_asin"),
        F.col("asin"),
        F.col("rating"),
        F.timestamp_millis(F.col("timestamp")).alias("ts"),
        F.col("helpful_vote"),
        F.col("verified_purchase"),
    )


def drop_exact_duplicates(df: DataFrame, cols: list[str] | None = None) -> DataFrame:
    """Collapse fully-identical rows (groupBy all silver columns, survivors identical).

    ``cols`` defaults to the Amazon :data:`SILVER_INTERACTION_COLS`; a sibling
    dataset with a different column set (``features/silver_ml32m.py``) passes its
    own so the D2 dedup rule is shared code, not a copy.
    """
    cols = SILVER_INTERACTION_COLS if cols is None else cols
    return df.dropDuplicates(cols)


def keep_latest(df: DataFrame, cols: list[str] | None = None) -> DataFrame:
    """Keep one row per (user_id, parent_asin): latest ``ts``, ties broken by a
    total order on ``xxhash64`` of all columns → partition-order independent (D2).

    ``cols`` (the hashed column set) defaults to :data:`SILVER_INTERACTION_COLS`.
    """
    cols = SILVER_INTERACTION_COLS if cols is None else cols
    order = [
        F.col("ts").desc(),
        F.xxhash64(*[F.col(c) for c in cols]).asc(),
    ]
    w = Window.partitionBy("user_id", "parent_asin").orderBy(*order)
    return (
        df.withColumn("__rn", F.row_number().over(w))
        .where(F.col("__rn") == F.lit(1))
        .drop("__rn")
    )


# --- dq_results helpers ------------------------------------------------------


def _gate_count_results(gate_res, contract, table, total, rid, rts) -> list[DqResult]:
    """One measured DqResult per quarantine check: how many rows it routed away."""
    kind_by_id = {c.check_id: c for c in contract.checks}
    out: list[DqResult] = []
    for cid, cnt in gate_res.violation_counts.items():
        chk = kind_by_id[cid]
        column = chk.columns[0] if len(chk.columns) == 1 else None
        out.append(
            DqResult(
                run_id=rid,
                run_ts=rts,
                table_name=table,
                contract_name=contract.name,
                contract_version=contract.version,
                check_id=f"gate__{cid}",
                check_kind=chk.kind,
                column=column,
                status="measured",
                violation_count=int(cnt),
                total_rows=int(total),
                metric_value=(cnt / total if total else 0.0),
                details=json.dumps({"stage": "gate_quarantine"}),
            )
        )
    return out


def _fail_action_failures(results: list[DqResult], contract) -> list[str]:
    """check_ids of fail-action contract checks whose audit status is 'fail'.

    Excludes schema-conformance rows (nullability of a required column always
    'fails' because Spark/Iceberg mark DataFrame columns nullable — that is a
    known cosmetic finding surfaced by the T8 audit CLI, not a build stopper).
    """
    fail_action_ids = {c.check_id for c in contract.checks if c.action == "fail"}
    return [
        r.check_id
        for r in results
        if r.check_id in fail_action_ids and r.status == "fail"
    ]


# --- Summary / conservation --------------------------------------------------


def _primary_reason_breakdown(quarantine_df: DataFrame) -> dict[str, int]:
    """Rows per primary_reason (a partition of quarantined rows → sums exact)."""
    from batch_recsys_lab.contracts.engine import PRIMARY_COL

    rows = quarantine_df.groupBy(PRIMARY_COL).count().collect()
    return {r[PRIMARY_COL]: int(r["count"]) for r in rows}


def _write_summary(summary: dict, path: str = BUILD_SUMMARY_LOG) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(summary) + "\n")


def _assert_conservation(summary: dict) -> None:
    got = (
        summary["kept"]
        + sum(summary["quarantined"].values())
        + summary["exact_duplicate"]
        + summary["superseded_by_later_review"]
    )
    if got != summary["input_rows"]:
        raise RuntimeError(
            f"waterfall conservation failed for {summary['table']}: "
            f"input_rows={summary['input_rows']} != kept+quarantined+exact_duplicate"
            f"+superseded={got} ({summary!r})"
        )


# --- Build: items ------------------------------------------------------------


def build_items(
    spark: SparkSession,
    run_id: str | None = None,
    bronze_table: str = BRONZE_ITEMS,
    silver_table: str = SILVER_ITEMS,
    quarantine_table: str = QUARANTINE_ITEMS,
    summary_path: str = BUILD_SUMMARY_LOG,
    write_summary: bool = True,
) -> dict:
    """Build ``local.silver.items`` (+ quarantine + dq_results + summary line)."""
    start = time.perf_counter()
    rid, rts = _resolve_run_id(run_id)
    contract = load_contract(ITEMS_CONTRACT)

    projected = transform_items(spark.table(bronze_table))
    projected = projected.localCheckpoint(eager=True)  # materialize once

    measure = projected.agg(
        F.count(F.lit(1)).alias("total"),
        F.sum(F.col("_price_unparseable").cast("long")).alias("price_unparseable"),
        F.sum((F.col("_brand_source") == F.lit("Brand")).cast("long")).alias("from_brand"),
        F.sum((F.col("_brand_source") == F.lit("Manufacturer")).cast("long")).alias("from_manufacturer"),
        F.sum((F.col("_brand_source") == F.lit("none")).cast("long")).alias("from_none"),
    ).collect()[0]
    total = int(measure["total"] or 0)

    silver_in = projected.select(*SILVER_ITEM_COLS)
    gate_res = gate(silver_in, contract)

    gate_res.kept_df.writeTo(silver_table).createOrReplace()
    gate_res.quarantine_df.withColumn(RUN_ID_COL, F.lit(rid)).writeTo(
        quarantine_table
    ).createOrReplace()

    quarantined = _primary_reason_breakdown(gate_res.quarantine_df)
    kept = gate_res.total_rows - gate_res.quarantined_rows

    # dq_results: schema/table-level audit + gate counts + custom measures.
    results = audit(spark, contract, silver_table, run_id=rid)
    results += _gate_count_results(gate_res, contract, silver_table, total, rid, rts)
    results.append(
        DqResult(
            run_id=rid, run_ts=rts, table_name=silver_table,
            contract_name=contract.name, contract_version=contract.version,
            check_id="price_unparseable", check_kind="measure", column="price_usd",
            status="measured", violation_count=int(measure["price_unparseable"] or 0),
            total_rows=total,
            metric_value=((measure["price_unparseable"] or 0) / total if total else 0.0),
            details=json.dumps({"regex": _PRICE_REGEX}),
        )
    )
    results.append(
        DqResult(
            run_id=rid, run_ts=rts, table_name=silver_table,
            contract_name=contract.name, contract_version=contract.version,
            check_id="brand_source_share", check_kind="measure", column="brand_norm",
            status="measured", violation_count=int(measure["from_manufacturer"] or 0),
            total_rows=total,
            metric_value=((measure["from_manufacturer"] or 0) / total if total else 0.0),
            details=json.dumps(
                {
                    "from_brand": int(measure["from_brand"] or 0),
                    "from_manufacturer": int(measure["from_manufacturer"] or 0),
                    "from_none": int(measure["from_none"] or 0),
                }
            ),
        )
    )
    write_dq_results(spark, results)

    failures = _fail_action_failures(results, contract)
    if failures:
        raise RuntimeError(f"{silver_table}: fail-action audit checks failed: {failures}")

    summary = {
        "table": "items",
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
    bronze_table: str = BRONZE_REVIEWS,
    silver_table: str = SILVER_INTERACTIONS,
    quarantine_table: str = QUARANTINE_INTERACTIONS,
    summary_path: str = BUILD_SUMMARY_LOG,
    write_summary: bool = True,
) -> dict:
    """Build ``local.silver.interactions`` (gate → exact-dup → keep-latest, D2)."""
    start = time.perf_counter()
    rid, rts = _resolve_run_id(run_id)
    contract = load_contract(INTERACTIONS_CONTRACT)

    projected = transform_interactions(spark.table(bronze_table))
    projected = projected.localCheckpoint(eager=True)  # 44M-row frame: materialize once

    gate_res = gate(projected, contract)
    total = gate_res.total_rows
    quarantined_rows = gate_res.quarantined_rows
    kept_n = total - quarantined_rows

    gate_res.quarantine_df.withColumn(RUN_ID_COL, F.lit(rid)).writeTo(
        quarantine_table
    ).createOrReplace()
    quarantined = _primary_reason_breakdown(gate_res.quarantine_df)

    # Deterministic dedup (D2): exact-dup drop, then keep-latest.
    deduped = drop_exact_duplicates(gate_res.kept_df).localCheckpoint(eager=True)
    deduped_n = deduped.count()
    exact_duplicate = kept_n - deduped_n

    final = keep_latest(deduped).localCheckpoint(eager=True)
    final_n = final.count()
    superseded = deduped_n - final_n

    final.writeTo(silver_table).createOrReplace()

    results = audit(spark, contract, silver_table, run_id=rid)
    results += _gate_count_results(gate_res, contract, silver_table, total, rid, rts)
    write_dq_results(spark, results)

    failures = _fail_action_failures(results, contract)
    if failures:
        raise RuntimeError(f"{silver_table}: fail-action audit checks failed: {failures}")

    summary = {
        "table": "interactions",
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


# --- CLI ---------------------------------------------------------------------

_BUILDERS = {"items": build_items, "interactions": build_interactions}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.features.silver")
    parser.add_argument("--table", required=True, choices=sorted(_BUILDERS))
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--run-id", default=None)
    # Bronze source overrides (tests point these at fixture-loaded tables).
    parser.add_argument("--bronze-table", default=None)
    args = parser.parse_args(argv)

    spark = get_spark(
        app_name=f"silver-build-{args.table}",
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
