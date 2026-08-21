"""``local.gold.item_text`` build + JVM-free export (Phase 4, T9; docs/engineering-log/UPGRADE_PLAN.md §8).

Two steps, mirroring the two-process design of ``eval/extract.py``:

* :func:`build_item_text` (Spark) — restricts to the distinct 5-core catalog
  (``local.gold.interactions_5core``), left-joins ``local.gold.item_features``
  (already-cleaned title/brand_norm/main_category/categories) with
  ``local.bronze.items`` (raw ``features``/``description`` arrays), sanitizes all
  text, and writes ``local.gold.item_text``. Row count is asserted to equal the
  distinct 5-core catalog count (also gives uniqueness on ``parent_asin``, since
  the join is 1 catalog row -> at most 1 output row).
* :func:`export_item_text` (Spark reads item_text + writes parquet; no JVM needed
  by anything downstream) — reorders the table's rows to exactly the eval cache's
  ``item_ids`` order for the live 5-core snapshot, asserting set equality first,
  and writes ``data/eval/text/<five_core_snapshot_id>/item_text.parquet`` plus an
  ``export_manifest.json`` recording provenance (snapshot ids, row count, hashes).

Text sanitization (title + each array element of ``features``/``description``):
strip C0/DEL control characters (``\\n`` -> space first, so a real line break
degrades to a space rather than vanishing), collapse runs of whitespace to a
single space, trim. Empty strings that result from an all-whitespace source are
left as empty strings (not renulled) — "measured", not silently hidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from batch_recsys_lab.contracts import load_contract, write_dq_results
from batch_recsys_lab.contracts.engine import DqResult, _resolve_run_id
from batch_recsys_lab.spark_session import get_spark

# --- Locations -----------------------------------------------------------------
CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts"
ITEM_TEXT_CONTRACT = CONTRACTS_DIR / "gold_item_text.yaml"

FIVE_CORE = "local.gold.interactions_5core"
ITEM_FEATURES = "local.gold.item_features"
BRONZE_ITEMS = "local.bronze.items"
ITEM_TEXT = "local.gold.item_text"

DEFAULT_EXPORT_ROOT = Path("data/eval/text")
DEFAULT_CACHE_ROOT = Path("data/eval/cache")

ITEM_TEXT_COLS = [
    "parent_asin",
    "title",
    "brand_norm",
    "main_category",
    "categories",
    "features",
    "description",
]


def _ensure_namespace(spark: SparkSession, table: str) -> None:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {table.rsplit('.', 1)[0]}")


def _snapshot_id(spark: SparkSession, table: str) -> int:
    row = spark.sql(f"SELECT snapshot_id FROM {table}.refs WHERE name = 'main'").first()
    if row is None:
        raise RuntimeError(f"table {table} has no 'main' ref snapshot")
    return int(row["snapshot_id"])


# --- text sanitization ---------------------------------------------------------


def _sanitize_scalar(col: F.Column) -> F.Column:
    """``\\n`` -> space, strip remaining C0/DEL control chars, collapse whitespace,
    trim. NULL passes through as NULL."""
    no_newlines = F.regexp_replace(col, "\n", " ")
    no_control = F.regexp_replace(no_newlines, "[\\x00-\\x09\\x0B-\\x1F\\x7F]", "")
    collapsed = F.regexp_replace(no_control, "\\s+", " ")
    return F.trim(collapsed)


def _sanitize_array(col: F.Column) -> F.Column:
    return F.transform(col, lambda x: _sanitize_scalar(x))


# --- build -----------------------------------------------------------------


def build_item_text(
    spark: SparkSession,
    five_core_table: str = FIVE_CORE,
    item_features_table: str = ITEM_FEATURES,
    bronze_items_table: str = BRONZE_ITEMS,
    out_table: str = ITEM_TEXT,
    run_id: str | None = None,
    contract_path: str | Path = ITEM_TEXT_CONTRACT,
) -> dict:
    """Build ``local.gold.item_text``. Raises ``AssertionError`` if the written
    row count does not equal the distinct 5-core catalog count."""
    start = time.perf_counter()
    rid, rts = _resolve_run_id(run_id)
    contract = load_contract(contract_path)

    catalog = spark.table(five_core_table).select("parent_asin").distinct()
    catalog_count = catalog.count()

    features = spark.table(item_features_table).select(
        "parent_asin", "title", "brand_norm", "main_category", "categories"
    )

    # bronze.items is raw metadata, keyed loosely by parent_asin (not guaranteed
    # unique). Dedup deterministically (first row per parent_asin, in a stable
    # parent_asin-sorted order) so the catalog join never fans out.
    bronze_raw = spark.table(bronze_items_table).select("parent_asin", "features", "description")
    bronze = (
        bronze_raw.where(F.col("parent_asin").isNotNull())
        .dropDuplicates(["parent_asin"])
    )

    joined = (
        catalog.join(features, "parent_asin", "left")
        .join(bronze, "parent_asin", "left")
        .select(
            "parent_asin",
            _sanitize_scalar(F.col("title")).alias("title"),
            "brand_norm",
            "main_category",
            "categories",
            _sanitize_array(F.col("features")).alias("features"),
            _sanitize_array(F.col("description")).alias("description"),
        )
        .select(*ITEM_TEXT_COLS)
    )

    _ensure_namespace(spark, out_table)
    joined.writeTo(out_table).createOrReplace()

    written = spark.table(out_table)
    row_count = written.count()
    if row_count != catalog_count:
        raise AssertionError(
            f"gold.item_text row count {row_count} != distinct 5-core catalog count "
            f"{catalog_count}"
        )
    distinct_asins = written.select("parent_asin").distinct().count()
    if distinct_asins != row_count:
        raise AssertionError(
            f"gold.item_text parent_asin not unique: {distinct_asins} distinct of "
            f"{row_count} rows"
        )

    # Measured (non-failing) metrics: empty-title / empty-features share.
    agg = written.agg(
        F.sum(F.col("title").isNull().cast("long")).alias("null_title"),
        F.sum(
            (F.col("features").isNull() | (F.size(F.col("features")) == 0)).cast("long")
        ).alias("empty_features"),
    ).collect()[0]
    null_title = int(agg["null_title"] or 0)
    empty_features = int(agg["empty_features"] or 0)
    title_share = (null_title / row_count) if row_count else 0.0
    features_share = (empty_features / row_count) if row_count else 0.0

    write_dq_results(
        spark,
        [
            DqResult(
                run_id=rid,
                run_ts=rts,
                table_name=out_table,
                contract_name=contract.name,
                contract_version=contract.version,
                check_id="gold_item_text_empty_title_share",
                check_kind="measure",
                column="title",
                status="measured",
                violation_count=null_title,
                total_rows=row_count,
                metric_value=float(title_share),
                details=json.dumps({"null_title": null_title, "rows": row_count}),
            ),
            DqResult(
                run_id=rid,
                run_ts=rts,
                table_name=out_table,
                contract_name=contract.name,
                contract_version=contract.version,
                check_id="gold_item_text_empty_features_share",
                check_kind="measure",
                column="features",
                status="measured",
                violation_count=empty_features,
                total_rows=row_count,
                metric_value=float(features_share),
                details=json.dumps({"empty_features": empty_features, "rows": row_count}),
            ),
        ],
    )

    return {
        "table": "item_text",
        "run_id": rid,
        "rows": row_count,
        "catalog_items": int(catalog_count),
        "empty_title_share": float(title_share),
        "empty_features_share": float(features_share),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }


# --- export ----------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export_item_text(
    spark: SparkSession,
    item_text_table: str = ITEM_TEXT,
    five_core_table: str = FIVE_CORE,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    export_root: str | Path = DEFAULT_EXPORT_ROOT,
) -> dict:
    """Reorder ``local.gold.item_text`` to the eval cache's ``item_ids`` order for
    the live 5-core snapshot and write a JVM-free parquet + manifest."""
    start = time.perf_counter()
    five_core_snapshot = _snapshot_id(spark, five_core_table)
    item_text_snapshot = _snapshot_id(spark, item_text_table)

    cache_dir = Path(cache_root) / str(five_core_snapshot)
    item_ids_path = cache_dir / "item_ids.parquet"
    if not item_ids_path.exists():
        raise FileNotFoundError(
            f"eval cache item_ids not found at {item_ids_path}; run `make eval-extract` first "
            f"for 5-core snapshot {five_core_snapshot}"
        )
    cache_item_ids = pq.read_table(item_ids_path).column("item_id").to_pylist()

    df = spark.table(item_text_table).select(*ITEM_TEXT_COLS)
    table_asins = {r["parent_asin"] for r in df.select("parent_asin").collect()}
    cache_asin_set = set(cache_item_ids)

    if table_asins != cache_asin_set:
        missing_in_table = cache_asin_set - table_asins
        missing_in_cache = table_asins - cache_asin_set
        raise AssertionError(
            "gold.item_text parent_asin set != eval cache item_ids set "
            f"(missing_in_table={len(missing_in_table)}, missing_in_cache={len(missing_in_cache)})"
        )

    order_df = spark.createDataFrame(
        [(asin, i) for i, asin in enumerate(cache_item_ids)],
        "parent_asin string, __order int",
    )
    ordered = (
        df.join(order_df, "parent_asin", "inner")
        .orderBy("__order")
        .select(*ITEM_TEXT_COLS)
    )
    pdf = ordered.toPandas()
    row_count = len(pdf)

    out_dir = Path(export_root) / str(five_core_snapshot)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "item_text.parquet"

    arrow_table = pa.Table.from_pandas(pdf, preserve_index=False)
    pq.write_table(arrow_table, parquet_path)

    parquet_sha256 = _sha256_bytes(parquet_path.read_bytes())
    item_ids_sha256 = hashlib.sha256(
        "\n".join(cache_item_ids).encode("utf-8")
    ).hexdigest()

    manifest = {
        "five_core_snapshot_id": five_core_snapshot,
        "item_text_snapshot_id": item_text_snapshot,
        "row_count": row_count,
        "item_ids_sha256": item_ids_sha256,
        "parquet_sha256": parquet_sha256,
        "aligned_to_cache": True,
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "status": "exported",
        "out_dir": str(out_dir),
        **manifest,
    }


# --- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.features.item_text")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--mode",
        choices=("build", "export"),
        default="build",
        help="build: write local.gold.item_text; export: reorder to the eval cache's "
        "item_ids order and write the JVM-free parquet + manifest",
    )
    parser.add_argument("--five-core-table", default=FIVE_CORE)
    parser.add_argument("--item-features-table", default=ITEM_FEATURES)
    parser.add_argument("--bronze-items-table", default=BRONZE_ITEMS)
    parser.add_argument("--item-text-table", default=ITEM_TEXT)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT))
    args = parser.parse_args(argv)

    spark = get_spark(
        app_name="gold-item-text",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        if args.mode == "build":
            summary = build_item_text(
                spark,
                five_core_table=args.five_core_table,
                item_features_table=args.item_features_table,
                bronze_items_table=args.bronze_items_table,
                out_table=args.item_text_table,
                run_id=args.run_id,
            )
        else:
            summary = export_item_text(
                spark,
                item_text_table=args.item_text_table,
                five_core_table=args.five_core_table,
                cache_root=args.cache_root,
                export_root=args.export_root,
            )
    finally:
        spark.stop()

    # Summary JSON MUST be the last stdout line (repo convention).
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
