"""``local.gold_ml32m.item_text`` build + JVM-free export (Phase 9, T9-3b).

Implements the T9-3b preregistration §3(a)–(g) verbatim (EXPERIMENT_LOG.md,
"Phase 9 T9-3b preregistration"). Same two-step shape as the Amazon lane's
``features/item_text.py``, and wholly additive to it: separate namespaces
(``gold_ml32m`` / ``dq_ml32m``), separate contract (``contracts/ml32m``),
separate export root — no Amazon table, artifact or record is touched.

* :func:`build_item_text_ml32m` (Spark) — restricts to the distinct 5-core
  catalog (``local.gold_ml32m.interactions_5core``), LEFT JOINs
  ``local.gold_ml32m.item_features`` (title + genres as stored) and the
  TRAIN-cutoff tag aggregation over ``local.silver_ml32m.tags``, and writes
  ``local.gold_ml32m.item_text``. Row count is asserted equal to the distinct
  5-core catalog count (which also gives ``parent_asin`` uniqueness, since each
  join is 1 catalog row -> at most 1 output row).
* :func:`export_item_text_ml32m` (Spark reads, writes parquet; nothing
  downstream needs a JVM) — reorders the table's rows to exactly the ML-32M
  eval cache's ``item_ids`` order, asserting set equality first, and writes
  ``data/eval/text_ml32m/<five_core_snapshot_id>/item_text.parquet`` plus an
  ``export_manifest.json`` recording provenance.

**Tag aggregation (§3a–e), the only new degree of freedom, frozen:**

* §3(a) leakage guard — only tags with ``ts <= train_end`` (INCLUSIVE; read from
  ``configs/splits_ml32m.yaml``, never hardcoded) may enter the recipe. Tags are
  timestamped user events, unlike Amazon's static product metadata, so an
  unfiltered tag join would inject post-cutoff information into the item
  representation and silently violate invariant #1.
* §3(b) normalization — ``lower()`` + ``trim()`` of the already-silver-sanitized
  value (silver strips C0/DEL, maps ``\\n``->space, collapses whitespace, trims,
  and quarantines empty text). Rows still empty after normalization are dropped
  and *counted* as a measure, a re-assertion of the silver gate.
* §3(c) weight — ``COUNT(DISTINCT user_id)`` per ``(parent_asin, tag_norm)``.
  One user tagging the same movie repeatedly counts once. Tag timestamps are
  used ONLY for the §3(a) cutoff, never for recency weighting (that would be a
  new tunable).
* §3(d) ranking/cap — ``tag_weight DESC, tag_norm ASC`` (Spark's default UTF-8
  binary string ordering) as the deterministic tie-break, top **K = 10**. No
  minimum-weight filter, no tag-length cap, no K sweep.
* §3(e) join order/coverage — LEFT JOIN from the catalog; a movie with no
  in-window tags gets an **empty list**, never a placeholder token. The
  zero-in-window-tag share and the empty-``genres`` share are published to
  ``local.dq_ml32m.dq_results`` as measures BEFORE the embedding job runs.
* §3(f) genres — taken as stored in ``gold_ml32m.item_features`` (already split
  on ``|``, ``(no genres listed)`` already an empty array). Order preserved, no
  re-sorting, mirroring how the Amazon recipe consumes ``features``.

The assembled embedding text itself (§3g) lives with the recipe, in
``models/minilm_embed.build_recipe_text_ml32m`` — this module produces the
columns, the recipe assembles the string, exactly as on the Amazon side.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from batch_recsys_lab.contracts import load_contract, write_dq_results
from batch_recsys_lab.contracts.engine import DqResult, _resolve_run_id
from batch_recsys_lab.features.item_text import (
    _ensure_namespace,
    _sha256_bytes,
    _snapshot_id,
)
from batch_recsys_lab.features.silver_ml32m import DQ_TABLE
from batch_recsys_lab.features.splits import SplitConfig, load_splits
from batch_recsys_lab.spark_session import get_spark

# --- Locations -----------------------------------------------------------------
CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts" / "ml32m"
ITEM_TEXT_CONTRACT = CONTRACTS_DIR / "gold_ml32m_item_text.yaml"
SPLITS_PATH = Path(__file__).resolve().parents[3] / "configs" / "splits_ml32m.yaml"

FIVE_CORE = "local.gold_ml32m.interactions_5core"
ITEM_FEATURES = "local.gold_ml32m.item_features"
SILVER_TAGS = "local.silver_ml32m.tags"
ITEM_TEXT = "local.gold_ml32m.item_text"

DEFAULT_EXPORT_ROOT = Path("data/eval/text_ml32m")
DEFAULT_CACHE_ROOT = Path("data/eval/cache_ml32m")

# §3(d): frozen a priori at 10 to keep title+genres+tags inside all-MiniLM-L6-v2's
# 256-word-piece window. NOT a tunable — a K sweep would tune a text recipe and
# multiply recipe hashes. Mirrored into the recipe's `extra` (§3h) so the artifact
# identity binds it.
TAG_TOP_K = 10

ITEM_TEXT_ML32M_COLS = ["parent_asin", "title", "genres", "tags_top10"]


# --- tag aggregation (§3a–d) ----------------------------------------------------


def aggregate_tags(
    tags: DataFrame,
    train_end: datetime,
    top_k: int = TAG_TOP_K,
) -> DataFrame:
    """``(parent_asin, tags_top10 array<string>)`` — the frozen §3(a)–(d) rule.

    ``train_end`` is INCLUSIVE, matching ``SplitConfig.split_label``'s
    ``ts <= train_end`` train boundary and the pop-t12m window semantics: a tag
    written at exactly ``2022-06-30T23:59:59.999Z`` is TRAIN-side and enters the
    recipe; one millisecond later does not.

    Items with no surviving tag row are simply absent from the result — the
    empty-list default is applied by the LEFT JOIN in
    :func:`build_item_text_ml32m` (§3e), not here.
    """
    in_window = tags.where(F.col("ts") <= F.lit(train_end).cast("timestamp"))
    normalized = in_window.select(
        F.col("parent_asin"),
        F.col("user_id"),
        F.trim(F.lower(F.col("tag"))).alias("tag_norm"),
    ).where(F.col("tag_norm").isNotNull() & (F.col("tag_norm") != F.lit("")))

    weighted = normalized.groupBy("parent_asin", "tag_norm").agg(
        F.count_distinct(F.col("user_id")).alias("tag_weight")
    )

    ranked = weighted.withColumn(
        "__rank",
        F.row_number().over(
            Window.partitionBy("parent_asin").orderBy(
                F.col("tag_weight").desc(), F.col("tag_norm").asc()
            )
        ),
    ).where(F.col("__rank") <= F.lit(int(top_k)))

    # collect_list has no ordering guarantee, so the rank travels INSIDE the
    # struct and sort_array re-imposes it (struct ordering is field-by-field, and
    # __rank is unique per parent_asin) — deterministic under any partitioning.
    return (
        ranked.groupBy("parent_asin")
        .agg(
            F.sort_array(
                F.collect_list(F.struct(F.col("__rank"), F.col("tag_norm")))
            ).alias("__ranked")
        )
        .select(
            F.col("parent_asin"),
            F.transform(F.col("__ranked"), lambda s: s["tag_norm"]).alias("tags_top10"),
        )
    )


# --- build ----------------------------------------------------------------------


def build_item_text_ml32m(
    spark: SparkSession,
    five_core_table: str = FIVE_CORE,
    item_features_table: str = ITEM_FEATURES,
    tags_table: str = SILVER_TAGS,
    out_table: str = ITEM_TEXT,
    run_id: str | None = None,
    contract_path: str | Path = ITEM_TEXT_CONTRACT,
    dq_table: str = DQ_TABLE,
    splits: SplitConfig | None = None,
    splits_path: str | Path = SPLITS_PATH,
    top_k: int = TAG_TOP_K,
) -> dict:
    """Build ``local.gold_ml32m.item_text``. Raises ``AssertionError`` if the
    written row count does not equal the distinct 5-core catalog count."""
    start = time.perf_counter()
    rid, rts = _resolve_run_id(run_id)
    contract = load_contract(contract_path)
    if splits is None:
        splits = load_splits(splits_path)

    catalog = spark.table(five_core_table).select("parent_asin").distinct()
    catalog_count = catalog.count()

    features = spark.table(item_features_table).select("parent_asin", "title", "genres")

    tags = spark.table(tags_table).select("user_id", "parent_asin", "tag", "ts")
    in_window_rows = tags.where(F.col("ts") <= F.lit(splits.train_end).cast("timestamp"))
    in_window_count = in_window_rows.count()
    # §3(b) re-assertion: the silver gate already quarantined empty tag text, so
    # this count is expected to be 0 — it is measured, not assumed.
    empty_after_norm = in_window_rows.where(
        F.col("tag").isNull() | (F.trim(F.lower(F.col("tag"))) == F.lit(""))
    ).count()

    tags_agg = aggregate_tags(tags, splits.train_end, top_k=top_k)

    joined = (
        catalog.join(features, "parent_asin", "left")
        .join(tags_agg, "parent_asin", "left")
        .select(
            F.col("parent_asin"),
            F.col("title"),
            F.col("genres"),
            # §3(e): no in-window tags -> EMPTY LIST, never NULL, never a
            # placeholder token.
            F.coalesce(F.col("tags_top10"), F.array().cast("array<string>")).alias(
                "tags_top10"
            ),
        )
        .select(*ITEM_TEXT_ML32M_COLS)
    )

    _ensure_namespace(spark, out_table)
    joined.writeTo(out_table).createOrReplace()

    written = spark.table(out_table)
    row_count = written.count()
    if row_count != catalog_count:
        raise AssertionError(
            f"gold_ml32m.item_text row count {row_count} != distinct 5-core catalog "
            f"count {catalog_count}"
        )
    distinct_asins = written.select("parent_asin").distinct().count()
    if distinct_asins != row_count:
        raise AssertionError(
            f"gold_ml32m.item_text parent_asin not unique: {distinct_asins} distinct "
            f"of {row_count} rows"
        )

    empty_tags = F.size(F.col("tags_top10")) == 0
    empty_genres = F.col("genres").isNull() | (F.size(F.col("genres")) == 0)
    empty_title = F.col("title").isNull() | (F.trim(F.col("title")) == F.lit(""))
    agg = written.agg(
        F.sum(empty_tags.cast("long")).alias("zero_tag_items"),
        F.sum(empty_genres.cast("long")).alias("empty_genres_items"),
        F.sum((empty_tags & empty_genres & empty_title).cast("long")).alias(
            "empty_text_items"
        ),
    ).collect()[0]
    zero_tag_items = int(agg["zero_tag_items"] or 0)
    empty_genres_items = int(agg["empty_genres_items"] or 0)
    empty_text_items = int(agg["empty_text_items"] or 0)
    zero_tag_share = (zero_tag_items / row_count) if row_count else 0.0
    empty_genres_share = (empty_genres_items / row_count) if row_count else 0.0
    empty_text_share = (empty_text_items / row_count) if row_count else 0.0

    def _measure(
        check_id: str,
        column: str | None,
        violations: int,
        total: int,
        metric: float,
        details: dict,
    ) -> DqResult:
        return DqResult(
            run_id=rid,
            run_ts=rts,
            table_name=out_table,
            contract_name=contract.name,
            contract_version=contract.version,
            check_id=check_id,
            check_kind="measure",
            column=column,
            status="measured",
            violation_count=int(violations),
            total_rows=int(total),
            metric_value=float(metric),
            details=json.dumps(details),
        )

    write_dq_results(
        spark,
        [
            # §3(e), published BEFORE the embedding job runs.
            _measure(
                "gold_ml32m_item_text_zero_tag_share",
                "tags_top10",
                zero_tag_items,
                row_count,
                zero_tag_share,
                {
                    "zero_tag_items": zero_tag_items,
                    "rows": row_count,
                    "tag_cutoff": splits.train_end.isoformat(),
                    "tag_top_k": int(top_k),
                },
            ),
            _measure(
                "gold_ml32m_item_text_empty_genres_share",
                "genres",
                empty_genres_items,
                row_count,
                empty_genres_share,
                {"empty_genres_items": empty_genres_items, "rows": row_count},
            ),
            # §3(b) re-assertion of the silver empty-text gate (expected 0).
            _measure(
                "gold_ml32m_item_text_empty_tag_rows",
                "tag",
                empty_after_norm,
                in_window_count,
                (empty_after_norm / in_window_count) if in_window_count else 0.0,
                {
                    "empty_after_norm": int(empty_after_norm),
                    "in_window_tag_rows": int(in_window_count),
                },
            ),
            # Coverage input to the §3(j) fail-closed threshold ("> 50% of the
            # 5-core catalog has an entirely empty assembled text string"). This
            # measure only PUBLISHES the share; §3(j)'s omit-the-arm decision is
            # not automated here.
            _measure(
                "gold_ml32m_item_text_empty_text_share",
                None,
                empty_text_items,
                row_count,
                empty_text_share,
                {"empty_text_items": empty_text_items, "rows": row_count},
            ),
        ],
        dq_table,
    )

    return {
        "table": "ml32m_item_text",
        "run_id": rid,
        "rows": row_count,
        "catalog_items": int(catalog_count),
        "tag_cutoff": splits.train_end.isoformat(),
        "tag_top_k": int(top_k),
        "in_window_tag_rows": int(in_window_count),
        "empty_after_norm_tag_rows": int(empty_after_norm),
        "zero_tag_share": float(zero_tag_share),
        "empty_genres_share": float(empty_genres_share),
        "empty_text_share": float(empty_text_share),
        "wall_clock_s": round(time.perf_counter() - start, 3),
    }


# --- export ---------------------------------------------------------------------


def export_item_text_ml32m(
    spark: SparkSession,
    item_text_table: str = ITEM_TEXT,
    five_core_table: str = FIVE_CORE,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    export_root: str | Path = DEFAULT_EXPORT_ROOT,
) -> dict:
    """Reorder ``local.gold_ml32m.item_text`` to the ML-32M eval cache's
    ``item_ids`` order and write a JVM-free parquet + manifest (§3e)."""
    start = time.perf_counter()
    five_core_snapshot = _snapshot_id(spark, five_core_table)
    item_text_snapshot = _snapshot_id(spark, item_text_table)

    cache_dir = Path(cache_root) / str(five_core_snapshot)
    item_ids_path = cache_dir / "item_ids.parquet"
    if not item_ids_path.exists():
        raise FileNotFoundError(
            f"ML-32M eval cache item_ids not found at {item_ids_path}; run the "
            f"ML-32M eval extract first for 5-core snapshot {five_core_snapshot}"
        )
    cache_item_ids = pq.read_table(item_ids_path).column("item_id").to_pylist()

    df = spark.table(item_text_table).select(*ITEM_TEXT_ML32M_COLS)
    table_asins = {r["parent_asin"] for r in df.select("parent_asin").collect()}
    cache_asin_set = set(cache_item_ids)

    if table_asins != cache_asin_set:
        missing_in_table = cache_asin_set - table_asins
        missing_in_cache = table_asins - cache_asin_set
        raise AssertionError(
            "gold_ml32m.item_text parent_asin set != eval cache item_ids set "
            f"(missing_in_table={len(missing_in_table)}, missing_in_cache={len(missing_in_cache)})"
        )

    order_df = spark.createDataFrame(
        [(asin, i) for i, asin in enumerate(cache_item_ids)],
        "parent_asin string, __order int",
    )
    ordered = (
        df.join(order_df, "parent_asin", "inner")
        .orderBy("__order")
        .select(*ITEM_TEXT_ML32M_COLS)
    )
    pdf = ordered.toPandas()
    row_count = len(pdf)

    out_dir = Path(export_root) / str(five_core_snapshot)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "item_text.parquet"

    arrow_table = pa.Table.from_pandas(pdf, preserve_index=False)
    pq.write_table(arrow_table, parquet_path)

    parquet_sha256 = _sha256_bytes(parquet_path.read_bytes())
    item_ids_sha256 = _sha256_bytes("\n".join(cache_item_ids).encode("utf-8"))

    manifest = {
        "dataset": "ml32m",
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
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.features.item_text_ml32m")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--mode",
        choices=("build", "export"),
        default="build",
        help="build: write local.gold_ml32m.item_text; export: reorder to the ML-32M "
        "eval cache's item_ids order and write the JVM-free parquet + manifest",
    )
    parser.add_argument("--five-core-table", default=FIVE_CORE)
    parser.add_argument("--item-features-table", default=ITEM_FEATURES)
    parser.add_argument("--tags-table", default=SILVER_TAGS)
    parser.add_argument("--item-text-table", default=ITEM_TEXT)
    parser.add_argument("--splits-path", default=str(SPLITS_PATH))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT))
    args = parser.parse_args(argv)

    spark = get_spark(
        app_name="gold-ml32m-item-text",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        if args.mode == "build":
            summary = build_item_text_ml32m(
                spark,
                five_core_table=args.five_core_table,
                item_features_table=args.item_features_table,
                tags_table=args.tags_table,
                out_table=args.item_text_table,
                run_id=args.run_id,
                splits_path=args.splits_path,
            )
        else:
            summary = export_item_text_ml32m(
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
