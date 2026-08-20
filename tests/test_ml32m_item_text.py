"""ML-32M item-text build/export tests (Phase 9, T9-3b §3a–f, §9).

Exercises the REAL tag aggregation and the REAL build/export against a
hand-computable micro-catalog in throwaway namespaces (no ML-32M download, and
no collision with ``test_ml32m_pipeline.py``'s ``local.gold_ml32m.*`` tables —
every table name here is passed explicitly).

Preregistration §9 names four of these gates directly: tag-cutoff boundary
(inclusive at ``train_end``), tag ranking determinism including the
lexicographic tie-break, empty-tag handling, and empty-genre handling.
"""

from __future__ import annotations

import os

# Force loopback before any SparkContext starts (see test_silver_build.py).
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batch_recsys_lab.features.item_text_ml32m import (
    ITEM_TEXT_ML32M_COLS,
    TAG_TOP_K,
    aggregate_tags,
    build_item_text_ml32m,
    export_item_text_ml32m,
)
from batch_recsys_lab.features.splits import load_splits

pytestmark = pytest.mark.spark

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_ML32M = REPO_ROOT / "configs" / "splits_ml32m.yaml"

MS = timedelta(milliseconds=1)
TAGS_DDL = "user_id string, parent_asin string, tag string, ts timestamp"

# Throwaway namespaces: this module must never read or write the tables that
# test_ml32m_pipeline.py builds in the same session-scoped warehouse.
NS = "local.t9b"
FIVE_CORE = f"{NS}.interactions_5core"
ITEM_FEATURES = f"{NS}.item_features"
SILVER_TAGS = f"{NS}.tags"
ITEM_TEXT = f"{NS}.item_text"
DQ_TABLE = f"{NS}.dq_results"


@pytest.fixture(scope="module")
def splits():
    return load_splits(SPLITS_ML32M)


# --------------------------------------------------------------------------- #
# §3(a)–(d): the aggregation rule, tested in isolation.
# --------------------------------------------------------------------------- #


def _tags(spark, rows):
    return spark.createDataFrame(rows, TAGS_DDL)


def _agg(spark, rows, splits, top_k=TAG_TOP_K):
    return {
        r["parent_asin"]: r["tags_top10"]
        for r in aggregate_tags(_tags(spark, rows), splits.train_end, top_k=top_k).collect()
    }


def test_tag_cutoff_is_inclusive_at_train_end(spark, splits):
    """§3(a): ts <= train_end, INCLUSIVE — the frozen split's own boundary."""
    rows = [
        ("u1", "m1", "on the boundary", splits.train_end),
        ("u2", "m1", "one ms later", splits.train_end + MS),
        ("u3", "m1", "well before", splits.train_end - timedelta(days=365)),
        ("u4", "m2", "val era", splits.val_end),
        ("u5", "m3", "test era", splits.val_end + timedelta(days=90)),
    ]
    out = _agg(spark, rows, splits)
    # m1 keeps the boundary tag and the older one; the +1ms tag is post-cutoff.
    assert out["m1"] == ["on the boundary", "well before"]
    # m2/m3 have no in-window tags at all -> absent here (empty list is applied
    # by the LEFT JOIN in the build, not by the aggregation).
    assert "m2" not in out and "m3" not in out


def test_weight_is_distinct_users_not_rows(spark, splits):
    """§3(c): one user tagging the same movie repeatedly counts once."""
    ts = splits.train_end - timedelta(days=1)
    rows = [
        ("u1", "m1", "funny", ts),
        ("u1", "m1", "funny", ts - timedelta(days=5)),  # same user again -> 1
        ("u1", "m1", "FUNNY", ts - timedelta(days=6)),  # same tag after lower() -> 1
        ("u2", "m1", "quirky", ts),
        ("u3", "m1", "quirky", ts),  # weight 2 -> ranks above "funny"
    ]
    assert _agg(spark, rows, splits)["m1"] == ["quirky", "funny"]


def test_ranking_is_weight_desc_then_lexicographic_ascending(spark, splits):
    """§3(d): ties break on tag_norm ASC (UTF-8 binary), deterministically."""
    ts = splits.train_end - timedelta(days=1)
    rows = [
        # Three tags at weight 1 -> pure lexicographic order; one at weight 2.
        ("u1", "m1", "zebra", ts),
        ("u2", "m1", "Apple", ts),  # lowercased -> "apple" sorts first
        ("u3", "m1", "middle", ts),
        ("u4", "m1", "heavy", ts),
        ("u5", "m1", "heavy", ts),
    ]
    assert _agg(spark, rows, splits)["m1"] == ["heavy", "apple", "middle", "zebra"]


def test_ranking_is_stable_under_input_row_order(spark, splits):
    ts = splits.train_end - timedelta(days=1)
    rows = [(f"u{i}", "m1", tag, ts) for i, tag in enumerate(["b", "a", "d", "c"])]
    first = _agg(spark, rows, splits)["m1"]
    second = _agg(spark, list(reversed(rows)), splits)["m1"]
    assert first == second == ["a", "b", "c", "d"]


def test_top_k_cap_keeps_exactly_the_ten_heaviest(spark, splits):
    """§3(d): K = 10, frozen a priori; no minimum-weight filter."""
    ts = splits.train_end - timedelta(days=1)
    rows = []
    # tag "t00".."t11": tag tNN gets NN+1 distinct taggers, so the ranking is the
    # reverse of the tag name order and the cap must drop t00 and t01.
    for n in range(12):
        for u in range(n + 1):
            rows.append((f"u{n}_{u}", "m1", f"t{n:02d}", ts))
    out = _agg(spark, rows, splits)["m1"]
    assert len(out) == TAG_TOP_K == 10
    assert out == [f"t{n:02d}" for n in range(11, 1, -1)]


def test_empty_and_whitespace_tags_are_dropped_not_tokenized(spark, splits):
    """§3(b): a row that normalizes to "" is dropped (re-assertion of silver)."""
    ts = splits.train_end - timedelta(days=1)
    rows = [
        ("u1", "m1", "   ", ts),
        ("u2", "m1", "", ts),
        ("u3", "m1", None, ts),
        ("u4", "m1", "  Real Tag  ", ts),  # trimmed + lowercased
    ]
    assert _agg(spark, rows, splits)["m1"] == ["real tag"]


# --------------------------------------------------------------------------- #
# §3(e)/(f): the build — catalog LEFT JOIN, empty list, coverage measures.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def built(spark, splits):
    ts = splits.train_end - timedelta(days=1)
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {NS}")

    # 5-core catalog: m1..m4 (m4 has no item_features row at all — the ML-32M
    # join-loss case that gold_ml32m.item_features measures).
    spark.createDataFrame(
        [
            ("u1", "m1", ts),
            ("u2", "m2", ts),
            ("u2", "m2", ts),  # duplicate: catalog is DISTINCT parent_asin
            ("u3", "m3", ts),
            ("u4", "m4", ts),
        ],
        "user_id string, parent_asin string, ts timestamp",
    ).writeTo(FIVE_CORE).createOrReplace()

    spark.createDataFrame(
        [
            ("m1", "Toy Story (1995)", ["Adventure", "Animation", "Children"]),
            ("m2", "American President, The (1995)", ["Comedy", "Drama"]),
            ("m3", "Mystery Film (2023)", []),  # "(no genres listed)" upstream
            # m5 is not in the 5-core: item_features rows outside the catalog
            # must not leak into item_text.
            ("m5", "Unrated Film (2001)", ["Drama"]),
        ],
        "parent_asin string, title string, genres array<string>",
    ).writeTo(ITEM_FEATURES).createOrReplace()

    spark.createDataFrame(
        [
            ("u1", "m1", "pixar", ts),
            ("u2", "m1", "pixar", ts),
            ("u3", "m1", "funny", ts),
            ("u9", "m1", "post cutoff", splits.train_end + MS),  # excluded
            ("u4", "m3", "obscure", ts),
            # m2 has only a post-cutoff tag -> zero in-window tags.
            ("u5", "m2", "later", splits.val_end),
        ],
        TAGS_DDL,
    ).writeTo(SILVER_TAGS).createOrReplace()

    summary = build_item_text_ml32m(
        spark,
        five_core_table=FIVE_CORE,
        item_features_table=ITEM_FEATURES,
        tags_table=SILVER_TAGS,
        out_table=ITEM_TEXT,
        run_id="testrun",
        dq_table=DQ_TABLE,
        splits=splits,
    )
    rows = {r["parent_asin"]: r for r in spark.table(ITEM_TEXT).collect()}
    return summary, rows


def test_build_covers_the_catalog_exactly_once(spark, built):
    summary, rows = built
    assert summary["rows"] == summary["catalog_items"] == 4
    assert set(rows) == {"m1", "m2", "m3", "m4"}  # m5 (non-catalog) excluded
    assert spark.table(ITEM_TEXT).columns == ITEM_TEXT_ML32M_COLS


def test_zero_in_window_tags_becomes_an_empty_list_never_a_placeholder(built):
    _, rows = built
    assert rows["m1"]["tags_top10"] == ["pixar", "funny"]  # weight 2 then 1
    assert rows["m2"]["tags_top10"] == []  # only a post-cutoff tag
    assert rows["m4"]["tags_top10"] == []  # no tags, no item_features row


def test_genres_are_taken_as_stored_and_missing_metadata_stays_null(built):
    _, rows = built
    # §3(f): order preserved, no re-sorting.
    assert rows["m1"]["genres"] == ["Adventure", "Animation", "Children"]
    assert rows["m3"]["genres"] == []  # empty, not a placeholder genre
    # m4 has no item_features row: LEFT JOIN leaves title/genres NULL rather
    # than dropping the catalog item.
    assert rows["m4"]["title"] is None
    assert rows["m4"]["genres"] is None


def test_coverage_measures_are_published_before_any_embedding(spark, built):
    summary, _ = built
    assert summary["zero_tag_share"] == pytest.approx(2 / 4)  # m2, m4
    assert summary["empty_genres_share"] == pytest.approx(2 / 4)  # m3 ([]), m4 (NULL)
    assert summary["empty_text_share"] == pytest.approx(1 / 4)  # m4 only
    assert summary["tag_cutoff"] == "2022-06-30T23:59:59.999000+00:00"
    assert summary["tag_top_k"] == 10
    assert summary["empty_after_norm_tag_rows"] == 0

    published = {
        r["check_id"]: r
        for r in spark.table(DQ_TABLE).where("table_name = '" + ITEM_TEXT + "'").collect()
    }
    assert {
        "gold_ml32m_item_text_zero_tag_share",
        "gold_ml32m_item_text_empty_genres_share",
        "gold_ml32m_item_text_empty_tag_rows",
        "gold_ml32m_item_text_empty_text_share",
    } <= set(published)
    assert published["gold_ml32m_item_text_zero_tag_share"]["status"] == "measured"
    assert published["gold_ml32m_item_text_zero_tag_share"]["metric_value"] == pytest.approx(0.5)
    assert published["gold_ml32m_item_text_empty_genres_share"]["contract_name"] == (
        "gold_ml32m_item_text"
    )


# --------------------------------------------------------------------------- #
# §3(e): the export is aligned to the eval cache's item_ids order.
# --------------------------------------------------------------------------- #


def _write_cache_item_ids(spark, tmp_path, item_ids):
    snapshot = spark.sql(
        f"SELECT snapshot_id FROM {FIVE_CORE}.refs WHERE name = 'main'"
    ).first()["snapshot_id"]
    cache_dir = tmp_path / "cache" / str(snapshot)
    cache_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"item_id": pa.array(item_ids, type=pa.string())}),
        cache_dir / "item_ids.parquet",
    )
    return tmp_path / "cache", str(snapshot)


def test_export_reorders_to_cache_item_ids_order(spark, built, tmp_path):
    order = ["m3", "m1", "m4", "m2"]
    cache_root, snapshot = _write_cache_item_ids(spark, tmp_path, order)
    out = export_item_text_ml32m(
        spark,
        item_text_table=ITEM_TEXT,
        five_core_table=FIVE_CORE,
        cache_root=cache_root,
        export_root=tmp_path / "text",
    )
    assert out["row_count"] == 4 and out["aligned_to_cache"] is True
    table = pq.read_table(tmp_path / "text" / snapshot / "item_text.parquet")
    assert table.column("parent_asin").to_pylist() == order
    assert table.column_names == ITEM_TEXT_ML32M_COLS
    # The empty list survives the parquet round trip as [] (not NULL).
    assert table.column("tags_top10").to_pylist()[order.index("m2")] == []


def test_export_aborts_on_a_cache_that_is_not_the_same_item_set(spark, built, tmp_path):
    cache_root, _ = _write_cache_item_ids(spark, tmp_path, ["m1", "m2", "m3"])
    with pytest.raises(AssertionError, match="!= eval cache item_ids set"):
        export_item_text_ml32m(
            spark,
            item_text_table=ITEM_TEXT,
            five_core_table=FIVE_CORE,
            cache_root=cache_root,
            export_root=tmp_path / "text",
        )
