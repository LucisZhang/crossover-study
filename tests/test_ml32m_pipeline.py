"""ML-32M silver → gold → churn-statistic tests on a 18-row micro-dataset
(Phase 9, T9-3a).

Runs the REAL builders end to end against the tmp Iceberg warehouse — no
downloaded data, no fixtures on disk — so the lane is proven runnable before it
ever meets the 32M-row source:

  bronze_ml32m (inline rows) → silver (gate/quarantine/dedup/contracts)
    → gold 5-core (k=2 here; k=5 in production) → user_stats / item_features /
      popularity → eval/churn_contrast.collect_inputs + compute_churn

The micro-dataset is designed so every branch that matters is exercised and every
expected number is hand-computable:

  * m1  TRAIN support 5  -> "high"        (1 TEST GT interaction)
  * m2  TRAIN support 3  -> "low"         (1 TEST GT interaction)
  * m3  TRAIN support 0  -> "zero"        (2 TEST GT interactions, first seen in TEST)
  * m99 TRAIN support 0, absent from movies.csv -> survives the 5-core but is NOT
        in the item catalog, so its 2 TEST interactions are the catalog-join loss
  * m9  in movies.csv with no ratings -> never reaches the 5-core catalog
  * three contract violators (half-star domain, null key, pre-1995 ts) -> quarantined

Expected churn share = (2 zero + 1 low) / 4 catalog-joined GT interactions = 0.75.
"""

from __future__ import annotations

import os

# Force loopback before any SparkContext starts (see test_silver_build.py).
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from pathlib import Path

import pytest

from batch_recsys_lab.eval.churn_contrast import collect_inputs, compute_churn
from batch_recsys_lab.features import gold_ml32m, silver_ml32m
from batch_recsys_lab.features.splits import load_splits

pytestmark = pytest.mark.spark

SPLITS_ML32M = Path(__file__).resolve().parents[1] / "configs" / "splits_ml32m.yaml"

TRAIN_TS = 1577836800  # 2020-01-01T00:00:00Z
TEST_TS = 1675209600  # 2023-02-01T00:00:00Z
PRE_1995_TS = 700000000  # 1992-03-08T00:00:00Z — below the contract's lower bound

BRONZE_RATINGS = "local.bronze_ml32m.ratings"
BRONZE_MOVIES = "local.bronze_ml32m.movies"
RATINGS_DDL = "userId long, movieId long, rating double, timestamp long"
MOVIES_DDL = "movieId long, title string, genres string"

RATING_ROWS = [
    # --- TRAIN -----------------------------------------------------------------
    (1, 1, 4.0, TRAIN_TS),
    (2, 1, 3.5, TRAIN_TS),
    (3, 1, 4.0, TRAIN_TS),
    (4, 1, 5.0, TRAIN_TS),
    (5, 1, 3.0, TRAIN_TS),  # m1 TRAIN support = 5 -> "high"
    (1, 2, 5.0, TRAIN_TS),
    (2, 2, 2.0, TRAIN_TS),
    (6, 2, 3.5, TRAIN_TS),  # m2 TRAIN support = 3 -> "low"
    # --- TEST ------------------------------------------------------------------
    (3, 2, 4.0, TEST_TS),  # GT on a low-support item
    (6, 1, 3.0, TEST_TS),  # GT on a high-support item
    (4, 3, 4.5, TEST_TS),
    (5, 3, 5.0, TEST_TS),  # m3: zero TRAIN support, first seen post-cutoff
    (4, 99, 4.0, TEST_TS),
    (5, 99, 3.5, TEST_TS),  # m99: no movies.csv row -> catalog-join loss
    # --- contract violators (quarantined at silver) -----------------------------
    (1, 3, 3.7, TEST_TS),  # rating outside the half-star domain
    (None, 5, 4.0, TRAIN_TS),  # null user key
    (7, 1, 4.0, PRE_1995_TS),  # ts below 1995-01-01
]

MOVIE_ROWS = [
    (1, "Toy Story (1995)", "Adventure|Animation|Children"),
    (2, "American President, The (1995)", "Comedy|Drama|Romance"),
    (3, "Mystery Film (2023)", "(no genres listed)"),
    (9, "Unrated Film (2001)", "Drama"),  # no ratings -> never in the 5-core catalog
]


@pytest.fixture(scope="module")
def ml32m_built(spark):
    """bronze → silver → gold, once per module. Returns every stage's summary."""
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.bronze_ml32m")
    spark.createDataFrame(RATING_ROWS, RATINGS_DDL).writeTo(BRONZE_RATINGS).createOrReplace()
    spark.createDataFrame(MOVIE_ROWS, MOVIES_DDL).writeTo(BRONZE_MOVIES).createOrReplace()

    # write_summary=False: the build summary log is a data/ artifact, not a test one.
    items = silver_ml32m.build_items(spark, run_id="testrun", write_summary=False)
    interactions = silver_ml32m.build_interactions(
        spark, run_id="testrun", write_summary=False
    )
    core = gold_ml32m.build_five_core(spark, k=2, run_id="testrun")
    features = gold_ml32m.build_gold_features(
        spark, run_id="testrun", splits_path=SPLITS_ML32M
    )
    return {
        "items": items,
        "interactions": interactions,
        "core": core,
        "features": features,
    }


# --------------------------------------------------------------------------- #
# Transforms.
# --------------------------------------------------------------------------- #


def test_transform_interactions_converts_epoch_seconds_and_stringifies_keys(spark):
    df = silver_ml32m.transform_interactions(
        spark.createDataFrame([(7, 42, 4.5, TEST_TS)], RATINGS_DDL)
    )
    assert dict(df.dtypes) == {
        "user_id": "string",
        "parent_asin": "string",
        "rating": "double",
        "ts": "timestamp",
    }
    row = df.collect()[0]
    assert (row["user_id"], row["parent_asin"]) == ("7", "42")
    # Epoch SECONDS, not millis: getting this wrong lands every rating in 1970.
    assert row["ts"].year == 2023 and row["ts"].month == 2


def test_transform_items_splits_genres_and_empties_the_no_genres_token(spark):
    df = silver_ml32m.transform_items(spark.createDataFrame(MOVIE_ROWS, MOVIES_DDL))
    rows = {r["parent_asin"]: r for r in df.collect()}
    assert rows["1"]["genres"] == ["Adventure", "Animation", "Children"]
    assert rows["2"]["title"] == "American President, The (1995)"
    assert rows["3"]["genres"] == []  # "(no genres listed)" -> empty array
    assert rows["3"]["_genres_missing"] is True
    assert rows["1"]["_genres_missing"] is False


# --------------------------------------------------------------------------- #
# Silver builds.
# --------------------------------------------------------------------------- #


def test_silver_conservation_and_quarantine_reasons(spark, ml32m_built):
    interactions = ml32m_built["interactions"]
    assert interactions["input_rows"] == len(RATING_ROWS) == 17
    assert interactions["kept"] == 14
    # One violator per quarantine check, and dedup is a no-op (MovieLens stores at
    # most one rating per (user, movie)) — verified, not assumed.
    assert interactions["quarantined"] == {
        "keys_non_null": 1,
        "rating_domain": 1,
        "ts_range": 1,
    }
    assert interactions["exact_duplicate"] == 0
    assert interactions["superseded_by_later_review"] == 0
    # Conservation is asserted inside the builder; re-assert it here explicitly.
    assert (
        interactions["kept"]
        + sum(interactions["quarantined"].values())
        + interactions["exact_duplicate"]
        + interactions["superseded_by_later_review"]
    ) == interactions["input_rows"]

    items = ml32m_built["items"]
    assert items["input_rows"] == len(MOVIE_ROWS)
    assert items["kept"] == len(MOVIE_ROWS)
    assert items["quarantined"] == {}

    assert spark.table(silver_ml32m.SILVER_INTERACTIONS).count() == 14
    assert spark.table(silver_ml32m.SILVER_ITEMS).count() == 4


def test_silver_dq_ledger_is_separate_and_carries_the_uniqueness_measure(spark):
    dq = spark.table(silver_ml32m.DQ_TABLE)
    rows = {
        r["check_id"]: r
        for r in dq.where(dq["table_name"] == silver_ml32m.SILVER_INTERACTIONS).collect()
    }
    uniqueness = rows["interaction_pair_uniqueness"]
    assert uniqueness["status"] == "measured"
    assert uniqueness["violation_count"] == 0
    # The Amazon ledger must not have been written to by this lane. (Other tests
    # in the shared tmp warehouse may have created it; what matters is that no
    # ML-32M row is in it, since that ledger backs the published DQ dashboard.)
    assert silver_ml32m.DQ_TABLE == "local.dq_ml32m.dq_results"
    if spark.catalog.tableExists("local.dq.dq_results"):
        amazon_dq = spark.table("local.dq.dq_results")
        assert amazon_dq.where(amazon_dq["table_name"].like("%_ml32m.%")).count() == 0


# --------------------------------------------------------------------------- #
# Gold.
# --------------------------------------------------------------------------- #


def test_five_core_keeps_the_designed_graph(ml32m_built):
    core = ml32m_built["core"]
    assert core["converged"] is True
    assert core["input_rows"] == 14
    assert core["output_rows"] == 14  # every user and item already has degree >= 2
    assert (core["output_users"], core["output_items"]) == (6, 4)


def test_item_features_measures_the_catalog_join_loss(ml32m_built):
    features = ml32m_built["features"]["item_features"]
    # m1, m2, m3 join; m99 is a 5-core item with no movies.csv row; m9 has a
    # movies.csv row but no interactions, so it is not in the catalog at all.
    assert features["catalog_items"] == 4
    assert features["rows"] == 3
    assert features["join_loss_items"] == 1
    assert features["join_loss_share"] == pytest.approx(0.25)


def test_user_stats_and_popularity_use_the_ml32m_splits(spark, ml32m_built):
    stats = {
        r["user_id"]: r for r in spark.table(gold_ml32m.USER_STATS).collect()
    }
    assert len(stats) == 6
    assert (stats["1"]["n_train"], stats["1"]["n_test"]) == (2, 0)
    assert (stats["4"]["n_train"], stats["4"]["n_test"]) == (1, 2)
    assert all(r["n_val"] == 0 for r in stats.values())

    pop = spark.table(gold_ml32m.POPULARITY)
    # The grid is {train_end, val_end} x {0, 30, 90, 365}, but a row exists only
    # where the window is non-empty: this micro-dataset's TRAIN traffic is all from
    # 2020, i.e. >365d before the 2022-06-30 cutoff, so only the all-history
    # (window_days == 0) slices are populated.
    assert {r["window_days"] for r in pop.select("window_days").distinct().collect()} == {0}
    assert pop.select("as_of").distinct().count() == 2
    # Leak-free: the 6 TEST-window interactions never enter popularity, at either
    # as_of — both all-history slices see exactly the 8 TRAIN rows.
    for as_of_row in pop.select("as_of").distinct().collect():
        slice_rows = pop.where(
            (pop["window_days"] == 0) & (pop["as_of"] == as_of_row["as_of"])
        ).collect()
        assert sum(r["n_interactions"] for r in slice_rows) == 8


# --------------------------------------------------------------------------- #
# The churn statistic (the T9-3a hinge), end to end.
# --------------------------------------------------------------------------- #


def test_collect_inputs_aligns_the_catalog_and_reports_the_join_edge(spark, ml32m_built):
    splits = load_splits(SPLITS_ML32M)
    data = collect_inputs(spark, splits, gold_ml32m.FIVE_CORE, gold_ml32m.ITEM_FEATURES)

    assert data["item_ids"] == ["1", "2", "3"]
    assert list(data["support"]) == [5, 3, 0]
    assert list(data["gt_counts"]) == [1, 1, 2]
    # 6 TEST-window 5-core interactions; the 2 on m99 fall outside the catalog.
    assert data["gt_interactions_all_5core"] == 6
    assert data["gt_interactions_total"] == 4
    assert data["catalog_join_loss_interactions"] == 2
    assert data["catalog_join_loss_items"] == 1
    assert data["n_users"] == 4
    assert data["coverage"] == {
        "catalog_size": 3,
        "stats_rows": 4,
        "catalog_items_covered": 3,
        "missing_from_stats": 0,
        "stats_rows_outside_catalog": 1,
    }
    assert data["support_bucket_counts_all_5core"] == {"zero": 2, "low": 1, "high": 1}


def test_churn_statistic_matches_the_hand_computed_micro_dataset(spark, ml32m_built):
    splits = load_splits(SPLITS_ML32M)
    train_end_ms = int(splits.train_end.timestamp() * 1000)
    data = collect_inputs(spark, splits, gold_ml32m.FIVE_CORE, gold_ml32m.ITEM_FEATURES)

    out = compute_churn(
        data["support"],
        data["last_train_ms"],
        data["first_seen_ms"],
        data["gt_counts"],
        train_end_ms,
    )
    headline, gate = out["headline"], out["gate"]

    by_support = headline["gt_interactions_by_support"]
    assert (by_support["zero"]["n"], by_support["low"]["n"], by_support["high"]["n"]) == (
        2,
        1,
        1,
    )
    assert gate["measured_share"] == pytest.approx(0.75)
    assert gate["band"] == ">=0.25"

    # The item axes carry over from T8-1 unchanged: m1/m2 last saw TRAIN traffic in
    # 2020 (>365d before the 2022-06-30 cutoff), m3 was never in TRAIN at all.
    by_recency = headline["gt_interactions_by_recency"]
    assert by_recency[">365d"]["n"] == 2
    assert by_recency["absent"]["n"] == 2
    by_first_seen = headline["gt_interactions_by_first_seen"]
    assert by_first_seen["2020"]["n"] == 2
    assert by_first_seen["post-cutoff"]["n"] == 2

    assert headline["distinct_gt_items_total"] == 3
    assert headline["catalog_size"] == 3
