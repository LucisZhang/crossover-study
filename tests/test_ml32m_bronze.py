"""ML-32M bronze ingestion tests on tiny synthetic CSVs (Phase 9, T9-3a).

Never touches ``data/raw/ml32m`` (nothing is downloaded on this machine) — writes
small CSV files to a tmp dir and ingests them against the tmp Iceberg warehouse.

Covers the three things that differ from the Amazon gz-jsonl path: the csv reader
branch, the corrupt-record accounting under PERMISSIVE csv parsing, and the
positional-schema header guard.
"""

from __future__ import annotations

import os

# Force loopback before any SparkContext starts (see test_silver_build.py).
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import pytest

from batch_recsys_lab.ingest.bronze import TABLE_SPECS as AMAZON_TABLE_SPECS
from batch_recsys_lab.ingest.bronze import ingest_table
from batch_recsys_lab.ingest.bronze_ml32m import TABLE_SPECS

pytestmark = pytest.mark.spark

RATINGS_HEADER = "userId,movieId,rating,timestamp"
MOVIES_HEADER = "movieId,title,genres"


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_amazon_specs_are_untouched_by_the_ml32m_lane():
    # The additive TableSpec fields must leave the Amazon lane on exactly the
    # code path it had before (json reader, no options, local.bronze).
    for name, spec in AMAZON_TABLE_SPECS.items():
        assert spec.fmt == "json", name
        assert spec.read_options == {}, name
        assert spec.namespace == "local.bronze", name
        assert spec.expected_header is None, name


def test_ingest_ratings_counts_corrupt_rows_and_keeps_raw_types(spark, tmp_path):
    src = _write(
        tmp_path / "ratings.csv",
        [
            RATINGS_HEADER,
            "1,1,4.0,1577836800",
            "1,2,0.5,1577836801",
            "2,1,5.0,1577836802",
            "notanid,3,4.0,1577836803",  # unparseable key -> corrupt
            "3,4",  # too few tokens -> corrupt
        ],
    )

    summary = ingest_table(spark, "ratings", src, repartition=2, specs=TABLE_SPECS)

    assert summary["table"] == "local.bronze_ml32m.ratings"
    assert summary["total_parsed"] == 5
    assert summary["corrupt"] == 2
    assert summary["written"] == 3

    df = spark.table("local.bronze_ml32m.ratings")
    assert df.count() == 3
    assert "_corrupt_record" not in df.columns
    dtypes = dict(df.dtypes)
    # Bronze is faithful: source column names, integral keys, raw epoch-SECONDS
    # long (silver converts), half-star double.
    assert dtypes == {
        "userId": "bigint",
        "movieId": "bigint",
        "rating": "double",
        "timestamp": "bigint",
    }
    assert {r["timestamp"] for r in df.collect()} == {1577836800, 1577836801, 1577836802}


def test_ingest_movies_keeps_quoted_titles_and_raw_genres(spark, tmp_path):
    src = _write(
        tmp_path / "movies.csv",
        [
            MOVIES_HEADER,
            "1,Toy Story (1995),Adventure|Animation|Children",
            '2,"American President, The (1995)",Comedy|Drama|Romance',
            "3,Mystery Film (2023),(no genres listed)",
        ],
    )

    summary = ingest_table(spark, "movies", src, repartition=1, specs=TABLE_SPECS)

    assert summary == {
        "table": "local.bronze_ml32m.movies",
        "total_parsed": 3,
        "corrupt": 0,
        "written": 3,
        "wall_clock_s": summary["wall_clock_s"],
    }
    rows = {r["movieId"]: r for r in spark.table("local.bronze_ml32m.movies").collect()}
    # The embedded comma survives (quoted field), and genres stay the raw
    # pipe-separated string — splitting is silver's job.
    assert rows[2]["title"] == "American President, The (1995)"
    assert rows[1]["genres"] == "Adventure|Animation|Children"
    assert rows[3]["genres"] == "(no genres listed)"


def test_header_drift_refuses_to_ingest(spark, tmp_path):
    src = _write(
        tmp_path / "renamed.csv",
        ["userId,itemId,rating,timestamp", "1,1,4.0,1577836800"],
    )
    with pytest.raises(ValueError, match="CSV header"):
        ingest_table(spark, "ratings", src, repartition=1, specs=TABLE_SPECS)


def test_column_reorder_is_caught_before_any_read(spark, tmp_path):
    # Same column NAMES, different order: a positional schema would silently swap
    # movieId and rating. The header check is the only thing standing in the way.
    src = _write(
        tmp_path / "reordered.csv",
        ["userId,rating,movieId,timestamp", "1,4.0,1,1577836800"],
    )
    with pytest.raises(ValueError, match="refusing to ingest"):
        ingest_table(spark, "ratings", src, repartition=1, specs=TABLE_SPECS)
