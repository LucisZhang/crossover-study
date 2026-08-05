"""Deterministic dedup tests (Phase 1, T3; D2).

Exercises the two silver-interactions dedup stages as pure DataFrame transforms
on synthetic frames (no Iceberg writes, no contract engine):

* exact-dup drop collapses fully-identical rows;
* keep-latest keeps the max-``ts`` row per (user_id, parent_asin);
* a pair tied on (user_id, parent_asin, ts) but differing elsewhere resolves to
  the *same* winner across two different repartitionings of the input — i.e. the
  ``xxhash64``-of-all-columns tie-break is a total order, partition-independent.

Uses the shared tmp-warehouse ``spark`` fixture (``tests/conftest.py``).
"""

from __future__ import annotations

import os

# This host cannot bind Spark to its own hostname; force loopback before any
# SparkContext starts (the session-scoped fixture builds it lazily on first use).
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from datetime import datetime

import pytest

from batch_recsys_lab.features.silver import (
    SILVER_INTERACTION_COLS,
    drop_exact_duplicates,
    keep_latest,
)

pytestmark = pytest.mark.spark

_DDL = (
    "user_id string, parent_asin string, asin string, rating double, "
    "ts timestamp, helpful_vote long, verified_purchase boolean"
)


def _row(user="U", parent="P", asin="A", rating=5.0, ts=datetime(2022, 1, 1), hv=0, vp=True):
    return (user, parent, asin, rating, ts, hv, vp)


def test_exact_duplicate_collapses(spark):
    a = _row(asin="A")
    b = _row(asin="B")
    df = spark.createDataFrame([a, a, a, b], _DDL)
    out = drop_exact_duplicates(df)
    rows = {tuple(r) for r in out.collect()}
    assert out.count() == 2
    assert rows == {a, b}


def test_keep_latest_keeps_max_ts(spark):
    old = _row(asin="A", ts=datetime(2020, 1, 1))
    mid = _row(asin="A", ts=datetime(2021, 6, 1))
    new = _row(asin="A", ts=datetime(2022, 12, 31))
    df = spark.createDataFrame([old, new, mid], _DDL)
    out = keep_latest(df).collect()
    assert len(out) == 1
    assert out[0]["ts"] == datetime(2022, 12, 31)


def test_tie_break_is_partition_order_independent(spark):
    ts = datetime(2022, 5, 5)
    a = _row(asin="AAA", ts=ts)
    b = _row(asin="BBB", ts=ts)  # tie on (user, parent, ts); differs on asin

    w1 = keep_latest(spark.createDataFrame([a, b], _DDL).repartition(1)).collect()
    w2 = keep_latest(spark.createDataFrame([b, a], _DDL).repartition(4)).collect()

    assert len(w1) == 1 and len(w2) == 1
    winner1 = tuple(w1[0][c] for c in SILVER_INTERACTION_COLS)
    winner2 = tuple(w2[0][c] for c in SILVER_INTERACTION_COLS)
    assert winner1 == winner2  # xxhash64 total order → deterministic survivor
