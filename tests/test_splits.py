"""Frozen split boundary tests (Phase 1, T5).

Verifies the loader parses the OWNER-FROZEN boundaries to tz-aware UTC instants
and that ``split_label`` places boundary instants on the correct side *exactly*:
the instant AT ``train_end`` is train, one millisecond later is val; AT ``val_end``
is val, one later is test; one before ``test_end`` is test, AT ``test_end`` is
out-of-range (NULL label).
"""

from __future__ import annotations

import os

# Force loopback before any SparkContext starts (see test_silver_build.py).
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from datetime import datetime, timedelta, timezone

import pytest
from pyspark.sql import functions as F

from batch_recsys_lab.features.splits import load_splits

pytestmark = pytest.mark.spark

MS = timedelta(milliseconds=1)


def test_loader_parses_frozen_boundaries():
    s = load_splits()
    assert s.version == 1
    assert s.train_end == datetime(2022, 6, 30, 23, 59, 59, 999000, tzinfo=timezone.utc)
    assert s.val_end == datetime(2022, 12, 31, 23, 59, 59, 999000, tzinfo=timezone.utc)
    assert s.test_end == datetime(2023, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
    # All tz-aware UTC.
    for dt in (s.train_end, s.val_end, s.test_end):
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timedelta(0)


def test_split_label_boundaries_exact(spark):
    s = load_splits()
    cases = [
        ("before_train", datetime(2020, 1, 1, tzinfo=timezone.utc), "train"),
        ("at_train_end", s.train_end, "train"),
        ("train_end_plus_1ms", s.train_end + MS, "val"),
        ("at_val_end", s.val_end, "val"),
        ("val_end_plus_1ms", s.val_end + MS, "test"),
        ("test_end_minus_1ms", s.test_end - MS, "test"),
        ("at_test_end", s.test_end, None),  # out of range post-contract
    ]
    df = spark.createDataFrame(
        [(name, ts) for name, ts, _ in cases], "name string, ts timestamp"
    ).withColumn("label", s.split_label("ts")).withColumn("oor", s.out_of_range("ts"))
    got = {r["name"]: (r["label"], r["oor"]) for r in df.collect()}

    for name, _, expected in cases:
        assert got[name][0] == expected, f"{name}: expected {expected}, got {got[name][0]}"
    # out_of_range flags exactly the at/after test_end instants.
    assert got["at_test_end"][1] is True
    assert got["test_end_minus_1ms"][1] is False
