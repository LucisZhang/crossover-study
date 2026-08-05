"""Frozen temporal split boundaries (Phase 1, T5; UPGRADE_PLAN.md §6.1, invariant #1).

The split boundaries are OWNER-FROZEN in ``configs/splits.yaml`` and must never be
recomputed or altered at runtime — this module only *loads* them into a frozen
dataclass and offers the two helpers the gold builds need:

* :meth:`SplitConfig.split_label` — a PySpark ``Column`` mapping a timestamp column
  to ``'train' | 'val' | 'test'`` per the frozen rule::

      train:  ts <= train_end                     (train_end inclusive)
      val:    train_end < ts <= val_end            (val_end inclusive)
      test:   val_end   < ts <  test_end           (test_end exclusive, snapshot end)

  Any ``ts >= test_end`` (or a NULL ``ts``) maps to NULL: post-contract such rows
  cannot exist (the silver ``ts_range`` check has ``max_exclusive == test_end``), so a
  non-NULL label partitions every real row into exactly one bucket. Use
  :meth:`out_of_range` to *flag* violators explicitly.

The three boundaries are parsed into tz-aware UTC ``datetime`` objects. The Spark
session pins its timezone to UTC (see ``spark_session.get_spark``), so literal
comparisons are instant-for-instant exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pyspark.sql import Column
from pyspark.sql import functions as F

# configs/splits.yaml at the repo root (…/features/splits.py -> parents[3] == root).
DEFAULT_SPLITS_PATH = Path(__file__).resolve().parents[3] / "configs" / "splits.yaml"

TRAIN = "train"
VAL = "val"
TEST = "test"


def _parse_ts(raw: str) -> datetime:
    """Parse an ISO-8601 boundary string to a tz-aware UTC ``datetime``.

    ``datetime.fromisoformat`` (Python 3.11+) accepts the trailing ``Z``. A naive
    result is rejected: the frozen boundaries are always zoned instants.
    """
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        raise ValueError(f"split boundary {raw!r} must carry a timezone")
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class SplitConfig:
    """The frozen temporal split boundaries plus the split-label helpers."""

    version: int
    frozen_at: str
    train_end: datetime
    val_end: datetime
    test_end: datetime

    def _lit(self, dt: datetime) -> Column:
        # tz-aware datetime -> timestamp literal; UTC session tz makes it exact.
        return F.lit(dt).cast("timestamp")

    def split_label(self, ts_col: str | Column) -> Column:
        """Map a timestamp column to ``'train' | 'val' | 'test'`` (else NULL).

        The ``when`` chain encodes the frozen inequalities exactly: an instant
        landing on ``train_end`` is ``train`` (``<=``), one microsecond later is
        ``val``; an instant on ``val_end`` is ``val``, one later is ``test``. NULL
        ``ts`` and ``ts >= test_end`` fall through to NULL.
        """
        ts = F.col(ts_col) if isinstance(ts_col, str) else ts_col
        return (
            F.when(ts <= self._lit(self.train_end), F.lit(TRAIN))
            .when(ts <= self._lit(self.val_end), F.lit(VAL))
            .when(ts < self._lit(self.test_end), F.lit(TEST))
            .otherwise(F.lit(None).cast("string"))
        )

    def out_of_range(self, ts_col: str | Column) -> Column:
        """``True`` where ``ts >= test_end`` — rows that must not exist post-contract."""
        ts = F.col(ts_col) if isinstance(ts_col, str) else ts_col
        return F.coalesce(ts >= self._lit(self.test_end), F.lit(False))


def load_splits(path: str | Path = DEFAULT_SPLITS_PATH) -> SplitConfig:
    """Load the frozen split boundaries from ``configs/splits.yaml``."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"splits {path!s}: top-level YAML must be a mapping")
    return SplitConfig(
        version=int(doc["version"]),
        frozen_at=str(doc["frozen_at"]),
        train_end=_parse_ts(doc["train_end"]),
        val_end=_parse_ts(doc["val_end"]),
        test_end=_parse_ts(doc["test_end"]),
    )
