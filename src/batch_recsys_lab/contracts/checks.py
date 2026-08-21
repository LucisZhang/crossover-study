"""Check kind → PySpark expression builders (Phase 1, T1; docs/engineering-log/UPGRADE_PLAN.md §8).

Two families:

* **Row-level kinds** (``not_null``, ``allowed_values``, ``forbidden_values``,
  ``range``, ``no_control_chars``) build a boolean *violation* Column — ``True``
  where the row violates the check. The gate ORs these together in a single
  pass; audit sums them.
* **Table-level kinds** (``no_all_null``, ``unknown_share``, ``orphan_rate``,
  and ``no_control_chars`` when audited) build aggregate expressions / counts.

NULL semantics (D5, D7): ``range`` and ``forbidden_values`` treat NULL as a
*pass* — absence of data is neither out of range nor a forbidden value.
``allowed_values`` treats NULL as a *violation* — a value declared to live in a
closed domain must be present and in it (there is no separate not_null check on
those columns, so allowed_values is the only guard). ``not_null`` is, of course,
the null guard itself.
"""

from __future__ import annotations

from functools import reduce

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from batch_recsys_lab.contracts.loader import Check

# Control chars: C0 range (0x00–0x1F, includes tab / newline / CR) plus DEL.
# Strict by design — a tab or newline embedded in a title or brand is a hygiene
# defect for this lab's text fields.
CONTROL_CHAR_REGEX = "[\\x00-\\x1F\\x7F]"

ROW_LEVEL_KINDS: frozenset[str] = frozenset(
    {"not_null", "allowed_values", "forbidden_values", "range", "no_control_chars"}
)
TABLE_LEVEL_KINDS: frozenset[str] = frozenset(
    {"no_all_null", "unknown_share", "orphan_rate", "no_control_chars"}
)


def _range_literal(value: object, dtype: str | None) -> Column:
    """Build a comparable literal for a range bound.

    Timestamp bounds arrive as ISO-8601 strings (e.g. ``2023-10-01T00:00:00Z``);
    cast to timestamp so the comparison is against the ts column's real type.
    Spark's string→timestamp cast handles the ``T`` separator and trailing ``Z``.
    Numeric bounds pass through unchanged.
    """
    if dtype == "timestamp":
        return F.lit(value).cast("timestamp")
    return F.lit(value)


def row_violation_column(check: Check, column_types: dict[str, str]) -> Column:
    """Return a boolean Column that is ``True`` exactly where a row violates ``check``.

    Only defined for row-level kinds. ``column_types`` maps column name → contract
    dtype and is consulted for ``range`` (to type timestamp bounds correctly).
    """
    kind = check.kind

    if kind == "not_null":
        # Violation if ANY of the listed columns is null.
        return reduce(
            lambda a, b: a | b,
            (F.col(c).isNull() for c in check.columns),
            F.lit(False),
        )

    if kind == "allowed_values":
        col = F.col(check.columns[0])
        allowed = list(check.values or ())
        # NULL is a violation (out of the declared closed domain).
        return col.isNull() | ~col.isin(allowed)

    if kind == "forbidden_values":
        col = F.col(check.columns[0])
        forbidden = list(check.values or ())
        # NULL passes: it is not one of the forbidden values.
        return F.coalesce(col.isin(forbidden), F.lit(False))

    if kind == "range":
        col = F.col(check.columns[0])
        dtype = column_types.get(check.columns[0])
        cond: Column = F.lit(False)
        if check.min is not None:
            cond = cond | (col < _range_literal(check.min, dtype))
        if check.max is not None:
            cond = cond | (col > _range_literal(check.max, dtype))
        if check.max_exclusive is not None:
            cond = cond | (col >= _range_literal(check.max_exclusive, dtype))
        # NULL passes: any comparison against NULL is NULL → coalesced to False.
        return F.coalesce(cond, F.lit(False))

    if kind == "no_control_chars":
        # Violation if ANY listed column contains a control character.
        return reduce(
            lambda a, b: a | b,
            (F.coalesce(F.col(c).rlike(CONTROL_CHAR_REGEX), F.lit(False)) for c in check.columns),
            F.lit(False),
        )

    raise ValueError(f"{kind!r} is not a row-level check kind")


# --- Table-level aggregate builders ------------------------------------------


def no_all_null_nonnull_exprs(column_names: list[str]) -> list[Column]:
    """One non-null-count expression per column (``count(col)`` skips NULLs).

    A column whose non-null count is 0 is a dead column (all-NULL).
    """
    return [F.count(F.col(name)).alias(f"nonnull__{name}") for name in column_names]


def unknown_share_agg_expr(check: Check) -> Column:
    """Sum-of-matches expression for ``unknown_share`` (share = matches / total)."""
    col = F.col(check.columns[0])
    return F.sum(F.when(col == F.lit(check.value), F.lit(1)).otherwise(F.lit(0)))


def control_chars_agg_expr(check: Check) -> Column:
    """Count of rows with a control-char violation (audit view of no_control_chars)."""
    return F.sum(row_violation_column(check, {}).cast("long"))


def orphan_stats(spark: SparkSession, df: DataFrame, check: Check) -> tuple[int, int]:
    """Return ``(orphan_count, non_null_fk_count)`` for an ``orphan_rate`` check.

    An orphan is a non-null FK value absent from the reference table's key column.
    NULL FKs are not orphans (they are excluded from the denominator too).
    """
    fk = check.columns[0]
    ref = (
        spark.table(check.ref_table)
        .select(F.col(check.ref_column).alias("__ref"))
        .where(F.col("__ref").isNotNull())
        .distinct()
    )
    fks = df.select(F.col(fk).alias("__fk")).where(F.col("__fk").isNotNull())
    denom = fks.count()
    orphans = fks.join(ref, fks["__fk"] == ref["__ref"], "left_anti").count()
    return orphans, denom
