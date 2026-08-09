"""DuckDB single-node reality check: rebuild BOTH silver tables, JVM-free
(Phase 7 stretch item 1; plan Part B, T-B2).

What this is
------------
A faithful DuckDB port of ``features/silver.py``'s two builds, gated by the same
``contracts/silver_items.yaml`` / ``contracts/silver_interactions.yaml`` (the
YAML is parsed with the production loader, so the *declared check order* — which
IS the ``primary_reason`` priority, D5 — cannot drift). It reads bronze
read-only at the CURRENT Iceberg snapshot, writes ONLY under
``data/bench/duckdb/``, never touches ``data/warehouse/``, and hard-asserts the
resulting waterfall against the integers recorded by the Spark build
(:data:`EXPECTED_WATERFALL`). A run whose counts do not reconcile exactly is a
failed run, not a published one.

What it is NOT (scope asymmetry — read before quoting the timings)
------------------------------------------------------------------
The recorded Spark ``wall_clock_s`` values in ``data/build_summary.jsonl``
(:data:`SPARK_REFERENCE`) cover *more* than this port does. ``build_items`` /
``build_interactions`` also run, inside the same timer:

* ``contracts.audit`` over the published table — a full re-scan plus, for
  interactions, the ``item_fk`` ``orphan_rate`` **join** of 43.4M FKs against
  the 1.6M-row item catalog;
* the ``dq_results`` ledger write, and the Iceberg commits (snapshot metadata,
  manifest lists) for silver + quarantine.

This module times only: bronze scan → typed projection → contract gate →
quarantine write → exact-dup drop → keep-latest → Parquet write. Any writeup
comparing the two numbers must say so; the record carries
``spark_reference.scope_note`` so the asymmetry travels with the evidence.

Known, measured engine divergences (never patched over)
-------------------------------------------------------
1. **Tie-break.** Spark's ``keep_latest`` breaks ``(user_id, parent_asin)``
   ties on the max ``ts`` with ``xxhash64`` of all seven columns; DuckDB cannot
   reproduce Spark's xxhash64. This port uses an explicit total order on the
   remaining columns instead. Row *counts* are tie-break-invariant (the parity
   spec is unaffected); survivor *content* can differ inside a tie group, by at
   most one row per group. Both ``tie_groups`` and ``tie_group_rows`` are
   measured and recorded, and ``--content-parity`` bounds the observed
   ``EXCEPT ALL`` diff by them.
2. **Regex dialect.** Spark's ``rlike`` is ``java.util.regex`` where ``$``
   also matches before a trailing newline; DuckDB's is RE2 where it does not.
   Only ``price_usd`` parsing can be affected (never the waterfall — an
   unparseable price becomes NULL and NULL passes the gate). The count of rows
   that Java would have parsed and RE2 would not is measured as
   ``price_regex_java_only`` and reported.
3. **Out-of-range epoch millis.** ``epoch_ms`` raises in DuckDB where Spark's
   ``timestamp_millis`` wraps. Left to raise on purpose: the recorded build
   quarantined 0 rows for ``ts_range``, so no such row exists in bronze, and a
   silent divergence would be worse than a loud stop.

No JVM is ever started (no Spark session, no JDK pin) — though ``pyspark`` does
get imported transitively, because ``batch_recsys_lab.contracts.__init__``
re-exports the Spark-based engine next to the pure-YAML ``load_contract`` this
module uses. The contract YAML itself is parsed with the production loader, so
check identity and order cannot drift.

Constants re-declared here (``SILVER_*_COLS``, ``_PRICE_REGEX``,
``CONTROL_CHAR_REGEX``) mirror ``features/silver.py`` / ``contracts/checks.py``
rather than importing them, so this module has no import-time dependency on the
Spark build code (the ``demo/dq_export_job.py`` precedent).
``tests/test_bench_duckdb.py`` asserts each mirror equals the Spark-side
original, so the copies cannot drift.

Usage::

    uv run --group bench python -m batch_recsys_lab.bench.duckdb_silver \\
        --runs 3 --content-parity --dry-run     # prints the record, appends nothing
    ... --runs 3 --content-parity --append      # appends one kind="bench" record
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from batch_recsys_lab.contracts.loader import Check, Contract, load_contract
from batch_recsys_lab.eval import runlog

# --- Locations ---------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_DIR = _REPO_ROOT / "contracts"
ITEMS_CONTRACT = CONTRACTS_DIR / "silver_items.yaml"
INTERACTIONS_CONTRACT = CONTRACTS_DIR / "silver_interactions.yaml"

DEFAULT_WAREHOUSE = "data/warehouse"
DEFAULT_OUT_DIR = "data/bench/duckdb"
DEFAULT_RESULTS_LOG = "results/runs.jsonl"
BUILD_SUMMARY_LOG = "data/build_summary.jsonl"

BRONZE_ITEMS = "local.bronze.items"
BRONZE_REVIEWS = "local.bronze.reviews"
SILVER_INTERACTIONS = "local.silver.interactions"

RECORD_KIND = "bench"

# --- Mirrored transform constants (drift-tested; see module docstring) --------

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
_PRICE_REGEX = r"^\$?\d+(\.\d+)?$"
# Java-regex ``$`` also matches before a single trailing newline; RE2's does not.
# Used only to MEASURE divergence 2 above, never to parse.
_PRICE_REGEX_JAVA_TAIL = r"^\$?\d+(\.\d+)?\n?$"
CONTROL_CHAR_REGEX = "[\\x00-\\x1F\\x7F]"

# Temp relations. Named (not anonymous CTEs) so each stage is separately timed,
# mirroring the ``localCheckpoint`` boundaries in features/silver.py.
_T_GATED = "bench_gated"
_T_DEDUPED = "bench_deduped"
_T_FINAL = "bench_final"

# --- Recorded Spark ledger (the parity spec; plan Part B) --------------------

EXPECTED_WATERFALL: dict[str, dict] = {
    "items": {
        "input_rows": 1610012,
        "kept": 1610012,
        "quarantined": {},
        "exact_duplicate": 0,
        "superseded_by_later_review": 0,
    },
    "interactions": {
        "input_rows": 43886944,
        "kept": 43365424,
        "quarantined": {"rating_domain": 2},
        "exact_duplicate": 477968,
        "superseded_by_later_review": 43550,
    },
}

# Recorded Spark *measures* for silver.items, from the committed contract-ledger
# export results/dq/dq_raw.json (measured_rates.price_unparseable_share and
# .brand_from_manufacturer_share, build run 20260805T143256Z-7406fc1;
# .unknown_brand_share, audit run 20260806T104111Z-5df9906). The items waterfall
# alone is a weak parity test (1,610,012 in, 1,610,012 out), so the transform
# itself is pinned here: price parsing and the Brand/Manufacturer fallback have
# to reproduce Spark's counts exactly, not just conserve rows.
EXPECTED_MEASURES: dict[str, dict[str, int]] = {
    "items": {
        "price_unparseable": 316,
        "brand_from_brand": 1153897,
        "brand_from_manufacturer": 384785,
        "brand_from_none": 71330,
        "brand_unknown": 73178,
    },
}

# Recorded Spark wall clocks, quoted from data/build_summary.jsonl (gitignored,
# hence hardcoded here and cross-checked against the file when it is present).
# NOTE: the ledger holds THREE items builds (15.645 / 23.609 / 12.553); the plan
# text quoted only two of them. All three are published.
SPARK_REFERENCE: dict[str, list[float]] = {
    "items_s": [15.645, 23.609, 12.553],
    "interactions_s": [474.25, 569.081, 316.413],
}
SPARK_SCOPE_NOTE = (
    "Recorded prior Spark local[10] runs; Spark was NOT re-run (no snapshot "
    "churn). These wall clocks additionally include the post-publish contract "
    "audit (for interactions, the item_fk orphan_rate join over 43.4M FKs), the "
    "dq_results ledger write and the Iceberg commits — none of which the DuckDB "
    "timings below cover. Compare scopes, not just numbers."
)
DUCKDB_SCOPE_NOTE = (
    "bronze iceberg scan -> typed projection -> contract gate -> quarantine "
    "parquet write -> exact-duplicate drop -> keep-latest -> silver parquet write"
)


# --- SQL helpers -------------------------------------------------------------


def _lit(value: object) -> str:
    """SQL literal for a contract-declared scalar."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _bound_literal(value: object, dtype: str | None) -> str:
    """Typed literal for a ``range`` bound (mirrors ``checks._range_literal``)."""
    if dtype == "timestamp":
        return f"CAST({_lit(value)} AS TIMESTAMP)"
    return _lit(value)


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def violation_sql(check: Check, column_types: dict[str, str]) -> str:
    """SQL boolean that is TRUE exactly where a row violates ``check``.

    Port of ``contracts.checks.row_violation_column`` including its NULL
    semantics (D5/D7): ``range``/``forbidden_values`` treat NULL as a pass,
    ``allowed_values`` treats NULL as a violation, ``not_null`` is the null
    guard. Every expression is COALESCEd to FALSE so it is never NULL.
    """
    kind = check.kind
    if kind == "not_null":
        inner = " OR ".join(f"{_ident(c)} IS NULL" for c in check.columns)
        return f"({inner})"

    if kind == "allowed_values":
        col = _ident(check.columns[0])
        values = ", ".join(_lit(v) for v in (check.values or ()))
        return f"({col} IS NULL OR NOT ({col} IN ({values})))"

    if kind == "forbidden_values":
        col = _ident(check.columns[0])
        values = ", ".join(_lit(v) for v in (check.values or ()))
        return f"COALESCE({col} IN ({values}), FALSE)"

    if kind == "range":
        col = _ident(check.columns[0])
        dtype = column_types.get(check.columns[0])
        parts: list[str] = []
        if check.min is not None:
            parts.append(f"{col} < {_bound_literal(check.min, dtype)}")
        if check.max is not None:
            parts.append(f"{col} > {_bound_literal(check.max, dtype)}")
        if check.max_exclusive is not None:
            parts.append(f"{col} >= {_bound_literal(check.max_exclusive, dtype)}")
        return "COALESCE(" + " OR ".join(parts) + ", FALSE)"

    if kind == "no_control_chars":
        inner = " OR ".join(
            f"COALESCE(regexp_matches({_ident(c)}, {_lit(CONTROL_CHAR_REGEX)}), FALSE)"
            for c in check.columns
        )
        return f"({inner})"

    raise ValueError(f"{kind!r} is not a row-level check kind")


def quarantine_checks(contract: Contract) -> list[Check]:
    """The contract's quarantine checks, in declared order (== D5 priority)."""
    row_level = {"not_null", "allowed_values", "forbidden_values", "range", "no_control_chars"}
    return [c for c in contract.checks if c.action == "quarantine" and c.kind in row_level]


def gate_columns_sql(contract: Contract) -> tuple[str, list[str]]:
    """``(select-list fragment, check_ids)`` adding one ``v__<id>`` boolean per
    quarantine check plus ``primary_reason`` (first violated, declared order)."""
    checks = quarantine_checks(contract)
    column_types = {cs.name: cs.dtype for cs in contract.columns}
    pieces = [
        f"{violation_sql(c, column_types)} AS {_ident('v__' + c.check_id)}" for c in checks
    ]
    when = " ".join(
        f"WHEN {violation_sql(c, column_types)} THEN {_lit(c.check_id)}" for c in checks
    )
    pieces.append(f"CASE {when} END AS primary_reason")
    pieces.append(
        "list_filter(["
        + ", ".join(
            f"CASE WHEN {violation_sql(c, column_types)} THEN {_lit(c.check_id)} END"
            for c in checks
        )
        + "], x -> x IS NOT NULL) AS violation_reasons"
    )
    return ",\n       ".join(pieces), [c.check_id for c in checks]


# --- Transform SQL (faithful ports of features/silver.py) --------------------


def build_items_sql(source: str) -> str:
    """Typed silver-items projection + measure helpers, from a bronze relation.

    Port of ``features.silver.transform_items``. ``source`` is any DuckDB
    relation expression (``iceberg_scan(...)``, ``read_parquet(...)``, a table
    name) exposing the bronze.items columns.
    """
    ctrl = _lit(CONTROL_CHAR_REGEX)
    price_raw = "trim(price)"
    parseable = f"regexp_matches({price_raw}, {_lit(_PRICE_REGEX)})"
    java_only = (
        f"(COALESCE(regexp_matches({price_raw}, {_lit(_PRICE_REGEX_JAVA_TAIL)}), FALSE) "
        f"AND NOT COALESCE({parseable}, FALSE))"
    )

    def blank_to_null(expr: str) -> str:
        return f"CASE WHEN trim({expr}) = '' THEN NULL ELSE {expr} END"

    brand_b = blank_to_null("details['Brand']")
    brand_m = blank_to_null("details['Manufacturer']")

    inner = f"""SELECT
       parent_asin,
       trim(regexp_replace(title, {ctrl}, ' ', 'g')) AS title,
       main_category,
       categories,
       store,
       average_rating,
       rating_number,
       CASE WHEN {parseable}
            THEN TRY_CAST(regexp_replace({price_raw}, '^\\$', '') AS DOUBLE)
            END AS price_usd,
       lower(trim(regexp_replace(COALESCE({brand_b}, {brand_m}), {ctrl}, ' ', 'g')))
            AS _brand_clean,
       ({price_raw} IS NOT NULL AND {price_raw} <> '' AND NOT COALESCE({parseable}, FALSE))
            AS _price_unparseable,
       {java_only} AS _price_regex_java_only,
       CASE WHEN {brand_b} IS NOT NULL THEN 'Brand'
            WHEN {brand_m} IS NOT NULL THEN 'Manufacturer'
            ELSE 'none' END AS _brand_source
     FROM {source}"""

    return f"""SELECT
       parent_asin, title, main_category, categories, store,
       average_rating, rating_number, price_usd,
       CASE WHEN _brand_clean IS NULL OR _brand_clean = '' THEN 'unknown'
            ELSE _brand_clean END AS brand_norm,
       _price_unparseable, _price_regex_java_only, _brand_source
     FROM ({inner})"""


def build_interactions_sql(source: str) -> str:
    """Typed silver-interactions projection from a bronze.reviews relation.

    Port of ``features.silver.transform_interactions``: ``ts`` is
    ``timestamp_millis(timestamp)``; the review ``title`` is dropped.
    """
    return f"""SELECT
       user_id,
       parent_asin,
       asin,
       rating,
       epoch_ms("timestamp") AS ts,
       helpful_vote,
       verified_purchase
     FROM {source}"""


def keep_latest_sql(source: str) -> str:
    """One row per ``(user_id, parent_asin)``: latest ``ts``.

    Spark breaks ties on ``xxhash64`` of all columns; that hash is not
    reproducible here, so ties are broken by an explicit total order on the
    remaining columns. After the exact-duplicate drop those five columns are
    unique within a partition, so the order is total and the result is
    deterministic — but it is NOT the same survivor Spark picks inside a tie
    group (see module docstring, divergence 1). Counts are unaffected.
    """
    tiebreak = ", ".join(
        f"{_ident(c)} ASC NULLS LAST"
        for c in SILVER_INTERACTION_COLS
        if c not in ("user_id", "parent_asin", "ts")
    )
    cols = ", ".join(_ident(c) for c in SILVER_INTERACTION_COLS)
    return f"""SELECT {cols}
     FROM ({source})
     QUALIFY row_number() OVER (
         PARTITION BY user_id, parent_asin
         ORDER BY ts DESC, {tiebreak}
     ) = 1"""


def tie_group_sql(source: str) -> str:
    """``(tie_groups, tie_group_rows)`` — partitions whose max-``ts`` row is not
    unique, i.e. exactly where the tie-break choice can change survivor content."""
    return f"""WITH d AS (SELECT * FROM ({source})),
     per_ts AS (SELECT user_id, parent_asin, ts, count(*) AS n FROM d GROUP BY 1, 2, 3),
     mx AS (SELECT user_id, parent_asin, max(ts) AS mts FROM d GROUP BY 1, 2)
     SELECT COALESCE(count(*), 0) AS tie_groups, COALESCE(sum(n), 0) AS tie_group_rows
     FROM per_ts JOIN mx USING (user_id, parent_asin)
     WHERE per_ts.ts = mx.mts AND n > 1"""


# --- Bronze access -----------------------------------------------------------


def _table_dir(warehouse: str | Path, table: str) -> Path:
    """``local.bronze.items`` -> ``<warehouse>/bronze/items`` (Hadoop catalog)."""
    return Path(warehouse).joinpath(*table.split(".")[1:])


def resolve_table(warehouse: str | Path, table: str) -> dict:
    """``{table, dir, metadata_json, snapshot_id}`` for a Hadoop-catalog table.

    The snapshot id comes from ``eval.runlog.iceberg_snapshot_id`` (the existing
    JVM-free version-hint reader) — never re-implemented here.
    """
    meta_dir = _table_dir(warehouse, table) / "metadata"
    version = int((meta_dir / "version-hint.text").read_text().strip())
    return {
        "table": table,
        "dir": str(_table_dir(warehouse, table)),
        "metadata_json": str(meta_dir / f"v{version}.metadata.json"),
        "snapshot_id": runlog.iceberg_snapshot_id(warehouse, table),
    }


def _quote_path(path: str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def iceberg_relation(ref: dict) -> str:
    return f"iceberg_scan({_quote_path(ref['dir'])}, allow_moved_paths=true)"


def parquet_fallback_relation(ref: dict) -> str:
    """``read_parquet([...])`` over the CURRENT snapshot's data files.

    Fallback for hosts where the DuckDB ``iceberg`` extension cannot be
    installed or cannot read the Hadoop layout (plan risk 8). File enumeration
    is unambiguous: the current snapshot is a full overwrite.
    """
    from pyiceberg.table import StaticTable

    table = StaticTable.from_metadata(ref["metadata_json"])
    files = sorted(task.file.file_path for task in table.scan().plan_files())
    if not files:
        raise RuntimeError(f"{ref['table']}: current snapshot lists no data files")
    cleaned = [f[len("file://") :] if f.startswith("file://") else f for f in files]
    return "read_parquet([" + ", ".join(_quote_path(f) for f in cleaned) + "])"


def source_relation(ref: dict, reader: str) -> str:
    if reader == "iceberg":
        return iceberg_relation(ref)
    if reader == "parquet-fallback":
        return parquet_fallback_relation(ref)
    raise ValueError(f"unknown reader {reader!r}")


# --- Waterfall accounting ----------------------------------------------------


_WATERFALL_KEYS = (
    "input_rows",
    "kept",
    "quarantined",
    "exact_duplicate",
    "superseded_by_later_review",
)


def assert_conservation(summary: dict) -> None:
    """``input_rows == kept + Σquarantined + exact_duplicate + superseded``.

    Same assertion, same message shape as ``features.silver._assert_conservation``.
    """
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


def assert_parity(
    summary: dict, expected: dict | None, expected_measures: dict | None = None
) -> None:
    """Hard parity gate against the recorded Spark waterfall integers (and, where
    the ledger recorded them, the transform's measures)."""
    if expected is None:
        return
    diffs = [
        f"{key}: duckdb={summary[key]!r} spark={expected[key]!r}"
        for key in _WATERFALL_KEYS
        if summary[key] != expected[key]
    ]
    if diffs:
        raise RuntimeError(
            f"waterfall parity failed for {summary['table']} vs the recorded Spark "
            f"build: " + "; ".join(diffs)
        )
    measure_diffs = [
        f"{key}: duckdb={summary['measures'].get(key)!r} spark={value!r}"
        for key, value in (expected_measures or {}).items()
        if summary["measures"].get(key) != value
    ]
    if measure_diffs:
        raise RuntimeError(
            f"measure parity failed for {summary['table']} vs the recorded Spark "
            f"contract ledger: " + "; ".join(measure_diffs)
        )


# --- Builds ------------------------------------------------------------------


class _Stopwatch:
    """Accumulates named stage timings and one total."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    def stage(self, name: str, fn):
        t = time.perf_counter()
        out = fn()
        self.stages[name] = round(time.perf_counter() - t, 3)
        return out

    @property
    def total(self) -> float:
        return round(time.perf_counter() - self._t0, 3)


def build_items(con, source: str, out_dir: Path, contract: Contract | None = None) -> dict:
    """DuckDB port of ``features.silver.build_items``. Returns a summary dict
    shaped exactly like the ``data/build_summary.jsonl`` line (plus extras)."""
    contract = contract or load_contract(ITEMS_CONTRACT)
    gate_cols, check_ids = gate_columns_sql(contract)
    projection = build_items_sql(source)
    cols = ", ".join(_ident(c) for c in SILVER_ITEM_COLS)
    sw = _Stopwatch()

    sw.stage(
        "scan_project_gate",
        lambda: con.execute(
            f"CREATE OR REPLACE TEMP TABLE {_T_GATED} AS "
            f"SELECT {cols}, _price_unparseable, _price_regex_java_only, _brand_source,\n"
            f"       {gate_cols}\n"
            f"FROM ({projection})"
        ),
    )

    def _stats():
        per_check = ", ".join(
            f"count(*) FILTER (WHERE {_ident('v__' + cid)}) AS {_ident(cid)}" for cid in check_ids
        )
        row = con.execute(
            f"SELECT count(*) AS input_rows, "
            f"count(*) FILTER (WHERE primary_reason IS NOT NULL) AS quarantined_rows, "
            f"count(*) FILTER (WHERE _price_unparseable) AS price_unparseable, "
            f"count(*) FILTER (WHERE _price_regex_java_only) AS price_regex_java_only, "
            f"count(*) FILTER (WHERE _brand_source = 'Brand') AS from_brand, "
            f"count(*) FILTER (WHERE _brand_source = 'Manufacturer') AS from_manufacturer, "
            f"count(*) FILTER (WHERE _brand_source = 'none') AS from_none, "
            f"count(*) FILTER (WHERE brand_norm = 'unknown') AS brand_unknown, "
            f"{per_check} FROM {_T_GATED}"
        ).fetchdf().iloc[0]
        breakdown = dict(
            con.execute(
                f"SELECT primary_reason, count(*) FROM {_T_GATED} "
                f"WHERE primary_reason IS NOT NULL GROUP BY 1"
            ).fetchall()
        )
        return row, {str(k): int(v) for k, v in breakdown.items()}

    row, quarantined = sw.stage("gate_stats", _stats)

    sw.stage(
        "write_quarantine",
        lambda: con.execute(
            f"COPY (SELECT {cols}, violation_reasons, primary_reason FROM {_T_GATED} "
            f"WHERE primary_reason IS NOT NULL) "
            f"TO {_quote_path(str(out_dir / 'quarantine_items.parquet'))} (FORMAT PARQUET)"
        ),
    )
    sw.stage(
        "write_silver",
        lambda: con.execute(
            f"COPY (SELECT {cols} FROM {_T_GATED} WHERE primary_reason IS NULL) "
            f"TO {_quote_path(str(out_dir / 'silver_items.parquet'))} (FORMAT PARQUET)"
        ),
    )
    total_s = sw.total
    con.execute(f"DROP TABLE IF EXISTS {_T_GATED}")

    input_rows = int(row["input_rows"])
    summary = {
        "table": "items",
        "input_rows": input_rows,
        "kept": input_rows - int(row["quarantined_rows"]),
        "quarantined": quarantined,
        "exact_duplicate": 0,
        "superseded_by_later_review": 0,
        "wall_clock_s": total_s,
        "stages_s": sw.stages,
        "violation_counts": {cid: int(row[cid]) for cid in check_ids},
        "measures": {
            "price_unparseable": int(row["price_unparseable"]),
            "price_regex_java_only": int(row["price_regex_java_only"]),
            "brand_from_brand": int(row["from_brand"]),
            "brand_from_manufacturer": int(row["from_manufacturer"]),
            "brand_from_none": int(row["from_none"]),
            "brand_unknown": int(row["brand_unknown"]),
        },
    }
    assert_conservation(summary)
    return summary


def build_interactions(
    con, source: str, out_dir: Path, contract: Contract | None = None
) -> dict:
    """DuckDB port of ``features.silver.build_interactions`` (gate → exact-dup →
    keep-latest, D2), with the tie-group measurement the port owes the reader."""
    contract = contract or load_contract(INTERACTIONS_CONTRACT)
    gate_cols, check_ids = gate_columns_sql(contract)
    projection = build_interactions_sql(source)
    cols = ", ".join(_ident(c) for c in SILVER_INTERACTION_COLS)
    sw = _Stopwatch()

    sw.stage(
        "scan_project_gate",
        lambda: con.execute(
            f"CREATE OR REPLACE TEMP TABLE {_T_GATED} AS "
            f"SELECT {cols},\n       {gate_cols}\nFROM ({projection})"
        ),
    )

    def _stats():
        per_check = ", ".join(
            f"count(*) FILTER (WHERE {_ident('v__' + cid)}) AS {_ident(cid)}" for cid in check_ids
        )
        row = con.execute(
            f"SELECT count(*) AS input_rows, "
            f"count(*) FILTER (WHERE primary_reason IS NOT NULL) AS quarantined_rows, "
            f"{per_check} FROM {_T_GATED}"
        ).fetchdf().iloc[0]
        breakdown = dict(
            con.execute(
                f"SELECT primary_reason, count(*) FROM {_T_GATED} "
                f"WHERE primary_reason IS NOT NULL GROUP BY 1"
            ).fetchall()
        )
        return row, {str(k): int(v) for k, v in breakdown.items()}

    row, quarantined = sw.stage("gate_stats", _stats)

    sw.stage(
        "write_quarantine",
        lambda: con.execute(
            f"COPY (SELECT {cols}, violation_reasons, primary_reason FROM {_T_GATED} "
            f"WHERE primary_reason IS NOT NULL) "
            f"TO {_quote_path(str(out_dir / 'quarantine_interactions.parquet'))} (FORMAT PARQUET)"
        ),
    )
    deduped_n = sw.stage(
        "exact_duplicate_drop",
        lambda: con.execute(
            f"CREATE OR REPLACE TEMP TABLE {_T_DEDUPED} AS "
            f"SELECT DISTINCT {cols} FROM {_T_GATED} WHERE primary_reason IS NULL"
        ).execute(f"SELECT count(*) FROM {_T_DEDUPED}").fetchone()[0],
    )
    con.execute(f"DROP TABLE IF EXISTS {_T_GATED}")
    final_n = sw.stage(
        "keep_latest",
        lambda: con.execute(
            f"CREATE OR REPLACE TEMP TABLE {_T_FINAL} AS "
            f"{keep_latest_sql('SELECT * FROM ' + _T_DEDUPED)}"
        ).execute(f"SELECT count(*) FROM {_T_FINAL}").fetchone()[0],
    )
    sw.stage(
        "write_silver",
        lambda: con.execute(
            f"COPY (SELECT {cols} FROM {_T_FINAL}) "
            f"TO {_quote_path(str(out_dir / 'silver_interactions.parquet'))} (FORMAT PARQUET)"
        ),
    )
    total_s = sw.total

    # Measured OUTSIDE the timed build: honesty instrumentation, not pipeline work.
    tie_groups, tie_group_rows = con.execute(
        tie_group_sql(f"SELECT * FROM {_T_DEDUPED}")
    ).fetchone()
    # Free both 43M-row temp tables before any --content-parity EXCEPT ALL runs:
    # the published silver is on disk now, and the memory budget is 12GB.
    con.execute(f"DROP TABLE IF EXISTS {_T_DEDUPED}")
    con.execute(f"DROP TABLE IF EXISTS {_T_FINAL}")

    input_rows = int(row["input_rows"])
    kept_after_gate = input_rows - int(row["quarantined_rows"])
    summary = {
        "table": "interactions",
        "input_rows": input_rows,
        "kept": int(final_n),
        "quarantined": quarantined,
        "exact_duplicate": kept_after_gate - int(deduped_n),
        "superseded_by_later_review": int(deduped_n) - int(final_n),
        "wall_clock_s": total_s,
        "stages_s": sw.stages,
        "violation_counts": {cid: int(row[cid]) for cid in check_ids},
        "measures": {
            "kept_after_gate": kept_after_gate,
            "distinct_rows": int(deduped_n),
            "tie_groups": int(tie_groups),
            "tie_group_rows": int(tie_group_rows),
        },
    }
    assert_conservation(summary)
    return summary


# --- Content parity ----------------------------------------------------------


def live_select_sql(relation: str) -> str:
    """Select the seven silver columns from the LIVE ``local.silver.interactions``.

    Spark writes its ``timestamp`` as Iceberg ``timestamptz`` (TIMESTAMP_LTZ is
    Spark's default), while this port's ``epoch_ms`` produces a naive TIMESTAMP.
    The session time zone is pinned to UTC in :func:`connect`, so casting the
    live column to naive TIMESTAMP is instant-preserving and lossless — and it
    is what makes the two frames set-comparable at all.
    """
    projected = ", ".join(
        f"CAST({_ident(c)} AS TIMESTAMP) AS {_ident(c)}" if c == "ts" else _ident(c)
        for c in SILVER_INTERACTION_COLS
    )
    return f"SELECT {projected} FROM {relation}"


def content_parity(con, live_relation: str, rebuilt_parquet: Path) -> dict:
    """Two-way ``EXCEPT ALL`` between the rebuilt silver and the live table.

    Bounded by the tie-group measurement: a diff above ``2 * tie_group_rows``
    means something other than the tie-break diverged.
    """
    cols = ", ".join(_ident(c) for c in SILVER_INTERACTION_COLS)
    rebuilt = f"SELECT {cols} FROM read_parquet({_quote_path(str(rebuilt_parquet))})"
    live = live_select_sql(live_relation)
    only_rebuilt = con.execute(
        f"SELECT count(*) FROM (({rebuilt}) EXCEPT ALL ({live}))"
    ).fetchone()[0]
    only_live = con.execute(
        f"SELECT count(*) FROM (({live}) EXCEPT ALL ({rebuilt}))"
    ).fetchone()[0]
    return {
        "only_in_duckdb": int(only_rebuilt),
        "only_in_spark": int(only_live),
        "diff_rows": int(only_rebuilt) + int(only_live),
    }


# --- Engine setup ------------------------------------------------------------


def connect(threads: int, memory_limit: str, temp_dir: Path, reader: str):
    """A fresh DuckDB connection configured per the plan (one per timed run)."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory={_quote_path(str(temp_dir))}")
    # Bronze `timestamp` is epoch millis; keep the session naive/UTC so
    # epoch_ms() lands on the same instants Spark's timestamp_millis produced.
    con.execute("SET TimeZone='UTC'")
    if reader == "iceberg":
        try:
            con.execute("INSTALL iceberg")
        except Exception as exc:  # already installed / offline: LOAD decides
            print(f"INSTALL iceberg: {exc}", file=sys.stderr)
        con.execute("LOAD iceberg")
    return con


def engine_info(con, threads: int, memory_limit: str, reader: str) -> dict:
    import duckdb

    info = {
        "name": "duckdb",
        "version": duckdb.__version__,
        "threads": int(threads),
        "memory_limit": memory_limit,
        "reader": reader,
    }
    if reader == "iceberg":
        row = con.execute(
            "SELECT extension_version, install_mode FROM duckdb_extensions() "
            "WHERE extension_name = 'iceberg'"
        ).fetchone()
        if row:
            info["iceberg_extension_version"] = row[0]
            info["iceberg_extension_install_mode"] = row[1]
    return info


# --- Orchestration -----------------------------------------------------------


def run_once(
    *,
    warehouse: str,
    out_dir: Path,
    reader: str,
    threads: int,
    memory_limit: str,
    temp_dir: Path,
    refs: dict,
    do_content_parity: bool,
    expected: dict | None,
) -> dict:
    """One timed run: fresh connection, both silver tables, parity asserted."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = connect(threads, memory_limit, temp_dir, reader)
    try:
        info = engine_info(con, threads, memory_limit, reader)
        items = build_items(con, source_relation(refs[BRONZE_ITEMS], reader), out_dir)
        assert_parity(
            items,
            expected["items"] if expected else None,
            EXPECTED_MEASURES.get("items") if expected else None,
        )
        interactions = build_interactions(
            con, source_relation(refs[BRONZE_REVIEWS], reader), out_dir
        )
        assert_parity(
            interactions,
            expected["interactions"] if expected else None,
            EXPECTED_MEASURES.get("interactions") if expected else None,
        )

        parity: dict | None = None
        if do_content_parity:
            live = source_relation(refs[SILVER_INTERACTIONS], reader)
            parity = content_parity(con, live, out_dir / "silver_interactions.parquet")
            bound = 2 * int(interactions["measures"]["tie_group_rows"])
            parity["tie_group_bound"] = bound
            parity["within_tie_bound"] = parity["diff_rows"] <= bound
            if not parity["within_tie_bound"]:
                raise RuntimeError(
                    "content parity outside the tie-break bound: "
                    f"{parity['diff_rows']} differing rows > 2 * tie_group_rows "
                    f"({bound}). Something other than the xxhash64 tie-break diverged."
                )
        return {"engine": info, "items": items, "interactions": interactions, "parity": parity}
    finally:
        con.close()


def _git_short_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover - git absent / not a repo
        pass
    return None


def resolve_run_id(run_id: str | None) -> tuple[str, str]:
    """``(run_id, run_ts)`` — the D3 form. Re-implemented (not imported from
    ``contracts.engine``) so this module never imports pyspark; same precedent
    and same output shape as ``demo.dq_export_job.resolve_run_id``."""
    now = datetime.now(timezone.utc)
    run_ts = now.isoformat()
    if run_id:
        return run_id, run_ts
    env = os.environ.get("RECSYS_RUN_ID")
    if env:
        return env, run_ts
    sha = _git_short_sha()
    return now.strftime("%Y%m%dT%H%M%SZ") + (f"-{sha}" if sha else ""), run_ts


def spark_reference(build_summary_path: str | Path = BUILD_SUMMARY_LOG) -> dict:
    """The recorded Spark timings, cross-checked against the ledger file.

    ``data/build_summary.jsonl`` is gitignored, so :data:`SPARK_REFERENCE` is the
    canonical copy; when the file IS present its wall clocks must agree exactly,
    else the constants have gone stale and the run stops.
    """
    block = {
        "source": str(build_summary_path),
        "items_s": list(SPARK_REFERENCE["items_s"]),
        "interactions_s": list(SPARK_REFERENCE["interactions_s"]),
        "scope_note": SPARK_SCOPE_NOTE,
        "ledger_present": False,
    }
    path = Path(build_summary_path)
    if not path.exists():
        return block
    seen: dict[str, list[float]] = {"items": [], "interactions": []}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        seen.setdefault(rec["table"], []).append(float(rec["wall_clock_s"]))
    for table in ("items", "interactions"):
        if sorted(seen.get(table, [])) != sorted(SPARK_REFERENCE[f"{table}_s"]):
            raise RuntimeError(
                f"SPARK_REFERENCE[{table}_s]={SPARK_REFERENCE[f'{table}_s']} does not "
                f"match {path} ({seen.get(table)}). Update the constant deliberately."
            )
    block["ledger_present"] = True
    return block


def build_record(
    *,
    run_id: str | None,
    runs: list[dict],
    refs: dict,
    warehouse: str,
    out_dir: Path,
    wall_clock_s: float,
    build_summary_path: str | Path = BUILD_SUMMARY_LOG,
    manifest_path: str | Path | None = None,
) -> dict:
    """Assemble the ``kind="bench"`` record (schema_version 1)."""
    rid, rts = resolve_run_id(run_id)
    git = runlog.git_info()
    last = runs[-1]
    contracts = {}
    for path in (ITEMS_CONTRACT, INTERACTIONS_CONTRACT):
        contract = load_contract(path)
        contracts[contract.table] = {
            "name": contract.name,
            "version": str(contract.version),
            "file_hash": runlog.sha256_file(path),
        }
    manifest = Path(manifest_path) if manifest_path else runlog.DEFAULT_MANIFEST_PATH
    return {
        "schema_version": runlog.record_schema_version,
        "kind": RECORD_KIND,
        "run_id": rid,
        "run_ts": rts,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "engine": last["engine"],
        "scope": DUCKDB_SCOPE_NOTE,
        "warehouse": warehouse,
        "out_dir": str(out_dir),
        "dataset_manifest_hash": runlog.dataset_manifest_hash(manifest),
        "iceberg_snapshots": {name: ref["snapshot_id"] for name, ref in refs.items()},
        "contracts": contracts,
        "n_runs": len(runs),
        "timings": {
            "items_s": [r["items"]["wall_clock_s"] for r in runs],
            "interactions_s": [r["interactions"]["wall_clock_s"] for r in runs],
            "items_stages_s": [r["items"]["stages_s"] for r in runs],
            "interactions_stages_s": [r["interactions"]["stages_s"] for r in runs],
        },
        "spark_reference": spark_reference(build_summary_path),
        "waterfall": {
            "items": {k: last["items"][k] for k in _WATERFALL_KEYS},
            "interactions": {k: last["interactions"][k] for k in _WATERFALL_KEYS},
        },
        "measures": {
            "items": last["items"]["measures"],
            "interactions": last["interactions"]["measures"],
        },
        "parity": {
            "expected_waterfall": EXPECTED_WATERFALL,
            "expected_measures": EXPECTED_MEASURES,
            "waterfall_matches_spark": True,
            "conservation_asserted": True,
            "content_parity": last["parity"],
            "tie_break_note": (
                "Spark breaks keep-latest ties on xxhash64(all columns), which "
                "DuckDB cannot reproduce; counts are tie-break-invariant, survivor "
                "content can differ by at most one row per tie group."
            ),
        },
        "hardware": runlog.hardware_string(),
        "wall_clock_s": round(wall_clock_s, 3),
    }


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="batch_recsys_lab.bench.duckdb_silver",
        description=__doc__.splitlines()[0],
    )
    ap.add_argument("--runs", type=int, default=3, help="timed runs (fresh connection each)")
    ap.add_argument("--warehouse", default=DEFAULT_WAREHOUSE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--reader",
        choices=("iceberg", "parquet-fallback"),
        default="iceberg",
        help="bronze access path: DuckDB iceberg extension, or pyiceberg file enumeration",
    )
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--memory-limit", default="12GB")
    ap.add_argument("--temp-dir", default=None, help="default: <out-dir>/tmp")
    ap.add_argument(
        "--content-parity",
        action="store_true",
        help="two-way EXCEPT ALL of the rebuilt interactions vs live local.silver.interactions",
    )
    ap.add_argument(
        "--expect",
        choices=("production", "none"),
        default="production",
        help="hard-assert the waterfall against the recorded Spark integers (default) or not",
    )
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--results-log", default=DEFAULT_RESULTS_LOG)
    ap.add_argument("--dry-run", action="store_true", help="build and print the record; append nothing")
    ap.add_argument("--append", action="store_true", help="append the record to results/runs.jsonl")
    args = ap.parse_args(argv)

    if args.append == args.dry_run:
        ap.error("pass exactly one of --dry-run / --append")
    if args.runs < 1:
        ap.error("--runs must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(args.temp_dir) if args.temp_dir else out_dir / "tmp"
    expected = EXPECTED_WATERFALL if args.expect == "production" else None

    tables = [BRONZE_ITEMS, BRONZE_REVIEWS]
    if args.content_parity:
        tables.append(SILVER_INTERACTIONS)
    refs = {t: resolve_table(args.warehouse, t) for t in tables}
    for name, ref in refs.items():
        print(f"{name}: snapshot {ref['snapshot_id']} ({ref['dir']})", file=sys.stderr)

    t0 = time.perf_counter()
    runs: list[dict] = []
    for i in range(args.runs):
        out = run_once(
            warehouse=args.warehouse,
            out_dir=out_dir,
            reader=args.reader,
            threads=args.threads,
            memory_limit=args.memory_limit,
            temp_dir=temp_dir,
            refs=refs,
            do_content_parity=args.content_parity,
            expected=expected,
        )
        runs.append(out)
        print(
            f"run {i + 1}/{args.runs}: items {out['items']['wall_clock_s']}s, "
            f"interactions {out['interactions']['wall_clock_s']}s",
            file=sys.stderr,
        )

    record = build_record(
        run_id=args.run_id,
        runs=runs,
        refs=refs,
        warehouse=args.warehouse,
        out_dir=out_dir,
        wall_clock_s=time.perf_counter() - t0,
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))

    if args.dry_run:
        print(
            f"\nDRY RUN — nothing appended. {args.results_log} is unchanged.\n"
            f"  compact line length: {len(json.dumps(record, separators=(',', ':')))} bytes\n"
            f"  to append (from a clean tree, after committing):\n"
            f"    uv run --group bench python -m batch_recsys_lab.bench.duckdb_silver "
            f"--runs {args.runs}{' --content-parity' if args.content_parity else ''} --append",
            file=sys.stderr,
        )
        return 0

    if record["git_dirty"]:
        print(
            "REFUSING TO APPEND from a dirty working tree: a bench record must name "
            "the commit that produced it (CLAUDE.md invariant #3). Commit first.",
            file=sys.stderr,
        )
        return 4

    runlog.append_record(record, args.results_log)
    print(
        f"\nappended kind={RECORD_KIND} run_id={record['run_id']} to {args.results_log}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
