"""Micro-case tests for the DuckDB silver port (Phase 7 stretch item 1, T-B3).

No Spark, no JVM, no warehouse: every case builds a tiny bronze-shaped Parquet
file in ``tmp_path`` and runs it through the SAME code paths production uses
(``build_items`` / ``build_interactions`` off ``bench.duckdb_silver``, gated by
the real ``contracts/*.yaml``). The point is that the *semantics* — quarantine
priority order, NULL handling, dedup accounting, price parsing, the ``'g'`` flag
on control-char normalization — are pinned at row level, where the 44M-row
parity run can only pin totals.

The module skips cleanly when the optional ``bench`` dependency group is not
installed (the ``embed``-group precedent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb", reason="bench group not installed (uv sync --group bench)")

from batch_recsys_lab.bench import duckdb_silver as ds  # noqa: E402
from batch_recsys_lab.contracts.loader import load_contract  # noqa: E402

CTRL_TAB = chr(9)
CTRL_DEL = chr(127)
CTRL_NUL = chr(0)

# Bronze-shaped column lists (subset of the real bronze schemas: the transform
# only ever reads these).
_REVIEW_COLS = (
    "user_id",
    "parent_asin",
    "asin",
    "rating",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
)
_ITEM_COLS = (
    "parent_asin",
    "title",
    "main_category",
    "categories",
    "store",
    "average_rating",
    "rating_number",
    "price",
    "details",
)


@pytest.fixture()
def con(tmp_path):
    c = ds.connect(
        threads=2, memory_limit="1GB", temp_dir=tmp_path / "tmp", reader="parquet-fallback"
    )
    yield c
    c.close()


def _ms(iso: str) -> int:
    """Epoch millis for a naive UTC ISO instant (bronze stores millis)."""
    from datetime import datetime, timezone

    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _review(
    user_id="U1",
    parent_asin="P1",
    asin="A1",
    rating=5.0,
    ts="2020-01-01T00:00:00",
    helpful_vote=0,
    verified_purchase=True,
):
    return {
        "user_id": user_id,
        "parent_asin": parent_asin,
        "asin": asin,
        "rating": rating,
        "timestamp": None if ts is None else _ms(ts),
        "helpful_vote": helpful_vote,
        "verified_purchase": verified_purchase,
    }


def _item(parent_asin="P1", title="Widget", price=None, details=None, **kw):
    row = {
        "parent_asin": parent_asin,
        "title": title,
        "main_category": "Electronics",
        "categories": ["Electronics"],
        "store": "Acme",
        "average_rating": 4.5,
        "rating_number": 10,
        "price": price,
        "details": details or {},
    }
    row.update(kw)
    return row


def _write_parquet(con, rows: list[dict], cols: tuple[str, ...], path: Path, ddl: str) -> str:
    """Materialize ``rows`` as a bronze-shaped Parquet file; return the relation SQL."""
    con.execute(f"CREATE OR REPLACE TEMP TABLE __src ({ddl})")
    placeholders = ", ".join("?" for _ in cols)
    con.executemany(
        f"INSERT INTO __src VALUES ({placeholders})",
        [[r[c] for c in cols] for r in rows],
    )
    con.execute(f"COPY (SELECT * FROM __src) TO '{path}' (FORMAT PARQUET)")
    con.execute("DROP TABLE __src")
    return f"read_parquet('{path}')"


_REVIEWS_DDL = (
    "user_id VARCHAR, parent_asin VARCHAR, asin VARCHAR, rating DOUBLE, "
    '"timestamp" BIGINT, helpful_vote BIGINT, verified_purchase BOOLEAN'
)
_ITEMS_DDL = (
    "parent_asin VARCHAR, title VARCHAR, main_category VARCHAR, categories VARCHAR[], "
    "store VARCHAR, average_rating DOUBLE, rating_number BIGINT, price VARCHAR, "
    "details MAP(VARCHAR, VARCHAR)"
)


def _run_interactions(con, tmp_path, rows):
    src = _write_parquet(con, rows, _REVIEW_COLS, tmp_path / "reviews.parquet", _REVIEWS_DDL)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    summary = ds.build_interactions(con, src, out)
    final = con.execute(
        f"SELECT * FROM read_parquet('{out / 'silver_interactions.parquet'}') "
        "ORDER BY user_id, parent_asin, ts"
    ).fetchall()
    quarantined = con.execute(
        f"SELECT user_id, primary_reason, violation_reasons "
        f"FROM read_parquet('{out / 'quarantine_interactions.parquet'}') ORDER BY user_id"
    ).fetchall()
    return summary, final, quarantined


def _run_items(con, tmp_path, rows):
    src = _write_parquet(con, rows, _ITEM_COLS, tmp_path / "items.parquet", _ITEMS_DDL)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    summary = ds.build_items(con, src, out)
    kept = con.execute(
        f"SELECT parent_asin, title, price_usd, brand_norm "
        f"FROM read_parquet('{out / 'silver_items.parquet'}') ORDER BY parent_asin"
    ).fetchall()
    quarantined = con.execute(
        f"SELECT parent_asin, primary_reason, violation_reasons "
        f"FROM read_parquet('{out / 'quarantine_items.parquet'}') ORDER BY parent_asin"
    ).fetchall()
    return summary, kept, quarantined


# --------------------------------------------------------------------------- #
# Constants mirrored from the Spark side must not drift (module docstring).
# --------------------------------------------------------------------------- #


def test_mirrored_constants_match_spark_side():
    from batch_recsys_lab.contracts.checks import CONTROL_CHAR_REGEX
    from batch_recsys_lab.features import silver

    assert ds.CONTROL_CHAR_REGEX == CONTROL_CHAR_REGEX
    assert ds.SILVER_ITEM_COLS == silver.SILVER_ITEM_COLS
    assert ds.SILVER_INTERACTION_COLS == silver.SILVER_INTERACTION_COLS
    assert ds._PRICE_REGEX == silver._PRICE_REGEX
    assert ds.ITEMS_CONTRACT == silver.ITEMS_CONTRACT
    assert ds.INTERACTIONS_CONTRACT == silver.INTERACTIONS_CONTRACT


def test_quarantine_priority_order_is_the_contract_order():
    """The declared check order IS the primary_reason priority (D5)."""
    ids = [c.check_id for c in ds.quarantine_checks(load_contract(ds.INTERACTIONS_CONTRACT))]
    assert ids == ["keys_non_null", "rating_domain", "ts_range"]
    item_ids = [c.check_id for c in ds.quarantine_checks(load_contract(ds.ITEMS_CONTRACT))]
    assert item_ids == ["key_non_null", "price_nonneg", "price_sentinel"]


# --------------------------------------------------------------------------- #
# Interactions: dedup accounting
# --------------------------------------------------------------------------- #


def test_exact_duplicate_collapse(con, tmp_path):
    rows = [_review(), _review(), _review(), _review(user_id="U2")]
    summary, final, _ = _run_interactions(con, tmp_path, rows)
    assert summary["input_rows"] == 4
    assert summary["exact_duplicate"] == 2
    assert summary["superseded_by_later_review"] == 0
    assert summary["kept"] == 2
    assert len(final) == 2
    ds.assert_conservation(summary)


def test_exact_duplicate_treats_nulls_as_equal(con, tmp_path):
    """Spark's dropDuplicates groups NULLs together; SELECT DISTINCT must too."""
    rows = [_review(asin=None, helpful_vote=None), _review(asin=None, helpful_vote=None)]
    summary, final, _ = _run_interactions(con, tmp_path, rows)
    assert summary["exact_duplicate"] == 1
    assert summary["kept"] == 1
    assert final[0][2] is None


def test_keep_latest_supersede(con, tmp_path):
    rows = [
        _review(ts="2020-01-01T00:00:00", asin="OLD"),
        _review(ts="2021-06-01T00:00:00", asin="NEW"),
        _review(ts="2019-01-01T00:00:00", asin="OLDER"),
    ]
    summary, final, _ = _run_interactions(con, tmp_path, rows)
    assert summary["exact_duplicate"] == 0
    assert summary["superseded_by_later_review"] == 2
    assert summary["kept"] == 1
    assert final[0][2] == "NEW"  # latest ts survives
    ds.assert_conservation(summary)


def test_keep_latest_partitions_by_user_and_item(con, tmp_path):
    rows = [
        _review(user_id="U1", parent_asin="P1", ts="2020-01-01T00:00:00"),
        _review(user_id="U1", parent_asin="P1", ts="2021-01-01T00:00:00"),
        _review(user_id="U1", parent_asin="P2", ts="2020-01-01T00:00:00"),
        _review(user_id="U2", parent_asin="P1", ts="2020-01-01T00:00:00"),
    ]
    summary, final, _ = _run_interactions(con, tmp_path, rows)
    assert summary["superseded_by_later_review"] == 1
    assert summary["kept"] == 3


def test_tie_group_measurement(con, tmp_path):
    """Rows tied at a partition's max ts are exactly where the tie-break (Spark
    xxhash64 vs this port's column order) can pick a different survivor."""
    rows = [
        _review(ts="2020-01-01T00:00:00", asin="A"),
        _review(ts="2020-01-01T00:00:00", asin="B"),
        _review(user_id="U2", ts="2020-01-01T00:00:00", asin="A"),
        _review(user_id="U2", ts="2021-01-01T00:00:00", asin="B"),
    ]
    summary, final, _ = _run_interactions(con, tmp_path, rows)
    assert summary["superseded_by_later_review"] == 2
    assert summary["measures"]["tie_groups"] == 1  # only U1/P1 is tied at max ts
    assert summary["measures"]["tie_group_rows"] == 2
    assert summary["kept"] == 2
    # Deterministic within this engine: the total order picks the smaller asin.
    assert sorted(r[2] for r in final) == ["A", "B"]


# --------------------------------------------------------------------------- #
# Interactions: quarantine reasons + priority
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "row, reason",
    [
        (_review(user_id=None), "keys_non_null"),
        (_review(parent_asin=None), "keys_non_null"),
        (_review(ts=None), "keys_non_null"),
        (_review(rating=0.0), "rating_domain"),
        (_review(rating=4.5), "rating_domain"),
        (_review(rating=None), "rating_domain"),
        (_review(ts="1995-12-31T23:59:59"), "ts_range"),
        (_review(ts="2023-10-01T00:00:00"), "ts_range"),
    ],
)
def test_each_quarantine_reason(con, tmp_path, row, reason):
    good = _review(user_id="ZZ")
    summary, final, quarantined = _run_interactions(con, tmp_path, [row, good])
    assert summary["quarantined"] == {reason: 1}
    assert summary["kept"] == 1
    assert len(quarantined) == 1
    assert quarantined[0][1] == reason
    ds.assert_conservation(summary)


def test_ts_range_bounds_are_inclusive_min_exclusive_max(con, tmp_path):
    rows = [
        _review(user_id="A", ts="1996-01-01T00:00:00"),
        _review(user_id="B", ts="2023-09-30T23:59:59"),
    ]
    summary, final, _ = _run_interactions(con, tmp_path, rows)
    assert summary["quarantined"] == {}
    assert summary["kept"] == 2


def test_primary_reason_is_the_first_violated_check(con, tmp_path):
    """A row violating all three reports keys_non_null; reasons list keeps order."""
    rows = [
        _review(user_id=None, rating=7.0, ts="2024-01-01T00:00:00"),
        _review(user_id="B", rating=7.0, ts="2024-01-01T00:00:00"),
    ]
    summary, _, quarantined = _run_interactions(con, tmp_path, rows)
    assert summary["quarantined"] == {"keys_non_null": 1, "rating_domain": 1}
    reasons = {q[0]: (q[1], list(q[2])) for q in quarantined}
    assert reasons[None] == ("keys_non_null", ["keys_non_null", "rating_domain", "ts_range"])
    assert reasons["B"] == ("rating_domain", ["rating_domain", "ts_range"])
    # Per-check counts are independent of the priority collapse.
    assert summary["violation_counts"] == {
        "keys_non_null": 1,
        "rating_domain": 2,
        "ts_range": 2,
    }
    ds.assert_conservation(summary)


def test_gate_runs_before_dedup(con, tmp_path):
    """Quarantined rows never reach the dedup stages (Spark orders it the same)."""
    rows = [_review(rating=0.0), _review(rating=0.0), _review()]
    summary, _, _ = _run_interactions(con, tmp_path, rows)
    assert summary["quarantined"] == {"rating_domain": 2}
    assert summary["exact_duplicate"] == 0
    assert summary["kept"] == 1
    ds.assert_conservation(summary)


def test_full_waterfall_conserves(con, tmp_path):
    rows = [
        _review(user_id="U1", ts="2020-01-01T00:00:00"),
        _review(user_id="U1", ts="2020-01-01T00:00:00"),  # exact dup
        _review(user_id="U1", ts="2021-01-01T00:00:00", asin="A2"),  # supersedes
        _review(user_id="U2", rating=9.0),  # quarantined
        _review(user_id="U3", parent_asin=None),  # quarantined
        _review(user_id="U4"),
    ]
    summary, final, _ = _run_interactions(con, tmp_path, rows)
    assert summary["input_rows"] == 6
    assert summary["quarantined"] == {"rating_domain": 1, "keys_non_null": 1}
    assert summary["exact_duplicate"] == 1
    assert summary["superseded_by_later_review"] == 1
    assert summary["kept"] == 2
    ds.assert_conservation(summary)
    assert len(final) == 2


def test_assert_conservation_rejects_a_broken_waterfall():
    bad = {
        "table": "interactions",
        "input_rows": 10,
        "kept": 5,
        "quarantined": {"rating_domain": 1},
        "exact_duplicate": 1,
        "superseded_by_later_review": 1,
    }
    with pytest.raises(RuntimeError, match="waterfall conservation failed"):
        ds.assert_conservation(bad)


def test_assert_parity_rejects_a_mismatch():
    summary = dict(ds.EXPECTED_WATERFALL["interactions"], table="interactions", measures={})
    ds.assert_parity(summary, ds.EXPECTED_WATERFALL["interactions"])  # exact match: silent
    summary["kept"] += 1
    with pytest.raises(RuntimeError, match="waterfall parity failed"):
        ds.assert_parity(summary, ds.EXPECTED_WATERFALL["interactions"])


def test_assert_parity_rejects_a_measure_mismatch():
    """The items waterfall is 1.61M in / 1.61M out, so the transform is pinned by
    the recorded contract-ledger measures instead."""
    summary = dict(
        ds.EXPECTED_WATERFALL["items"],
        table="items",
        measures=dict(ds.EXPECTED_MEASURES["items"]),
    )
    ds.assert_parity(summary, ds.EXPECTED_WATERFALL["items"], ds.EXPECTED_MEASURES["items"])
    summary["measures"]["price_unparseable"] += 1
    with pytest.raises(RuntimeError, match="measure parity failed"):
        ds.assert_parity(summary, ds.EXPECTED_WATERFALL["items"], ds.EXPECTED_MEASURES["items"])


# --------------------------------------------------------------------------- #
# Items: price parsing, brand, control chars
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, parsed, unparseable",
    [
        ("$12.99", 12.99, False),
        ("12.99", 12.99, False),
        (" $19 ", 19.0, False),  # trim() before the regex
        ("19", 19.0, False),
        ("12.99 - 19.99", None, True),
        ("see price in cart", None, True),
        ("$-1.0", None, True),  # a negative can never be PARSED (regex has no sign)
        ("-1.0", None, True),
        ("", None, False),  # empty is not "unparseable", it is absent
        (None, None, False),
    ],
)
def test_price_parse(con, tmp_path, raw, parsed, unparseable):
    summary, kept, quarantined = _run_items(con, tmp_path, [_item(price=raw)])
    assert kept[0][2] == parsed
    assert summary["measures"]["price_unparseable"] == int(unparseable)
    assert summary["quarantined"] == {}  # price violations are unreachable from bronze
    assert quarantined == []
    ds.assert_conservation(summary)


def test_price_gate_checks_are_unreachable_from_bronze_but_correct(con, tmp_path):
    """No bronze price string can yield a negative ``price_usd``, which is why the
    recorded items waterfall quarantines nothing. Exercise the gate SQL directly
    against a synthetic silver-shaped relation to pin the priority order anyway:
    ``-1.0`` violates BOTH price_nonneg and price_sentinel, and the contract's
    declared order makes price_nonneg the primary reason.
    """
    contract = load_contract(ds.ITEMS_CONTRACT)
    gate_cols, check_ids = ds.gate_columns_sql(contract)
    rows = con.execute(
        f"SELECT price_usd, primary_reason, violation_reasons FROM ("
        f"  SELECT * , {gate_cols} FROM (VALUES"
        f"    (NULL::VARCHAR, 5.0::DOUBLE), ('P2', -1.0), ('P3', -5.0), ('P4', 0.0), ('P5', NULL)"
        f"  ) AS t(parent_asin, price_usd)"
        f") ORDER BY parent_asin NULLS FIRST"
    ).fetchall()
    assert check_ids == ["key_non_null", "price_nonneg", "price_sentinel"]
    assert [(r[1], list(r[2] or [])) for r in rows] == [
        ("key_non_null", ["key_non_null"]),
        ("price_nonneg", ["price_nonneg", "price_sentinel"]),
        ("price_nonneg", ["price_nonneg"]),
        (None, []),
        (None, []),  # NULL price passes range and forbidden_values (D7)
    ]


def test_control_chars_normalized_everywhere_not_just_first(con, tmp_path):
    """Multi-occurrence proof for the DuckDB ``'g'`` flag: without it only the
    first control char would be replaced and the published title would still
    violate the contract's ``text_hygiene`` check."""
    dirty = f"a{CTRL_TAB}b{CTRL_TAB}c{CTRL_NUL}d{CTRL_DEL}"
    summary, kept, _ = _run_items(
        con, tmp_path, [_item(title=f" {dirty} ", details={"Brand": f"So{CTRL_TAB}n{CTRL_TAB}y"})]
    )
    assert kept[0][1] == "a b c d"  # every control char -> space, then trim
    assert kept[0][3] == "so n y"
    # The published columns must satisfy the fail-action text_hygiene check.
    contract = load_contract(ds.ITEMS_CONTRACT)
    check = next(c for c in contract.checks if c.check_id == "text_hygiene")
    violations = con.execute(
        f"SELECT count(*) FROM read_parquet('{tmp_path / 'out' / 'silver_items.parquet'}') "
        f"WHERE {ds.violation_sql(check, {})}"
    ).fetchone()[0]
    assert violations == 0
    assert summary["kept"] == 1


def test_title_of_only_control_chars_becomes_empty_string(con, tmp_path):
    _, kept, _ = _run_items(con, tmp_path, [_item(title=CTRL_TAB + CTRL_NUL)])
    assert kept[0][1] == ""


@pytest.mark.parametrize(
    "details, brand, source",
    [
        ({"Brand": "Sony"}, "sony", "Brand"),
        ({"Brand": "  SONY  "}, "sony", "Brand"),
        ({"Brand": "", "Manufacturer": "Acme"}, "acme", "Manufacturer"),
        ({"Brand": "   ", "Manufacturer": "Acme"}, "acme", "Manufacturer"),
        ({"Manufacturer": "Acme"}, "acme", "Manufacturer"),
        ({}, "unknown", "none"),
        ({"Brand": None, "Manufacturer": None}, "unknown", "none"),
        ({"Brand": CTRL_TAB}, "unknown", "Brand"),  # non-blank, normalizes to empty
    ],
)
def test_brand_norm_and_source(con, tmp_path, details, brand, source):
    summary, kept, _ = _run_items(con, tmp_path, [_item(details=details)])
    assert kept[0][3] == brand
    key = {"Brand": "brand_from_brand", "Manufacturer": "brand_from_manufacturer", "none": "brand_from_none"}[source]
    assert summary["measures"][key] == 1


def test_items_key_non_null_quarantine(con, tmp_path):
    summary, kept, quarantined = _run_items(
        con, tmp_path, [_item(parent_asin=None), _item(parent_asin="P2")]
    )
    assert summary["input_rows"] == 2
    assert summary["kept"] == 1
    assert summary["quarantined"] == {"key_non_null": 1}
    assert quarantined[0][1] == "key_non_null"
    assert summary["exact_duplicate"] == 0  # items has no dedup stage
    ds.assert_conservation(summary)


def test_price_regex_java_only_divergence_is_measured(con, tmp_path):
    """Java's ``$`` matches before a trailing newline, RE2's does not. The port
    keeps RE2 semantics (NULL price) and MEASURES the divergence instead of
    hiding it; such a row cannot move the waterfall (NULL passes the gate)."""
    summary, kept, _ = _run_items(con, tmp_path, [_item(price="12.99\n")])
    assert kept[0][2] is None
    assert summary["measures"]["price_regex_java_only"] == 1
    assert summary["quarantined"] == {}


# --------------------------------------------------------------------------- #
# Bronze access + content parity plumbing
# --------------------------------------------------------------------------- #


def test_resolve_table_reads_the_version_hint(tmp_path):
    """Snapshot ids come from the existing JVM-free runlog reader, not a copy."""
    import json

    meta = tmp_path / "bronze" / "items" / "metadata"
    meta.mkdir(parents=True)
    (meta / "version-hint.text").write_text("7\n")
    (meta / "v7.metadata.json").write_text(json.dumps({"current-snapshot-id": 4242}))
    ref = ds.resolve_table(tmp_path, "local.bronze.items")
    assert ref["snapshot_id"] == 4242
    assert Path(ref["dir"]) == tmp_path / "bronze" / "items"
    assert Path(ref["metadata_json"]).name == "v7.metadata.json"
    assert ds.iceberg_relation(ref).startswith("iceberg_scan('")
    with pytest.raises(ValueError, match="unknown reader"):
        ds.source_relation(ref, "nope")


def test_live_select_sql_casts_the_timestamptz_column(con, tmp_path):
    """Spark stores silver ``ts`` as Iceberg timestamptz; the rebuilt frame is a
    naive UTC TIMESTAMP. Without the cast the two are not set-comparable."""
    con.execute(
        "CREATE OR REPLACE TEMP TABLE live AS SELECT "
        "'U1' AS user_id, 'P1' AS parent_asin, 'A1' AS asin, 5.0::DOUBLE AS rating, "
        "TIMESTAMPTZ '2020-01-01 00:00:00+00' AS ts, 0::BIGINT AS helpful_vote, "
        "TRUE AS verified_purchase"
    )
    row = con.execute(ds.live_select_sql("live")).fetchall()[0]
    types = con.execute(f"SELECT typeof(ts) FROM ({ds.live_select_sql('live')})").fetchone()[0]
    assert types == "TIMESTAMP"
    assert row[4].tzinfo is None


def test_content_parity_counts_both_directions(con, tmp_path):
    rows = [_review(user_id="U1"), _review(user_id="U2"), _review(user_id="U3")]
    summary, _, _ = _run_interactions(con, tmp_path, rows)
    rebuilt = tmp_path / "out" / "silver_interactions.parquet"
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE live AS "
        f"SELECT user_id, parent_asin, asin, rating, ts::TIMESTAMPTZ AS ts, helpful_vote, "
        f"verified_purchase FROM read_parquet('{rebuilt}')"
    )
    assert ds.content_parity(con, "live", rebuilt)["diff_rows"] == 0
    con.execute("DELETE FROM live WHERE user_id = 'U1'")
    con.execute("INSERT INTO live SELECT 'U9', 'P9', 'A9', 5.0, ts, 0, TRUE FROM live LIMIT 1")
    parity = ds.content_parity(con, "live", rebuilt)
    assert parity == {"only_in_duckdb": 1, "only_in_spark": 1, "diff_rows": 2}


# --------------------------------------------------------------------------- #
# Record assembly
# --------------------------------------------------------------------------- #


def test_expected_measures_match_the_committed_contract_ledger():
    """EXPECTED_MEASURES['items'] must be exactly what results/dq/dq_raw.json (the
    committed contract-ledger export) recorded for the Spark build."""
    import json

    path = Path(ds._REPO_ROOT) / "results" / "dq" / "dq_raw.json"
    if not path.exists():  # pragma: no cover - published artifact always present
        pytest.skip("results/dq/dq_raw.json not published")
    rates = json.loads(path.read_text())["measured_rates"]
    expected = ds.EXPECTED_MEASURES["items"]
    assert rates["price_unparseable_share"]["rows"] == expected["price_unparseable"]
    assert rates["unknown_brand_share"]["rows"] == expected["brand_unknown"]
    details = rates["brand_from_manufacturer_share"]["details"]
    assert details["from_brand"] == expected["brand_from_brand"]
    assert details["from_manufacturer"] == expected["brand_from_manufacturer"]
    assert details["from_none"] == expected["brand_from_none"]
    # Every share is over the full items table.
    assert rates["price_unparseable_share"]["denominator"] == ds.EXPECTED_WATERFALL["items"]["input_rows"]


def test_spark_reference_matches_the_recorded_ledger(tmp_path):
    ledger = tmp_path / "build_summary.jsonl"
    lines = [
        {"table": "items", "wall_clock_s": s} for s in ds.SPARK_REFERENCE["items_s"]
    ] + [{"table": "interactions", "wall_clock_s": s} for s in ds.SPARK_REFERENCE["interactions_s"]]
    ledger.write_text("\n".join(__import__("json").dumps(x) for x in lines) + "\n")
    block = ds.spark_reference(ledger)
    assert block["ledger_present"] is True
    assert block["interactions_s"] == [474.25, 569.081, 316.413]
    assert "audit" in block["scope_note"]

    ledger.write_text('{"table": "items", "wall_clock_s": 1.0}\n')
    with pytest.raises(RuntimeError, match="does not match"):
        ds.spark_reference(ledger)


def test_spark_reference_falls_back_when_the_ledger_is_absent(tmp_path):
    block = ds.spark_reference(tmp_path / "missing.jsonl")
    assert block["ledger_present"] is False
    assert block["items_s"] == ds.SPARK_REFERENCE["items_s"]


def test_build_record_shape(con, tmp_path, monkeypatch):
    rows = [_review(), _review(user_id="U2", rating=0.0)]
    summary, _, _ = _run_interactions(con, tmp_path, rows)
    items, _, _ = _run_items(con, tmp_path, [_item()])
    runs = [
        {
            "engine": ds.engine_info(con, 2, "1GB", "parquet-fallback"),
            "items": items,
            "interactions": summary,
            "parity": None,
        }
    ]
    refs = {"local.bronze.items": {"snapshot_id": 1}, "local.bronze.reviews": {"snapshot_id": 2}}
    manifest = tmp_path / "MANIFEST.md"
    manifest.write_text("# fixture manifest\n")
    record = ds.build_record(
        run_id="TESTRUN",
        runs=runs,
        refs=refs,
        warehouse="data/warehouse",
        out_dir=tmp_path / "out",
        wall_clock_s=1.5,
        build_summary_path=tmp_path / "missing.jsonl",
        manifest_path=manifest,
    )
    assert record["kind"] == "bench"
    assert record["schema_version"] == 1
    assert record["run_id"] == "TESTRUN"
    assert record["iceberg_snapshots"] == {
        "local.bronze.items": 1,
        "local.bronze.reviews": 2,
    }
    assert set(record["contracts"]) == {"local.silver.items", "local.silver.interactions"}
    assert record["timings"]["interactions_s"] == [summary["wall_clock_s"]]
    assert record["waterfall"]["interactions"]["kept"] == 1
    assert record["parity"]["expected_waterfall"] == ds.EXPECTED_WATERFALL
    assert record["engine"]["name"] == "duckdb"
    assert record["dataset_manifest_hash"].startswith("sha256:")
    # The scope asymmetry must travel with the evidence.
    assert "orphan_rate" in record["spark_reference"]["scope_note"]
    assert record["scope"].startswith("bronze iceberg scan")


@pytest.fixture()
def tiny_warehouse(con, tmp_path, monkeypatch):
    """Point the reader plumbing at tiny local Parquet files.

    Exercises ``run_once``/``main`` — the orchestration a multi-hour production
    run cannot afford to discover a bug in — without touching data/warehouse/.
    """
    reviews = _write_parquet(
        con,
        [_review(user_id="U1"), _review(user_id="U1"), _review(user_id="U2", rating=0.0)],
        _REVIEW_COLS,
        tmp_path / "b_reviews.parquet",
        _REVIEWS_DDL,
    )
    items = _write_parquet(
        con, [_item()], _ITEM_COLS, tmp_path / "b_items.parquet", _ITEMS_DDL
    )
    live = tmp_path / "live.parquet"
    con.execute(
        f"COPY (SELECT 'U1' AS user_id, 'P1' AS parent_asin, 'A1' AS asin, 5.0::DOUBLE AS rating, "
        f"TIMESTAMPTZ '2020-01-01 00:00:00+00' AS ts, 0::BIGINT AS helpful_vote, "
        f"TRUE AS verified_purchase) TO '{live}' (FORMAT PARQUET)"
    )
    relations = {
        ds.BRONZE_REVIEWS: reviews,
        ds.BRONZE_ITEMS: items,
        ds.SILVER_INTERACTIONS: f"read_parquet('{live}')",
    }
    monkeypatch.setattr(
        ds, "resolve_table", lambda wh, t: {"table": t, "dir": str(tmp_path), "snapshot_id": 1}
    )
    monkeypatch.setattr(ds, "source_relation", lambda ref, reader: relations[ref["table"]])
    return relations


def test_run_once_orchestrates_both_builds_and_parity(tiny_warehouse, tmp_path):
    refs = {t: {"table": t, "snapshot_id": 1} for t in tiny_warehouse}
    out = ds.run_once(
        warehouse="unused",
        out_dir=tmp_path / "bench_out",
        reader="parquet-fallback",
        threads=2,
        memory_limit="1GB",
        temp_dir=tmp_path / "tmp2",
        refs=refs,
        do_content_parity=True,
        expected=None,
    )
    assert out["items"]["kept"] == 1
    assert out["interactions"]["kept"] == 1  # one dup collapsed, one quarantined
    assert out["interactions"]["quarantined"] == {"rating_domain": 1}
    assert out["parity"]["diff_rows"] == 0
    assert out["parity"]["within_tie_bound"] is True
    assert (tmp_path / "bench_out" / "silver_interactions.parquet").exists()


def test_main_dry_run_appends_nothing(tiny_warehouse, tmp_path, capsys):
    log = tmp_path / "runs.jsonl"
    rc = ds.main(
        [
            "--runs", "2",
            "--expect", "none",
            "--reader", "parquet-fallback",
            "--out-dir", str(tmp_path / "bench_out"),
            "--threads", "2",
            "--memory-limit", "1GB",
            "--content-parity",
            "--run-id", "DRY",
            "--results-log", str(log),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not log.exists()
    record = __import__("json").loads(capsys.readouterr().out)
    assert record["kind"] == "bench" and record["run_id"] == "DRY"
    assert record["n_runs"] == 2 and len(record["timings"]["interactions_s"]) == 2
    assert record["parity"]["content_parity"]["diff_rows"] == 0
    assert set(record["iceberg_snapshots"]) == {
        ds.BRONZE_ITEMS,
        ds.BRONZE_REVIEWS,
        ds.SILVER_INTERACTIONS,
    }


def test_cli_requires_exactly_one_of_dry_run_or_append():
    with pytest.raises(SystemExit):
        ds.main(["--runs", "1"])
    with pytest.raises(SystemExit):
        ds.main(["--runs", "1", "--dry-run", "--append"])
