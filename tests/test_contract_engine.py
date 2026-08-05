"""Contract engine tests (Phase 1, T1).

Covers: every check kind fires on planted violations; clean data passes; the
multi-violation → single quarantine row with declared-order primary_reason
invariant (D5); NULL semantics for range / forbidden_values; loader round-trips
the real YAMLs and rejects unknown kinds / actions.

Spark-backed tests use the shared tmp-warehouse session (``tests/conftest.py``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from batch_recsys_lab.contracts import audit, gate, load_contract, write_dq_results
from batch_recsys_lab.contracts.loader import parse_contract

pytestmark = pytest.mark.spark

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"
INTERACTIONS_YAML = CONTRACTS_DIR / "silver_interactions.yaml"
ITEMS_YAML = CONTRACTS_DIR / "silver_items.yaml"


# --------------------------------------------------------------------------- #
# Synthetic-frame builders (explicit schemas so dtypes/nullability are stable).
# --------------------------------------------------------------------------- #

_INTERACTIONS_DDL = (
    "user_id string, parent_asin string, asin string, rating double, "
    "ts timestamp, helpful_vote long, verified_purchase boolean"
)
_ITEMS_DDL = (
    "parent_asin string, title string, main_category string, "
    "categories array<string>, store string, average_rating double, "
    "rating_number long, price_usd double, brand_norm string"
)


def _interaction(
    user_id="U1",
    parent_asin="P1",
    rating=5.0,
    ts=datetime(2022, 1, 1),
    asin="A1",
    helpful_vote=0,
    verified_purchase=True,
):
    return (user_id, parent_asin, asin, rating, ts, helpful_vote, verified_purchase)


def _item(
    parent_asin="P1",
    title="Widget",
    main_category="Electronics",
    categories=None,
    store="StoreX",
    average_rating=4.0,
    rating_number=10,
    price_usd=9.99,
    brand_norm="acme",
):
    return (
        parent_asin,
        title,
        main_category,
        categories if categories is not None else ["c1"],
        store,
        average_rating,
        rating_number,
        price_usd,
        brand_norm,
    )


def _interactions_df(spark, rows):
    return spark.createDataFrame(rows, schema=_INTERACTIONS_DDL)


def _items_df(spark, rows):
    return spark.createDataFrame(rows, schema=_ITEMS_DDL)


def _quarantine_rows(gate_result):
    return gate_result.quarantine_df.collect()


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def test_loader_roundtrips_interactions():
    c = load_contract(INTERACTIONS_YAML)
    assert c.name == "silver_interactions"
    assert c.version == 1
    assert c.table == "local.silver.interactions"
    assert [col.name for col in c.columns][:3] == ["user_id", "parent_asin", "asin"]
    # Declared check order is the fixed priority — assert it verbatim.
    assert [chk.check_id for chk in c.checks] == [
        "keys_non_null",
        "rating_domain",
        "ts_range",
        "no_dead_columns",
        "item_fk",
    ]
    keys = c.checks[0]
    assert keys.kind == "not_null" and keys.action == "quarantine"
    assert keys.columns == ("user_id", "parent_asin", "ts")
    rating = c.checks[1]
    assert rating.kind == "allowed_values"
    assert rating.values == (1.0, 2.0, 3.0, 4.0, 5.0)
    ts = c.checks[2]
    assert ts.kind == "range"
    assert ts.min == "1996-01-01T00:00:00Z"
    assert ts.max_exclusive == "2023-10-01T00:00:00Z"
    fk = c.checks[4]
    assert fk.kind == "orphan_rate" and fk.action == "measure"
    assert fk.ref_table == "local.silver.items" and fk.ref_column == "parent_asin"


def test_loader_roundtrips_items():
    c = load_contract(ITEMS_YAML)
    assert c.name == "silver_items"
    assert c.version == 1
    assert c.table == "local.silver.items"
    categories = next(col for col in c.columns if col.name == "categories")
    assert categories.dtype == "array<string>" and categories.nullable is True
    brand = next(col for col in c.columns if col.name == "brand_norm")
    assert brand.nullable is False
    assert [chk.check_id for chk in c.checks] == [
        "key_non_null",
        "price_nonneg",
        "price_sentinel",
        "text_hygiene",
        "no_dead_columns",
        "brand_unknown_share",
    ]
    sentinel = c.checks[2]
    assert sentinel.kind == "forbidden_values" and sentinel.values == (-1.0,)
    hygiene = c.checks[3]
    assert hygiene.kind == "no_control_chars" and hygiene.columns == ("title", "brand_norm")
    share = c.checks[5]
    assert share.kind == "unknown_share" and share.value == "unknown"


def test_loader_rejects_unknown_kind():
    doc = {
        "name": "x",
        "version": 1,
        "table": "local.x.y",
        "columns": [{"name": "a", "dtype": "string", "nullable": False}],
        "checks": [{"id": "bad", "kind": "not_a_real_kind", "action": "quarantine", "column": "a"}],
    }
    with pytest.raises(ValueError, match="unknown check kind"):
        parse_contract(doc)


def test_loader_rejects_unknown_action():
    doc = {
        "name": "x",
        "version": 1,
        "table": "local.x.y",
        "columns": [{"name": "a", "dtype": "string", "nullable": False}],
        "checks": [{"id": "bad", "kind": "not_null", "action": "drop", "column": "a"}],
    }
    with pytest.raises(ValueError, match="unknown action"):
        parse_contract(doc)


# --------------------------------------------------------------------------- #
# gate: each row-level kind fires
# --------------------------------------------------------------------------- #


def test_gate_not_null_fires(spark):
    contract = load_contract(INTERACTIONS_YAML)
    df = _interactions_df(spark, [_interaction(), _interaction(user_id=None)])
    gr = gate(df, contract)
    assert gr.total_rows == 2
    assert gr.quarantined_rows == 1
    assert gr.kept_df.count() == 1
    assert gr.violation_counts["keys_non_null"] == 1
    (row,) = _quarantine_rows(gr)
    assert row["primary_reason"] == "keys_non_null"
    assert row["violation_reasons"] == ["keys_non_null"]


def test_gate_allowed_values_fires_including_null(spark):
    contract = load_contract(INTERACTIONS_YAML)
    df = _interactions_df(
        spark,
        [_interaction(rating=5.0), _interaction(rating=9.0), _interaction(rating=None)],
    )
    gr = gate(df, contract)
    # 9.0 is out of domain; NULL is a violation for allowed_values.
    assert gr.violation_counts["rating_domain"] == 2
    assert gr.quarantined_rows == 2
    for row in _quarantine_rows(gr):
        assert row["primary_reason"] == "rating_domain"


def test_gate_range_fires_on_timestamp_bounds(spark):
    contract = load_contract(INTERACTIONS_YAML)
    df = _interactions_df(
        spark,
        [
            _interaction(ts=datetime(2022, 1, 1)),  # in range
            _interaction(ts=datetime(1990, 1, 1)),  # below min
            _interaction(ts=datetime(2024, 1, 1)),  # >= max_exclusive
        ],
    )
    gr = gate(df, contract)
    assert gr.violation_counts["ts_range"] == 2
    assert gr.kept_df.count() == 1


def test_gate_forbidden_values_and_numeric_range(spark):
    contract = load_contract(ITEMS_YAML)
    df = _items_df(
        spark,
        [
            _item(parent_asin="Pok", price_usd=10.0),  # clean
            _item(parent_asin="Pneg", price_usd=-5.0),  # range only
            _item(parent_asin="Psent", price_usd=-1.0),  # range AND forbidden sentinel
            _item(parent_asin="Pnull", price_usd=None),  # NULL passes both
        ],
    )
    gr = gate(df, contract)
    assert gr.violation_counts["price_nonneg"] == 2  # -5.0 and -1.0
    assert gr.violation_counts["price_sentinel"] == 1  # -1.0
    assert gr.quarantined_rows == 2
    assert gr.kept_df.count() == 2  # clean + NULL-price
    by_asin = {r["parent_asin"]: r for r in _quarantine_rows(gr)}
    # -1.0 violates both; primary_reason is the first declared (price_nonneg).
    assert by_asin["Psent"]["primary_reason"] == "price_nonneg"
    assert by_asin["Psent"]["violation_reasons"] == ["price_nonneg", "price_sentinel"]
    assert by_asin["Pneg"]["violation_reasons"] == ["price_nonneg"]


def test_gate_clean_data_passes(spark):
    contract = load_contract(INTERACTIONS_YAML)
    df = _interactions_df(
        spark,
        [_interaction(user_id="U1"), _interaction(user_id="U2", rating=3.0)],
    )
    gr = gate(df, contract)
    assert gr.quarantined_rows == 0
    assert gr.quarantine_df.count() == 0
    assert gr.kept_df.count() == 2
    assert all(v == 0 for v in gr.violation_counts.values())


def test_gate_multi_violation_quarantined_once_with_priority(spark):
    contract = load_contract(INTERACTIONS_YAML)
    # One row violating BOTH keys_non_null (null user_id) and rating_domain (9.0).
    df = _interactions_df(spark, [_interaction(user_id=None, rating=9.0)])
    gr = gate(df, contract)
    assert gr.quarantined_rows == 1
    (row,) = _quarantine_rows(gr)
    # Stored exactly once, reasons in declared order, primary = first declared.
    assert row["violation_reasons"] == ["keys_non_null", "rating_domain"]
    assert row["primary_reason"] == "keys_non_null"


def test_range_and_forbidden_values_null_passes(spark):
    contract = load_contract(ITEMS_YAML)
    # NULL price must pass BOTH price_nonneg (range) and price_sentinel (forbidden).
    df = _items_df(spark, [_item(price_usd=None)])
    gr = gate(df, contract)
    assert gr.quarantined_rows == 0
    assert gr.violation_counts["price_nonneg"] == 0
    assert gr.violation_counts["price_sentinel"] == 0


# --------------------------------------------------------------------------- #
# audit: table-level kinds fire (no_all_null, no_control_chars, unknown_share,
# orphan_rate) + zero-violation re-assertion + dq_results sink.
# --------------------------------------------------------------------------- #


def _publish(spark, namespace, table, df):
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS local.{namespace}")
    df.writeTo(f"local.{namespace}.{table}").createOrReplace()


def _by_check(results):
    return {r.check_id: r for r in results}


def test_audit_items_table_level_checks(spark):
    contract = load_contract(ITEMS_YAML)
    rows = [
        _item(parent_asin="P1", title="Clean Widget", store=None, brand_norm="acme"),
        _item(parent_asin="P2", title="BadTitle", store=None, brand_norm="acme"),
        _item(parent_asin="P3", title="Third", store=None, brand_norm="unknown"),
    ]
    _publish(spark, "silver", "items", _items_df(spark, rows))

    results = _by_check(audit(spark, contract))

    # no_control_chars: exactly one row has a control char → fail.
    hygiene = results["text_hygiene"]
    assert hygiene.check_kind == "no_control_chars"
    assert hygiene.status == "fail" and hygiene.violation_count == 1

    # no_all_null: store is entirely NULL across all rows → fail, listed dead.
    dead = results["no_dead_columns"]
    assert dead.status == "fail"
    assert "store" in dead.details and "store" in dead.details  # JSON string

    # unknown_share: one of three items is brand "unknown".
    share = results["brand_unknown_share"]
    assert share.status == "measured" and share.violation_count == 1
    assert abs(share.metric_value - (1 / 3)) < 1e-9

    # Zero-violation re-assertion of quarantine rules holds on clean-ish data.
    assert results["key_non_null"].status == "pass"
    assert results["price_nonneg"].status == "pass"
    assert results["price_sentinel"].status == "pass"


def test_audit_interactions_orphan_rate(spark):
    items_contract = load_contract(ITEMS_YAML)
    inter_contract = load_contract(INTERACTIONS_YAML)

    _publish(spark, "silver", "items", _items_df(spark, [_item(parent_asin="P1"), _item(parent_asin="P2")]))
    _publish(
        spark,
        "silver",
        "interactions",
        _interactions_df(
            spark,
            [
                _interaction(parent_asin="P1"),
                _interaction(parent_asin="P2"),
                _interaction(parent_asin="PX"),  # orphan: not in items
            ],
        ),
    )

    results = _by_check(audit(spark, inter_contract))
    fk = results["item_fk"]
    assert fk.check_kind == "orphan_rate" and fk.status == "measured"
    assert fk.violation_count == 1
    assert abs(fk.metric_value - (1 / 3)) < 1e-9
    # Row-level rules re-assert clean on this valid data.
    assert results["keys_non_null"].status == "pass"
    assert results["rating_domain"].status == "pass"
    assert results["ts_range"].status == "pass"

    # Keep items_contract referenced (audit reads it as the FK target).
    assert items_contract.table == "local.silver.items"


def test_write_dq_results_creates_and_appends(spark):
    contract = load_contract(ITEMS_YAML)
    _publish(spark, "silver", "items", _items_df(spark, [_item(parent_asin="P1")]))
    results = audit(spark, contract)
    assert results  # non-empty

    table = "local.dq.dq_results_test"
    write_dq_results(spark, results, table=table)
    first = spark.table(table).count()
    assert first == len(results)

    write_dq_results(spark, results, table=table)  # append-only
    assert spark.table(table).count() == 2 * len(results)

    cols = set(spark.table(table).columns)
    assert {"run_id", "check_id", "status", "violation_count", "metric_value", "details"} <= cols
