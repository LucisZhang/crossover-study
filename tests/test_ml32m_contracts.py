"""ML-32M contract + frozen-split loading tests (Phase 9, T9-3a).

JVM-free except for the boundary-label check. Covers:

* every ``contracts/ml32m/*.yaml`` round-trips through the closed-vocabulary
  loader, and declares the table this lane actually builds;
* the ML-32M-specific check parameters (half-star rating domain, 1995 → Oct-2023
  ts bound, key non-null, no_dead_columns, orphan_rate vs the movies catalog);
* the ts upper bound EQUALS ``configs/splits_ml32m.yaml``'s ``test_end``, so a row
  that survives the contract always carries a non-NULL split label;
* the frozen ML-32M boundaries parse to the declared instants and label boundary
  timestamps exactly;
* **the Amazon contract directory is untouched** — ``contracts/*.yaml`` still
  globs to exactly the seven Amazon contracts, which is what keeps
  ``make contracts-audit`` (and the demo DQ export, which globs the same way)
  from trying to grade ML-32M tables.
"""

from __future__ import annotations

import os

# Force loopback before any SparkContext starts (see test_silver_build.py).
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from batch_recsys_lab.contracts.loader import load_contract
from batch_recsys_lab.features.splits import load_splits

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = REPO_ROOT / "contracts"
ML32M_CONTRACTS_DIR = CONTRACTS_DIR / "ml32m"
SPLITS_ML32M = REPO_ROOT / "configs" / "splits_ml32m.yaml"

MS = timedelta(milliseconds=1)

AMAZON_CONTRACTS = {
    "gold_interactions_5core.yaml",
    "gold_item_features.yaml",
    "gold_item_text.yaml",
    "gold_popularity.yaml",
    "gold_user_stats.yaml",
    "silver_interactions.yaml",
    "silver_items.yaml",
}

EXPECTED_ML32M = {
    "silver_ml32m_items.yaml": "local.silver_ml32m.items",
    "silver_ml32m_interactions.yaml": "local.silver_ml32m.interactions",
    "silver_ml32m_tags.yaml": "local.silver_ml32m.tags",
    "gold_ml32m_interactions_5core.yaml": "local.gold_ml32m.interactions_5core",
    "gold_ml32m_user_stats.yaml": "local.gold_ml32m.user_stats",
    "gold_ml32m_item_features.yaml": "local.gold_ml32m.item_features",
    "gold_ml32m_popularity.yaml": "local.gold_ml32m.popularity",
    # T9-3b: the content arm's item-text table (title + genres + TRAIN-cutoff
    # tags). Every contract in this directory is graded by
    # `make contracts-audit-ml32m`, so its table must be built before that
    # target runs — `make data-ml32m` orders gold-ml32m-item-text ahead of it.
    "gold_ml32m_item_text.yaml": "local.gold_ml32m.item_text",
}

HALF_STARS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def _checks(contract) -> dict:
    return {c.check_id: c for c in contract.checks}


def test_amazon_contract_glob_is_unchanged():
    # run_audit and demo/dq_export_job both do Path(dir).glob("*.yaml") (non
    # recursive). An ML-32M YAML dropped in the root would make `make
    # contracts-audit` hard-fail on a missing table and would change the DQ
    # dashboard's contract inventory. The subdirectory is what prevents that.
    assert {p.name for p in CONTRACTS_DIR.glob("*.yaml")} == AMAZON_CONTRACTS


def test_every_ml32m_contract_loads_and_names_its_table():
    found = {p.name for p in ML32M_CONTRACTS_DIR.glob("*.yaml")}
    assert found == set(EXPECTED_ML32M)
    for name, table in EXPECTED_ML32M.items():
        contract = load_contract(ML32M_CONTRACTS_DIR / name)
        assert contract.table == table
        assert contract.name == Path(name).stem
        assert contract.version == 1
        assert contract.columns and contract.checks
        # Every ML-32M table publishes the lab-wide item identity column.
        if "user_stats" not in name:
            assert any(c.name == "parent_asin" for c in contract.columns)


@pytest.mark.parametrize(
    "name",
    ["silver_ml32m_interactions.yaml", "gold_ml32m_interactions_5core.yaml"],
)
def test_interaction_contracts_declare_the_ml32m_domains(name):
    contract = load_contract(ML32M_CONTRACTS_DIR / name)
    checks = _checks(contract)

    assert list(checks["rating_domain"].values) == HALF_STARS
    assert checks["keys_non_null"].columns == ("user_id", "parent_asin", "ts")
    assert checks["ts_range"].min == "1995-01-01T00:00:00Z"
    assert checks["ts_range"].max_exclusive == "2023-11-01T00:00:00Z"
    assert checks["no_dead_columns"].kind == "no_all_null"
    # Referential health vs the movies catalog is MEASURED, never dropped.
    assert checks["item_fk"].kind == "orphan_rate"
    assert checks["item_fk"].action == "measure"
    assert checks["item_fk"].ref_table == "local.silver_ml32m.items"
    assert checks["item_fk"].ref_column == "parent_asin"


def test_items_contract_normalizes_genres_and_guards_text():
    contract = load_contract(ML32M_CONTRACTS_DIR / "silver_ml32m_items.yaml")
    dtypes = {c.name: c.dtype for c in contract.columns}
    assert dtypes == {
        "parent_asin": "string",
        "title": "string",
        "genres": "array<string>",
    }
    checks = _checks(contract)
    assert checks["key_non_null"].columns == ("parent_asin",)
    assert checks["text_hygiene"].kind == "no_control_chars"
    assert checks["no_dead_columns"].kind == "no_all_null"


def test_tags_contract_gates_the_t9_3b_text_source():
    # §8c T9-3b's content arm is title+genres+TAGS, so tags is a gated silver
    # table, not raw bronze text the model stage reaches back for.
    contract = load_contract(ML32M_CONTRACTS_DIR / "silver_ml32m_tags.yaml")
    assert {c.name: c.dtype for c in contract.columns} == {
        "user_id": "string",
        "parent_asin": "string",
        "tag": "string",
        "ts": "timestamp",
    }
    assert all(c.nullable is False for c in contract.columns)

    checks = _checks(contract)
    # The tag TEXT is part of the key: an empty tag is not an observation.
    assert checks["keys_non_null"].kind == "not_null"
    assert checks["keys_non_null"].action == "quarantine"
    assert checks["keys_non_null"].columns == ("user_id", "parent_asin", "tag", "ts")
    # Same frozen ts window as interactions.
    assert checks["ts_range"].min == "1995-01-01T00:00:00Z"
    assert checks["ts_range"].max_exclusive == "2023-11-01T00:00:00Z"
    assert checks["no_dead_columns"].kind == "no_all_null"
    # Measured (never dropped): empty-text rate and referential health.
    assert checks["empty_tag_share"].kind == "unknown_share"
    assert checks["empty_tag_share"].action == "measure"
    assert checks["empty_tag_share"].value == ""
    assert checks["item_fk"].kind == "orphan_rate"
    assert checks["item_fk"].action == "measure"
    assert checks["item_fk"].ref_table == "local.silver_ml32m.items"


def test_ts_upper_bound_equals_the_frozen_test_end():
    splits = load_splits(SPLITS_ML32M)
    for name in (
        "silver_ml32m_interactions.yaml",
        "silver_ml32m_tags.yaml",
        "gold_ml32m_interactions_5core.yaml",
    ):
        bound = _checks(load_contract(ML32M_CONTRACTS_DIR / name))["ts_range"].max_exclusive
        assert datetime.fromisoformat(bound) == splits.test_end


def test_frozen_ml32m_boundaries():
    s = load_splits(SPLITS_ML32M)
    assert s.version == 1
    assert s.frozen_at == "2026-08-19"
    assert s.train_end == datetime(2022, 6, 30, 23, 59, 59, 999000, tzinfo=timezone.utc)
    assert s.val_end == datetime(2022, 12, 31, 23, 59, 59, 999000, tzinfo=timezone.utc)
    # Covers the published snapshot: ML-32M ratings run through Oct 2023.
    assert s.test_end == datetime(2023, 11, 1, tzinfo=timezone.utc)
    for dt in (s.train_end, s.val_end, s.test_end):
        assert dt.tzinfo is not None and dt.utcoffset() == timedelta(0)
    # The Amazon split is a different frozen file and must not have moved.
    amazon = load_splits()
    assert amazon.test_end == datetime(2023, 10, 1, tzinfo=timezone.utc)


@pytest.mark.spark
def test_ml32m_split_label_boundaries_exact(spark):
    s = load_splits(SPLITS_ML32M)
    cases = [
        ("movielens_epoch", datetime(1995, 1, 9, tzinfo=timezone.utc), "train"),
        ("at_train_end", s.train_end, "train"),
        ("train_end_plus_1ms", s.train_end + MS, "val"),
        ("at_val_end", s.val_end, "val"),
        ("val_end_plus_1ms", s.val_end + MS, "test"),
        ("oct_2023", datetime(2023, 10, 15, tzinfo=timezone.utc), "test"),
        ("at_test_end", s.test_end, None),  # out of range post-contract
    ]
    df = (
        spark.createDataFrame(
            [(name, ts) for name, ts, _ in cases], "name string, ts timestamp"
        )
        .withColumn("label", s.split_label("ts"))
        .withColumn("oor", s.out_of_range("ts"))
    )
    got = {r["name"]: (r["label"], r["oor"]) for r in df.collect()}
    for name, _, expected in cases:
        assert got[name][0] == expected, f"{name}: expected {expected}, got {got[name][0]}"
    assert got["at_test_end"][1] is True
    assert got["oct_2023"][1] is False
