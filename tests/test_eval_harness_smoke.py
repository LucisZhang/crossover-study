"""Eval-harness CI smoke test (Phase 2, T5 — the plan's acceptance vehicle).

The bundled ~50k-row fixture's 5-core can legitimately be EMPTY (5-core pruning
removes every interaction below the k-core threshold in a tiny sample), so it is
not a usable harness substrate. Instead this test builds a small, deterministic
synthetic ``gold`` layer (~200 users x 50 items) directly in the session's tmp
Iceberg warehouse — interactions straddling the frozen TRAIN/VAL/TEST boundaries
of ``configs/splits.yaml`` — extracts it to a tmp snapshot-keyed cache (exactly
as ``test_eval_extract.py`` does, with explicit table names), then runs the full
config -> JSONL harness for popularity and random.

Because the tmp warehouse is LIVE, ``allow_stale=False`` must PASS (the
stale-cache guard is genuinely exercised: cache snapshot IDs are re-verified
against the on-disk Iceberg metadata via the pure-numpy resolver). An ORACLE
user is constructed so its single VAL GT item is the globally most-popular item
it has never seen in TRAIN — popularity must rank it #1, giving NDCG@10 == 1.0.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

import json
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.compare import compare
from batch_recsys_lab.eval.extract import extract
from batch_recsys_lab.eval.harness import run_eval
from batch_recsys_lab.features.splits import load_splits

pytestmark = pytest.mark.spark

UTC = timezone.utc
DAY = timedelta(days=1)

FIVE_CORE_DDL = "user_id string, parent_asin string, ts timestamp, rating double"
USER_STATS_DDL = (
    "user_id string, n_total long, n_train long, n_val long, n_test long, "
    "first_ts timestamp, last_ts timestamp, tenure_days long"
)
ITEM_FEATURES_DDL = (
    "parent_asin string, title string, brand_norm string, price_usd double, "
    "main_category string, categories array<string>, average_rating double, "
    "rating_number long"
)
POPULARITY_DDL = (
    "as_of timestamp, window_days int, parent_asin string, n_interactions long, "
    "n_unique_users long"
)

FIVE_CORE = "local.gold.five_core_smoke"
USER_STATS = "local.gold.user_stats_smoke"
ITEM_FEATURES = "local.gold.item_features_smoke"
POPULARITY = "local.gold.popularity_smoke"

N_USERS = 200
N_ITEMS = 50
WINDOWS = (0, 30, 90, 365)


def _write(spark, rows, ddl, table):
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.gold")
    spark.createDataFrame(rows, ddl).writeTo(table).createOrReplace()


def _stamp_contract(spark, table, name, version):
    spark.sql(
        f"ALTER TABLE {table} SET TBLPROPERTIES "
        f"('contracts.name'='{name}', 'contracts.version'='{version}')"
    )


def _n_train_for(u: int, rng: np.random.Generator) -> int:
    """Span all five segments deterministically."""
    if u < 15:
        return 0
    if u < 70:
        return int(rng.integers(1, 5))
    if u < 130:
        return int(rng.integers(5, 10))
    if u < 170:
        return int(rng.integers(10, 20))
    return int(rng.integers(20, 25))


@pytest.fixture()
def synthetic_gold(spark):
    """~200 users x 50 items straddling TRAIN/VAL/TEST. Item ``I00`` is the global
    popularity max; user ``U000`` (0 TRAIN history) has VAL GT == ``I00`` -> the
    oracle whose popularity NDCG@10 must be exactly 1.0."""
    s = load_splits()
    train_ts = s.train_end - 10 * DAY
    val_ts = s.train_end + 30 * DAY
    test_ts = s.val_end + 30 * DAY

    items = [f"I{j:02d}" for j in range(N_ITEMS)]  # sorted catalog order I00..I49
    rng = np.random.default_rng(2026)

    five_core_rows = []
    user_stats_rows = []
    for u in range(N_USERS):
        uid = f"U{u:03d}"
        if u == 0:
            # Oracle: no TRAIN, VAL GT = I00 (global pop max), TEST GT = I01.
            train_idx = np.array([], dtype=int)
            val_item, test_item = 0, 1
        else:
            kt = _n_train_for(u, rng)
            perm = rng.permutation(N_ITEMS)
            train_idx = perm[:kt]
            val_item = int(perm[kt])
            test_item = int(perm[kt + 1])

        for it in train_idx:
            rating = float((int(it) % 5) + 1)
            five_core_rows.append((uid, items[int(it)], train_ts, rating))
        five_core_rows.append((uid, items[val_item], val_ts, float((val_item % 5) + 1)))
        five_core_rows.append((uid, items[test_item], test_ts, float((test_item % 5) + 1)))

        n_tr = int(len(train_idx))
        first_ts = train_ts if n_tr > 0 else val_ts
        user_stats_rows.append((uid, n_tr + 2, n_tr, 1, 1, first_ts, test_ts, 100))

    _write(spark, five_core_rows, FIVE_CORE_DDL, FIVE_CORE)
    _write(spark, user_stats_rows, USER_STATS_DDL, USER_STATS)

    item_features_rows = [
        (items[j], f"title{j}", "acme", 9.99, ["Cat A", "Cat B", "Cat C"][j % 3], ["c"], 4.0, 10)
        for j in range(N_ITEMS)
    ]
    _write(spark, item_features_rows, ITEM_FEATURES_DDL, ITEM_FEATURES)

    # Strictly decreasing popularity: I00 -> 50 (max), ..., I49 -> 1.
    popularity_rows = []
    for w in WINDOWS:
        for j in range(N_ITEMS):
            n_int = N_ITEMS - j
            popularity_rows.append((s.train_end, w, items[j], n_int, min(n_int, N_USERS)))
    _write(spark, popularity_rows, POPULARITY_DDL, POPULARITY)

    for t, name in (
        (FIVE_CORE, "gold_interactions_5core"),
        (USER_STATS, "gold_user_stats"),
        (ITEM_FEATURES, "gold_item_features"),
        (POPULARITY, "gold_popularity"),
    ):
        _stamp_contract(spark, t, name, "1")
    return s


def _tables_block():
    return {
        "five_core": FIVE_CORE,
        "user_stats": USER_STATS,
        "item_features": ITEM_FEATURES,
        "popularity": POPULARITY,
    }


def _base_cfg(model_block, split, seeds, cache_dir, warehouse, per_user_dir):
    return {
        "kind": "eval",
        "model": model_block,
        "protocol": {
            "eval_split": split,
            "knowledge_cutoff": "train_end",
            "k_list": [10, 20, 50],
            "batch_size": 64,
        },
        "bootstrap": {"n_resamples": 200, "seed": 20260805},
        "seeds": seeds,
        "tables": _tables_block(),
        "cache_dir": str(cache_dir),
        "warehouse": str(warehouse),
        "per_user_dir": str(per_user_dir),
    }


def _write_cfg(cfg, path):
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _run(cfg, tmp_path, name, results, run_id, allow_stale=False):
    cfg_path = _write_cfg(cfg, tmp_path / f"{name}.yaml")
    return run_eval(
        cfg, config_path=cfg_path, results_path=results, allow_stale=allow_stale, run_id=run_id
    )


def test_dirty_from_porcelain_excludes_run_outputs():
    """A tree whose ONLY changes are results/runs.jsonl / EXPERIMENT_LOG.md is
    treated as clean; any other change still counts as dirty."""
    assert runlog._dirty_from_porcelain([]) is False
    assert runlog._dirty_from_porcelain(["?? results/runs.jsonl"]) is False
    assert runlog._dirty_from_porcelain([" M EXPERIMENT_LOG.md"]) is False
    assert (
        runlog._dirty_from_porcelain(
            ["?? results/runs.jsonl", " M EXPERIMENT_LOG.md"]
        )
        is False
    )
    assert runlog._dirty_from_porcelain([" M src/batch_recsys_lab/eval/runlog.py"]) is True
    assert (
        runlog._dirty_from_porcelain(
            ["?? results/runs.jsonl", " M some/other/file.py"]
        )
        is True
    )


def test_harness_smoke(spark, synthetic_gold, tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    summary = extract(
        spark,
        out=cache_root,
        five_core_table=FIVE_CORE,
        user_stats_table=USER_STATS,
        item_features_table=ITEM_FEATURES,
        popularity_table=POPULARITY,
    )
    assert summary["status"] == "built"

    # The active session's warehouse is whatever the FIRST get_spark call in the
    # process pinned (session-scoped fixture; another spark test may have started
    # it) — spark.conf can report a stale value. Derive the true warehouse root
    # from an on-disk Iceberg metadata path so the stale-cache guard reads the
    # same metadata the tables were actually written to. Layout:
    # <warehouse>/gold/<table>/metadata/vN.metadata.json.
    meta_file = (
        spark.sql(f"SELECT file FROM {FIVE_CORE}.metadata_log_entries ORDER BY timestamp DESC LIMIT 1")
        .first()[0]
    )
    for scheme in ("file://", "file:"):
        if meta_file.startswith(scheme):
            meta_file = meta_file[len(scheme):]
            break
    # .../gold/five_core_smoke/metadata/vN.metadata.json -> parents[3] == <warehouse>
    warehouse = str(Path(meta_file).parents[3])
    results = tmp_path / "runs.jsonl"
    per_user = tmp_path / "per_user"

    pop_model = {"name": "popularity", "params": {"as_of": "train_end", "window_days": 365}}
    rand_model = {"name": "random", "params": {}}

    pop_cfg = _base_cfg(pop_model, "val", {"model": None}, cache_root, warehouse, per_user)
    rand_cfg = _base_cfg(rand_model, "val", {"model": 13}, cache_root, warehouse, per_user)

    # --- run popularity + random; allow_stale=False must PASS (live tmp warehouse) ---
    rec_pop1 = _run(pop_cfg, tmp_path, "pop", results, run_id="pop1")
    _run(rand_cfg, tmp_path, "rand", results, run_id="rand1")

    lines = results.read_text().splitlines()
    assert len(lines) == 2
    line1_bytes = (results.read_bytes().split(b"\n")[0])

    # --- record schema completeness (persisted line, not just the return value) ---
    rec = json.loads(lines[0])
    expected_keys = {
        "schema_version", "kind", "run_id", "run_ts", "git_sha", "git_dirty",
        "config_path", "config_hash", "splits", "dataset_manifest_hash",
        "iceberg_snapshots", "contracts", "protocol", "model", "seeds",
        "metrics", "beyond_accuracy", "per_user_artifact", "wall_clock_s", "hardware",
    }
    assert expected_keys.issubset(rec.keys())
    assert rec["kind"] == "eval"
    assert rec["config_hash"].startswith("sha256:")
    assert rec["dataset_manifest_hash"].startswith("sha256:")
    assert len(rec["iceberg_snapshots"]) == 4
    assert rec["seeds"]["bootstrap"] == 20260805
    assert rec["protocol"]["exclusion"] == "train_seen"
    assert rec["protocol"]["catalog_size"] == N_ITEMS

    # metric values in range; per-segment blocks carry n_users
    for m, d in rec["metrics"]["global"].items():
        for key in ("value", "ci_lo", "ci_hi"):
            assert 0.0 <= d[key] <= 1.0, (m, key, d[key])
    assert rec["metrics"]["per_segment"], "expected non-empty per-segment blocks"
    for label, blk in rec["metrics"]["per_segment"].items():
        assert blk["n_users"] >= 1
        assert "ndcg@10" in blk
    assert rec["beyond_accuracy"]["novelty@10"]["value"] >= 0.0
    assert 0.0 <= rec["beyond_accuracy"]["coverage@10"] <= 1.0

    # --- oracle: U000's sole VAL GT is I00 (global pop max, unseen in TRAIN) -> NDCG@10 == 1 ---
    art = pq.read_table(rec_pop1["per_user_artifact"])
    row = {name: art.column(name).to_pylist() for name in art.column_names}
    u0 = row["user_id"].index("U000")
    assert row["ndcg@10"][u0] == 1.0
    assert row["recall@10"][u0] == 1.0
    assert row["mrr"][u0] == 1.0

    # --- append-only: re-run popularity -> 3 lines, first line byte-unchanged ---
    rec_pop3 = _run(pop_cfg, tmp_path, "pop", results, run_id="pop3")
    lines3 = results.read_text().splitlines()
    assert len(lines3) == 3
    assert results.read_bytes().split(b"\n")[0] == line1_bytes

    # --- compare the two popularity runs -> paired_delta record, delta == 0 exactly ---
    cmp_cfg = {
        "kind": "paired_delta",
        "a": {"run_id": "pop1"},
        "b": {"run_id": "pop3"},
        "metrics": ["recall@20", "ndcg@10", "mrr", "hitrate@10"],
        "bootstrap": {"n_resamples": 200, "seed": 20260805},
    }
    cmp_path = _write_cfg(cmp_cfg, tmp_path / "cmp.yaml")
    cmp_rec = compare(cmp_cfg, config_path=cmp_path, results_path=results)
    assert cmp_rec["kind"] == "paired_delta"
    assert cmp_rec["n_common_users"] == N_USERS
    for m, d in cmp_rec["deltas"]["global"].items():
        assert d["delta"] == 0.0, (m, d)
    # per-segment deltas also present and zero
    assert cmp_rec["deltas"]["per_segment"]
    for label, mets in cmp_rec["deltas"]["per_segment"].items():
        for m, d in mets.items():
            assert d["delta"] == 0.0

    # compare appended exactly one line (append-only)
    assert len(results.read_text().splitlines()) == 4

    # --- git-dirty TEST-refusal guard (monkeypatch git_info to a dirty tree) ---
    monkeypatch.setattr(
        runlog, "git_info", lambda: {"git_sha": "deadbeef", "git_dirty": True}
    )
    dirty_cfg = _base_cfg(pop_model, "test", {"model": None}, cache_root, warehouse, per_user)
    with pytest.raises(RuntimeError, match="TEST-split"):
        _run(dirty_cfg, tmp_path, "dirty", results, run_id="dirtytest")
    # guard fired before any append
    assert len(results.read_text().splitlines()) == 4
