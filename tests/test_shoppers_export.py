"""Pick-a-shopper pipeline (Phase 6, T28): selection determinism + top10/hit mapping.

Everything here runs against a synthetic mini-repo built in ``tmp_path`` — five
fake eval records, five per-user parquets and a fake eval cache — so the tests
exercise the real modules without needing the 43.9M-row warehouse. The Spark
pull (``shopper_history_job``) is represented by a hand-built
``shoppers_raw.json``: what is under test is the *composition* logic, which is
where a wrong item id or a mis-attributed hit would come from.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from batch_recsys_lab.demo.export_shoppers import build
from batch_recsys_lab.demo.select_shoppers import (
    Context,
    select_shoppers,
    shopper_id_for,
    write_selection,
)

SEGMENTS = ("0", "1-4", "5-9", "10-19", "20+")
SEG_NTRAIN = {"0": 0, "1-4": 2, "5-9": 7, "10-19": 12, "20+": 25}
MODELS = ("blend", "pop_t12m", "als", "item_knn", "content")
COLD_MODELS = ("als", "item_knn")
MODEL_NAME = {
    "blend": "content_pop_blend",
    "pop_t12m": "popularity",
    "als": "als",
    "item_knn": "item_knn",
    "content": "content",
}
SNAPSHOT_ID = 111222333
SNAPSHOTS = {"local.gold.interactions_5core": SNAPSHOT_ID, "local.gold.item_features": 44}

N_PER_SEGMENT = 40
N_ITEMS = 60
TOP50_LEN = 12


# --- synthetic repo -----------------------------------------------------------


def _users() -> list[dict]:
    """200 users: 40 per segment, ascending user_idx, deterministic GT."""
    out = []
    for s_i, seg in enumerate(SEGMENTS):
        for j in range(N_PER_SEGMENT):
            u = s_i * N_PER_SEGMENT + j
            # every 7th user has a single GT item; the rest have two (so the
            # ">=2 TEST ground-truth items" preference has something to prefer)
            n_gt = 1 if j % 7 == 6 else 2
            out.append(
                {
                    "user_idx": u,
                    "user_id": f"USER{u:04d}",
                    "segment": seg,
                    "n_train": SEG_NTRAIN[seg],
                    # GT items live in the top half of the catalog
                    "gt": [(u * 3 + k) % (N_ITEMS // 2) for k in range(n_gt)],
                    # the first 10 users of each segment are blend hits
                    "hit": j < 10,
                }
            )
    for u in out:
        u["gt"] = sorted(set(u["gt"]))
    return out


def _top50_for(user: dict, hit: bool) -> list[int]:
    """A ranking whose first slot is a GT item when ``hit``, else all misses."""
    misses = [i for i in range(N_ITEMS) if i not in set(user["gt"])]
    if hit:
        return [user["gt"][0]] + misses[: TOP50_LEN - 1]
    return misses[:TOP50_LEN]


def _write_artifact(path: Path, users: list[dict], model: str) -> None:
    rows = []
    for u in users:
        cold = model in COLD_MODELS and u["n_train"] == 0
        # only blend's hit flag drives the curation; the other arms hit for a
        # different slice of users, so a bug that mixes arms up is visible
        hit = u["hit"] if model in ("blend", "pop_t12m") else (u["user_idx"] % 5 == 0)
        top50 = _top50_for(u, hit and not cold)
        n_hits = 0 if cold else len(set(top50[:10]) & set(u["gt"]))
        rows.append(
            {
                "user_id": u["user_id"],
                "user_idx": u["user_idx"],
                "segment": u["segment"],
                "recall@10": 0.0 if cold else n_hits / len(u["gt"]),
                "recall@20": 0.0 if cold else n_hits / len(u["gt"]),
                "ndcg@10": 0.0 if cold else (0.5 if n_hits else 0.0),
                "hitrate@10": 0.0 if cold else float(n_hits > 0),
                "novelty@10": 1.0,
                "top50": top50,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array([r["user_id"] for r in rows], pa.string()),
                "user_idx": pa.array([r["user_idx"] for r in rows], pa.int32()),
                "segment": pa.array([r["segment"] for r in rows], pa.string()),
                "recall@10": pa.array([r["recall@10"] for r in rows], pa.float64()),
                "recall@20": pa.array([r["recall@20"] for r in rows], pa.float64()),
                "ndcg@10": pa.array([r["ndcg@10"] for r in rows], pa.float64()),
                "hitrate@10": pa.array([r["hitrate@10"] for r in rows], pa.float64()),
                "novelty@10": pa.array([r["novelty@10"] for r in rows], pa.float64()),
                "top50": pa.array([r["top50"] for r in rows], pa.list_(pa.int32())),
            }
        ),
        path,
    )


def _record(model: str, users: list[dict]) -> dict:
    per_segment = {}
    for seg in SEGMENTS:
        members = [u for u in users if u["segment"] == seg]
        rate = sum(1 for u in members if u["hit"]) / len(members)
        per_segment[seg] = {
            "n_users": len(members),
            "ndcg@10": {"value": 0.01, "ci_lo": 0.009, "ci_hi": 0.011},
            "recall@20": {"value": 0.02, "ci_lo": 0.019, "ci_hi": 0.021},
            "hitrate@10": {"value": rate, "ci_lo": rate - 0.001, "ci_hi": rate + 0.001},
        }
    run_id = f"2026TEST-{model}"
    return {
        "schema_version": 1,
        "kind": "eval",
        "run_id": run_id,
        "git_sha": "0" * 40,
        "iceberg_snapshots": dict(SNAPSHOTS),
        "protocol": {"eval_split": "test", "n_users": len(users), "catalog_size": N_ITEMS},
        "model": {"name": MODEL_NAME[model], "params": {}},
        "seeds": {"bootstrap": 20260805, "model": None},
        "metrics": {
            "global": {"ndcg@10": {"value": 0.01, "ci_lo": 0.009, "ci_hi": 0.011}},
            "per_segment": per_segment,
        },
        "per_user_artifact": f"data/eval/per_user/{run_id}_{MODEL_NAME[model]}.parquet",
    }


def build_repo(root: Path, users: list[dict] | None = None) -> dict:
    """Materialize a mini-repo and return the shoppers-export config dict."""
    users = users if users is not None else _users()
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)

    records = [_record(m, users) for m in MODELS]
    (root / "results" / "runs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    for model, rec in zip(MODELS, records):
        _write_artifact(root / rec["per_user_artifact"], users, model)

    demo_cfg = {
        "runs_log": "results/runs.jsonl",
        "out_dir": "demo/data",
        "manifest": "demo/data/trace_manifest.json",
        "headline_run_id": "2026TEST-blend",
        "split": "test",
        "segments": list(SEGMENTS),
        "metrics": ["ndcg@10", "recall@20"],
        "models": [{"key": m, "label": m, "run_id": f"2026TEST-{m}"} for m in MODELS],
    }
    (root / "configs" / "demo_export.yaml").write_text(yaml.safe_dump(demo_cfg))

    cache = root / "data" / "eval" / "cache" / str(SNAPSHOT_ID)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "cache_manifest.json").write_text(json.dumps({"snapshot_ids": SNAPSHOTS}))
    pq.write_table(
        pa.table({"item_id": pa.array([f"ASIN{i:04d}" for i in range(N_ITEMS)], pa.string())}),
        cache / "item_ids.parquet",
    )
    pq.write_table(
        pa.table({"user_id": pa.array([u["user_id"] for u in users], pa.string())}),
        cache / "user_ids.parquet",
    )
    np.save(cache / "n_train.npy", np.array([u["n_train"] for u in users], dtype=np.int32))
    np.save(
        cache / "test_user_idx.npy",
        np.array([u["user_idx"] for u in users for _ in u["gt"]], dtype=np.int32),
    )
    np.save(
        cache / "test_item_idx.npy",
        np.array([g for u in users for g in u["gt"]], dtype=np.int32),
    )

    return {
        "demo_export_config": "configs/demo_export.yaml",
        "models": list(MODELS),
        "cold_collapse_models": list(COLD_MODELS),
        "cache_dir": None,
        "seed": 20260805,
        "per_segment": 6,
        "min_test_gt": 2,
        "min_blend_hits": 2,
        "min_blend_misses": 1,
        "max_attempts": 50,
        "salt_path": "data/demo_salt.txt",
        "work_dir": "data/demo_export",
        "out_dir": "demo/data",
        "manifest": "demo/data/trace_manifest.json",
    }


def make_raw(root: Path, ctx: Context, selection: dict, users: list[dict]) -> dict:
    """Stand-in for shopper_history_job's output, built from the same truth."""
    by_idx = {u["user_idx"]: u for u in users}
    item_ids = [f"ASIN{i:04d}" for i in range(N_ITEMS)]
    members = [m for seg in selection["by_segment"].values() for m in seg["members"]]
    items = {
        item_ids[i]: {
            "title": f"Item {i}",
            "brand_norm": "acme",
            "price_usd": float(i),
            "main_category": "All Electronics",
        }
        for i in range(N_ITEMS)
    }
    shoppers = {}
    for m in members:
        u = by_idx[m["user_idx"]]
        shoppers[m["shopper_id"]] = {
            "shopper_id": m["shopper_id"],
            "segment": u["segment"],
            "user_stats": {"n_train": u["n_train"], "n_test": len(u["gt"])},
            "history": [
                {
                    "item_id": item_ids[(u["user_idx"] + k) % N_ITEMS],
                    "ts": f"2021-01-{k % 28 + 1:02d}T00:00:00",
                    "rating": 5.0,
                }
                for k in range(u["n_train"])
            ],
            "test_rows": [
                {"item_id": item_ids[g], "ts": "2023-02-01T00:00:00", "rating": 4.0}
                for g in u["gt"]
            ],
        }
    return {
        "schema_version": 1,
        "headline_run_id": ctx.headline_run_id,
        "rule_id": selection["rule_id"],
        "iceberg_snapshots": dict(SNAPSHOTS),
        "shopper_order": selection["shopper_order"],
        "items": items,
        "shoppers": shoppers,
    }


@pytest.fixture
def repo(tmp_path):
    users = _users()
    cfg = build_repo(tmp_path, users)
    return tmp_path, cfg, users


def _select(root: Path, cfg: dict):
    ctx = Context(cfg, repo_root=root)
    return ctx, select_shoppers(ctx)


# --- selection rule -----------------------------------------------------------


def test_selection_shape_and_strata(repo):
    root, cfg, _ = repo
    _, sel = _select(root, cfg)
    assert len(sel["shopper_order"]) == 30
    assert len(set(sel["shopper_order"])) == 30
    for seg in SEGMENTS:
        s = sel["by_segment"][seg]
        assert len(s["members"]) == 6
        # the pre-declared predicate, satisfied by construction (v2)
        assert sum(m["blend_hit_at_10"] for m in s["members"]) == 2
        assert sum(not m["blend_hit_at_10"] for m in s["members"]) == 4
        assert s["attempts"] == 1
        assert [m["user_idx"] for m in s["members"]] == sorted(
            m["user_idx"] for m in s["members"]
        )


def test_selection_is_deterministic(repo):
    root, cfg, _ = repo
    _, first = _select(root, cfg)
    _, second = _select(root, cfg)
    assert first["shopper_order"] == second["shopper_order"]
    assert first["by_segment"] == second["by_segment"]


def test_selection_depends_on_the_seed(repo):
    root, cfg, _ = repo
    _, base = _select(root, cfg)
    _, moved = _select(root, {**cfg, "seed": 20260806})
    assert base["shopper_order"] != moved["shopper_order"]


def test_selection_prefers_users_with_two_ground_truth_items(repo):
    root, cfg, _ = repo
    _, sel = _select(root, cfg)
    for seg in SEGMENTS:
        s = sel["by_segment"][seg]
        assert not s["pool_fallback_to_all_users"]
        assert all(m["n_test_gt"] >= 2 for m in s["members"])


def test_selection_aborts_when_the_hit_stratum_is_too_small(tmp_path):
    users = _users()
    for u in users:  # only one blend hit in segment "5-9"
        if u["segment"] == "5-9":
            u["hit"] = u["user_idx"] == 80
    cfg = build_repo(tmp_path, users)
    ctx = Context(cfg, repo_root=tmp_path)
    with pytest.raises(RuntimeError, match="stratum too small"):
        select_shoppers(ctx)


def test_shopper_id_is_salted_hmac(repo):
    root, cfg, _ = repo
    ctx, sel = _select(root, cfg)
    salt = (root / "data" / "demo_salt.txt").read_text().strip().encode()
    for m in (m for s in sel["by_segment"].values() for m in s["members"]):
        assert m["shopper_id"] == shopper_id_for(salt, m["user_id"])
        assert len(m["shopper_id"]) == 12
    assert shopper_id_for(b"other-salt", "USER0001") != shopper_id_for(salt, "USER0001")
    # the mapping never leaves data/ — the id alone does not reveal the user
    assert all(m["user_id"] not in m["shopper_id"] for s in sel["by_segment"].values() for m in s["members"])


def test_selection_artifacts_are_written(repo):
    root, cfg, _ = repo
    ctx, sel = _select(root, cfg)
    json_path, map_path = write_selection(ctx, sel)
    assert json.loads(json_path.read_text())["shopper_order"] == sel["shopper_order"]
    table = pq.read_table(map_path)
    assert table.num_rows == 30
    assert set(table.column_names) >= {"shopper_id", "user_id", "user_idx", "segment"}


# --- export: top10 / hit mapping ----------------------------------------------


def _export(root: Path, cfg: dict, users: list[dict]):
    ctx, sel = _select(root, cfg)
    raw = make_raw(root, ctx, sel, users)
    writer = build(ctx, sel, raw)
    return ctx, sel, writer


def test_export_shape(repo):
    root, cfg, users = repo
    _, sel, writer = _export(root, cfg, users)
    doc = writer.document
    assert len(doc["shoppers"]) == 30
    counts: dict[str, int] = {}
    for s in doc["shoppers"].values():
        counts[s["segment"]] = counts.get(s["segment"], 0) + 1
    assert counts == {seg: 6 for seg in SEGMENTS}
    assert writer.untraced_numeric_leaves() == []


def test_top10_maps_catalog_indices_through_item_ids(repo):
    root, cfg, users = repo
    ctx, sel, writer = _export(root, cfg, users)
    doc = writer.document
    item_ids = [f"ASIN{i:04d}" for i in range(N_ITEMS)]
    members = {m["shopper_id"]: m for s in sel["by_segment"].values() for m in s["members"]}
    artifacts = {
        key: pq.read_table(path).to_pydict() for key, path in ctx.artifacts.items()
    }
    for sid, shopper in doc["shoppers"].items():
        u = members[sid]["user_idx"]
        for key, rec in shopper["recommendations"].items():
            if rec["cold_collapse"]:
                continue
            table = artifacts[key]
            row = table["user_idx"].index(u)
            expected = [int(v) for v in table["top50"][row][:10]]
            assert [e["catalog_index"] for e in rec["top10"]] == expected
            assert [e["rank"] for e in rec["top10"]] == list(range(1, 11))
            for e in rec["top10"]:
                assert e["item_id"] == item_ids[e["catalog_index"]]


def test_hit_flags_are_ground_truth_membership(repo):
    root, cfg, users = repo
    ctx, sel, writer = _export(root, cfg, users)
    doc = writer.document
    by_idx = {u["user_idx"]: u for u in users}
    members = {m["shopper_id"]: m for s in sel["by_segment"].values() for m in s["members"]}
    saw_hit = False
    for sid, shopper in doc["shoppers"].items():
        gt = set(by_idx[members[sid]["user_idx"]]["gt"])
        gt_asins = {f"ASIN{i:04d}" for i in gt}
        assert {p["item_id"] for p in shopper["test_purchases"]} == gt_asins
        for rec in shopper["recommendations"].values():
            if rec["cold_collapse"]:
                continue
            for e in rec["top10"]:
                assert e["hit"] == (e["catalog_index"] in gt)
                saw_hit |= e["hit"]
            assert (rec["hitrate@10"] == 1.0) == any(e["hit"] for e in rec["top10"])
    assert saw_hit, "fixture produced no hits at all — the test would be vacuous"


def test_cold_users_collapse_without_a_top10(repo):
    root, cfg, users = repo
    _, _, writer = _export(root, cfg, users)
    doc = writer.document
    cold_shoppers = [s for s in doc["shoppers"].values() if s["segment"] == "0"]
    assert len(cold_shoppers) == 6
    for s in cold_shoppers:
        assert s["n_train"] == 0
        for key in COLD_MODELS:
            rec = s["recommendations"][key]
            assert rec["cold_collapse"] is True
            assert "top10" not in rec
            assert rec["ndcg@10"] == 0.0 and rec["hitrate@10"] == 0.0
        for key in ("blend", "pop_t12m", "content"):
            assert s["recommendations"][key]["cold_collapse"] is False
            assert len(s["recommendations"][key]["top10"]) == 10
    for s in doc["shoppers"].values():
        if s["segment"] != "0":
            assert all(not r["cold_collapse"] for r in s["recommendations"].values())


def test_traced_leaves_cite_the_right_parquet_row(repo):
    root, cfg, users = repo
    ctx, sel, writer = _export(root, cfg, users)
    members = {m["shopper_id"]: m for s in sel["by_segment"].values() for m in s["members"]}
    per_user = [e for e in writer.entries if e["source"]["kind"] == "per_user_artifact"]
    assert per_user, "no per-user traces written"
    for e in per_user:
        sid = e["pointer"].split("/")[2]
        u = members[sid]["user_idx"]
        assert e["source"]["row_pointer"].startswith(f"user_idx={u}/")
        assert e["source"]["parquet_path"] == ctx.artifact_rel(
            e["source"]["run_id"].removeprefix("2026TEST-")
        )
    # every displayed recommendation carries its own trace
    top10_traces = [e for e in per_user if e["source"]["row_pointer"].count("/top50/")]
    assert len(top10_traces) == (30 * len(MODELS) - 6 * len(COLD_MODELS)) * 10


def test_export_rejects_a_hitrate_that_contradicts_the_ranking(repo):
    root, cfg, users = repo
    ctx, sel = _select(root, cfg)
    raw = make_raw(root, ctx, sel, users)
    # flip one recorded hitrate@10 so it disagrees with the stored top50
    path = ctx.artifacts["content"]
    table = pq.read_table(path).to_pydict()
    table["hitrate@10"] = [1.0 - v for v in table["hitrate@10"]]
    pq.write_table(pa.table(table), path)
    ctx2 = Context(cfg, repo_root=root)
    with pytest.raises(AssertionError, match="hitrate@10"):
        build(ctx2, sel, raw)


def test_export_rejects_a_mismatched_n_train(repo):
    root, cfg, users = repo
    ctx, sel = _select(root, cfg)
    raw = make_raw(root, ctx, sel, users)
    victim = sel["shopper_order"][-1]
    raw["shoppers"][victim]["user_stats"]["n_train"] += 1
    with pytest.raises(AssertionError, match="n_train disagrees"):
        build(ctx, sel, raw)


def test_export_rejects_ground_truth_drift(repo):
    root, cfg, users = repo
    ctx, sel = _select(root, cfg)
    raw = make_raw(root, ctx, sel, users)
    victim = sel["shopper_order"][0]
    raw["shoppers"][victim]["test_rows"].pop()
    with pytest.raises(AssertionError, match="TEST ground truth"):
        build(ctx, sel, raw)


def test_curation_discloses_the_real_hit_rate(repo):
    root, cfg, users = repo
    ctx, sel, writer = _export(root, cfg, users)
    doc = writer.document
    for seg in SEGMENTS:
        block = doc["curation"][seg]
        assert block["drawn"]["from_hit_stratum"] == 2
        assert block["drawn"]["from_miss_stratum"] == 4
        # traced verbatim to the blend record, not recomputed here
        recorded = ctx.runs[ctx.run_ids["blend"]]["metrics"]["per_segment"][seg]
        assert block["blend_hitrate@10"]["value"] == recorded["hitrate@10"]["value"]
        assert block["eval_users"] == recorded["n_users"]
