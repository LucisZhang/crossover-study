"""Read-only Spark pull for the pick-a-shopper exhibit (Phase 6, T28).

The only Spark touch the shopper pipeline needs. It **writes nothing to the
warehouse**: it reads three gold tables *at the exact snapshot ids the headline
record was scored against* and dumps the descriptive payload the exporter needs
to gitignored ``data/demo_export/``.

Snapshot guard (runs before the JVM starts, so a drifted warehouse fails in a
second, not after a minute of Spark): the live snapshot id of every table read
here must equal ``iceberg_snapshots`` in the headline eval record. On top of
that, each read is pinned with Iceberg's ``snapshot-id`` option, so even a
concurrent commit between the guard and the scan cannot change what is
exported.

    JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home \
    PATH="$JAVA_HOME/bin:$PATH" SPARK_LOCAL_IP=127.0.0.1 \
    uv run python -m batch_recsys_lab.demo.shopper_history_job \
        --config configs/shoppers_export.yaml

(Spark 4 supports Java 17/21 only; the host default is 25. The Makefile pins
the same values for every other Spark target.)

Outputs, both gitignored:

``data/demo_export/shoppers_raw.json``
    per shopper: TRAIN timeline (``parent_asin``, ``ts``, ``rating``) and
    TEST-window rows; plus item metadata (title, brand, price, category) for the
    union of history items, TEST ground-truth items, and every item in the five
    models' top-10s.
``data/demo_export/search_items_raw.parquet``
    the top-50k items by ``pop_train_end_365`` with the same metadata columns —
    T35's semantic-search input, exported here so the demo costs one Spark job
    rather than two.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from batch_recsys_lab.demo.select_shoppers import Context, load_config
from batch_recsys_lab.eval.runlog import iceberg_snapshot_id
from batch_recsys_lab.features.splits import TEST, TRAIN, load_splits

RAW_SCHEMA_VERSION = 1
ITEM_COLS = ("parent_asin", "title", "brand_norm", "price_usd", "main_category")


# --- guards -------------------------------------------------------------------


def assert_pinned_snapshots(
    warehouse: str | Path, tables: dict[str, str], expected: dict[str, int]
) -> dict[str, int]:
    """Abort unless every table's live snapshot id equals the headline record's.

    JVM-free (reads Iceberg metadata files directly), so it runs before Spark.
    """
    live: dict[str, int] = {}
    problems: list[str] = []
    for role, table in tables.items():
        if table not in expected:
            problems.append(f"{role}: {table} is not in the headline record's iceberg_snapshots")
            continue
        try:
            sid = iceberg_snapshot_id(warehouse, table)
        except Exception as exc:  # noqa: BLE001 - reported, then re-raised as one message
            problems.append(f"{role}: cannot read snapshot id of {table} ({exc})")
            continue
        live[table] = int(sid)
        if int(sid) != int(expected[table]):
            problems.append(
                f"{role}: {table} is at snapshot {sid}, headline record pins {expected[table]}"
            )
    if problems:
        raise SystemExit(
            "SNAPSHOT GUARD FAILED — the warehouse is not at the state the headline run was "
            "scored against; refusing to export.\n  " + "\n  ".join(problems)
        )
    print("snapshot guard OK — every table read is at the headline record's pinned snapshot:")
    for table in sorted(live):
        print(f"  {table} @ {live[table]}")
    return live


# --- helpers ------------------------------------------------------------------


def top10_catalog_indices(ctx: Context) -> dict[str, dict[int, list[int]]]:
    """``model -> user_idx -> first 10 catalog indices of its stored top50``."""
    out: dict[str, dict[int, list[int]]] = {}
    for key, path in ctx.artifacts.items():
        table = pq.read_table(path, columns=["user_idx", "top50"])
        idx = table.column("user_idx").to_pylist()
        rows = table.column("top50").to_pylist()
        out[key] = {int(u): [int(v) for v in r[:10]] for u, r in zip(idx, rows)}
    return out


def _read_pinned(spark, table: str, snapshot_id: int):
    """Read an Iceberg table at an explicit snapshot — never 'whatever is live'."""
    return spark.read.option("snapshot-id", str(int(snapshot_id))).table(table)


def _iso(value) -> str | None:
    return None if value is None else value.isoformat()


# --- main ---------------------------------------------------------------------


def run(cfg: dict) -> dict:
    ctx = Context(cfg)
    work = ctx.work_dir
    selection = json.loads((work / "shopper_selection.json").read_text())
    if selection["run_ids"] != ctx.run_ids:
        raise SystemExit(
            "shopper_selection.json cites different run_ids than the demo export config — "
            "re-run select_shoppers"
        )

    tables = {role: name for role, name in cfg["tables"].items()}
    expected = ctx.snapshot_ids
    live = assert_pinned_snapshots(cfg["warehouse"], tables, expected)

    members = [m for seg in selection["by_segment"].values() for m in seg["members"]]
    by_user_id = {m["user_id"]: m for m in members}
    user_idxs = [m["user_idx"] for m in members]

    item_ids = ctx.item_ids()
    gt = ctx.test_gt()
    tops = top10_catalog_indices(ctx)

    gt_asins = {str(item_ids[i]) for u in user_idxs for i in gt.get(u, [])}
    top_asins = {
        str(item_ids[i]) for per_user in tops.values() for u in user_idxs for i in per_user[u]
    }

    splits = load_splits()

    from batch_recsys_lab.spark_session import get_spark

    spark = get_spark(app_name="t28-shopper-history", warehouse=cfg["warehouse"])
    try:
        from pyspark.sql import functions as F

        five = _read_pinned(spark, tables["five_core"], live[tables["five_core"]])
        label = splits.split_label(F.col("ts"))
        rows = (
            five.where(F.col("user_id").isin(list(by_user_id)))
            .select("user_id", "parent_asin", "ts", "rating", label.alias("split"))
            .where(F.col("split").isin([TRAIN, TEST]))
            .orderBy("user_id", "ts", "parent_asin")
            .collect()
        )

        history: dict[str, list[dict]] = {m["shopper_id"]: [] for m in members}
        test_rows: dict[str, list[dict]] = {m["shopper_id"]: [] for m in members}
        history_asins: set[str] = set()
        for r in rows:
            sid = by_user_id[r["user_id"]]["shopper_id"]
            entry = {
                "item_id": r["parent_asin"],
                "ts": _iso(r["ts"]),
                "rating": None if r["rating"] is None else float(r["rating"]),
            }
            history_asins.add(r["parent_asin"])
            (history if r["split"] == TRAIN else test_rows)[sid].append(entry)

        # user_stats is the definition of record for n_train (eval/protocol.py
        # buckets exactly this column); cross-checked against the row counts
        # above and the eval cache in the exporter.
        stats = _read_pinned(spark, tables["user_stats"], live[tables["user_stats"]])
        stat_rows = (
            stats.where(F.col("user_id").isin(list(by_user_id)))
            .select("user_id", "n_train", "n_test")
            .collect()
        )
        user_stats = {
            by_user_id[r["user_id"]]["shopper_id"]: {
                "n_train": int(r["n_train"]),
                "n_test": int(r["n_test"]),
            }
            for r in stat_rows
        }

        # item metadata: history ∪ TEST ground truth ∪ every model's top-10
        wanted = sorted(history_asins | gt_asins | top_asins)
        feats = _read_pinned(spark, tables["item_features"], live[tables["item_features"]])
        keys = spark.createDataFrame([(a,) for a in wanted], "parent_asin string")
        meta_rows = feats.join(keys, "parent_asin", "inner").select(*ITEM_COLS).collect()
        items = {
            r["parent_asin"]: {
                "title": r["title"],
                "brand_norm": r["brand_norm"],
                "price_usd": None if r["price_usd"] is None else float(r["price_usd"]),
                "main_category": r["main_category"],
            }
            for r in meta_rows
        }

        # --- T35 input: top-N items by pop_train_end_365 ---------------------
        slice_cfg = cfg.get("search_slice") or {}
        search_path = None
        if slice_cfg:
            top_n = int(slice_cfg["top_n"])
            vec_name = slice_cfg["popularity_vector"]
            pop = np.load(ctx.cache_dir / f"{vec_name}.npy", allow_pickle=False)
            if len(pop) != len(item_ids):
                raise SystemExit(f"{vec_name}.npy has {len(pop)} rows, catalog has {len(item_ids)}")
            top_n = min(top_n, len(pop))
            # ties broken by ascending catalog index — deterministic, documented
            order = np.lexsort((np.arange(len(pop)), -pop.astype(np.float64)))[:top_n]
            slice_asins = [str(item_ids[i]) for i in order]
            keys2 = spark.createDataFrame([(a,) for a in slice_asins], "parent_asin string")
            slice_rows = feats.join(keys2, "parent_asin", "inner").select(*ITEM_COLS).collect()
            got = {r["parent_asin"]: r for r in slice_rows}
            table = pa.table(
                {
                    "catalog_index": pa.array([int(i) for i in order], pa.int32()),
                    "item_id": pa.array(slice_asins, pa.string()),
                    vec_name: pa.array([float(pop[i]) for i in order], pa.float64()),
                    "title": pa.array(
                        [got[a]["title"] if a in got else None for a in slice_asins], pa.string()
                    ),
                    "brand_norm": pa.array(
                        [got[a]["brand_norm"] if a in got else None for a in slice_asins], pa.string()
                    ),
                    "price_usd": pa.array(
                        [
                            float(got[a]["price_usd"])
                            if a in got and got[a]["price_usd"] is not None
                            else None
                            for a in slice_asins
                        ],
                        pa.float64(),
                    ),
                    "main_category": pa.array(
                        [got[a]["main_category"] if a in got else None for a in slice_asins],
                        pa.string(),
                    ),
                }
            )
            search_path = work / slice_cfg.get("out", "search_items_raw.parquet")
            pq.write_table(table, search_path)
            print(
                f"wrote {search_path} ({table.num_rows} items by {vec_name}, "
                f"{len(got)} with gold.item_features metadata)"
            )
    finally:
        spark.stop()

    doc = {
        "schema_version": RAW_SCHEMA_VERSION,
        "generated_by": "batch_recsys_lab.demo.shopper_history_job",
        "headline_run_id": ctx.headline_run_id,
        "rule_id": selection["rule_id"],
        "iceberg_snapshots": {t: int(s) for t, s in sorted(live.items())},
        "tables": tables,
        "splits": {
            "version": splits.version,
            "frozen_at": splits.frozen_at,
            "train_end": splits.train_end.isoformat(),
            "val_end": splits.val_end.isoformat(),
            "test_end": splits.test_end.isoformat(),
        },
        "shopper_order": selection["shopper_order"],
        "items": dict(sorted(items.items())),
        "shoppers": {
            m["shopper_id"]: {
                "shopper_id": m["shopper_id"],
                "segment": m["segment"],
                "user_stats": user_stats[m["shopper_id"]],
                "history": history[m["shopper_id"]],
                "test_rows": test_rows[m["shopper_id"]],
            }
            for m in members
        },
    }
    out = work / "shoppers_raw.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(
        f"wrote {out} ({len(doc['shoppers'])} shoppers · "
        f"{sum(len(s['history']) for s in doc['shoppers'].values())} TRAIN rows · "
        f"{sum(len(s['test_rows']) for s in doc['shoppers'].values())} TEST rows · "
        f"{len(items)} items of {len(wanted)} requested)"
    )
    return doc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/shoppers_export.yaml")
    args = ap.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
