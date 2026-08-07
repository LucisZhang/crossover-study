"""Per-stage lineage table (Phase 5, T24).

Pure python + pyarrow over a synthetic mini-universe in tmp: fake MANIFEST,
fake ingest/build ledgers, fake Hadoop-catalog Iceberg metadata dirs (the same
``version-hint.text`` + ``vN.metadata.json`` shape the real warehouse has, so the
JVM-free readers in ``ops.snapshot_metrics`` work unchanged), a fake k-core funnel
parquet, and a fake ``runs.jsonl``. No Spark, no JVM, and nothing under the real
``data/`` or ``results/`` is read or written.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batch_recsys_lab.ops import lineage

FIVE_CORE_SNAPSHOT = 999888777
HEADLINE_RUN = "20260101T000000Z-abc1234"
REPRO_RUN = "20260102T000000Z-def5678"
GOLD_RUN = "20260101T000000Z-run2"
OLD_GOLD_RUN = "20251231T000000Z-run1"

# (table, snapshot_id, total_records, total_files_size)
TABLES = [
    ("bronze/reviews", 111, 1000, 5000),
    ("bronze/items", 112, 200, 800),
    ("silver/items", 121, 200, 700),
    ("silver/interactions", 122, 900, 4000),
    ("gold/interactions_5core", FIVE_CORE_SNAPSHOT, 500, 2000),
    ("gold/user_stats", 131, 50, 300),
    ("gold/item_features", 132, 40, 200),
    ("gold/popularity", 133, 60, 100),
    ("gold/item_text", 134, 40, 1700),
    ("dq/kcore_funnel", 141, 4, 900),
]

EXPECTED_STAGES = [
    "raw_download",
    "bronze.reviews",
    "bronze.items",
    "silver.items",
    "silver.interactions",
    "gold.interactions_5core",
    "gold.user_stats",
    "gold.item_features",
    "gold.popularity",
    "gold.item_text",
    "eval_extract_cache",
    "headline_eval",
    "reproduce_headline",
    # One row per ops RECORD, in log order — repeats and no-ops included.
    "ops.backfill",
    "ops.append[2023-07]",
    "ops.append[2023-08]",
    "ops.append[2023-09]",
    "ops.upsert",
    "ops.compact[noop]",
    "ops.expire[retain=2,deleted=0]",
    "ops.fragment[2023-06]",
    "ops.compact[30->1]",
    "ops.expire[retain=2,deleted=3]",
    "ops.expire[retain=1,deleted=30]",
]

# Stages whose wall clock is legitimately null and therefore must be footnoted.
FOOTNOTED_STAGES = {
    "raw_download",
    "gold.user_stats",
    "gold.item_features",
    "gold.popularity",
    "gold.item_text",
    "eval_extract_cache",
}


# --- synthetic universe -------------------------------------------------------


def _write_metadata(warehouse: Path, rel: str, snapshot_id: int, rows: int, size: int):
    meta_dir = warehouse / rel / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "format-version": 2,
        "table-uuid": f"uuid-{snapshot_id}",
        "current-snapshot-id": snapshot_id,
        "default-spec-id": 0,
        "partition-specs": [{"spec-id": 0, "fields": []}],
        "snapshots": [
            {
                "snapshot-id": snapshot_id,
                "parent-snapshot-id": None,
                "sequence-number": 1,
                "timestamp-ms": 1700000000000,
                "summary": {
                    "operation": "overwrite",
                    "total-records": str(rows),
                    "total-data-files": "1",
                    "total-files-size": str(size),
                },
            }
        ],
    }
    (meta_dir / "v3.metadata.json").write_text(json.dumps(doc))
    (meta_dir / "version-hint.text").write_text("3\n")


def _write_funnel(warehouse: Path):
    """Two build runs' worth of funnel rows, as two parquet files in one data dir
    — exactly like the real table, where older snapshots' files stay on disk."""
    data_dir = warehouse / "dq" / "kcore_funnel" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    def rows(run_id, walls, row_counts):
        return pa.table(
            {
                "run_id": [run_id] * len(walls),
                "iteration": list(range(len(walls))),
                "rows": row_counts,
                "users": [10] * len(walls),
                "items": [5] * len(walls),
                "converged": [False] * (len(walls) - 1) + [True],
                "wall_clock_s": walls,
            }
        )

    pq.write_table(rows(OLD_GOLD_RUN, [99.0, 99.0], [900, 501]), data_dir / "00000-a.parquet")
    pq.write_table(rows(GOLD_RUN, [3.0, 4.0], [900, 500]), data_dir / "00001-b.parquet")
    # Hidden checksum sidecars exist in the real dir; pyarrow must skip them.
    (data_dir / ".00001-b.parquet.crc").write_bytes(b"junk")


def _ops_record(
    scenario, run_id, rows_before, rows_after, nbytes, wall, snap, month=None, **extras
):
    params = {}
    if month:
        params["month"] = month
    if "retain_last" in extras:
        params["retain_last"] = extras.pop("retain_last")
    return {
        "schema_version": 1,
        "kind": "ops",
        "run_id": run_id,
        "run_ts": "2026-01-03T00:00:00+00:00",
        "git_sha": "0" * 40,
        "git_dirty": False,
        "scenario": scenario,
        "table": "local.ops.interactions_monthly",
        "params": params,
        "snapshot_before": snap - 1,
        "snapshot_after": snap,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "files_before": 2,
        "files_after": 3,
        "bytes_before": nbytes - 10,
        "bytes_after": nbytes,
        "wall_clock_s": wall,
        **extras,
    }


@pytest.fixture
def universe(tmp_path: Path) -> dict:
    root = tmp_path
    data = root / "data"
    warehouse = data / "warehouse"
    (data / "raw").mkdir(parents=True)

    (data / "MANIFEST.md").write_text(
        "# Manifest\n\n"
        "## Files\n\n"
        "### Electronics.jsonl.gz\n\n"
        "- URL: https://example.invalid/a.gz\n"
        "- Size (bytes): 1000\n\n"
        "### meta_Electronics.jsonl.gz\n\n"
        "- Size (bytes): 500\n\n"
        "## Bronze notes\n\n"
        "Ingest wall-clock: reviews=509s, items=926s\n"
    )

    (data / "ingest_summary.jsonl").write_text(
        json.dumps(
            {"table": "local.bronze.reviews", "total_parsed": 1000, "corrupt": 0,
             "written": 1000, "wall_clock_s": 10.5}
        )
        + "\n"
        + json.dumps(
            {"table": "local.bronze.items", "total_parsed": 200, "corrupt": 0,
             "written": 200, "wall_clock_s": 2.0}
        )
        + "\n"
    )

    (data / "build_summary.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"table": "items", "run_id": OLD_GOLD_RUN, "input_rows": 200,
                 "kept": 200, "wall_clock_s": 90.0},
                {"table": "interactions", "run_id": OLD_GOLD_RUN, "input_rows": 1000,
                 "kept": 900, "wall_clock_s": 90.0},
                {"table": "items", "run_id": GOLD_RUN, "input_rows": 200,
                 "kept": 200, "wall_clock_s": 1.5},
                {"table": "interactions", "run_id": GOLD_RUN, "input_rows": 1000,
                 "kept": 900, "wall_clock_s": 20.0},
            ]
        )
        + "\n"
    )

    for rel, snap, rows, size in TABLES:
        _write_metadata(warehouse, rel, snap, rows, size)
    _write_funnel(warehouse)

    cache_dir = data / "eval" / "cache" / str(FIVE_CORE_SNAPSHOT)
    cache_dir.mkdir(parents=True)
    (cache_dir / "train_user_idx.npy").write_bytes(b"x" * 64)
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps({"snapshot_ids": {"local.gold.interactions_5core": FIVE_CORE_SNAPSHOT}})
    )
    cache_bytes = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())

    per_user = data / "eval" / "per_user"
    per_user.mkdir(parents=True)
    head_artifact = per_user / f"{HEADLINE_RUN}_content_pop_blend.parquet"
    head_artifact.write_bytes(b"p" * 128)

    repro_cache = data / "eval" / "cache_repro" / str(FIVE_CORE_SNAPSHOT)
    repro_cache.mkdir(parents=True)
    repro_per_user = data / "eval" / "cache_repro" / "per_user"
    repro_per_user.mkdir(parents=True)
    repro_artifact = repro_per_user / f"{REPRO_RUN}_content_pop_blend.parquet"
    repro_artifact.write_bytes(b"p" * 128)

    (root / "configs").mkdir()
    (root / "configs" / "headline.yaml").write_text(
        f'headline_run_id: "{HEADLINE_RUN}"\nresults_path: "results/runs.jsonl"\n'
    )

    records = [
        {"schema_version": 1, "kind": "eval", "run_id": "20251231T000000Z-old",
         "wall_clock_s": 5.0, "per_user_artifact": "nope.parquet",
         "protocol": {"n_users": 1}, "model": {"name": "popularity"},
         "iceberg_snapshots": {}},
        {"schema_version": 1, "kind": "eval", "run_id": HEADLINE_RUN,
         "run_ts": "2026-01-01T00:00:00+00:00", "git_sha": "abc1234" + "0" * 33,
         "git_dirty": False,
         "per_user_artifact": f"data/eval/per_user/{HEADLINE_RUN}_content_pop_blend.parquet",
         "protocol": {"eval_split": "test", "n_users": 7},
         "model": {"name": "content_pop_blend"},
         "iceberg_snapshots": {"local.gold.interactions_5core": FIVE_CORE_SNAPSHOT},
         "wall_clock_s": 100.0},
        {"schema_version": 1, "kind": "reproduce", "run_id": REPRO_RUN,
         "git_sha": "def5678" + "0" * 33, "git_dirty": False,
         "reproduces_run_id": HEADLINE_RUN, "verdict": "byte_exact",
         "repro_cache_dir": str(repro_cache),
         "per_user_compare_detail": {"n_rows_repro": 7},
         "extract_wall_clock_s": 1.0, "eval_wall_clock_s": 99.0},
        # The real chain (T21-T23): a backfill, three monthly appends, a late-data
        # MERGE, a compaction that measurably does nothing, a near-no-op expiry,
        # then the fragmentation exhibit — 30 daily slices, a real 30->1 rewrite,
        # and a two-stage expiry whose second pass reclaims the pinned files.
        _ops_record("backfill", "20260103T000000Z-aaa", 0, 900, 4000, 48.0, 201),
        _ops_record("append", "20260103T000100Z-aaa", 900, 910, 4100, 6.3, 202, "2023-07"),
        _ops_record("append", "20260103T000200Z-aaa", 910, 920, 4200, 6.0, 203, "2023-08"),
        _ops_record("append", "20260103T000300Z-aaa", 920, 930, 4300, 5.9, 204, "2023-09"),
        _ops_record("upsert", "20260103T000400Z-aaa", 930, 940, 4400, 19.1, 205),
        _ops_record("compact", "20260103T000500Z-aaa", 940, 940, 4400, 0.1, 206,
                    rewritten_files=0, added_files=0, rewritten_bytes=0),
        _ops_record("expire", "20260103T000600Z-aaa", 940, 940, 4400, 2.4, 207,
                    retain_last=2, deleted_data_files=0),
        _ops_record("fragment", "20260103T000700Z-aaa", 940, 940, 4500, 13.5, 208,
                    "2023-06", n_slices=30, files_added=30),
        _ops_record("compact", "20260103T000800Z-aaa", 940, 940, 4450, 2.2, 209,
                    rewritten_files=30, added_files=1, rewritten_bytes=3321687),
        _ops_record("expire", "20260103T000900Z-aaa", 940, 940, 4450, 3.0, 210,
                    retain_last=2, deleted_data_files=3),
        _ops_record("expire", "20260103T001000Z-aaa", 940, 940, 4450, 3.1, 211,
                    retain_last=1, deleted_data_files=30),
    ]
    results = root / "results" / "runs.jsonl"
    results.parent.mkdir(parents=True)
    results.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    return {
        "root": root,
        "records": records,
        "results": results,
        "cache_bytes": cache_bytes,
        "kwargs": {
            "warehouse": str(warehouse),
            "results": str(results),
            "manifest": str(data / "MANIFEST.md"),
            "ingest_summary": str(data / "ingest_summary.jsonl"),
            "build_summary": str(data / "build_summary.jsonl"),
            "cache_root": str(data / "eval" / "cache"),
            "headline_config": str(root / "configs" / "headline.yaml"),
            "root": str(root),
        },
        "argv": [
            f"--warehouse={warehouse}",
            f"--results={results}",
            f"--manifest={data / 'MANIFEST.md'}",
            f"--ingest-summary={data / 'ingest_summary.jsonl'}",
            f"--build-summary={data / 'build_summary.jsonl'}",
            f"--cache-root={data / 'eval' / 'cache'}",
            f"--headline-config={root / 'configs' / 'headline.yaml'}",
            f"--root={root}",
        ],
    }


def _rewrite(universe, records):
    universe["results"].write_text("\n".join(json.dumps(r) for r in records) + "\n")


# --- the complete table -------------------------------------------------------


def test_table_is_complete_and_every_stage_present(universe):
    table = lineage.assemble(**universe["kwargs"])

    assert table["problems"] == []
    assert table["complete"] is True
    assert [r["stage"] for r in table["stages"]] == EXPECTED_STAGES
    assert table["stages_count"] == len(EXPECTED_STAGES)


def test_every_row_carries_every_required_field(universe):
    table = lineage.assemble(**universe["kwargs"])
    for row in table["stages"]:
        assert tuple(row) == lineage.ROW_KEYS, row["stage"]
        assert row["complete"] is True and row["missing"] == []
        assert row["layer"] and row["table"]
        assert row["source_of_truth"], row["stage"]
        # A number is only ever present with a source naming where it came from.
        for field in ("rows_in", "rows_out", "bytes", "wall_clock_s", "snapshot_id"):
            if row[field] is not None and field in row["required"]:
                assert any(
                    field in key or key == "rows" for key in row["source_of_truth"]
                ), (row["stage"], field)


def test_null_wall_clocks_are_exactly_the_footnoted_stages(universe):
    table = lineage.assemble(**universe["kwargs"])
    nulls = {r["stage"] for r in table["stages"] if r["wall_clock_s"] is None}
    assert nulls == FOOTNOTED_STAGES
    for row in table["stages"]:
        if row["wall_clock_s"] is None:
            assert row["wall_clock_source"] == lineage.NOT_PERSISTED
        else:
            assert row["wall_clock_source"] not in (None, lineage.NOT_PERSISTED)
    assert lineage.NOT_PERSISTED in table["footnotes"]


def test_numbers_come_from_the_declared_ledgers(universe):
    table = lineage.assemble(**universe["kwargs"])
    rows = {r["stage"]: r for r in table["stages"]}

    # raw: MANIFEST byte sizes summed; the free-form "Ingest wall-clock" prose
    # in the same file is deliberately not trusted.
    assert rows["raw_download"]["bytes"] == 1500
    assert rows["raw_download"]["wall_clock_s"] is None

    # bronze: rows + runtime from ingest_summary, bytes/snapshot from Iceberg.
    assert rows["bronze.reviews"]["rows_in"] == 1000
    assert rows["bronze.reviews"]["rows_out"] == 1000
    assert rows["bronze.reviews"]["wall_clock_s"] == 10.5
    assert rows["bronze.reviews"]["bytes"] == 5000
    assert rows["bronze.reviews"]["snapshot_id"] == 111

    # silver: LAST build_summary record per table wins (append-only ledger).
    assert rows["silver.interactions"]["rows_in"] == 1000
    assert rows["silver.interactions"]["rows_out"] == 900
    assert rows["silver.interactions"]["wall_clock_s"] == 20.0

    # gold core: runtime summed over the funnel iterations of THAT build run
    # only (3.0 + 4.0), never the older run's rows in the same data dir.
    assert rows["gold.interactions_5core"]["wall_clock_s"] == 7.0
    assert rows["gold.interactions_5core"]["rows_in"] == 900
    assert rows["gold.interactions_5core"]["rows_out"] == 500

    # gold projections: rows_in is the 5-core row count; runtime never persisted.
    for stage, out in [("gold.user_stats", 50), ("gold.item_features", 40),
                       ("gold.popularity", 60)]:
        assert rows[stage]["rows_in"] == 500
        assert rows[stage]["rows_out"] == out

    # item_text is built off the DISTINCT 5-core catalog (== item_features rows),
    # not the 5-core interaction count.
    assert rows["gold.item_text"]["rows_in"] == 40
    assert rows["gold.item_text"]["rows_out"] == 40
    assert rows["gold.item_text"]["bytes"] == 1700
    assert rows["gold.item_text"]["snapshot_id"] == 134
    assert rows["gold.item_text"]["wall_clock_s"] is None

    assert rows["eval_extract_cache"]["bytes"] == universe["cache_bytes"]
    assert rows["eval_extract_cache"]["snapshot_id"] == FIVE_CORE_SNAPSHOT

    assert rows["headline_eval"]["wall_clock_s"] == 100.0
    assert rows["headline_eval"]["bytes"] == 128
    assert rows["headline_eval"]["rows_out"] == 7

    # reproduce: extract + scoring runtime, bytes from the reproduced artifact.
    assert rows["reproduce_headline"]["wall_clock_s"] == 100.0
    assert rows["reproduce_headline"]["bytes"] == 128

    assert rows["ops.append[2023-08]"]["rows_in"] == 910
    assert rows["ops.append[2023-08]"]["rows_out"] == 920
    assert rows["ops.append[2023-08]"]["bytes"] == 4200
    assert rows["ops.append[2023-08]"]["snapshot_id"] == 203


def test_kcore_funnel_disagreeing_with_the_table_is_fatal(universe):
    """A funnel whose final row count != the live table's total-records means one
    of the two is stale; the row would be fiction, so the stage fails."""
    warehouse = Path(universe["kwargs"]["warehouse"])
    _write_metadata(warehouse, "gold/interactions_5core", FIVE_CORE_SNAPSHOT, 12345, 2000)
    table = lineage.assemble(**universe["kwargs"])
    assert table["complete"] is False
    assert any("interactions_5core" in p and "12,345" in p for p in table["problems"])


# --- completeness failures ----------------------------------------------------


def test_missing_ops_scenario_fails_check_only_naming_it(universe, capsys):
    records = [
        r for r in universe["records"]
        if not (r.get("kind") == "ops" and (r.get("params") or {}).get("month") == "2023-08")
    ]
    _rewrite(universe, records)

    code = lineage.main([*universe["argv"], "--check-only"])
    captured = capsys.readouterr()

    assert code != 0
    assert "append" in captured.err
    assert "expected at least 3, found 2" in captured.err
    summary = json.loads(captured.out.strip().splitlines()[-1])
    assert summary["complete"] is False


@pytest.mark.parametrize("scenario", ["upsert", "fragment", "backfill"])
def test_missing_scenario_is_named(universe, capsys, scenario):
    records = [r for r in universe["records"] if r.get("scenario") != scenario]
    _rewrite(universe, records)
    assert lineage.main([*universe["argv"], "--check-only"]) != 0
    err = capsys.readouterr().err
    assert scenario in err and "expected at least 1, found 0" in err


def test_expected_ops_set_is_a_parameter(universe):
    """Declaring a scenario the chain never ran must fail; declaring fewer must
    not."""
    fewer = lineage.assemble(**universe["kwargs"], expected_ops=("backfill",))
    assert fewer["complete"] is True

    more = lineage.assemble(
        **universe["kwargs"], expected_ops=(*lineage.DEFAULT_EXPECTED_OPS, "rollback")
    )
    assert more["complete"] is False
    assert any("rollback" in p for p in more["problems"])


def test_expected_ops_is_a_floor_not_an_equality(universe):
    """The chain runs compact and expire more than the minimum. Extra records are
    the exhibit, not an error: they keep their own rows and are enumerated."""
    table = lineage.assemble(**universe["kwargs"])

    assert table["complete"] is True
    assert table["ops_records"] == 11
    assert table["ops_observed"] == {
        "append": 3, "backfill": 1, "compact": 2, "expire": 3, "fragment": 1,
        "upsert": 1,
    }
    assert Counter(lineage.DEFAULT_EXPECTED_OPS)["compact"] == 1
    ops_rows = [r for r in table["stages"] if r["layer"] == "ops"]
    assert len(ops_rows) == 11


def test_one_row_per_ops_record_in_log_order(universe):
    table = lineage.assemble(**universe["kwargs"])
    ops_rows = [r for r in table["stages"] if r["layer"] == "ops"]
    logged = [r for r in universe["records"] if r.get("kind") == "ops"]

    assert len(ops_rows) == len(logged)
    for row, rec in zip(ops_rows, logged, strict=True):
        assert row["rows_in"] == rec["rows_before"]
        assert row["rows_out"] == rec["rows_after"]
        assert row["bytes"] == rec["bytes_after"]
        assert row["wall_clock_s"] == rec["wall_clock_s"]
        assert row["snapshot_id"] == rec["snapshot_after"]
        assert rec["run_id"] in row["source_of_truth"]["rows_out"]


def test_no_op_scenarios_are_kept_and_labelled_from_record_data(universe):
    """The measured no-op compaction is a finding; it must survive into the table
    distinguishable from the effective one."""
    table = lineage.assemble(**universe["kwargs"])
    rows = {r["stage"]: r for r in table["stages"]}

    assert "ops.compact[noop]" in rows
    assert "ops.compact[30->1]" in rows
    # The no-op moved nothing: same rows, same bytes, same snapshot as its input.
    noop = rows["ops.compact[noop]"]
    assert noop["rows_in"] == noop["rows_out"] == 940
    assert noop["bytes"] == 4400

    # Two-stage expiry: same retain_last, different reclaim — kept apart.
    assert "ops.expire[retain=2,deleted=0]" in rows
    assert "ops.expire[retain=2,deleted=3]" in rows
    assert "ops.expire[retain=1,deleted=30]" in rows

    assert "ops.fragment[2023-06]" in rows


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"scenario": "append", "params": {"month": "2023-07"}}, "ops.append[2023-07]"),
        ({"scenario": "fragment", "params": {"month": "2023-06"}}, "ops.fragment[2023-06]"),
        ({"scenario": "compact", "params": {}, "rewritten_files": 0, "added_files": 0},
         "ops.compact[noop]"),
        ({"scenario": "compact", "params": {}, "rewritten_files": 30, "added_files": 1},
         "ops.compact[30->1]"),
        ({"scenario": "expire", "params": {"retain_last": 1}, "deleted_data_files": 30},
         "ops.expire[retain=1,deleted=30]"),
        ({"scenario": "backfill", "params": {}}, "ops.backfill"),
        # Nothing is hardcoded to a known chain: an unseen scenario with no
        # discriminators still gets a usable label.
        ({"scenario": "rollback", "params": {}}, "ops.rollback"),
        ({"scenario": "compact", "params": {}}, "ops.compact"),
    ],
)
def test_ops_labels_are_derived_from_the_record(record, expected):
    assert lineage.ops_label(record) == expected


def test_indistinguishable_ops_records_still_get_unique_rows(universe):
    """Two records a label cannot tell apart must not silently become one row."""
    dupes = [
        _ops_record("compact", f"20260104T00000{i}Z-aaa", 1, 1, 10, 0.5, 300 + i)
        for i in range(2)
    ]
    _rewrite(universe, universe["records"] + dupes)
    table = lineage.assemble(**universe["kwargs"])
    stages = [r["stage"] for r in table["stages"]]

    assert len(stages) == len(set(stages))
    assert "ops.compact" in stages and "ops.compact#2" in stages
    assert table["ops_records"] == 13


@pytest.mark.parametrize(
    "ledger, stage_hint",
    [
        ("data/MANIFEST.md", "raw_download"),
        ("data/ingest_summary.jsonl", "bronze"),
        ("data/build_summary.jsonl", "silver"),
    ],
)
def test_missing_ledger_fails_naming_the_stage(universe, capsys, ledger, stage_hint):
    (universe["root"] / ledger).unlink()
    assert lineage.main([*universe["argv"], "--check-only"]) != 0
    err = capsys.readouterr().err
    assert stage_hint in err
    assert Path(ledger).name in err


def test_missing_table_metadata_fails_naming_the_table(universe, capsys):
    meta = Path(universe["kwargs"]["warehouse"]) / "gold" / "popularity" / "metadata"
    (meta / "version-hint.text").unlink()
    assert lineage.main([*universe["argv"], "--check-only"]) != 0
    err = capsys.readouterr().err
    assert "local.gold.popularity" in err


def test_missing_reproduce_record_fails(universe, capsys):
    _rewrite(universe, [r for r in universe["records"] if r.get("kind") != "reproduce"])
    assert lineage.main([*universe["argv"], "--check-only"]) != 0
    assert "reproduce" in capsys.readouterr().err


def test_missing_per_user_artifact_fails(universe, capsys):
    art = universe["root"] / "data" / "eval" / "per_user"
    for p in art.iterdir():
        p.unlink()
    assert lineage.main([*universe["argv"], "--check-only"]) != 0
    assert "per-user artifact" in capsys.readouterr().err


def test_ambiguous_reproduce_artifact_is_an_error_not_a_guess(universe):
    """Two candidate reproductions with the same model + sha: refuse to pick."""
    d = universe["root"] / "data" / "eval" / "cache_repro" / "per_user"
    (d / f"{REPRO_RUN}_other_content_pop_blend.parquet").write_bytes(b"q" * 9)
    table = lineage.assemble(**universe["kwargs"])
    assert table["complete"] is False
    assert any("disambiguate" in p for p in table["problems"])

    override = d / f"{REPRO_RUN}_content_pop_blend.parquet"
    fixed = lineage.assemble(**universe["kwargs"], repro_per_user=str(override))
    assert fixed["complete"] is True


# --- outputs ------------------------------------------------------------------


def test_check_only_writes_nothing(universe, tmp_path):
    out_json = tmp_path / "out" / "lineage.json"
    out_md = tmp_path / "out" / "lineage.md"
    before = universe["results"].read_bytes()

    code = lineage.main(
        [*universe["argv"], "--check-only", "--append-record",
         f"--out-json={out_json}", f"--out-md={out_md}"]
    )

    assert code == 0
    assert not out_json.exists() and not out_md.exists()
    assert universe["results"].read_bytes() == before


def test_json_serialisation_is_deterministic(universe, tmp_path):
    a = lineage.to_json_bytes(lineage.assemble(**universe["kwargs"]))
    b = lineage.to_json_bytes(lineage.assemble(**universe["kwargs"]))
    assert a == b

    outs = []
    for i in range(2):
        out_json = tmp_path / f"run{i}" / "lineage.json"
        out_md = tmp_path / f"run{i}" / "lineage.md"
        assert lineage.main(
            [*universe["argv"], f"--out-json={out_json}", f"--out-md={out_md}"]
        ) == 0
        outs.append((out_json.read_bytes(), out_md.read_bytes()))
    assert outs[0][0] == outs[1][0]
    assert outs[0][1] == outs[1][1]
    assert outs[0][0] == a


def test_markdown_has_one_line_per_stage_plus_footnotes(universe):
    table = lineage.assemble(**universe["kwargs"])
    md = lineage.to_markdown(table)
    lines = md.splitlines()

    for stage in EXPECTED_STAGES:
        matches = [ln for ln in lines if ln.startswith(f"| {stage} ")]
        assert len(matches) == 1, stage

    body = [ln for ln in lines if ln.startswith("| ") and not ln.startswith("| Stage")]
    assert len(body) == len(EXPECTED_STAGES)

    # Aligned columns: every table line has the same width and same pipe layout.
    widths = {len(ln) for ln in body}
    assert len(widths) == 1

    assert "## Footnotes" in lines
    assert any(lineage.NOT_PERSISTED in ln and ln.startswith("[^") for ln in lines)
    # Every footnote marker used in the table is defined below it.
    assert "[^1]" in md.split("## Footnotes")[0]
    assert "## Incomplete" not in md


def test_append_record_is_behind_an_explicit_flag(universe, tmp_path):
    out_json = tmp_path / "lineage.json"
    out_md = tmp_path / "lineage.md"
    base = universe["results"].read_text().count("\n")

    assert lineage.main(
        [*universe["argv"], f"--out-json={out_json}", f"--out-md={out_md}"]
    ) == 0
    assert universe["results"].read_text().count("\n") == base

    assert lineage.main(
        [*universe["argv"], "--append-record",
         f"--out-json={out_json}", f"--out-md={out_md}", "--run-id=T24-test"]
    ) == 0

    lines = universe["results"].read_text().splitlines()
    assert len(lines) == base + 1
    record = json.loads(lines[-1])
    assert record["kind"] == "lineage"
    assert record["run_id"] == "T24-test"
    assert record["complete"] is True
    assert record["stages_count"] == len(EXPECTED_STAGES)
    assert record["stage_completeness"] == {s: True for s in EXPECTED_STAGES}
    assert record["footnoted_stages"] == [
        s for s in EXPECTED_STAGES if s in FOOTNOTED_STAGES
    ]
    assert record["expected_ops"] == list(lineage.DEFAULT_EXPECTED_OPS)

    import hashlib

    expected = "sha256:" + hashlib.sha256(out_json.read_bytes()).hexdigest()
    assert record["artifact_sha256"] == expected


def test_incomplete_table_is_never_published(universe, tmp_path):
    _rewrite(universe, [r for r in universe["records"] if r.get("scenario") != "expire"])
    out_json = tmp_path / "lineage.json"
    out_md = tmp_path / "lineage.md"
    base = universe["results"].read_text().count("\n")

    code = lineage.main(
        [*universe["argv"], "--append-record",
         f"--out-json={out_json}", f"--out-md={out_md}"]
    )

    assert code != 0
    assert not out_json.exists() and not out_md.exists()
    assert universe["results"].read_text().count("\n") == base


# --- parser units -------------------------------------------------------------


def test_manifest_parser_ignores_prose_sections(tmp_path):
    p = tmp_path / "MANIFEST.md"
    p.write_text(
        "### a.gz\n- Size (bytes): 7\n\n"
        "### Bronze layer notes\n\nprose, no size line\n\n"
        "### b.gz\n- URL: x\n- Size (bytes): 8\n"
    )
    assert lineage.parse_manifest_sizes(p) == {"a.gz": 7, "b.gz": 8}


def test_manifest_without_sizes_is_an_error(tmp_path):
    p = tmp_path / "MANIFEST.md"
    p.write_text("# Manifest\n\nno files here\n")
    with pytest.raises(lineage.LineageError, match="Size"):
        lineage.parse_manifest_sizes(p)


def test_funnel_run_filter_rejects_an_unknown_run(universe):
    warehouse = universe["kwargs"]["warehouse"]
    assert len(lineage.kcore_funnel_iterations(warehouse, GOLD_RUN)) == 2
    with pytest.raises(lineage.LineageError, match="no rows for run_id"):
        lineage.kcore_funnel_iterations(warehouse, "nope")
