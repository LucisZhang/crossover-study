"""Read-only DQ pull + ``kind="dq_export"`` record for exhibit 4 (Phase 6, T29).

Two phases, deliberately separate processes' worth of work:

``--phase collect`` (Spark, STRICTLY READ-ONLY)
    Reads the latest complete contract audit out of ``local.dq.dq_results``,
    the k-core funnel, the reconciliation waterfall ledger and both quarantine
    ledgers — every read pinned to a snapshot id captured *before* the JVM
    starts — and writes ``data/demo_export/dq_raw.json``. It writes nothing to
    the warehouse: no table is created, altered, appended to or stamped.

``--phase record`` (JVM-free)
    Hashes ``data/waterfall.json``, ``data/build_summary.jsonl`` and
    ``dq_raw.json``, publishes byte-identical copies of the two pointer-
    addressable artifacts under ``results/dq/``, re-checks every reconciliation
    identity, and appends ONE ``kind="dq_export"`` record to
    ``results/runs.jsonl``. ``--dry-run`` does all of that except the append.

The split exists so the record's ``git_sha`` contains the code that produced
the artifact it attests to: collect and dry-run first, commit, then append.
``dq_raw.json`` carries no timestamp and no wall clock, so it is byte-stable
across re-runs and the digest printed by ``--dry-run`` is the digest the
appended record carries.

    make demo-dq                                          # phase collect (pins JAVA_HOME)
    uv run python -m batch_recsys_lab.demo.dq_export_job \
        --config configs/dq_export.yaml --phase record --dry-run    # prints, appends NOTHING
    uv run python -m batch_recsys_lab.demo.dq_export_job \
        --config configs/dq_export.yaml --phase record --append     # from a clean tree

``--phase all --dry-run`` does the whole thing in one process (Spark pull, then
the printed record) and still appends nothing. Spark 4 supports Java 17/21 only
and the host default is 25, so the collect phase must run under the Makefile's
JDK pin; the record phase never imports pyspark and needs no JVM.

The record it appends carries, besides the standard provenance block:
``waterfall_sha256`` / ``build_summary_sha256`` / ``dq_raw_sha256`` (the three
artifacts it attests to), ``published_artifacts`` (their committed copies),
``iceberg_snapshots`` (the headline-pinned tables the guard checked),
``table_snapshots`` (the ledger tables it read), ``audit_run_id`` /
``build_run_id``, ``headline_counts``, ``contract_summary``,
``quarantine_totals``, ``measured_rates`` and ``reconciliation``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from batch_recsys_lab.eval import runlog
from batch_recsys_lab.eval.runlog import iceberg_snapshot_id, sha256_file

RAW_SCHEMA_VERSION = 1
RECORD_KIND = "dq_export"

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Measured rates lifted from the contract ledger. ``scope`` is how the row is
# selected: "audit" = must come from the selected audit run (a check the audit
# re-runs every time); "latest" = the most recent row for that (table, check_id)
# across ALL runs, because the measure is written by the build job that produced
# the table and is not part of a contract YAML (so a later audit does not
# re-emit it). Every entry records the run_id it actually came from.
LEDGER_RATES: tuple[tuple[str, str, str, str], ...] = (
    ("unknown_brand_share", "local.silver.items", "brand_unknown_share", "audit"),
    ("item_fk_orphan_rate", "local.silver.interactions", "item_fk", "audit"),
    ("price_unparseable_share", "local.silver.items", "price_unparseable", "latest"),
    ("brand_from_manufacturer_share", "local.silver.items", "brand_source_share", "latest"),
    ("gold_item_features_join_loss", "local.gold.item_features", "gold_item_features_join_loss", "latest"),
    ("gold_item_text_empty_title_share", "local.gold.item_text", "gold_item_text_empty_title_share", "latest"),
    ("gold_item_text_empty_features_share", "local.gold.item_text", "gold_item_text_empty_features_share", "latest"),
)

_REQUIRED_CONFIG = (
    "demo_export_config",
    "warehouse",
    "contracts_dir",
    "tables",
    "waterfall_json",
    "build_summary",
    "dq_raw",
    "published_dir",
)


class DqExportError(RuntimeError):
    """A guard failed: the warehouse, the ledgers and the waterfall disagree."""


# --- config -------------------------------------------------------------------


def load_config(path: str | Path, *, repo_root: Path | None = None) -> dict:
    """Load ``configs/dq_export.yaml`` and resolve the shared demo-export keys."""
    # Default to cwd, not the module's install location: production invokes from
    # the repo root, and tests chdir into a hermetic fixture repo. Resolving to
    # _REPO_ROOT here made main() read the REAL results/runs.jsonl from inside
    # the fixture — the duplicate guard then fired against the real dq_export
    # record and the dry-run test broke the moment the real append landed.
    root = Path(repo_root) if repo_root else Path.cwd()
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in _REQUIRED_CONFIG if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    cfg = dict(cfg)
    cfg["_config_path"] = str(path)
    cfg["_root"] = root

    demo_cfg = yaml.safe_load((root / cfg["demo_export_config"]).read_text())
    for key in ("runs_log", "out_dir", "manifest", "headline_run_id"):
        if key not in demo_cfg:
            raise ValueError(f"{cfg['demo_export_config']}: missing {key!r}")
        cfg.setdefault(key, demo_cfg[key])
    return cfg


def rel(cfg: dict, path: str | Path) -> Path:
    """Resolve a config path against the repo root (never the cwd)."""
    p = Path(path)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


# --- guards -------------------------------------------------------------------


def contract_tables(contracts_dir: str | Path) -> dict[str, dict]:
    """``{table: {name, version}}`` for every contract YAML on disk.

    Read straight from the YAML (not via the loader) so this stays import-light
    and the record phase never pulls Spark in transitively.
    """
    out: dict[str, dict] = {}
    for path in sorted(Path(contracts_dir).glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        out[str(doc["table"])] = {"name": str(doc["name"]), "version": int(doc["version"])}
    if not out:
        raise DqExportError(f"no contract YAMLs under {contracts_dir}")
    return out


def assert_headline_snapshots(warehouse: str | Path, expected: dict[str, int]) -> dict[str, int]:
    """Abort unless every table the headline record pins is still at that snapshot.

    This job does not read those tables (it reads the DQ ledgers), but the audit
    rows and the waterfall it publishes describe *that* state of the warehouse.
    JVM-free, so a drifted warehouse fails in a second rather than after Spark
    has started.
    """
    live: dict[str, int] = {}
    problems: list[str] = []
    for table, want in sorted(expected.items()):
        try:
            sid = int(iceberg_snapshot_id(warehouse, table))
        except Exception as exc:  # noqa: BLE001 - collected, re-raised as one message
            problems.append(f"{table}: cannot read snapshot id ({exc})")
            continue
        live[table] = sid
        if sid != int(want):
            problems.append(f"{table}: live snapshot {sid}, headline record pins {want}")
    if problems:
        raise SystemExit(
            "SNAPSHOT GUARD FAILED — the warehouse is not at the state the headline run was "
            "scored against; refusing to publish a DQ exhibit that claims it is.\n  "
            + "\n  ".join(problems)
        )
    print("snapshot guard OK — headline-pinned tables are unmoved:")
    for table in sorted(live):
        print(f"  {table} @ {live[table]}")
    return live


def live_snapshots(warehouse: str | Path, tables: dict[str, str]) -> dict[str, int]:
    """``{table: current snapshot id}`` for the tables this job reads (JVM-free)."""
    out: dict[str, int] = {}
    for role, table in sorted(tables.items()):
        try:
            out[str(table)] = int(iceberg_snapshot_id(warehouse, table))
        except Exception as exc:  # noqa: BLE001
            raise DqExportError(f"{role}: cannot read snapshot id of {table} ({exc})") from exc
    return out


# --- audit selection ------------------------------------------------------------


def _details(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": raw}


def select_audit_run(rows: list[dict], required_tables: set[str], pinned: str | None = None) -> str:
    """The audit run whose matrix the exhibit shows.

    ``dq_results`` is append-only and holds rows from every build *and* every
    audit ever run, including partial ones (a single-table re-audit) and runs
    where a build job's gate measures share a run_id with a later audit. The
    exhibit needs one coherent matrix, so a run qualifies only if it covers
    every contract on disk with exactly one row per ``(table, check_id)``. The
    latest qualifying run wins; if none qualifies the job aborts with the
    candidates listed rather than showing a half matrix.
    """
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(str(r["run_id"]), []).append(r)

    def _ts(run_id: str) -> str:
        return min(str(r["run_ts"]) for r in by_run[run_id])

    reasons: dict[str, str] = {}
    qualifying: list[str] = []
    for run_id, run_rows in by_run.items():
        tables = {str(r["table_name"]) for r in run_rows}
        gap = required_tables - tables
        if gap:
            reasons[run_id] = f"does not cover {sorted(gap)}"
            continue
        keys = [(str(r["table_name"]), str(r["check_id"])) for r in run_rows]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            reasons[run_id] = f"duplicate (table, check) rows: {dupes[:4]}"
            continue
        qualifying.append(run_id)

    if pinned is not None:
        if pinned not in by_run:
            raise DqExportError(f"audit_run_id {pinned!r} has no rows in dq_results")
        if pinned not in qualifying:
            raise DqExportError(f"audit_run_id {pinned!r} is not a complete audit: {reasons[pinned]}")
        return pinned

    if not qualifying:
        detail = "\n  ".join(f"{r} ({_ts(r)}): {reasons[r]}" for r in sorted(by_run, key=_ts))
        raise DqExportError(
            "no complete contract audit in dq_results — every run is partial or duplicated. "
            "Run `make contracts-audit` first.\n  " + detail
        )
    return max(qualifying, key=lambda r: (_ts(r), r))


def build_matrix(rows: list[dict], contracts: dict[str, dict]) -> dict:
    """``{table: {contract, total_rows, counts, checks: {check_id: {...}}}}``."""
    matrix: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: (str(r["table_name"]), str(r["check_kind"]), str(r["check_id"]))):
        table = str(r["table_name"])
        entry = matrix.setdefault(
            table,
            {
                "contract_name": str(r["contract_name"]),
                "contract_version": int(r["contract_version"]),
                "contract_declared_version": contracts.get(table, {}).get("version"),
                "total_rows": int(r["total_rows"]),
                "counts": {"pass": 0, "measured": 0, "fail": 0},
                "checks": {},
            },
        )
        if int(r["total_rows"]) != entry["total_rows"]:
            raise DqExportError(
                f"{table}: audit rows disagree on total_rows "
                f"({entry['total_rows']} vs {r['total_rows']})"
            )
        declared = entry["contract_declared_version"]
        if declared is not None and int(r["contract_version"]) != int(declared):
            raise DqExportError(
                f"{table}: the audit ran contract version {r['contract_version']} but "
                f"{table}'s contract YAML on disk declares version {declared}. The matrix "
                "would not match `make contracts-audit` — re-run the audit."
            )
        details = _details(r.get("details"))
        status = str(r["status"])
        entry["counts"][status] = entry["counts"].get(status, 0) + 1
        entry["checks"][str(r["check_id"])] = {
            "kind": str(r["check_kind"]),
            "column": None if r.get("column") is None else str(r["column"]),
            "status": status,
            "violations": int(r["violation_count"]),
            "measured": None if r.get("metric_value") is None else float(r["metric_value"]),
            "note": details.get("note"),
            "details": details,
        }
    return matrix


def summarize_matrix(matrix: dict) -> dict:
    """Tallies for the dashboard header.

    ``failing`` is enumerated (a failing check is the one thing a reader must be
    able to name); ``measured`` checks are only counted, broken down by kind —
    26 of them are the recorded T8 nullability downgrade and listing each would
    bury the signal in the record without adding evidence the matrix itself
    does not already carry.
    """
    counts = {"pass": 0, "measured": 0, "fail": 0}
    by_kind_status: dict[str, dict[str, int]] = {}
    failing: list[dict] = []
    n_checks = 0
    for table, entry in matrix.items():
        for check_id, chk in entry["checks"].items():
            n_checks += 1
            status = chk["status"]
            counts[status] = counts.get(status, 0) + 1
            by_kind_status.setdefault(chk["kind"], {})[status] = (
                by_kind_status.setdefault(chk["kind"], {}).get(status, 0) + 1
            )
            if status == "fail":
                failing.append(
                    {
                        "table": table,
                        "check_id": check_id,
                        "kind": chk["kind"],
                        "violations": chk["violations"],
                        "measured": chk["measured"],
                    }
                )
    return {
        "tables": len(matrix),
        "checks": n_checks,
        "pass": counts["pass"],
        "measured": counts["measured"],
        "fail": counts["fail"],
        "any_fail": counts["fail"] > 0,
        "by_kind_status": {
            kind: {s: by_kind_status[kind][s] for s in sorted(by_kind_status[kind])}
            for kind in sorted(by_kind_status)
        },
        "failing": sorted(failing, key=lambda d: (d["table"], d["check_id"])),
    }


# --- waterfall ------------------------------------------------------------------


def waterfall_edges(waterfall: dict) -> list[dict]:
    """Flatten ``data/waterfall.json`` into one list of edges, dataset-tagged."""
    out: list[dict] = []
    for dataset, block in waterfall["datasets"].items():
        for i, edge in enumerate(block["edges"]):
            e = dict(edge)
            e["dataset"] = dataset
            e["index"] = i
            out.append(e)
    return out


def _dominant_reason(edge: dict) -> str:
    drops = [r for r in edge["reasons"] if r["reason"] != "kept" and int(r["rows"]) > 0]
    if not drops:
        return "none"
    return str(max(drops, key=lambda r: int(r["rows"]))["reason"])


def waterfall_block(waterfall: dict, ledger_rows: list[dict]) -> dict:
    """The waterfall as the exhibit shows it, cross-checked against the ledger.

    ``data/waterfall.json`` is the committed artifact; ``local.dq.waterfall`` is
    the Iceberg ledger the same build wrote. Publishing the artifact while
    proving it still equals the live ledger is the whole point of the panel.
    """
    from_ledger: dict[tuple[str, str, str, str], int] = {}
    for r in ledger_rows:
        key = (str(r["dataset"]), str(r["stage_from"]), str(r["stage_to"]), str(r["reason"]))
        if key in from_ledger:
            raise DqExportError(f"local.dq.waterfall has duplicate rows for {key}")
        from_ledger[key] = int(r["rows"])

    stages: list[dict] = []
    reconciles = True
    for edge in waterfall_edges(waterfall):
        rows_in = int(edge["source_rows"])
        rows_out = int(edge["kept_rows"])
        reasons = []
        for reason in edge["reasons"]:
            key = (edge["dataset"], edge["stage_from"], edge["stage_to"], str(reason["reason"]))
            if key not in from_ledger:
                raise DqExportError(f"local.dq.waterfall has no row for {key} (in data/waterfall.json)")
            if from_ledger[key] != int(reason["rows"]):
                raise DqExportError(
                    f"waterfall drift at {key}: ledger={from_ledger[key]}, "
                    f"data/waterfall.json={reason['rows']}"
                )
            reasons.append(
                {
                    "reason": str(reason["reason"]),
                    "rows": int(reason["rows"]),
                    "share_of_rows_in": (int(reason["rows"]) / rows_in) if rows_in else 0.0,
                }
            )
        ok = bool(edge["sum_ok"]) and bool(edge["count_ok"])
        reconciles = reconciles and ok
        stages.append(
            {
                "stage": f"{edge['dataset']}:{edge['stage_from']}->{edge['stage_to']}",
                "dataset": edge["dataset"],
                "stage_from": str(edge["stage_from"]),
                "stage_to": str(edge["stage_to"]),
                "rows_in": rows_in,
                "rows_out": rows_out,
                "delta": rows_in - rows_out,
                "reason": _dominant_reason(edge),
                "reasons": reasons,
                "target_table": str(edge["target_table"]),
                "target_count": int(edge["target_count"]),
                "sum_ok": bool(edge["sum_ok"]),
                "count_ok": bool(edge["count_ok"]),
                "matches_ledger": True,
            }
        )
    return {
        "run_id": str(waterfall["run_id"]),
        "stages": stages,
        "reconciles": reconciles,
        "ledger_rows_checked": len(from_ledger),
    }


def headline_counts(waterfall: dict, quarantine_total: int) -> dict:
    """raw → bronze → silver → 5-core, with the drop reasons that explain it."""
    edges = {(e["dataset"], e["stage_from"], e["stage_to"]): e for e in waterfall_edges(waterfall)}

    def reason_rows(edge: dict, name: str) -> int:
        for r in edge["reasons"]:
            if r["reason"] == name:
                return int(r["rows"])
        return 0

    rb = edges[("reviews", "raw", "bronze")]
    bs = edges[("reviews", "bronze", "silver")]
    sg = edges[("reviews", "silver", "gold")]
    irb = edges[("items", "raw", "bronze")]
    ibs = edges[("items", "bronze", "silver")]

    return {
        "raw_reviews_rows": int(rb["source_rows"]),
        "bronze_reviews_rows": int(rb["kept_rows"]),
        "silver_interactions_rows": int(bs["kept_rows"]),
        "bronze_to_silver_dropped_rows": int(bs["source_rows"]) - int(bs["kept_rows"]),
        "exact_duplicate_rows": reason_rows(bs, "exact_duplicate"),
        "superseded_by_later_review_rows": reason_rows(bs, "superseded_by_later_review"),
        "quarantined_interaction_rows": reason_rows(bs, "quarantine:rating_domain"),
        "quarantine_ledger_rows": int(quarantine_total),
        "kcore_pruned_rows": reason_rows(sg, "kcore_pruned"),
        "gold_interactions_5core_rows": int(sg["kept_rows"]),
        "raw_items_rows": int(irb["source_rows"]),
        "silver_items_rows": int(ibs["kept_rows"]),
    }


def reconciliation_checks(hc: dict, funnel: list[dict], matrix: dict) -> dict:
    """Every identity the exhibit's numbers must satisfy, checked and recorded."""
    checks: list[dict] = []

    def add(name: str, lhs: int | float, rhs: int | float, note: str) -> None:
        checks.append({"name": name, "lhs": lhs, "rhs": rhs, "ok": lhs == rhs, "note": note})

    add(
        "raw_equals_bronze",
        hc["raw_reviews_rows"],
        hc["bronze_reviews_rows"],
        "ingest is lossless: 0 corrupt lines",
    )
    add(
        "bronze_minus_drops_equals_silver",
        hc["bronze_reviews_rows"]
        - hc["exact_duplicate_rows"]
        - hc["superseded_by_later_review_rows"]
        - hc["quarantined_interaction_rows"],
        hc["silver_interactions_rows"],
        "dedup + supersede + quarantine account for every dropped row",
    )
    add(
        "quarantine_ledger_equals_waterfall",
        hc["quarantine_ledger_rows"],
        hc["quarantined_interaction_rows"],
        "the quarantine table holds exactly the rows the waterfall says it does",
    )
    add(
        "silver_minus_kcore_equals_gold",
        hc["silver_interactions_rows"] - hc["kcore_pruned_rows"],
        hc["gold_interactions_5core_rows"],
        "the k-core prune accounts for the whole silver->gold delta",
    )
    if funnel:
        add(
            "funnel_starts_at_silver",
            int(funnel[0]["rows"]),
            hc["silver_interactions_rows"],
            "iteration 0 of the k-core funnel is the silver row count",
        )
        add(
            "funnel_ends_at_gold",
            int(funnel[-1]["rows"]),
            hc["gold_interactions_5core_rows"],
            "the converged funnel iteration is the published 5-core row count",
        )
    for table, expected in (
        ("local.silver.interactions", hc["silver_interactions_rows"]),
        ("local.gold.interactions_5core", hc["gold_interactions_5core_rows"]),
        ("local.silver.items", hc["silver_items_rows"]),
    ):
        if table in matrix:
            add(
                f"audit_total_rows::{table}",
                int(matrix[table]["total_rows"]),
                int(expected),
                "the audited table is the table the waterfall describes",
            )
    return {"checks": checks, "all_ok": all(c["ok"] for c in checks)}


# --- collect (Spark) ------------------------------------------------------------


def _rows(df, columns: list[str]) -> list[dict]:
    return [{c: r[c] for c in columns} for r in df.select(*columns).collect()]


def collect(cfg: dict) -> dict:
    """Phase 1: read-only Spark pull → ``dq_raw.json``. Writes nothing to Iceberg."""
    warehouse = rel(cfg, cfg["warehouse"])
    runs = {}
    runs_log = rel(cfg, cfg["runs_log"])
    for line in runs_log.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            runs[rec["run_id"]] = rec
    headline = runs[cfg["headline_run_id"]]
    assert_headline_snapshots(warehouse, {t: int(s) for t, s in headline["iceberg_snapshots"].items()})

    tables = {role: str(name) for role, name in cfg["tables"].items()}
    null_price_tables = [str(t) for t in (cfg.get("null_price_tables") or [])]
    if not cfg.get("measure_null_price", True):
        null_price_tables = []
    pins = live_snapshots(
        warehouse, {**tables, **{f"null_price_{i}": t for i, t in enumerate(null_price_tables)}}
    )
    print("read-only snapshot pins:")
    for table in sorted(pins):
        print(f"  {table} @ {pins[table]}")

    waterfall = json.loads(rel(cfg, cfg["waterfall_json"]).read_text())
    build_run_id = str(cfg.get("build_run_id") or waterfall["run_id"])
    contracts = contract_tables(rel(cfg, cfg["contracts_dir"]))

    from batch_recsys_lab.spark_session import get_spark

    spark = get_spark(app_name="t29-dq-export", warehouse=str(warehouse))
    try:
        from pyspark.sql import functions as F

        def read(table: str):
            return spark.read.option("snapshot-id", str(pins[table])).table(table)

        audit_rows = _rows(
            read(tables["dq_results"]),
            [
                "run_id",
                "run_ts",
                "table_name",
                "contract_name",
                "contract_version",
                "check_id",
                "check_kind",
                "column",
                "status",
                "violation_count",
                "total_rows",
                "metric_value",
                "details",
            ],
        )
        funnel_rows = _rows(
            read(tables["kcore_funnel"]),
            ["run_id", "iteration", "rows", "users", "items", "converged", "wall_clock_s"],
        )
        ledger_rows = _rows(
            read(tables["waterfall"]), ["run_id", "dataset", "stage_from", "stage_to", "reason", "rows"]
        )

        quarantine: dict[str, dict] = {}
        for role in ("quarantine_interactions", "quarantine_items"):
            table = tables[role]
            df = read(table)
            total_all = df.count()
            scoped = df.where(F.col("run_id") == F.lit(build_run_id))
            by_primary = _rows(
                scoped.groupBy("primary_reason").count().withColumnRenamed("count", "rows"),
                ["primary_reason", "rows"],
            )
            by_any = _rows(
                scoped.select(F.explode("violation_reasons").alias("reason"))
                .groupBy("reason")
                .count()
                .withColumnRenamed("count", "rows"),
                ["reason", "rows"],
            )
            quarantine[table] = {
                "rows": sum(int(r["rows"]) for r in by_primary),
                "rows_all_runs": int(total_all),
                "by_primary_reason": sorted(
                    ({"reason": str(r["primary_reason"]), "rows": int(r["rows"])} for r in by_primary),
                    key=lambda d: (-d["rows"], d["reason"]),
                ),
                "by_violation_reason": sorted(
                    ({"reason": str(r["reason"]), "rows": int(r["rows"])} for r in by_any),
                    key=lambda d: (-d["rows"], d["reason"]),
                ),
                "snapshot_id": int(pins[table]),
            }

        null_price: dict[str, dict] = {}
        for table in null_price_tables:
            df = read(table)
            agg = df.agg(
                F.count(F.lit(1)).alias("rows"),
                F.sum(F.col("price_usd").isNull().cast("long")).alias("nulls"),
            ).collect()[0]
            total = int(agg["rows"])
            nulls = int(agg["nulls"] or 0)
            null_price[table] = {
                "rows": nulls,
                "denominator": total,
                "rate": (nulls / total) if total else 0.0,
                "snapshot_id": int(pins[table]),
            }
    finally:
        spark.stop()

    # --- assemble (pure python from here) ------------------------------------
    audit_run_id = select_audit_run(audit_rows, set(contracts), cfg.get("audit_run_id"))
    selected = [r for r in audit_rows if str(r["run_id"]) == audit_run_id]
    audit_run_ts = min(str(r["run_ts"]) for r in selected)
    matrix = build_matrix(selected, contracts)
    summary = summarize_matrix(matrix)

    funnel = sorted(
        (
            {
                "iteration": int(r["iteration"]),
                "rows": int(r["rows"]),
                "users": int(r["users"]),
                "items": int(r["items"]),
                "converged": bool(r["converged"]),
                "wall_clock_s": float(r["wall_clock_s"]),
            }
            for r in funnel_rows
            if str(r["run_id"]) == build_run_id
        ),
        key=lambda d: d["iteration"],
    )
    if not funnel:
        raise DqExportError(f"local.dq.kcore_funnel has no rows for build run {build_run_id!r}")
    committed_funnel = waterfall["datasets"]["reviews"]["kcore_funnel"]
    if [{k: f[k] for k in ("iteration", "rows", "users", "items", "converged", "wall_clock_s")} for f in committed_funnel] != funnel:
        raise DqExportError(
            "local.dq.kcore_funnel disagrees with the funnel embedded in data/waterfall.json"
        )

    wf = waterfall_block(waterfall, [r for r in ledger_rows if str(r["run_id"]) == build_run_id])
    q_total = sum(int(v["rows"]) for v in quarantine.values())
    hc = headline_counts(waterfall, q_total)
    recon = reconciliation_checks(hc, funnel, matrix)

    rates = measured_rates(audit_rows, audit_run_id, null_price)

    doc = {
        "schema_version": RAW_SCHEMA_VERSION,
        "generated_by": "batch_recsys_lab.demo.dq_export_job",
        "warehouse": str(cfg["warehouse"]),
        "headline_run_id": str(cfg["headline_run_id"]),
        "audit_run_id": audit_run_id,
        "audit_run_ts": audit_run_ts,
        "build_run_id": build_run_id,
        "table_snapshots": {t: int(s) for t, s in sorted(pins.items())},
        "headline_snapshots": {t: int(s) for t, s in sorted(headline["iceberg_snapshots"].items())},
        "contracts": {t: contracts[t] for t in sorted(contracts)},
        "contract_matrix": {t: matrix[t] for t in sorted(matrix)},
        "contract_summary": summary,
        "quarantine": {
            "build_run_id": build_run_id,
            "total_rows": q_total,
            "by_reason": _quarantine_by_reason(quarantine, hc),
            "tables": {t: quarantine[t] for t in sorted(quarantine)},
        },
        "kcore_funnel": funnel,
        "waterfall": wf,
        "measured_rates": rates,
        "headline_counts": hc,
        "reconciliation": recon,
    }
    if not recon["all_ok"]:
        bad = [c for c in recon["checks"] if not c["ok"]]
        raise DqExportError(
            "reconciliation FAILED — refusing to publish a DQ exhibit whose numbers do not add up:\n  "
            + "\n  ".join(f"{c['name']}: {c['lhs']} != {c['rhs']} ({c['note']})" for c in bad)
        )

    out = rel(cfg, cfg["dq_raw"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    print(
        f"wrote {out} (audit {audit_run_id}: {summary['tables']} tables x {summary['checks']} checks, "
        f"{summary['fail']} fail / {summary['measured']} measured; "
        f"{q_total} quarantined rows; {len(funnel)} k-core iterations)\n"
        f"  sha256 {sha256_file(out)}"
    )
    return doc


def _quarantine_by_reason(quarantine: dict, hc: dict) -> list[dict]:
    """Flat by-reason list across both quarantine ledgers (each row counted once)."""
    total = sum(int(v["rows"]) for v in quarantine.values())
    inputs = {
        "local.quarantine.interactions": hc["raw_reviews_rows"],
        "local.quarantine.items": hc["raw_items_rows"],
    }
    out: list[dict] = []
    for table in sorted(quarantine):
        for entry in quarantine[table]["by_primary_reason"]:
            denom = inputs.get(table, 0)
            out.append(
                {
                    "table": table,
                    "reason": entry["reason"],
                    "rows": int(entry["rows"]),
                    "share": (int(entry["rows"]) / total) if total else 0.0,
                    "share_of_input": (int(entry["rows"]) / denom) if denom else 0.0,
                }
            )
    return sorted(out, key=lambda d: (-d["rows"], d["table"], d["reason"]))


def measured_rates(audit_rows: list[dict], audit_run_id: str, null_price: dict) -> dict:
    """The §9 rate panel: ledger measures + the job's own null-price counts."""
    out: dict[str, dict] = {}
    for key, table, check_id, scope in LEDGER_RATES:
        candidates = [
            r
            for r in audit_rows
            if str(r["table_name"]) == table
            and str(r["check_id"]) == check_id
            and (scope != "audit" or str(r["run_id"]) == audit_run_id)
        ]
        if not candidates:
            if scope == "audit":
                raise DqExportError(
                    f"audit {audit_run_id} has no {check_id} row for {table} "
                    "(the §9 rate panel cannot be assembled)"
                )
            continue
        row = max(candidates, key=lambda r: (str(r["run_ts"]), str(r["run_id"])))
        out[key] = {
            "source": "contract_ledger",
            "run_id": str(row["run_id"]),
            "is_selected_audit": str(row["run_id"]) == audit_run_id,
            "table": table,
            "check_id": check_id,
            "rows": int(row["violation_count"]),
            "denominator": int(row["total_rows"]),
            "rate": None if row.get("metric_value") is None else float(row["metric_value"]),
            "details": _details(row.get("details")),
        }
    for table, stats in sorted(null_price.items()):
        key = "null_price_share::" + table
        out[key] = {
            # NOT a contract check: §9 asks for a null-price rate and the ledger
            # has none (price_unparseable counts unparseable strings, not absent
            # prices). Measured here, read-only, and labelled as such.
            "source": "dq_export_job",
            "run_id": None,
            "is_selected_audit": False,
            "table": table,
            "check_id": None,
            "predicate": "price_usd IS NULL",
            "rows": int(stats["rows"]),
            "denominator": int(stats["denominator"]),
            "rate": float(stats["rate"]),
            "details": {"snapshot_id": int(stats["snapshot_id"])},
        }
    return out


# --- record (JVM-free) ------------------------------------------------------------


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
    """``(run_id, run_ts)`` — same D3 form the contract engine generates.

    Deliberately re-implemented here rather than imported from
    ``contracts.engine``: that module imports ``pyspark`` at module scope, and
    the record phase must stay importable (and runnable) without it.
    """
    now = datetime.now(timezone.utc)
    run_ts = now.isoformat()
    if run_id:
        return run_id, run_ts
    env = os.environ.get("RECSYS_RUN_ID")
    if env:
        return env, run_ts
    sha = _git_short_sha()
    return now.strftime("%Y%m%dT%H%M%SZ") + (f"-{sha}" if sha else ""), run_ts


def publish(cfg: dict, *, dq_raw_path: Path) -> dict:
    """Copy the two pointer-addressable artifacts into the committed results dir.

    Byte-identical copies, so one sha256 is valid for both paths. ``data/`` is
    gitignored and the traceability verifier re-reads a ``results_artifact``
    source file in record mode too — so the evidence ``dq.json`` cites has to
    live somewhere CI can see it (the ``results/lineage.json`` precedent).
    """
    published_dir = rel(cfg, cfg["published_dir"])
    published_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    for key, src in (("waterfall", rel(cfg, cfg["waterfall_json"])), ("dq_raw", dq_raw_path)):
        dst = published_dir / f"{key}.json"
        payload = src.read_bytes()
        if not dst.exists() or dst.read_bytes() != payload:
            dst.write_bytes(payload)
        digest = sha256_file(dst)
        if digest != sha256_file(src):
            raise DqExportError(f"published copy {dst} does not match {src}")
        out[key] = {
            "source_path": str(src.relative_to(cfg["_root"])),
            "path": str(dst.relative_to(cfg["_root"])),
            "sha256": digest,
        }
    return out


def build_record(cfg: dict, *, run_id: str | None = None) -> dict:
    """Assemble the ``kind="dq_export"`` record from the artifacts on disk."""
    dq_raw_path = rel(cfg, cfg["dq_raw"])
    if not dq_raw_path.exists():
        raise DqExportError(
            f"{dq_raw_path} does not exist — run `make demo-dq` (phase collect) first"
        )
    raw = json.loads(dq_raw_path.read_text())
    waterfall_path = rel(cfg, cfg["waterfall_json"])
    build_summary_path = rel(cfg, cfg["build_summary"])
    for label, path in (("waterfall_json", waterfall_path), ("build_summary", build_summary_path)):
        if not path.exists():
            raise DqExportError(
                f"{label} {path} does not exist — the record anchors its sha256; "
                "run `make data` / `make waterfall` before recording."
            )
    waterfall = json.loads(waterfall_path.read_text())

    # Re-check the identities JVM-free, against the files as they are right now.
    hc = headline_counts(waterfall, raw["quarantine"]["total_rows"])
    if hc != raw["headline_counts"]:
        raise DqExportError(
            "data/waterfall.json no longer yields the headline counts recorded in dq_raw.json — "
            "re-run the collect phase"
        )
    recon = reconciliation_checks(hc, raw["kcore_funnel"], raw["contract_matrix"])
    if not recon["all_ok"]:
        bad = [c for c in recon["checks"] if not c["ok"]]
        raise DqExportError(
            "reconciliation FAILED at record time:\n  "
            + "\n  ".join(f"{c['name']}: {c['lhs']} != {c['rhs']}" for c in bad)
        )

    published = publish(cfg, dq_raw_path=dq_raw_path)
    rid, rts = resolve_run_id(run_id)
    git = runlog.git_info()
    config_path = str(Path(cfg["_config_path"]))
    manifest_path = rel(cfg, cfg["dataset_manifest"]) if cfg.get("dataset_manifest") else runlog.DEFAULT_MANIFEST_PATH

    return {
        "schema_version": runlog.record_schema_version,
        "kind": RECORD_KIND,
        "run_id": rid,
        "run_ts": rts,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "config_path": config_path,
        "config_hash": runlog.config_hash(rel(cfg, config_path)),
        "dataset_manifest_hash": runlog.dataset_manifest_hash(manifest_path),
        "headline_run_id": str(raw["headline_run_id"]),
        "audit_run_id": str(raw["audit_run_id"]),
        "audit_run_ts": str(raw["audit_run_ts"]),
        "build_run_id": str(raw["build_run_id"]),
        "waterfall_path": str(waterfall_path.relative_to(cfg["_root"])),
        "waterfall_sha256": sha256_file(waterfall_path),
        "build_summary_path": str(build_summary_path.relative_to(cfg["_root"])),
        "build_summary_sha256": sha256_file(build_summary_path),
        "dq_raw_path": str(dq_raw_path.relative_to(cfg["_root"])),
        "dq_raw_sha256": sha256_file(dq_raw_path),
        "published_artifacts": published,
        # `iceberg_snapshots` is the repo-wide key for "the warehouse state this
        # record describes" (export_receipts copies it verbatim into the drawer):
        # for a DQ export that is the headline-pinned set the guard asserted.
        # `table_snapshots` additionally pins the ledger tables actually READ.
        "iceberg_snapshots": {t: int(s) for t, s in sorted(raw["headline_snapshots"].items())},
        "table_snapshots": {t: int(s) for t, s in sorted(raw["table_snapshots"].items())},
        "headline_counts": copy.deepcopy(raw["headline_counts"]),
        "contract_summary": copy.deepcopy(raw["contract_summary"]),
        "quarantine_totals": {
            "build_run_id": str(raw["quarantine"]["build_run_id"]),
            "total_rows": int(raw["quarantine"]["total_rows"]),
            "by_reason": copy.deepcopy(raw["quarantine"]["by_reason"]),
        },
        "measured_rates": {
            k: {kk: v[kk] for kk in ("source", "table", "rows", "denominator", "rate")}
            for k, v in raw["measured_rates"].items()
        },
        "kcore_iterations": len(raw["kcore_funnel"]),
        "reconciliation": recon,
        "hardware": runlog.hardware_string(),
    }


IDENTITY_FIELDS = (
    "waterfall_sha256",
    "build_summary_sha256",
    "dq_raw_sha256",
    "audit_run_id",
    "build_run_id",
    "headline_counts",
)


def find_equivalent(records: list[dict], record: dict) -> dict | None:
    """An already-appended ``dq_export`` record attesting the same evidence.

    The append is not idempotent by construction (run_id and git_sha move), so
    the guard is content-based: same artifact digests, same audit, same counts.
    Invariant #3 forbids editing the log, so a duplicate would be permanent.
    """
    for rec in records:
        if rec.get("kind") != RECORD_KIND:
            continue
        if all(rec.get(f) == record.get(f) for f in IDENTITY_FIELDS):
            return rec
    return None


def read_records(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# --- CLI ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/dq_export.yaml")
    ap.add_argument(
        "--phase",
        choices=("collect", "record", "all"),
        default="all",
        help="collect: Spark read-only pull. record: hash, publish, append (JVM-free).",
    )
    ap.add_argument("--dry-run", action="store_true", help="build and print the record; append nothing")
    ap.add_argument("--append", action="store_true", help="append the record to results/runs.jsonl")
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--force",
        action="store_true",
        help="append even if an equivalent dq_export record already exists",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    if args.phase in ("collect", "all"):
        collect(cfg)

    if args.phase == "collect":
        return 0

    if args.append == args.dry_run:
        ap.error("pass exactly one of --dry-run / --append for the record phase")

    record = build_record(cfg, run_id=args.run_id)
    print(json.dumps(record, indent=2, ensure_ascii=False))

    runs_log = rel(cfg, cfg["runs_log"])
    existing = find_equivalent(read_records(runs_log), record)
    if existing is not None and not args.force:
        print(
            f"\nAN EQUIVALENT RECORD ALREADY EXISTS: {existing['run_id']} attests the same "
            f"artifacts and counts. Not appending (pass --force to override; the log is "
            f"append-only, so a duplicate cannot be removed).",
            file=sys.stderr,
        )
        return 0 if args.dry_run else 3

    if args.dry_run:
        print(
            f"\nDRY RUN — nothing appended. {runs_log} is unchanged.\n"
            f"  compact line length: {len(json.dumps(record, separators=(',', ':')))} bytes\n"
            f"  to append (from a clean tree, after committing):\n"
            f"    uv run python -m batch_recsys_lab.demo.dq_export_job "
            f"--config {args.config} --phase record --append"
        )
        return 0

    if record["git_dirty"]:
        print(
            "REFUSING TO APPEND from a dirty working tree: a dq_export record must name the "
            "commit that produced dq_raw.json (CLAUDE.md invariant #3). Commit first.",
            file=sys.stderr,
        )
        return 4

    runlog.append_record(record, runs_log)
    print(f"\nappended kind={RECORD_KIND} run_id={record['run_id']} to {runs_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
