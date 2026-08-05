"""Reconciliation waterfall (Phase 1, T4b; UPGRADE_PLAN.md §8, acceptance #2).

The waterfall is the *proof* that no row is silently lost between layers. For
each dataset chain we assemble one edge per stage transition, attach a reason to
every row that leaves the stage, and then HARD-ASSERT three things:

1. **Conservation** — on every edge, ``Σ reason-rows == rows(stage_from)`` (the
   source-of-truth count recorded by the producing job).
2. **Live agreement** — the ``kept`` reason-count of every edge equals the
   *actual* ``count()`` of the target Iceberg table re-read at runtime (not a
   cached summary number). This catches a summary that drifted from the table.
3. **Chaining** — ``kept(edge N) == rows(stage_from)`` of ``edge N+1`` within a
   chain, so the layers compose end-to-end.

Any failure raises :class:`WaterfallError` (non-zero exit) with a message naming
the offending edge and the two disagreeing numbers — that message is the whole
point of the task.

Chains (D3):
    reviews = raw → bronze → silver → gold   (gold = 5-core row accounting)
    items   = raw → bronze → silver          (gold.item_features is a projection,
                                               not part of the row waterfall)

Inputs consumed (all produced by earlier tasks):
    * ``data/ingest_summary.jsonl``  — raw → bronze edge (raw = written + corrupt)
    * ``data/build_summary.jsonl``   — bronze → silver edge (silver.py conservation)
    * ``local.dq.kcore_funnel``      — silver → gold edge for reviews (kcore.py)

Outputs:
    * append edge rows to ``local.dq.waterfall``
    * idempotent ``## Reconciliation waterfall`` section in ``data/MANIFEST.md``
    * ``data/waterfall.json`` (same structure) for the demo DQ dashboard
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    LongType,
    StringType,
    StructField,
    StructType,
)

# run_id resolution reused (import, don't reimplement) from kcore.
from batch_recsys_lab.features.kcore import _resolve_run_id
from batch_recsys_lab.spark_session import get_spark

# --- Locations / defaults ----------------------------------------------------
DEFAULT_INGEST_SUMMARY = "data/ingest_summary.jsonl"
DEFAULT_BUILD_SUMMARY = "data/build_summary.jsonl"
DEFAULT_MANIFEST = "data/MANIFEST.md"
DEFAULT_JSON_OUT = "data/waterfall.json"

DEFAULT_BRONZE_REVIEWS = "local.bronze.reviews"
DEFAULT_BRONZE_ITEMS = "local.bronze.items"
DEFAULT_SILVER_INTERACTIONS = "local.silver.interactions"
DEFAULT_SILVER_ITEMS = "local.silver.items"
DEFAULT_GOLD_5CORE = "local.gold.interactions_5core"
DEFAULT_WATERFALL_TABLE = "local.dq.waterfall"
DEFAULT_FUNNEL_TABLE = "local.dq.kcore_funnel"

# The reason that names the surviving rows of an edge. Its row-count MUST equal
# the live count of the target table (assertion #2).
KEPT_REASON = "kept"

SECTION_HEADER = "## Reconciliation waterfall"
# Section-upsert regex — same shape as ingest/reconcile.py's SECTION_RE idiom
# (header line, non-greedy body, up to the next '## ' header or EOF). Adapted to
# *replace-or-append* (reconcile only replaces); we do not fork the pattern.
SECTION_RE = re.compile(
    rf"{re.escape(SECTION_HEADER)}\n(.*?)(\n## |\Z)", flags=re.DOTALL
)

WATERFALL_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), False),
        StructField("dataset", StringType(), False),
        StructField("stage_from", StringType(), False),
        StructField("stage_to", StringType(), False),
        StructField("reason", StringType(), False),
        StructField("rows", LongType(), False),
    ]
)


class WaterfallError(RuntimeError):
    """A reconciliation assertion failed. The message names the edge + numbers."""


# --- Edge model --------------------------------------------------------------


@dataclass
class Edge:
    dataset: str
    stage_from: str
    stage_to: str
    source_rows: int
    reasons: list[tuple[str, int]]  # (reason, rows); one entry named KEPT_REASON
    target_table: str  # Iceberg table whose live count must equal kept_rows

    @property
    def edge_label(self) -> str:
        return f"{self.dataset}: {self.stage_from} -> {self.stage_to}"

    @property
    def reason_sum(self) -> int:
        return sum(n for _, n in self.reasons)

    @property
    def kept_rows(self) -> int:
        for reason, n in self.reasons:
            if reason == KEPT_REASON:
                return n
        raise WaterfallError(f"{self.edge_label}: no '{KEPT_REASON}' reason on edge")


# --- Input readers -----------------------------------------------------------


def _latest_per_key(path: str, key: str) -> dict[str, dict]:
    """Read a JSONL summary log, returning the LAST record per ``key`` value.

    The logs are append-only, so the last line for a key is the latest run.
    """
    p = Path(path)
    if not p.exists():
        raise WaterfallError(f"summary log not found: {path}")
    latest: dict[str, dict] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        latest[rec[key]] = rec
    return latest


def _read_funnel(spark: SparkSession, funnel_table: str, run_id: str | None):
    """Return the k-core funnel rows (list of dicts) for ``run_id``.

    If ``run_id`` is None, the latest run_id present in the table is used
    (run_ids sort chronologically). Rows are returned ordered by iteration.
    """
    if not spark.catalog.tableExists(funnel_table):
        raise WaterfallError(
            f"k-core funnel table {funnel_table} does not exist; run features.kcore first"
        )
    df = spark.table(funnel_table)
    if run_id is None:
        picked = df.selectExpr("max(run_id) as r").first()["r"]
        if picked is None:
            raise WaterfallError(f"k-core funnel table {funnel_table} is empty")
        run_id = picked
    rows = (
        df.where(df.run_id == run_id)
        .orderBy("iteration")
        .select("run_id", "iteration", "rows", "users", "items", "converged", "wall_clock_s")
        .collect()
    )
    if not rows:
        raise WaterfallError(
            f"k-core funnel table {funnel_table} has no rows for run_id={run_id!r}"
        )
    return run_id, [r.asDict() for r in rows]


def _count(spark: SparkSession, table: str) -> int:
    return int(spark.table(table).count())


# --- Edge assembly -----------------------------------------------------------


def _build_edges(
    spark: SparkSession,
    *,
    ingest_summary_path: str,
    build_summary_path: str,
    bronze_reviews_table: str,
    bronze_items_table: str,
    silver_interactions_table: str,
    silver_items_table: str,
    gold_5core_table: str,
    funnel_table: str,
    run_id: str | None,
    allow_missing_gold: bool,
) -> tuple[dict[str, list[Edge]], list[dict], int, str | None]:
    """Assemble per-dataset edge lists from the summaries + live tables.

    Returns (edges_by_dataset, kcore_funnel_rows, gold_present_flag, kcore_run_id).
    """
    ingest = _latest_per_key(ingest_summary_path, "table_name")
    build = _latest_per_key(build_summary_path, "table")

    def _ingest_rec(name: str) -> dict:
        if name not in ingest:
            raise WaterfallError(
                f"no ingest_summary record for table_name={name!r} in {ingest_summary_path}"
            )
        return ingest[name]

    def _build_rec(name: str) -> dict:
        if name not in build:
            raise WaterfallError(
                f"no build_summary record for table={name!r} in {build_summary_path}"
            )
        return build[name]

    def _raw_bronze_edge(dataset: str, ing: dict, target_table: str) -> Edge:
        written = int(ing["written"])
        corrupt = int(ing["corrupt"])
        return Edge(
            dataset=dataset,
            stage_from="raw",
            stage_to="bronze",
            source_rows=written + corrupt,
            reasons=[(KEPT_REASON, written), ("corrupt", corrupt)],
            target_table=target_table,
        )

    def _bronze_silver_edge(dataset: str, b: dict, target_table: str) -> Edge:
        reasons: list[tuple[str, int]] = [(KEPT_REASON, int(b["kept"]))]
        for cid, n in b.get("quarantined", {}).items():
            reasons.append((f"quarantine:{cid}", int(n)))
        reasons.append(("exact_duplicate", int(b.get("exact_duplicate", 0))))
        reasons.append(
            ("superseded_by_later_review", int(b.get("superseded_by_later_review", 0)))
        )
        return Edge(
            dataset=dataset,
            stage_from="bronze",
            stage_to="silver",
            source_rows=int(b["input_rows"]),
            reasons=reasons,
            target_table=target_table,
        )

    edges: dict[str, list[Edge]] = {"reviews": [], "items": []}

    # --- items chain: raw -> bronze -> silver
    edges["items"].append(
        _raw_bronze_edge("items", _ingest_rec("items"), bronze_items_table)
    )
    edges["items"].append(
        _bronze_silver_edge("items", _build_rec("items"), silver_items_table)
    )

    # --- reviews chain: raw -> bronze -> silver -> gold
    edges["reviews"].append(
        _raw_bronze_edge("reviews", _ingest_rec("reviews"), bronze_reviews_table)
    )
    edges["reviews"].append(
        _bronze_silver_edge(
            "reviews", _build_rec("interactions"), silver_interactions_table
        )
    )

    # silver -> gold (reviews only): k-core row accounting.
    kcore_run_id, funnel = _read_funnel(spark, funnel_table, run_id)
    iter0 = funnel[0]
    final = funnel[-1]
    if not bool(final["converged"]):
        raise WaterfallError(
            f"k-core funnel run_id={kcore_run_id!r} did not converge; refusing to "
            "publish a waterfall against a non-converged gold table"
        )
    pruned = int(iter0["rows"]) - int(final["rows"])
    edges["reviews"].append(
        Edge(
            dataset="reviews",
            stage_from="silver",
            stage_to="gold",
            source_rows=int(iter0["rows"]),
            reasons=[(KEPT_REASON, int(final["rows"])), ("kcore_pruned", pruned)],
            target_table=gold_5core_table,
        )
    )

    gold_present = spark.catalog.tableExists(gold_5core_table)
    if not gold_present and not allow_missing_gold:
        raise WaterfallError(
            f"gold table {gold_5core_table} does not exist; run features.kcore, "
            "or pass --allow-missing-gold to skip the gold live-count check"
        )

    return edges, funnel, gold_present, kcore_run_id


# --- Assertions --------------------------------------------------------------


def _assert_edges(
    spark: SparkSession,
    edges_by_dataset: dict[str, list[Edge]],
    gold_5core_table: str,
    gold_present: bool,
) -> dict[str, dict[str, int]]:
    """Run the three exactness assertions; return live target counts per edge.

    Result maps ``edge_label -> {"target_count": int | -1}`` (-1 = check skipped).
    """
    live: dict[str, dict[str, int]] = {}
    for dataset, edges in edges_by_dataset.items():
        prev_kept: int | None = None
        prev_label: str | None = None
        for e in edges:
            # (1) conservation: Σ reason-rows == source.
            if e.reason_sum != e.source_rows:
                raise WaterfallError(
                    f"[{e.edge_label}] conservation FAILED: "
                    f"Σ reason-rows={e.reason_sum} != source rows={e.source_rows} "
                    f"(reasons={e.reasons})"
                )

            # (2) live agreement: kept == actual target-table count.
            skip_gold = (
                e.target_table == gold_5core_table and not gold_present
            )
            if skip_gold:
                print(
                    f"[warn] gold table {gold_5core_table} missing; skipping live "
                    f"count check for edge [{e.edge_label}] (--allow-missing-gold)"
                )
                target_count = -1
            else:
                target_count = _count(spark, e.target_table)
                if e.kept_rows != target_count:
                    raise WaterfallError(
                        f"[{e.edge_label}] live-count FAILED: kept={e.kept_rows} != "
                        f"count({e.target_table})={target_count}"
                    )

            # (3) chaining: kept(edge N) == source(edge N+1).
            if prev_kept is not None and prev_kept != e.source_rows:
                raise WaterfallError(
                    f"[{dataset}] chaining FAILED between [{prev_label}] and "
                    f"[{e.edge_label}]: kept={prev_kept} != next source={e.source_rows}"
                )
            prev_kept = e.kept_rows
            prev_label = e.edge_label
            live[e.edge_label] = {"target_count": target_count}
    return live


# --- dq.waterfall append -----------------------------------------------------


def _append_waterfall_table(
    spark: SparkSession, waterfall_table: str, run_id: str, edges_by_dataset: dict
) -> None:
    rows = [
        (run_id, e.dataset, e.stage_from, e.stage_to, reason, int(n))
        for edges in edges_by_dataset.values()
        for e in edges
        for reason, n in e.reasons
    ]
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {waterfall_table.rsplit('.', 1)[0]}")
    sdf = spark.createDataFrame(rows, WATERFALL_SCHEMA)
    if spark.catalog.tableExists(waterfall_table):
        sdf.writeTo(waterfall_table).append()
    else:
        sdf.writeTo(waterfall_table).create()


# --- JSON structure ----------------------------------------------------------

_CHAINS = {
    "reviews": ["raw", "bronze", "silver", "gold"],
    "items": ["raw", "bronze", "silver"],
}


def _build_result(
    run_id: str,
    edges_by_dataset: dict[str, list[Edge]],
    live: dict[str, dict[str, int]],
    funnel: list[dict],
    kcore_run_id: str | None,
) -> dict:
    datasets: dict[str, dict] = {}
    for dataset, edges in edges_by_dataset.items():
        edge_jsons = []
        for e in edges:
            tc = live[e.edge_label]["target_count"]
            edge_jsons.append(
                {
                    "stage_from": e.stage_from,
                    "stage_to": e.stage_to,
                    "source_rows": e.source_rows,
                    "kept_rows": e.kept_rows,
                    "reason_sum": e.reason_sum,
                    "target_table": e.target_table,
                    "target_count": tc,
                    "sum_ok": e.reason_sum == e.source_rows,
                    "count_ok": tc == -1 or tc == e.kept_rows,
                    "reasons": [{"reason": r, "rows": n} for r, n in e.reasons],
                }
            )
        entry: dict = {"chain": _CHAINS[dataset], "edges": edge_jsons}
        if dataset == "reviews":
            entry["kcore_run_id"] = kcore_run_id
            entry["kcore_funnel"] = [
                {
                    "iteration": int(r["iteration"]),
                    "rows": int(r["rows"]),
                    "users": int(r["users"]),
                    "items": int(r["items"]),
                    "converged": bool(r["converged"]),
                    "wall_clock_s": float(r["wall_clock_s"]),
                }
                for r in funnel
            ]
        datasets[dataset] = entry
    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
    }


# --- MANIFEST rendering (idempotent section) ---------------------------------


def _fmt(n: int) -> str:
    return "n/a" if n == -1 else f"{n:,}"


def _render_section(result: dict) -> str:
    lines: list[str] = [SECTION_HEADER, ""]
    lines.append(f"Run ID: `{result['run_id']}` · generated {result['generated_at']}")
    lines.append("")
    lines.append(
        "Every dropped row carries a reason; per edge, Σ reason-rows == the source "
        "count AND the `kept` count == the live Iceberg table count (re-read at "
        "publish time). Assertions are enforced in code (non-zero exit on drift)."
    )
    for dataset, entry in result["datasets"].items():
        chain = " → ".join(entry["chain"])
        lines.append("")
        lines.append(f"### {dataset}  ({chain})")
        lines.append("")
        lines.append("| stage_from | stage_to | reason | rows |")
        lines.append("|---|---|---|---|")
        for e in entry["edges"]:
            for r in e["reasons"]:
                lines.append(
                    f"| {e['stage_from']} | {e['stage_to']} | {r['reason']} | {r['rows']:,} |"
                )
        lines.append("")
        lines.append("Reconciliation checks:")
        for e in entry["edges"]:
            sum_mark = "✓" if e["sum_ok"] else "✗"
            cnt = (
                "target not checked (missing gold)"
                if e["target_count"] == -1
                else f"target `{e['target_table']}` count = {_fmt(e['target_count'])} "
                + ("✓" if e["count_ok"] else "✗")
            )
            lines.append(
                f"- {e['stage_from']} → {e['stage_to']}: "
                f"Σ = {e['reason_sum']:,} = source {e['source_rows']:,} {sum_mark}; {cnt}"
            )
        if dataset == "reviews" and entry.get("kcore_funnel"):
            lines.append("")
            lines.append(f"#### k-core funnel (reviews, run `{entry.get('kcore_run_id')}`)")
            lines.append("")
            lines.append("| iteration | rows | users | items | converged | wall_clock_s |")
            lines.append("|---|---|---|---|---|---|")
            for f in entry["kcore_funnel"]:
                lines.append(
                    f"| {f['iteration']} | {f['rows']:,} | {f['users']:,} | "
                    f"{f['items']:,} | {f['converged']} | {f['wall_clock_s']} |"
                )
    lines.append("")
    return "\n".join(lines)


def _upsert_section(text: str, new_section: str) -> str:
    """Replace an existing ``## Reconciliation waterfall`` section, else append it.

    Adapted from ingest/reconcile.py's SECTION_RE idiom; that helper errors on a
    missing section (its section is created by `make manifest`). Ours may be the
    first writer, so a missing section appends instead of raising.
    """
    if SECTION_RE.search(text):
        def _sub(match: re.Match) -> str:
            trailing = match.group(2)
            if trailing == "\n## ":
                return new_section + "\n## "
            return new_section
        return SECTION_RE.sub(_sub, text, count=1)
    if not text:
        return new_section
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return text + sep + new_section


def _write_manifest(manifest_path: str, result: dict) -> None:
    p = Path(manifest_path)
    text = p.read_text() if p.exists() else ""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_upsert_section(text, _render_section(result)))


def _write_json(json_out: str, result: dict) -> None:
    p = Path(json_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2) + "\n")


# --- Orchestration -----------------------------------------------------------


def run_waterfall(
    spark: SparkSession,
    run_id: str | None = None,
    *,
    ingest_summary_path: str = DEFAULT_INGEST_SUMMARY,
    build_summary_path: str = DEFAULT_BUILD_SUMMARY,
    manifest_path: str = DEFAULT_MANIFEST,
    json_out: str = DEFAULT_JSON_OUT,
    bronze_reviews_table: str = DEFAULT_BRONZE_REVIEWS,
    bronze_items_table: str = DEFAULT_BRONZE_ITEMS,
    silver_interactions_table: str = DEFAULT_SILVER_INTERACTIONS,
    silver_items_table: str = DEFAULT_SILVER_ITEMS,
    gold_5core_table: str = DEFAULT_GOLD_5CORE,
    waterfall_table: str = DEFAULT_WATERFALL_TABLE,
    funnel_table: str = DEFAULT_FUNNEL_TABLE,
    allow_missing_gold: bool = False,
    write_manifest: bool = True,
    write_json: bool = True,
    append_table: bool = True,
) -> dict:
    """Assemble, assert, persist and publish the reconciliation waterfall.

    ``run_id`` doubles as the k-core funnel filter (which run's gold to reconcile
    against) and the ``dq.waterfall`` row key. If None, the latest funnel run_id
    is used and a fresh run_id is generated for the appended rows.
    """
    wf_run_id = _resolve_run_id(run_id)

    edges_by_dataset, funnel, gold_present, kcore_run_id = _build_edges(
        spark,
        ingest_summary_path=ingest_summary_path,
        build_summary_path=build_summary_path,
        bronze_reviews_table=bronze_reviews_table,
        bronze_items_table=bronze_items_table,
        silver_interactions_table=silver_interactions_table,
        silver_items_table=silver_items_table,
        gold_5core_table=gold_5core_table,
        funnel_table=funnel_table,
        run_id=run_id,
        allow_missing_gold=allow_missing_gold,
    )

    live = _assert_edges(spark, edges_by_dataset, gold_5core_table, gold_present)

    if append_table:
        _append_waterfall_table(spark, waterfall_table, wf_run_id, edges_by_dataset)

    result = _build_result(wf_run_id, edges_by_dataset, live, funnel, kcore_run_id)

    if write_manifest:
        _write_manifest(manifest_path, result)
    if write_json:
        _write_json(json_out, result)

    return result


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.features.waterfall")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--master", default="local[10]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--allow-missing-gold", action="store_true")
    # jsonl path + table overrides (tests point these at tmp paths/tables).
    parser.add_argument("--ingest-summary", default=DEFAULT_INGEST_SUMMARY)
    parser.add_argument("--build-summary", default=DEFAULT_BUILD_SUMMARY)
    parser.add_argument("--bronze-reviews-table", default=DEFAULT_BRONZE_REVIEWS)
    parser.add_argument("--bronze-items-table", default=DEFAULT_BRONZE_ITEMS)
    parser.add_argument("--silver-interactions-table", default=DEFAULT_SILVER_INTERACTIONS)
    parser.add_argument("--silver-items-table", default=DEFAULT_SILVER_ITEMS)
    parser.add_argument("--gold-5core-table", default=DEFAULT_GOLD_5CORE)
    parser.add_argument("--waterfall-table", default=DEFAULT_WATERFALL_TABLE)
    parser.add_argument("--funnel-table", default=DEFAULT_FUNNEL_TABLE)
    args = parser.parse_args(argv)

    spark = get_spark(
        app_name="waterfall",
        warehouse=args.warehouse,
        master=args.master,
        driver_memory=args.driver_memory,
    )
    try:
        result = run_waterfall(
            spark,
            run_id=args.run_id,
            ingest_summary_path=args.ingest_summary,
            build_summary_path=args.build_summary,
            manifest_path=args.manifest_path,
            json_out=args.json_out,
            bronze_reviews_table=args.bronze_reviews_table,
            bronze_items_table=args.bronze_items_table,
            silver_interactions_table=args.silver_interactions_table,
            silver_items_table=args.silver_items_table,
            gold_5core_table=args.gold_5core_table,
            waterfall_table=args.waterfall_table,
            funnel_table=args.funnel_table,
            allow_missing_gold=args.allow_missing_gold,
        )
    except WaterfallError as exc:
        print(f"[error] waterfall reconciliation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        spark.stop()

    # Summary JSON MUST be the last stdout line (repo convention).
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
