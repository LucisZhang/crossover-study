"""Per-stage lineage table: rows / bytes / wall-clock for every pipeline stage
(Phase 5, T24).

    python -m batch_recsys_lab.ops.lineage [--check-only]

One row per stage, from raw download through the ops chain, each number carrying
the ledger or Iceberg metadata file it came from. Nothing here re-runs, re-counts
or re-measures anything: every figure is *read back* from an artifact that some
earlier run already committed to disk. That is the point of the exhibit — if a
number was never persisted at build time it stays ``null`` and is footnoted, it
is never recovered by re-running the stage (re-running gold would move snapshots
that recorded eval records cite).

Sources, in the order the table lists them:

======================  ==========================================================
stage                   numbers come from
======================  ==========================================================
raw download            ``data/MANIFEST.md`` § Files (per-file byte sizes)
bronze.*                ``data/ingest_summary.jsonl`` (rows, wall clock)
silver.*                ``data/build_summary.jsonl`` (rows, wall clock)
gold.interactions_5core ``dq.kcore_funnel`` parquet (wall clock summed over iters)
gold.{user_stats,…}     Iceberg metadata only — build runtimes were never persisted
gold.item_text          Iceberg metadata only — see the note below
eval extract cache      on-disk file sizes + ``cache_manifest.json``
headline eval/reproduce ``results/runs.jsonl`` (wall clock) + artifact file size
ops chain               ``results/runs.jsonl`` ``kind="ops"`` records
======================  ==========================================================

``gold.item_text`` deserves the note: ``data/eval/text/<snapshot>/export_manifest.json``
does carry a ``wall_clock_s``, but that is the JVM-free *export* step (reordering
the table to the cache's item order), a different stage from the table build.
``features.item_text.build_item_text`` measures its own runtime and prints it in
the summary line — it never writes it anywhere — so the build row's wall clock is
null and footnoted like the other gold projections. Borrowing the export's number
for the build would be exactly the fiction this module exists to prevent.

The ops chain is one row per ``kind="ops"`` RECORD, in log order — never one row
per scenario. The exhibit deliberately runs ``compact`` and ``expire`` more than
once, including runs that measurably do nothing (a compaction with
``rewritten_files == 0`` is the *point* of the no-op exhibit), so collapsing
repeats or dropping no-ops would delete the finding. Row labels disambiguate from
the record's own data (month, rewritten/added file counts, retain_last and
deleted file counts), never from a hardcoded list.

Bytes and snapshot IDs for every Iceberg table come from the current snapshot's
summary (``total-files-size`` / ``total-records``) read JVM-free through
:mod:`batch_recsys_lab.ops.snapshot_metrics` — this module never starts Spark.

**Completeness contract.** The generator exits non-zero, naming what is missing,
if any expected stage cannot be assembled: a ledger file that is absent, a table
whose metadata is gone, or fewer ops scenarios than the expected MINIMUM
(:data:`DEFAULT_EXPECTED_OPS`). Only the fields listed as nullable per stage (see
``required`` in each builder) may be ``null``, and every null wall clock must
carry a footnote flag. A partial table is never written. The expected ops set is
a floor, not an equality: extra records are enumerated in ``ops_observed`` and
rendered as their own rows, never treated as an error and never dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

from batch_recsys_lab.contracts.engine import _resolve_run_id
from batch_recsys_lab.eval import runlog
from batch_recsys_lab.ops.snapshot_metrics import table_metadata_for

schema_version = 1

DEFAULT_WAREHOUSE = "data/warehouse"
DEFAULT_RESULTS = "results/runs.jsonl"
DEFAULT_MANIFEST = "data/MANIFEST.md"
DEFAULT_INGEST_SUMMARY = "data/ingest_summary.jsonl"
DEFAULT_BUILD_SUMMARY = "data/build_summary.jsonl"
DEFAULT_CACHE_ROOT = "data/eval/cache"
DEFAULT_HEADLINE_CONFIG = "configs/headline.yaml"
DEFAULT_OUT_JSON = "results/lineage.json"
DEFAULT_OUT_MD = "results/lineage.md"

#: The MINIMUM ops chain the exhibit must contain, as a multiset of scenario
#: names. A floor, not an equality: the compaction exhibit deliberately runs
#: compact and expire more than once, and those extra records get their own rows.
DEFAULT_EXPECTED_OPS = (
    "backfill",
    "append",
    "append",
    "append",
    "upsert",
    "fragment",
    "compact",
    "expire",
)

#: Wall-clock footnote flag. Used verbatim as ``wall_clock_source`` wherever the
#: runtime is null because nobody persisted it, not because the stage was fast.
NOT_PERSISTED = "runtime_not_persisted_at_build"

FOOTNOTES = {
    NOT_PERSISTED: (
        "Wall clock is null because this stage never wrote a runtime to any "
        "ledger. It is deliberately NOT recovered by re-running: re-running a "
        "gold build would move Iceberg snapshots that recorded eval records "
        "cite, and re-running the extract would overwrite the snapshot-keyed "
        "cache the headline run was scored against."
    ),
}

# Every row carries exactly these keys, in this order (stable JSON key order).
ROW_KEYS = (
    "stage",
    "layer",
    "table",
    "rows_in",
    "rows_out",
    "bytes",
    "wall_clock_s",
    "wall_clock_source",
    "snapshot_id",
    "source_of_truth",
    "required",
    "missing",
    "complete",
)


class LineageError(RuntimeError):
    """A stage could not be assembled from the artifacts on disk."""


# --- ledger parsers -----------------------------------------------------------


_MANIFEST_FILE_RE = re.compile(r"^###\s+(?P<name>\S+)\s*$")
_MANIFEST_SIZE_RE = re.compile(r"^-\s+Size\s+\(bytes\):\s*(?P<size>\d+)\s*$")


def parse_manifest_sizes(manifest_path: str | Path) -> dict[str, int]:
    """``{filename: size_bytes}`` from the ``## Files`` section of MANIFEST.md.

    Only ``### <name>`` blocks that actually carry a ``- Size (bytes): N`` line
    are returned, so prose sections cannot leak in as phantom files.

    NOTE: the manifest's free-form ``Ingest wall-clock:`` line is deliberately
    NOT read here — it is an operator-supplied string (``make manifest
    --wall-clock``), and it disagrees with the machine-written
    ``ingest_summary.jsonl``. The machine ledger wins.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise LineageError(f"missing ledger: {path} (raw download byte sizes)")
    sizes: dict[str, int] = {}
    current: str | None = None
    for line in path.read_text().splitlines():
        head = _MANIFEST_FILE_RE.match(line.strip())
        if head:
            current = head.group("name")
            continue
        size = _MANIFEST_SIZE_RE.match(line.strip())
        if size and current:
            sizes[current] = int(size.group("size"))
            current = None
    if not sizes:
        raise LineageError(
            f"{path} has no '### <file>' block with a '- Size (bytes): N' line; "
            "raw download bytes cannot be sourced"
        )
    return sizes


def read_jsonl(path: str | Path, *, what: str) -> list[dict]:
    """All records of a JSONL ledger, in file order."""
    p = Path(path)
    if not p.exists():
        raise LineageError(f"missing ledger: {p} ({what})")
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    if not out:
        raise LineageError(f"empty ledger: {p} ({what})")
    return out


def last_by(records: list[dict], key: str) -> dict[str, dict]:
    """Last record per ``key`` value — these ledgers are append-only, so the
    last entry for a table is the one that produced the live table."""
    out: dict[str, dict] = {}
    for rec in records:
        k = rec.get(key)
        if k is not None:
            out[str(k)] = rec
    return out


def dir_bytes(path: str | Path) -> int:
    """Total size of every regular file under ``path`` (recursive)."""
    root = Path(path)
    if not root.is_dir():
        raise LineageError(f"missing directory: {root}")
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def kcore_funnel_iterations(warehouse: str | Path, run_id: str) -> list[dict]:
    """Funnel rows for one build run, iteration-ordered, read straight from the
    table's parquet files with pyarrow (the table is ~34 rows; no Spark).

    The data dir holds files from every snapshot ever appended, so rows from
    older ``make data`` runs are present too — hence the ``run_id`` filter,
    which is what actually selects the build that produced the live gold table.
    """
    import pyarrow.dataset as ds

    data_dir = Path(warehouse) / "dq" / "kcore_funnel" / "data"
    if not data_dir.is_dir():
        raise LineageError(
            f"missing k-core funnel data dir: {data_dir} "
            "(gold.interactions_5core wall clock)"
        )
    table = ds.dataset(str(data_dir), format="parquet").to_table()
    cols = table.to_pydict()
    rows = [
        {k: cols[k][i] for k in cols}
        for i in range(table.num_rows)
        if str(cols["run_id"][i]) == str(run_id)
    ]
    if not rows:
        raise LineageError(
            f"k-core funnel has no rows for run_id {run_id!r} "
            f"(present: {sorted({str(r) for r in cols['run_id']})})"
        )
    return sorted(rows, key=lambda r: int(r["iteration"]))


# --- row assembly -------------------------------------------------------------


def make_row(
    stage: str,
    layer: str,
    table: str,
    *,
    rows_in: int | None = None,
    rows_out: int | None = None,
    n_bytes: int | None = None,
    wall_clock_s: float | None = None,
    wall_clock_source: str | None = None,
    snapshot_id: int | None = None,
    source_of_truth: dict[str, str] | None = None,
    required: tuple[str, ...] = (),
) -> dict:
    """One lineage row, with its own completeness contract attached.

    ``required`` names the fields that MUST be non-null for this stage; anything
    else is nullable by design. A null ``wall_clock_s`` is only legal when
    ``wall_clock_source`` carries the :data:`NOT_PERSISTED` footnote flag, and
    that rule is enforced here rather than at the call sites.
    """
    if wall_clock_s is None and wall_clock_source is None:
        wall_clock_source = NOT_PERSISTED
    row = {
        "stage": stage,
        "layer": layer,
        "table": table,
        "rows_in": None if rows_in is None else int(rows_in),
        "rows_out": None if rows_out is None else int(rows_out),
        "bytes": None if n_bytes is None else int(n_bytes),
        "wall_clock_s": None if wall_clock_s is None else round(float(wall_clock_s), 3),
        "wall_clock_source": wall_clock_source,
        "snapshot_id": None if snapshot_id is None else int(snapshot_id),
        "source_of_truth": dict(source_of_truth or {}),
        "required": list(required),
    }
    row["missing"] = [f for f in required if row.get(f) is None]
    row["complete"] = not row["missing"]
    return {k: row[k] for k in ROW_KEYS}


def iceberg_stats(warehouse: str | Path, full_name: str) -> dict:
    """``{rows, bytes, snapshot_id, source}`` from the CURRENT snapshot summary."""
    meta = table_metadata_for(warehouse, full_name)
    if not meta["exists"]:
        raise LineageError(
            f"missing Iceberg metadata for {full_name} at {meta['metadata_dir']}"
        )
    current = meta["current_snapshot_id"]
    snap = next(
        (s for s in meta["snapshots"] if s["snapshot_id"] == current),
        None,
    )
    if snap is None:
        raise LineageError(
            f"{full_name}: current snapshot {current} not present in "
            f"v{meta['metadata_version']}.metadata.json"
        )
    rel = Path(meta["metadata_dir"]).name
    source = f"{Path(meta['metadata_dir']).parent.name}/{rel}/v{meta['metadata_version']}.metadata.json"
    return {
        "rows": snap["total_records"],
        "bytes": snap["total_files_size"],
        "snapshot_id": current,
        "source": f"iceberg:{full_name}@{source}",
    }


# --- stage builders -----------------------------------------------------------


def stage_raw_download(manifest_path: str | Path) -> dict:
    sizes = parse_manifest_sizes(manifest_path)
    names = sorted(sizes)
    return make_row(
        "raw_download",
        "raw",
        f"data/raw/ ({len(names)} files: {', '.join(names)})",
        n_bytes=sum(sizes.values()),
        wall_clock_source=NOT_PERSISTED,
        source_of_truth={"bytes": f"{manifest_path} § Files"},
        required=("bytes",),
    )


def stages_bronze(warehouse, ingest_summary_path) -> list[dict]:
    records = last_by(
        read_jsonl(ingest_summary_path, what="bronze rows + wall clock"), "table"
    )
    rows = []
    for full_name in ("local.bronze.reviews", "local.bronze.items"):
        rec = records.get(full_name)
        if rec is None:
            raise LineageError(
                f"{ingest_summary_path} has no record for {full_name} "
                "(bronze rows + wall clock)"
            )
        ice = iceberg_stats(warehouse, full_name)
        rows.append(
            make_row(
                full_name.split(".", 1)[1],
                "bronze",
                full_name,
                rows_in=rec["total_parsed"],
                rows_out=rec["written"],
                n_bytes=ice["bytes"],
                wall_clock_s=rec.get("wall_clock_s"),
                wall_clock_source=str(ingest_summary_path),
                snapshot_id=ice["snapshot_id"],
                source_of_truth={
                    "rows": str(ingest_summary_path),
                    "wall_clock_s": str(ingest_summary_path),
                    "bytes": ice["source"],
                    "snapshot_id": ice["source"],
                },
                required=("rows_in", "rows_out", "bytes", "wall_clock_s", "snapshot_id"),
            )
        )
    return rows


def stages_silver(warehouse, build_summary_path) -> list[dict]:
    records = last_by(
        read_jsonl(build_summary_path, what="silver rows + wall clock"), "table"
    )
    rows = []
    for short in ("items", "interactions"):
        rec = records.get(short)
        if rec is None:
            raise LineageError(
                f"{build_summary_path} has no record for table {short!r} "
                "(silver rows + wall clock)"
            )
        full_name = f"local.silver.{short}"
        ice = iceberg_stats(warehouse, full_name)
        rows.append(
            make_row(
                f"silver.{short}",
                "silver",
                full_name,
                rows_in=rec["input_rows"],
                rows_out=rec["kept"],
                n_bytes=ice["bytes"],
                wall_clock_s=rec.get("wall_clock_s"),
                wall_clock_source=f"{build_summary_path} (run {rec.get('run_id')})",
                snapshot_id=ice["snapshot_id"],
                source_of_truth={
                    "rows": str(build_summary_path),
                    "wall_clock_s": str(build_summary_path),
                    "bytes": ice["source"],
                    "snapshot_id": ice["source"],
                },
                required=("rows_in", "rows_out", "bytes", "wall_clock_s", "snapshot_id"),
            )
        )
    return rows


def stage_gold_core(warehouse, build_summary_path, gold_run_id: str | None) -> dict:
    """``gold.interactions_5core``: the only gold stage whose runtime survives —
    the k-core funnel table persisted per-iteration wall clocks, so the build
    time is their sum."""
    if gold_run_id is None:
        build = read_jsonl(build_summary_path, what="gold build run id")
        gold_run_id = str(build[-1].get("run_id"))
    iters = kcore_funnel_iterations(warehouse, gold_run_id)
    full_name = "local.gold.interactions_5core"
    ice = iceberg_stats(warehouse, full_name)

    final_rows = int(iters[-1]["rows"])
    if ice["rows"] is not None and final_rows != int(ice["rows"]):
        raise LineageError(
            f"{full_name}: k-core funnel run {gold_run_id} converged at "
            f"{final_rows:,} rows but the live table's snapshot summary reports "
            f"{int(ice['rows']):,} — the funnel and the table disagree, so the "
            "lineage row would be fiction"
        )

    funnel_src = f"iceberg:local.dq.kcore_funnel (run {gold_run_id}, {len(iters)} iterations)"
    return make_row(
        "gold.interactions_5core",
        "gold",
        full_name,
        rows_in=int(iters[0]["rows"]),
        rows_out=ice["rows"],
        n_bytes=ice["bytes"],
        wall_clock_s=sum(float(r["wall_clock_s"]) for r in iters),
        wall_clock_source=funnel_src,
        snapshot_id=ice["snapshot_id"],
        source_of_truth={
            "rows_in": funnel_src,
            "rows_out": ice["source"],
            "bytes": ice["source"],
            "wall_clock_s": funnel_src,
            "snapshot_id": ice["source"],
        },
        required=("rows_in", "rows_out", "bytes", "wall_clock_s", "snapshot_id"),
    )


def stages_gold_features(warehouse, core_rows: int | None) -> list[dict]:
    """The three projections off the 5-core table. Their build runtimes were
    never written to any ledger, so wall clock is null + footnoted."""
    rows = []
    for short in ("user_stats", "item_features", "popularity"):
        full_name = f"local.gold.{short}"
        ice = iceberg_stats(warehouse, full_name)
        rows.append(
            make_row(
                f"gold.{short}",
                "gold",
                full_name,
                rows_in=core_rows,
                rows_out=ice["rows"],
                n_bytes=ice["bytes"],
                wall_clock_s=None,
                wall_clock_source=NOT_PERSISTED,
                snapshot_id=ice["snapshot_id"],
                source_of_truth={
                    "rows_in": "iceberg:local.gold.interactions_5core",
                    "rows_out": ice["source"],
                    "bytes": ice["source"],
                    "snapshot_id": ice["source"],
                },
                required=("rows_in", "rows_out", "bytes", "snapshot_id"),
            )
        )
    return rows


def stage_gold_item_text(warehouse, catalog_rows: int | None) -> dict:
    """``gold.item_text`` (Phase 4, T9): the 5-core catalog joined to
    ``item_features`` and the raw ``bronze.items`` text fields.

    ``rows_in`` is the DISTINCT 5-core catalog count, not the 5-core interaction
    count — ``build_item_text`` joins off ``select(parent_asin).distinct()`` and
    asserts the written row count equals it. ``gold.item_features`` is that same
    per-item projection, so its row count is the honest, already-published source
    for this stage's input.

    Wall clock is null: the build times itself but only prints the number (see
    the module docstring).
    """
    full_name = "local.gold.item_text"
    ice = iceberg_stats(warehouse, full_name)
    return make_row(
        "gold.item_text",
        "gold",
        full_name,
        rows_in=catalog_rows,
        rows_out=ice["rows"],
        n_bytes=ice["bytes"],
        wall_clock_s=None,
        wall_clock_source=NOT_PERSISTED,
        snapshot_id=ice["snapshot_id"],
        source_of_truth={
            "rows_in": "iceberg:local.gold.item_features (= distinct 5-core catalog)",
            "rows_out": ice["source"],
            "bytes": ice["source"],
            "snapshot_id": ice["source"],
        },
        required=("rows_in", "rows_out", "bytes", "snapshot_id"),
    )


def stage_eval_cache(cache_root: str | Path, snapshot_id: int) -> dict:
    """The snapshot-keyed numpy/parquet extract cache the scoring process reads.

    Not an Iceberg table: bytes are the on-disk total, and the snapshot ID is the
    directory key (re-read from ``cache_manifest.json`` when present, so the row
    cites the cache's own claim rather than a path convention).
    """
    cache_dir = Path(cache_root) / str(snapshot_id)
    total = dir_bytes(cache_dir)
    manifest = cache_dir / "cache_manifest.json"
    sid = int(snapshot_id)
    sid_source = f"{cache_root}/<snapshot>/ directory key"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        claimed = (data.get("snapshot_ids") or {}).get("local.gold.interactions_5core")
        if claimed is not None:
            sid = int(claimed)
            sid_source = str(manifest)
    return make_row(
        "eval_extract_cache",
        "cache",
        str(cache_dir),
        n_bytes=total,
        wall_clock_s=None,
        wall_clock_source=NOT_PERSISTED,
        snapshot_id=sid,
        source_of_truth={"bytes": f"on-disk file sizes under {cache_dir}", "snapshot_id": sid_source},
        required=("bytes", "snapshot_id"),
    )


def _artifact_bytes(path: str | Path, *, root: Path) -> tuple[int, str]:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    if not p.is_file():
        raise LineageError(f"missing per-user artifact: {p}")
    return p.stat().st_size, str(path)


def stage_headline_eval(record: dict, *, root: Path) -> dict:
    n_bytes, art = _artifact_bytes(record["per_user_artifact"], root=root)
    return make_row(
        "headline_eval",
        "eval",
        art,
        rows_out=(record.get("protocol") or {}).get("n_users"),
        n_bytes=n_bytes,
        wall_clock_s=record.get("wall_clock_s"),
        wall_clock_source=f"results/runs.jsonl kind=eval run_id={record['run_id']}",
        snapshot_id=(record.get("iceberg_snapshots") or {}).get(
            "local.gold.interactions_5core"
        ),
        source_of_truth={
            "rows_out": f"runs.jsonl run_id={record['run_id']} protocol.n_users",
            "wall_clock_s": f"runs.jsonl run_id={record['run_id']} wall_clock_s",
            "bytes": f"on-disk size of {art}",
            "snapshot_id": f"runs.jsonl run_id={record['run_id']} iceberg_snapshots",
        },
        required=("rows_out", "bytes", "wall_clock_s", "snapshot_id"),
    )


def resolve_repro_artifact(
    record: dict, *, root: Path, model_name: str | None, override: str | None
) -> Path:
    """Locate the per-user parquet the reproduction wrote.

    The ``kind="reproduce"`` record does not carry the path: the inner eval run
    mints its OWN run id (seconds before the reproduce record's), so the filename
    cannot be derived from the record. Resolution order: explicit override, then
    the one file under ``<cache_repro>/per_user/`` matching the model name and the
    reproduce record's short git sha, then a sole file if there is exactly one.
    Ambiguity is an error, never a guess.
    """
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = root / p
        if not p.is_file():
            raise LineageError(f"--repro-per-user {p} does not exist")
        return p

    repro_cache = record.get("repro_cache_dir")
    if not repro_cache:
        raise LineageError(
            "reproduce record has no repro_cache_dir; pass --repro-per-user"
        )
    per_user_dir = Path(repro_cache).parent / "per_user"
    if not per_user_dir.is_dir():
        raise LineageError(f"missing reproduce per-user dir: {per_user_dir}")
    candidates = sorted(per_user_dir.glob("*.parquet"))
    if not candidates:
        raise LineageError(f"no per-user parquet under {per_user_dir}")

    narrowed = candidates
    if model_name:
        by_model = [c for c in narrowed if c.stem.endswith(model_name)]
        if by_model:
            narrowed = by_model
    sha = (record.get("git_sha") or "")[:7]
    if sha:
        by_sha = [c for c in narrowed if sha in c.name]
        if by_sha:
            narrowed = by_sha
    if len(narrowed) == 1:
        return narrowed[0]
    if len(candidates) == 1:
        return candidates[0]
    raise LineageError(
        f"cannot disambiguate the reproduce per-user artifact under {per_user_dir} "
        f"({[c.name for c in narrowed]}); pass --repro-per-user"
    )


def stage_reproduce(
    record: dict, *, root: Path, model_name: str | None, override: str | None
) -> dict:
    artifact = resolve_repro_artifact(
        record, root=root, model_name=model_name, override=override
    )
    # The reproduce record splits its runtime into the pinned time-travel extract
    # and the scoring pass; the stage's wall clock is their sum.
    extract = record.get("extract_wall_clock_s")
    score = record.get("eval_wall_clock_s")
    parts = [v for v in (extract, score) if v is not None]
    wall = sum(float(v) for v in parts) if parts else record.get("wall_clock_s")
    detail = record.get("per_user_compare_detail") or {}
    return make_row(
        "reproduce_headline",
        "eval",
        str(artifact),
        rows_out=detail.get("n_rows_repro"),
        n_bytes=artifact.stat().st_size,
        wall_clock_s=wall,
        wall_clock_source=(
            f"results/runs.jsonl kind=reproduce run_id={record['run_id']} "
            "(extract_wall_clock_s + eval_wall_clock_s)"
        ),
        source_of_truth={
            "rows_out": (
                f"runs.jsonl run_id={record['run_id']} "
                "per_user_compare_detail.n_rows_repro"
            ),
            "wall_clock_s": f"runs.jsonl run_id={record['run_id']}",
            "bytes": f"on-disk size of {artifact}",
        },
        required=("rows_out", "bytes", "wall_clock_s"),
    )


def ops_label(rec: dict) -> str:
    """Row label for one ops record, disambiguated from the RECORD'S OWN data.

    The chain runs several scenarios more than once, so ``ops.compact`` alone
    would be ambiguous — and which compaction was the measured no-op is the whole
    point of that exhibit. Every discriminator below is read off the record:

    * ``month`` (append, fragment) -> ``ops.append[2023-07]``
    * compact -> ``ops.compact[noop]`` when ``rewritten_files == 0``, else
      ``ops.compact[30->1]`` (rewritten -> added files)
    * expire -> ``ops.expire[retain=2,deleted=3]`` (retain_last + deleted data
      files), which separates the two-stage expiry's pinned-predecessor run from
      the run that actually reclaimed files

    Nothing here is hardcoded to a known chain; an unrecognised scenario falls
    back to the bare ``ops.<scenario>``, and :func:`stages_ops` appends ``#n`` if
    two rows still collide.
    """
    scenario = str(rec.get("scenario"))
    params = rec.get("params") or {}
    month = params.get("month")
    if month:
        return f"ops.{scenario}[{month}]"

    if scenario == "compact":
        rewritten = rec.get("rewritten_files")
        added = rec.get("added_files")
        if rewritten == 0:
            return "ops.compact[noop]"
        if rewritten is not None and added is not None:
            return f"ops.compact[{rewritten}->{added}]"

    if scenario == "expire":
        bits = []
        if params.get("retain_last") is not None:
            bits.append(f"retain={params['retain_last']}")
        if rec.get("deleted_data_files") is not None:
            bits.append(f"deleted={rec['deleted_data_files']}")
        if bits:
            return f"ops.expire[{','.join(bits)}]"

    return f"ops.{scenario}"


def stages_ops(ops_records: list[dict]) -> list[dict]:
    """One row per ``kind="ops"`` RECORD, in log order.

    Never one row per scenario and never a de-duplicated summary: repeats and
    measured no-ops are the exhibit, so each record keeps its own row.
    """
    rows = []
    seen: Counter = Counter()
    for rec in ops_records:
        stage = ops_label(rec)
        seen[stage] += 1
        if seen[stage] > 1:
            stage = f"{stage}#{seen[stage]}"
        src = f"results/runs.jsonl kind=ops run_id={rec.get('run_id')}"
        rows.append(
            make_row(
                stage,
                "ops",
                str(rec.get("table")),
                rows_in=rec.get("rows_before"),
                rows_out=rec.get("rows_after"),
                n_bytes=rec.get("bytes_after"),
                wall_clock_s=rec.get("wall_clock_s"),
                wall_clock_source=src,
                snapshot_id=rec.get("snapshot_after"),
                source_of_truth={
                    "rows_in": f"{src} rows_before",
                    "rows_out": f"{src} rows_after",
                    "bytes": f"{src} bytes_after",
                    "wall_clock_s": f"{src} wall_clock_s",
                    "snapshot_id": f"{src} snapshot_after",
                },
                required=("rows_out", "bytes", "wall_clock_s", "snapshot_id"),
            )
        )
    return rows


# --- assembly -----------------------------------------------------------------


def headline_run_id(headline_config: str | Path) -> str:
    p = Path(headline_config)
    if not p.exists():
        raise LineageError(f"missing headline pin: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    rid = data.get("headline_run_id")
    if not rid:
        raise LineageError(f"{p} has no headline_run_id")
    return str(rid)


def assemble(
    *,
    warehouse: str | Path = DEFAULT_WAREHOUSE,
    results: str | Path = DEFAULT_RESULTS,
    manifest: str | Path = DEFAULT_MANIFEST,
    ingest_summary: str | Path = DEFAULT_INGEST_SUMMARY,
    build_summary: str | Path = DEFAULT_BUILD_SUMMARY,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    headline_config: str | Path = DEFAULT_HEADLINE_CONFIG,
    headline_run: str | None = None,
    gold_run_id: str | None = None,
    repro_per_user: str | None = None,
    expected_ops=DEFAULT_EXPECTED_OPS,
    root: str | Path = ".",
) -> dict:
    """Build the whole table. Never raises for a missing stage — the failure is
    reported in ``problems`` so the CLI can list ALL of them at once."""
    root = Path(root)
    stages: list[dict] = []
    problems: list[str] = []

    def add(name: str, builder):
        try:
            out = builder()
        except LineageError as exc:
            problems.append(f"{name}: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001 - surfaced as a stage problem
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            return None
        if isinstance(out, list):
            stages.extend(out)
        else:
            stages.append(out)
        return out

    add("raw_download", lambda: stage_raw_download(manifest))
    add("bronze.*", lambda: stages_bronze(warehouse, ingest_summary))
    add("silver.*", lambda: stages_silver(warehouse, build_summary))
    core = add(
        "gold.interactions_5core",
        lambda: stage_gold_core(warehouse, build_summary, gold_run_id),
    )
    core_rows = core["rows_out"] if core else None
    features = add("gold features", lambda: stages_gold_features(warehouse, core_rows))
    catalog_rows = None
    for row in features or []:
        if row["stage"] == "gold.item_features":
            catalog_rows = row["rows_out"]
    add("gold.item_text", lambda: stage_gold_item_text(warehouse, catalog_rows))

    # runs.jsonl-derived stages.
    records: list[dict] = []
    try:
        records = read_jsonl(results, what="eval / reproduce / ops records")
    except LineageError as exc:
        problems.append(f"results log: {exc}")

    by_run = {str(r.get("run_id")): r for r in records}
    head_rec = None
    try:
        rid = headline_run or headline_run_id(headline_config)
        head_rec = by_run.get(rid)
        if head_rec is None:
            raise LineageError(f"no record with run_id={rid} in {results}")
    except LineageError as exc:
        problems.append(f"headline_eval: {exc}")

    cache_snapshot = None
    if head_rec is not None:
        cache_snapshot = (head_rec.get("iceberg_snapshots") or {}).get(
            "local.gold.interactions_5core"
        )
    if cache_snapshot is None and core is not None:
        cache_snapshot = core["snapshot_id"]
    if cache_snapshot is None:
        problems.append(
            "eval_extract_cache: cannot determine the 5-core snapshot that keys "
            "the cache directory"
        )
    else:
        add("eval_extract_cache", lambda: stage_eval_cache(cache_root, cache_snapshot))

    if head_rec is not None:
        add("headline_eval", lambda: stage_headline_eval(head_rec, root=root))

    repro_recs = [r for r in records if r.get("kind") == "reproduce"]
    if not repro_recs:
        problems.append(f"reproduce_headline: no kind='reproduce' record in {results}")
    else:
        repro = repro_recs[-1]
        model_name = ((head_rec or {}).get("model") or {}).get("name")
        add(
            "reproduce_headline",
            lambda: stage_reproduce(
                repro, root=root, model_name=model_name, override=repro_per_user
            ),
        )

    ops_records = [r for r in records if r.get("kind") == "ops"]
    add("ops chain", lambda: stages_ops(ops_records))

    # The expected set is a FLOOR. A scenario run more times than expected is the
    # compaction exhibit doing its job, not an error — it is enumerated, and each
    # record already has its own row above.
    expected = Counter(str(s) for s in expected_ops)
    observed = Counter(str(r.get("scenario")) for r in ops_records)
    short = {s: expected[s] - observed.get(s, 0) for s in expected if expected[s] > observed.get(s, 0)}
    if short:
        problems.append(
            "ops chain: missing "
            + ", ".join(
                f"{n}x {s} (expected at least {expected[s]}, found {observed.get(s, 0)})"
                for s, n in sorted(short.items())
            )
        )

    for row in stages:
        if row["missing"]:
            problems.append(
                f"{row['stage']}: null in required field(s) {', '.join(row['missing'])}"
            )

    used = sorted({row["wall_clock_source"] for row in stages if row["wall_clock_s"] is None})
    return {
        "schema_version": schema_version,
        "expected_ops": [str(s) for s in expected_ops],
        "ops_observed": {s: observed[s] for s in sorted(observed)},
        "ops_records": len(ops_records),
        "stages_count": len(stages),
        "complete": not problems,
        "problems": problems,
        "footnotes": {k: FOOTNOTES[k] for k in used if k in FOOTNOTES},
        "stages": stages,
    }


# --- serialisation ------------------------------------------------------------


def to_json_bytes(table: dict) -> bytes:
    """Stable, byte-deterministic serialisation.

    Key order is the insertion order the builders produce (never re-sorted, so
    the table reads top-down like the pipeline), and nothing time-, host- or
    random-dependent is in the document — two runs over identical inputs produce
    identical bytes.
    """
    return (json.dumps(table, indent=2, sort_keys=False, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _fmt_int(v) -> str:
    return "—" if v is None else f"{int(v):,}"


def _fmt_wall(row: dict, marks: dict[str, str]) -> str:
    if row["wall_clock_s"] is None:
        mark = marks.get(row["wall_clock_source"] or "", "")
        return f"— {mark}".strip()
    return f"{row['wall_clock_s']:,.3f}"


def _short_source(row: dict) -> str:
    """Per-field sources, with fields sharing one source collapsed onto it.

    The JSON keeps the full per-field mapping; the markdown would otherwise
    repeat the same run id five times on every ops row.
    """
    grouped: dict[str, list[str]] = {}
    for key in ("rows", "rows_in", "rows_out", "bytes", "wall_clock_s", "snapshot_id"):
        val = row["source_of_truth"].get(key)
        if val:
            grouped.setdefault(val, []).append(key)
    if not grouped:
        return "—"
    return "; ".join(f"{','.join(keys)}={src}" for src, keys in grouped.items())


def to_markdown(table: dict) -> str:
    """Human table: one line per stage, aligned columns, footnotes section."""
    marks = {name: f"[^{i + 1}]" for i, name in enumerate(sorted(table["footnotes"]))}
    header = [
        "Stage",
        "Layer",
        "Table / artifact",
        "Rows in",
        "Rows out",
        "Bytes",
        "Wall clock (s)",
        "Snapshot",
        "Source of truth",
    ]
    body = [
        [
            row["stage"],
            row["layer"],
            row["table"],
            _fmt_int(row["rows_in"]),
            _fmt_int(row["rows_out"]),
            _fmt_int(row["bytes"]),
            _fmt_wall(row, marks),
            "—" if row["snapshot_id"] is None else str(row["snapshot_id"]),
            _short_source(row),
        ]
        for row in table["stages"]
    ]
    widths = [
        max(len(header[i]), *(len(r[i]) for r in body)) if body else len(header[i])
        for i in range(len(header))
    ]

    def line(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [
        "# Pipeline lineage — rows, bytes and wall clock per stage",
        "",
        "Every number below is read back from an artifact an earlier run committed "
        "to disk (ledger JSONL, Iceberg snapshot summary, or a file's size on "
        "disk). Nothing here is re-measured, and nothing is estimated: a runtime "
        "that was never persisted stays blank and is footnoted.",
        "",
        f"Stages: {table['stages_count']} · complete: "
        f"{str(table['complete']).lower()} · ops records: {table['ops_records']} "
        "("
        + ", ".join(f"{s}x{n}" for s, n in table["ops_observed"].items())
        + ")",
        "",
        line(header),
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    out.extend(line(r) for r in body)

    if table["footnotes"]:
        out += ["", "## Footnotes", ""]
        for name in sorted(table["footnotes"]):
            out.append(f"{marks[name]} **{name}** — {table['footnotes'][name]}")

    if table["problems"]:
        out += ["", "## Incomplete", ""]
        out += [f"- {p}" for p in table["problems"]]

    return "\n".join(out) + "\n"


def build_lineage_record(
    table: dict,
    *,
    artifact_sha256: str,
    out_json: str | Path,
    out_md: str | Path,
    run_id: str | None = None,
) -> dict:
    rid, rts = _resolve_run_id(run_id)
    git = runlog.git_info()
    return {
        "schema_version": runlog.record_schema_version,
        "kind": "lineage",
        "run_id": rid,
        "run_ts": rts,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "stages_count": table["stages_count"],
        "complete": table["complete"],
        "artifact_sha256": artifact_sha256,
        "out_json": str(out_json),
        "out_md": str(out_md),
        "stage_completeness": {r["stage"]: r["complete"] for r in table["stages"]},
        "expected_ops": table["expected_ops"],
        "ops_observed": table["ops_observed"],
        "ops_records": table["ops_records"],
        "footnoted_stages": [
            r["stage"] for r in table["stages"] if r["wall_clock_s"] is None
        ],
        "hardware": runlog.hardware_string(),
    }


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.ops.lineage")
    parser.add_argument("--warehouse", default=DEFAULT_WAREHOUSE)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--ingest-summary", default=DEFAULT_INGEST_SUMMARY)
    parser.add_argument("--build-summary", default=DEFAULT_BUILD_SUMMARY)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--headline-config", default=DEFAULT_HEADLINE_CONFIG)
    parser.add_argument("--headline-run-id", default=None)
    parser.add_argument("--gold-run-id", default=None)
    parser.add_argument("--repro-per-user", default=None)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--expect-ops",
        default=",".join(DEFAULT_EXPECTED_OPS),
        help="comma-separated ops scenarios, repeated for multiplicity",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="assemble and grade completeness; write nothing",
    )
    parser.add_argument(
        "--append-record",
        action="store_true",
        help="also append one kind='lineage' record to --results",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    expected_ops = tuple(s.strip() for s in args.expect_ops.split(",") if s.strip())
    table = assemble(
        warehouse=args.warehouse,
        results=args.results,
        manifest=args.manifest,
        ingest_summary=args.ingest_summary,
        build_summary=args.build_summary,
        cache_root=args.cache_root,
        headline_config=args.headline_config,
        headline_run=args.headline_run_id,
        gold_run_id=args.gold_run_id,
        repro_per_user=args.repro_per_user,
        expected_ops=expected_ops,
        root=args.root,
    )

    payload = to_json_bytes(table)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()

    if not table["complete"]:
        print("LINEAGE INCOMPLETE — refusing to publish a partial table:", file=sys.stderr)
        for problem in table["problems"]:
            print(f"  - {problem}", file=sys.stderr)

    summary = {
        "stages_count": table["stages_count"],
        "complete": table["complete"],
        "problems": table["problems"],
        "artifact_sha256": digest,
        "check_only": bool(args.check_only),
        "out_json": None,
        "out_md": None,
        "record_appended": False,
    }

    if not args.check_only and table["complete"]:
        out_json = Path(args.out_json)
        out_md = Path(args.out_md)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_bytes(payload)
        out_md.write_text(to_markdown(table), encoding="utf-8")
        summary["out_json"] = str(out_json)
        summary["out_md"] = str(out_md)
        if args.append_record:
            record = build_lineage_record(
                table,
                artifact_sha256=digest,
                out_json=out_json,
                out_md=out_md,
                run_id=args.run_id,
            )
            runlog.append_record(record, args.results)
            summary["record_appended"] = True
            summary["record_run_id"] = record["run_id"]

    # Summary JSON MUST be the last stdout line (repo convention).
    print(json.dumps(summary))
    return 0 if table["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
