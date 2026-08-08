"""Independent traceability checker for ``demo/data/`` (Phase 6, T26).

    make demo-verify                 # full mode (reads per-user parquets too)
    uv run python -m batch_recsys_lab.demo.verify_traceability --mode=record

This module deliberately shares NO code with the writing path: it does not
import :mod:`batch_recsys_lab.demo.export_core`, and re-implements pointer
parsing and leaf walking so a bug in the writer cannot excuse itself. The only
import from the repo is ``eval.runlog.sha256_file`` — the same digest function
the append-only records were written with, which is the point of the check.

What it proves (exit 0) about ``demo/data/*.json``:

STALE          the manifest's ``runs_jsonl_sha256`` still matches the log on disk
FILESET        the manifest describes exactly the documents present
FILE_HASH      each document still hashes to what the manifest recorded
UNCOVERED      every numeric leaf of every document has a trace entry, or sits
               under a declared ``descriptive`` path
ORPHAN         every manifest entry's pointer exists in its document
DOC_MISMATCH   entry value == document value, exactly (same type, no epsilon)
SOURCE_*       entry value == the value re-read from its source: a runs.jsonl
               record, a record-anchored results artifact (re-hashed against
               the SHA-256 the anchoring record carries), or — in ``full`` mode
               — the per-user parquet a record names
RECEIPTS       every run_id the manifest depends on has a receipts.json card

Any failure prints a grouped report and exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from batch_recsys_lab.eval.runlog import sha256_file

MODES = ("record", "full")


# --- independent JSON plumbing ------------------------------------------------


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _get(doc: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 pointer. Raises LookupError if it does not exist."""
    node = doc
    if pointer and not pointer.startswith("/"):
        raise LookupError(f"malformed pointer {pointer!r}")
    for token in ([] if pointer == "" else [_unescape(t) for t in pointer[1:].split("/")]):
        if isinstance(node, dict):
            if token not in node:
                raise LookupError(f"{pointer!r}: missing key {token!r}")
            node = node[token]
        elif isinstance(node, list):
            if not token.isdigit() or int(token) >= len(node):
                raise LookupError(f"{pointer!r}: bad index {token!r}")
            node = node[int(token)]
        else:
            raise LookupError(f"{pointer!r}: cannot descend into a scalar at {token!r}")
    return node


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _leaves(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.extend(_leaves(v, prefix + "/" + str(k).replace("~", "~0").replace("/", "~1")))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(_leaves(v, prefix + "/" + str(i)))
    else:
        out.append((prefix, node))
    return out


def _same(a: Any, b: Any) -> bool:
    """Exact equality, no epsilon and no int/float coercion.

    ``1 == 1.0`` in Python; here it is a mismatch, because a demo leaf must be
    the record's value, not a numerically equal restatement of it.
    """
    if type(a) is not type(b):
        return False
    return a == b


def _fmt(value: Any) -> str:
    return repr(value)


# --- source resolution --------------------------------------------------------


class _Resolver:
    """Caches the runs index, artifact documents and parquet reads."""

    def __init__(self, runs_log: Path, repo_root: Path, mode: str) -> None:
        self.runs_log = runs_log
        self.repo_root = repo_root
        self.mode = mode
        self.runs: dict[str, dict] = {}
        with open(runs_log) as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                rid = rec.get("run_id")
                if rid is None:
                    raise ValueError(f"{runs_log}:{lineno}: record without run_id")
                self.runs[rid] = rec
        self._artifacts: dict[str, Any] = {}
        self._artifact_hashes: dict[str, str] = {}
        self._tables: dict[str, Any] = {}
        # (parquet_rel, column) -> materialized python list; (parquet_rel, key_col)
        # -> {str(key): [rows]}. A 228k x 50 list column costs ~1.6s to materialize;
        # without these two caches full mode re-pays that per manifest entry.
        self._columns: dict[tuple[str, str], list] = {}
        self._key_index: dict[tuple[str, str], dict[str, list[int]]] = {}
        self.skipped_per_user = 0

    def _path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.repo_root / p

    def record(self, run_id: str) -> dict:
        if run_id not in self.runs:
            raise LookupError(f"run_id {run_id!r} is not in {self.runs_log}")
        return self.runs[run_id]

    def artifact_hash(self, rel: str) -> str:
        if rel not in self._artifact_hashes:
            p = self._path(rel)
            if not p.exists():
                raise LookupError(f"artifact {rel} does not exist")
            self._artifact_hashes[rel] = sha256_file(p)
        return self._artifact_hashes[rel]

    def artifact_doc(self, rel: str) -> Any:
        if rel not in self._artifacts:
            self._artifacts[rel] = json.loads(self._path(rel).read_text())
        return self._artifacts[rel]

    def per_user_value(self, parquet_rel: str, row_pointer: str) -> Any:
        """Resolve ``<key_col>=<key_value>[/<column>[/<index>]]`` in a parquet."""
        import pyarrow.parquet as pq

        if parquet_rel not in self._tables:
            p = self._path(parquet_rel)
            if not p.exists():
                raise LookupError(f"per-user artifact {parquet_rel} does not exist")
            self._tables[parquet_rel] = pq.read_table(p)
        table = self._tables[parquet_rel]

        parts = row_pointer.split("/")
        if "=" not in parts[0]:
            raise LookupError(f"row_pointer {row_pointer!r}: first segment must be '<column>=<value>'")
        key_col, key_val = parts[0].split("=", 1)
        if key_col not in table.column_names:
            raise LookupError(f"row_pointer {row_pointer!r}: no column {key_col!r}")
        idx_key = (parquet_rel, key_col)
        if idx_key not in self._key_index:
            keys = table.column(key_col).to_pylist()
            index: dict[str, list[int]] = {}
            for i, v in enumerate(keys):
                index.setdefault(str(v), []).append(i)
            self._key_index[idx_key] = index
        rows = self._key_index[idx_key].get(key_val, [])
        if len(rows) != 1:
            raise LookupError(f"row_pointer {row_pointer!r}: matched {len(rows)} rows, expected exactly 1")
        row = rows[0]
        if len(parts) == 1:
            raise LookupError(f"row_pointer {row_pointer!r}: no column selected")
        col = parts[1]
        if col not in table.column_names:
            raise LookupError(f"row_pointer {row_pointer!r}: no column {col!r}")
        col_key = (parquet_rel, col)
        if col_key not in self._columns:
            self._columns[col_key] = table.column(col).to_pylist()
        value = self._columns[col_key][row]
        for token in parts[2:]:
            if not token.isdigit():
                raise LookupError(f"row_pointer {row_pointer!r}: {token!r} is not a list index")
            value = value[int(token)]
        return value


# --- checks -------------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.failures: list[tuple[str, str]] = []
        self.counts: dict[str, int] = {}

    def fail(self, kind: str, message: str) -> None:
        self.failures.append((kind, message))

    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    @property
    def ok(self) -> bool:
        return not self.failures


def verify(
    data_dir: str | Path,
    manifest_path: str | Path,
    *,
    runs_log: str | Path | None = None,
    mode: str = "full",
    repo_root: str | Path | None = None,
) -> Report:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path)
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    rep = Report()

    if not manifest_path.exists():
        rep.fail("MANIFEST", f"{manifest_path} does not exist — run `make demo-export` first")
        return rep
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        rep.fail("MANIFEST", f"unsupported manifest schema_version {manifest.get('schema_version')!r}")
        return rep

    log_path = Path(runs_log) if runs_log else Path(manifest.get("runs_log", "results/runs.jsonl"))
    if not log_path.is_absolute():
        log_path = repo_root / log_path
    if not log_path.exists():
        rep.fail("MANIFEST", f"runs log {log_path} does not exist")
        return rep

    # -- staleness guard --
    actual_log_hash = sha256_file(log_path)
    if manifest.get("runs_jsonl_sha256") != actual_log_hash:
        rep.fail(
            "STALE",
            f"manifest runs_jsonl_sha256={manifest.get('runs_jsonl_sha256')} but {log_path} "
            f"hashes to {actual_log_hash} — the log moved since the export; re-run `make demo-export`",
        )

    resolver = _Resolver(log_path, repo_root, mode)

    # -- file set --
    on_disk = {p.name for p in sorted(data_dir.glob("*.json"))} - {manifest_path.name}
    declared = {f["name"] for f in manifest.get("files", [])}
    for name in sorted(on_disk - declared):
        rep.fail("FILESET", f"{name} is in {data_dir} but has no manifest entry set")
    for name in sorted(declared - on_disk):
        rep.fail("FILESET", f"manifest describes {name} but it is not in {data_dir}")

    docs: dict[str, Any] = {}
    for f in manifest.get("files", []):
        name = f["name"]
        path = data_dir / name
        if not path.exists():
            continue
        docs[name] = json.loads(path.read_text())
        actual = sha256_file(path)
        if actual != f.get("sha256"):
            rep.fail(
                "FILE_HASH",
                f"{name}: sha256 {actual} != manifest {f.get('sha256')} — the document was "
                "modified after export",
            )

    entries = manifest.get("entries", [])
    descriptive = manifest.get("descriptive", [])
    by_file: dict[str, dict[str, dict]] = {}
    for e in entries:
        dup = by_file.setdefault(e["file"], {})
        if e["pointer"] in dup:
            rep.fail("DUPLICATE", f"{e['file']}{e['pointer']}: two manifest entries for one pointer")
        dup[e["pointer"]] = e

    # -- coverage: every numeric leaf is traced or declared descriptive --
    for name, doc in docs.items():
        traced = set(by_file.get(name, {}))
        exact = {d["pointer"] for d in descriptive if d["file"] == name and not d.get("subtree")}
        prefixes = [d["pointer"] for d in descriptive if d["file"] == name and d.get("subtree")]
        absorbed = 0
        for ptr, val in _leaves(doc):
            if not _numeric(val):
                continue
            rep.count("numeric_leaves")
            if ptr in traced:
                continue
            if ptr in exact:
                rep.count("descriptive_leaves")
                continue
            if any(ptr == p or ptr.startswith(p + "/") for p in prefixes):
                rep.count("descriptive_leaves")
                absorbed += 1
                continue
            rep.fail("UNCOVERED", f"{name}{ptr} = {_fmt(val)} has no trace entry and is not declared descriptive")
        if absorbed:
            rep.count("subtree_absorbed_leaves", absorbed)

    # -- every entry: present in the doc, equal to the doc, equal to its source --
    for e in entries:
        name, ptr = e["file"], e["pointer"]
        where = f"{name}{ptr}"
        rep.count("entries")
        if name not in docs:
            rep.fail("ORPHAN", f"{where}: entry for a document that is not present")
            continue
        try:
            doc_value = _get(docs[name], ptr)
        except LookupError as exc:
            rep.fail("ORPHAN", f"{where}: pointer does not resolve in the document ({exc})")
            continue
        if isinstance(doc_value, (dict, list)):
            rep.fail("ORPHAN", f"{where}: entry points at a container, not a leaf")
            continue
        if not _same(doc_value, e["value"]):
            rep.fail(
                "DOC_MISMATCH",
                f"{where}: document has {_fmt(doc_value)} but the manifest recorded {_fmt(e['value'])}",
            )
            continue
        _verify_source(e, where, resolver, rep, mode)

    # -- receipts closure --
    needed = {e["source"]["run_id"] for e in entries if "run_id" in e.get("source", {})}
    if "receipts.json" in docs:
        have = set(docs["receipts.json"].get("runs", {}))
        for rid in sorted(needed - have):
            rep.fail("RECEIPTS", f"run_id {rid} is cited by the manifest but has no receipts.json card")
    elif needed:
        rep.fail("RECEIPTS", f"{len(needed)} run_ids are cited but receipts.json is not present")

    rep.count("run_ids", len(needed))
    if resolver.skipped_per_user:
        rep.count("per_user_skipped", resolver.skipped_per_user)
    return rep


def _verify_source(e: dict, where: str, resolver: _Resolver, rep: Report, mode: str) -> None:
    src = e.get("source") or {}
    kind = src.get("kind")
    value = e["value"]
    try:
        if kind == "runs_record":
            rec = resolver.record(src["run_id"])
            got = _get(rec, src["source_pointer"])
            if not _same(got, value):
                rep.fail(
                    "SOURCE_MISMATCH",
                    f"{where}: run {src['run_id']}{src['source_pointer']} is {_fmt(got)}, "
                    f"exported as {_fmt(value)}",
                )
            rep.count("src_runs_record")

        elif kind == "results_artifact":
            rec = resolver.record(src["run_id"])
            anchored = _get(rec, src["anchor_pointer"])
            if anchored != src["sha256"]:
                rep.fail(
                    "SOURCE_ANCHOR",
                    f"{where}: manifest says {src['source_file']} is {src['sha256']} but run "
                    f"{src['run_id']}{src['anchor_pointer']} records {anchored}",
                )
                return
            actual = resolver.artifact_hash(src["source_file"])
            if actual != anchored:
                rep.fail(
                    "SOURCE_HASH",
                    f"{where}: {src['source_file']} hashes to {actual}, but run {src['run_id']} "
                    f"anchored {anchored} — the artifact drifted from the record",
                )
                return
            got = _get(resolver.artifact_doc(src["source_file"]), src["pointer"])
            if not _same(got, value):
                rep.fail(
                    "SOURCE_MISMATCH",
                    f"{where}: {src['source_file']}{src['pointer']} is {_fmt(got)}, exported as {_fmt(value)}",
                )
            rep.count("src_results_artifact")

        elif kind == "per_user_artifact":
            rec = resolver.record(src["run_id"])
            declared = rec.get("per_user_artifact")
            if declared != src["parquet_path"]:
                rep.fail(
                    "SOURCE_ANCHOR",
                    f"{where}: run {src['run_id']} names per_user_artifact {declared!r}, "
                    f"manifest cites {src['parquet_path']!r}",
                )
                return
            if mode != "full":
                resolver.skipped_per_user += 1
                return
            actual = resolver.artifact_hash(src["parquet_path"])
            if actual != src["sha256"]:
                rep.fail(
                    "SOURCE_HASH",
                    f"{where}: {src['parquet_path']} hashes to {actual}, manifest recorded {src['sha256']}",
                )
                return
            got = resolver.per_user_value(src["parquet_path"], src["row_pointer"])
            if not _same(got, value):
                rep.fail(
                    "SOURCE_MISMATCH",
                    f"{where}: {src['parquet_path']}[{src['row_pointer']}] is {_fmt(got)}, "
                    f"exported as {_fmt(value)}",
                )
            rep.count("src_per_user_artifact")

        else:
            rep.fail("SOURCE_KIND", f"{where}: unknown source kind {kind!r}")
    except LookupError as exc:
        rep.fail("SOURCE_MISSING", f"{where}: {exc}")


def print_report(rep: Report, *, mode: str, data_dir: Path, out=sys.stdout) -> None:
    grouped: dict[str, list[str]] = {}
    for kind, msg in rep.failures:
        grouped.setdefault(kind, []).append(msg)
    print(f"demo traceability check · {data_dir} · mode={mode}", file=out)
    stats = ", ".join(f"{k}={v}" for k, v in sorted(rep.counts.items()))
    print(f"  {stats}", file=out)
    if rep.ok:
        print("  OK — every numeric leaf re-resolves to its recorded source", file=out)
        return
    print(f"  FAILED — {len(rep.failures)} problem(s) in {len(grouped)} class(es)", file=out)
    for kind in sorted(grouped):
        msgs = grouped[kind]
        print(f"\n  [{kind}] {len(msgs)}", file=out)
        for m in msgs[:20]:
            print(f"    - {m}", file=out)
        if len(msgs) > 20:
            print(f"    … {len(msgs) - 20} more", file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="demo/data")
    ap.add_argument("--manifest", default=None, help="default: <data-dir>/trace_manifest.json")
    ap.add_argument("--runs-log", default=None, help="default: the manifest's recorded runs_log")
    ap.add_argument(
        "--mode",
        choices=MODES,
        default="full",
        help="record: skip per-user parquet reads (CI). full: read them too.",
    )
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    manifest = Path(args.manifest) if args.manifest else data_dir / "trace_manifest.json"
    rep = verify(data_dir, manifest, runs_log=args.runs_log, mode=args.mode)
    print_report(rep, mode=args.mode, data_dir=data_dir)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
