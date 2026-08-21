"""Export foundation for the static demo (Phase 6, T26): run index, JSON
pointers, ``TracedWriter``, and the cumulative ``trace_manifest.json``.

Design rule (plan §9 "Architecture"): **an untraced number must be impossible
to write**. Every leaf of a ``demo/data/*.json`` document is set through one of
two ``TracedWriter`` methods:

``copy_from_record`` / ``copy_from_artifact`` / ``put``
    Traced. The value is recorded in the manifest together with the source it
    came from, so :mod:`verify_traceability` can re-resolve it independently.
``put_descriptive``
    Untraced, and registered in the manifest's ``descriptive`` list as a
    declared non-evidence path (display labels, ordering hints, the document's
    own ``schema_version``). Numeric leaves reached this way are the only ones
    the verifier will accept without a trace entry.

Values are written at FULL precision — ``json.dump`` emits the shortest
round-tripping repr of a float, so a demo leaf and its source record leaf
compare exactly (``==``, no epsilon). Nothing here rounds, ever.

Documents carry no timestamp, so re-exporting unchanged evidence is byte-stable
and the manifest can pin each document's SHA-256. Only the manifest carries
``generated_at``.

The manifest is cumulative and read-modify-write: each exporter replaces its
own file's entries and leaves other files' entries alone, so
``make demo-export`` can run the exporters as separate processes (receipts last
— it reads the manifest to find the run_id closure it must document).
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# The run index is the append-only log's reader used by the T14 chart; reused
# verbatim so the demo and the figures resolve run_ids the same way.
from batch_recsys_lab.eval.crossover_chart import index_runs
from batch_recsys_lab.eval.runlog import sha256_file

__all__ = [
    "TraceManifest",
    "TracedWriter",
    "index_runs",
    "index_runs_multi",
    "is_numeric",
    "iter_leaves",
    "jp",
    "load_export_config",
    "manifest_schema_version",
    "parse_pointer",
    "resolve_pointer",
    "select_record",
    "sha256_file",
    "write_document",
]

manifest_schema_version = 1

_REPO_ROOT = Path(__file__).resolve().parents[3]


# --- JSON pointers (RFC 6901) -------------------------------------------------


def jp(*tokens: object) -> str:
    """Build a JSON pointer from raw tokens (``~`` and ``/`` escaped)."""
    out = ""
    for t in tokens:
        s = str(t).replace("~", "~0").replace("/", "~1")
        out += "/" + s
    return out


def parse_pointer(pointer: str) -> list[str]:
    """Split an RFC 6901 pointer into unescaped tokens. ``""`` is the root."""
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer {pointer!r}: must start with '/'")
    return [t.replace("~1", "/").replace("~0", "~") for t in pointer[1:].split("/")]


def resolve_pointer(doc: Any, pointer: str) -> Any:
    """Resolve a JSON pointer; ``KeyError`` (with the full pointer) if absent."""
    node = doc
    for i, token in enumerate(parse_pointer(pointer)):
        here = "/" + "/".join(parse_pointer(pointer)[: i + 1])
        if isinstance(node, dict):
            if token not in node:
                raise KeyError(f"pointer {pointer!r}: no key {token!r} at {here!r}")
            node = node[token]
        elif isinstance(node, list):
            if not token.isdigit() or int(token) >= len(node):
                raise KeyError(f"pointer {pointer!r}: bad list index {token!r} at {here!r}")
            node = node[int(token)]
        else:
            raise KeyError(f"pointer {pointer!r}: {here!r} is a scalar, cannot descend")
    return node


def is_numeric(value: Any) -> bool:
    """True for JSON numbers. ``bool`` is a Python ``int`` but a JSON boolean —
    excluded, so ``true`` never counts as a number needing a trace."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def iter_leaves(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Every scalar leaf of a JSON value as ``(pointer, value)`` pairs.

    Empty containers contribute no leaf (there is nothing to verify in them).
    """
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.extend(iter_leaves(v, prefix + jp(k)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(iter_leaves(v, prefix + jp(i)))
    else:
        out.append((prefix, node))
    return out


# --- run index: run_id collisions are resolved, never guessed -----------------


def index_runs_multi(runs_log: str | Path) -> dict[str, list[dict]]:
    """``run_id -> [record, ...]`` in log order.

    ``index_runs`` (shared with the figures) collapses to last-occurrence-wins.
    That is fine while run_ids are unique, but the append-only log has at least
    one COLLIDED id (two ML-32M TEST evals whose run_ids were minted in the same
    second: ``20260820T221701Z-20d8ff9`` names both an ALS-decay run and an
    item-kNN-t12m run). "Whichever the dict happened to keep" is not evidence,
    so the demo path keeps every candidate and forces the citation to say which
    record it means — see :func:`select_record`.
    """
    out: dict[str, list[dict]] = {}
    with open(runs_log) as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            rid = rec.get("run_id")
            if rid is None:
                raise ValueError(f"{runs_log}:{lineno}: record without run_id")
            out.setdefault(rid, []).append(rec)
    return out


def select_record(
    runs_multi: dict[str, list[dict]], run_id: str, selector: dict[str, Any] | None = None
) -> dict:
    """Resolve ``run_id`` to EXACTLY ONE record.

    ``selector`` is a mapping of JSON pointer -> expected value (e.g.
    ``{"/config_path": "configs/eval_itemknn_t12m_ml32m_test.yaml"}``). A
    collided run_id with no selector is a hard error, not a coin flip; a
    selector that matches zero or several records is a hard error too. The
    selector travels with the manifest entry, so the independent verifier
    re-resolves the same record without re-deriving the tie-break.
    """
    candidates = runs_multi.get(run_id)
    if not candidates:
        raise KeyError(f"run_id {run_id!r} not found in the runs log")
    if selector:
        matched = []
        for rec in candidates:
            try:
                if all(resolve_pointer(rec, ptr) == val for ptr, val in selector.items()):
                    matched.append(rec)
            except KeyError:
                continue
    else:
        matched = list(candidates)
    if len(matched) == 1:
        return matched[0]
    if not selector:
        raise KeyError(
            f"run_id {run_id!r} is COLLIDED: {len(candidates)} records in the append-only log "
            f"share it ({[r.get('config_path') for r in candidates]}). Cite it with a "
            "record_selector (e.g. {'/config_path': ...}); positional resolution is not evidence."
        )
    raise KeyError(
        f"run_id {run_id!r} with record_selector {selector!r} matched {len(matched)} of "
        f"{len(candidates)} record(s); expected exactly 1"
    )


# --- source descriptors -------------------------------------------------------


def source_runs_record(
    run_id: str, source_pointer: str, record_selector: dict[str, Any] | None = None
) -> dict:
    """A value copied out of one ``results/runs.jsonl`` record."""
    src = {"kind": "runs_record", "run_id": run_id, "source_pointer": source_pointer}
    if record_selector:
        src["record_selector"] = dict(record_selector)
    return src


def source_results_artifact(
    *, source_file: str, sha256: str, pointer: str, run_id: str, anchor_pointer: str
) -> dict:
    """A value read from a committed results artifact whose SHA-256 the record
    ``run_id`` carries at ``anchor_pointer`` (the transitive-trace pattern the
    ``kind="lineage"`` record already established)."""
    return {
        "kind": "results_artifact",
        "source_file": source_file,
        "sha256": sha256,
        "pointer": pointer,
        "run_id": run_id,
        "anchor_pointer": anchor_pointer,
    }


def source_derived_artifact(
    *,
    source_file: str,
    sha256: str,
    pointer: str,
    artifact_kind: str,
    git_sha: str,
    config_hash: str,
    input_anchors: list[dict],
) -> dict:
    """A value read from a committed DERIVED analysis artifact that, by design,
    appends nothing to ``results/runs.jsonl`` — so no append-only record can
    carry its SHA-256 and the stronger ``results_artifact`` kind is unavailable.

    Anchoring instead (see ``export_contrast`` for the full rationale):

    * the file must still hash to the digest pinned here (drift is caught),
    * it must self-declare ``derived=true`` / ``appends_to_runs_jsonl=false``
      — this weaker kind may never stand in for a record-anchored artifact,
    * its ``git_sha`` / ``config_hash`` must be the ones pinned here, and
    * ``input_anchors`` ties its INPUTS to the append-only log: each anchor
      names a run_id (with a ``record_selector`` where the id is collided) plus
      a pointer in the artifact and a pointer in the record that must agree —
      e.g. the artifact's per-user parquet path == the path the eval record
      published at ``/per_user_artifact``.
    """
    return {
        "kind": "derived_artifact",
        "source_file": source_file,
        "sha256": sha256,
        "pointer": pointer,
        "artifact_kind": artifact_kind,
        "git_sha": git_sha,
        "config_hash": config_hash,
        "input_anchors": copy.deepcopy(input_anchors),
    }


def source_per_user_artifact(
    *, parquet_path: str, sha256: str, row_pointer: str, run_id: str
) -> dict:
    """A value read from the per-user parquet the record ``run_id`` names.

    ``row_pointer`` grammar (resolved by ``verify_traceability --mode=full``)::

        <key_column>=<key_value>[/<column>[/<list_index>]]

    e.g. ``user_index=17/ndcg@10`` or ``user_index=17/top50/0``. The key must
    select exactly one row.
    """
    return {
        "kind": "per_user_artifact",
        "parquet_path": parquet_path,
        "sha256": sha256,
        "row_pointer": row_pointer,
        "run_id": run_id,
    }


# --- traced writer ------------------------------------------------------------


class TracedWriter:
    """Builds one ``demo/data/<file_name>`` document plus its manifest entries."""

    def __init__(
        self,
        file_name: str,
        runs: dict[str, dict],
        *,
        doc_schema_version: int = 1,
        generated_by: str | None = None,
        runs_multi: dict[str, list[dict]] | None = None,
        record_selectors: dict[str, dict] | None = None,
    ) -> None:
        self.file_name = file_name
        self.runs = runs
        # Opt-in strict resolution. Exporters that pass ``runs_multi`` get
        # "a collided run_id must be disambiguated"; exporters that do not keep
        # the historical last-occurrence-wins dict (no citation of a collided id
        # exists in their documents, and the verifier now rejects one anyway).
        self.runs_multi = runs_multi
        self.record_selectors = dict(record_selectors or {})
        self._doc: dict = {}
        self._entries: list[dict] = []
        self._descriptive: list[dict] = []
        self._artifacts: dict[str, dict] = {}
        self.put_descriptive("/schema_version", doc_schema_version, note="document schema version")
        if generated_by:
            self.put_descriptive("/generated_by", generated_by, note="exporter module")

    # -- document plumbing --

    def _set(self, pointer: str, value: Any) -> None:
        tokens = parse_pointer(pointer)
        if not tokens:
            raise ValueError("cannot write the document root")
        node: Any = self._doc
        for i, token in enumerate(tokens[:-1]):
            if isinstance(node, list):
                if not token.isdigit() or int(token) >= len(node):
                    raise KeyError(f"{pointer}: list index {token!r} does not exist (pre-create the list)")
                node = node[int(token)]
            else:
                node = node.setdefault(token, {})
                if not isinstance(node, (dict, list)):
                    raise KeyError(f"{pointer}: /{'/'.join(tokens[: i + 1])} is already a scalar")
        last = tokens[-1]
        if isinstance(node, list):
            if not last.isdigit() or int(last) >= len(node):
                raise KeyError(f"{pointer}: list index {last!r} does not exist (pre-create the list)")
            node[int(last)] = value
        else:
            if last in node:
                raise KeyError(f"{pointer}: already written (documents are write-once)")
            node[last] = value

    def _selector(self, run_id: str, selector: dict | None) -> dict | None:
        return selector if selector is not None else self.record_selectors.get(run_id)

    def record(self, run_id: str, selector: dict | None = None) -> dict:
        """The single record ``run_id`` (+ selector) resolves to. Read-only."""
        return self._record(run_id, selector)

    def _record(self, run_id: str, selector: dict | None = None) -> dict:
        if self.runs_multi is not None:
            return select_record(self.runs_multi, run_id, self._selector(run_id, selector))
        if run_id not in self.runs:
            raise KeyError(f"run_id {run_id!r} not found in the runs log")
        return self.runs[run_id]

    # -- writing --

    def put(self, pointer: str, value: Any, source: dict) -> None:
        """Write one traced leaf. ``source`` must be a source descriptor."""
        if isinstance(value, (dict, list)):
            raise TypeError(f"{pointer}: put() writes leaves only; use copy_from_record for subtrees")
        if "kind" not in source:
            raise ValueError(f"{pointer}: source descriptor without 'kind'")
        self._set(pointer, value)
        self._entries.append(
            {"file": self.file_name, "pointer": pointer, "value": value, "source": source}
        )

    def put_descriptive(self, pointer: str, value: Any, *, note: str = "", subtree: bool = False) -> None:
        """Write an UNTRACED, declared-descriptive value (labels, ordering,
        document metadata).

        ``subtree=True`` declares the whole subtree descriptive — use it only
        for bulk non-evidence payloads (e.g. a shopper's item titles); the
        verifier reports how many numeric leaves each subtree declaration
        absorbs so a broad declaration cannot hide a metric unnoticed.
        """
        self._set(pointer, copy.deepcopy(value))
        self._descriptive.append(
            {"file": self.file_name, "pointer": pointer, "subtree": bool(subtree), "note": note}
        )

    def ensure_list(self, pointer: str, length: int) -> None:
        """Pre-create a list of ``length`` empty objects so traced leaves can be
        written into ``<pointer>/<i>/…``. Structure only — no values, nothing to
        trace (``_set`` refuses to auto-create list indices, on purpose)."""
        self._set(pointer, [{} for _ in range(length)])

    def copy_from_record(
        self, pointer: str, run_id: str, source_pointer: str, *, selector: dict | None = None
    ) -> Any:
        """Copy a value (scalar OR subtree) verbatim out of a runs.jsonl record.

        Every leaf of the copied subtree gets its own trace entry pointing at
        the matching leaf of the record, so the whole subtree is re-resolvable
        leaf by leaf. Returns the copied value.

        ``selector`` (JSON pointer -> expected value) disambiguates a collided
        run_id and is written into every entry it produces, so the verifier
        resolves the same record from the manifest alone.
        """
        record = self._record(run_id, selector)
        effective = self._selector(run_id, selector)
        value = copy.deepcopy(resolve_pointer(record, source_pointer))
        self._set(pointer, value)
        for leaf_ptr, leaf_val in iter_leaves(value):
            self._entries.append(
                {
                    "file": self.file_name,
                    "pointer": pointer + leaf_ptr,
                    "value": leaf_val,
                    "source": source_runs_record(run_id, source_pointer + leaf_ptr, effective),
                }
            )
        return value

    def register_artifact(
        self, key: str, path: str | Path, *, run_id: str, anchor_pointer: str
    ) -> dict:
        """Bind a results artifact to the record that attests to its SHA-256.

        Hashes the file now and refuses the binding unless the digest matches
        the record's recorded value — the artifact is only usable as evidence
        while it still is the artifact the record signed.
        """
        record = self._record(run_id)
        recorded = resolve_pointer(record, anchor_pointer)
        actual = sha256_file(path)
        if actual != recorded:
            raise ValueError(
                f"artifact {path}: sha256 {actual} != {recorded} recorded by run {run_id} "
                f"at {anchor_pointer}. The artifact drifted from the record that anchors it."
            )
        art = {
            "key": key,
            "path": str(path),
            "sha256": actual,
            "run_id": run_id,
            "anchor_pointer": anchor_pointer,
            "doc": json.loads(Path(path).read_text()),
        }
        self._artifacts[key] = art
        return art

    def register_derived_artifact(self, key: str, path: str | Path, *, input_anchors: list[dict]) -> dict:
        """Bind a committed DERIVED artifact that appends nothing to the log.

        Refuses the binding unless the file self-declares ``derived=true`` and
        ``appends_to_runs_jsonl=false`` (anything that DOES append must be
        anchored the strong way, via :meth:`register_artifact`), and unless
        every input anchor holds: the pointer the artifact names as an input
        equals the pointer the append-only record publishes.
        """
        doc = json.loads(Path(path).read_text())
        if doc.get("derived") is not True or doc.get("appends_to_runs_jsonl") is not False:
            raise ValueError(
                f"derived artifact {path}: expected derived=true and appends_to_runs_jsonl=false "
                f"(got {doc.get('derived')!r}/{doc.get('appends_to_runs_jsonl')!r}). An artifact a "
                "runs.jsonl record attests to must use register_artifact() instead."
            )
        for anchor in input_anchors:
            rec = self._record(anchor["run_id"], anchor.get("record_selector"))
            in_artifact = resolve_pointer(doc, anchor["artifact_pointer"])
            in_record = resolve_pointer(rec, anchor["record_pointer"])
            if in_artifact != in_record:
                raise ValueError(
                    f"derived artifact {path}: input anchor {anchor['artifact_pointer']} is "
                    f"{in_artifact!r} but run {anchor['run_id']}{anchor['record_pointer']} is "
                    f"{in_record!r}. The derived file does not name the record's artifact."
                )
        art = {
            "key": key,
            "path": str(path),
            "sha256": sha256_file(path),
            "artifact_kind": doc.get("kind"),
            "git_sha": doc.get("git_sha"),
            "config_hash": doc.get("config_hash"),
            "input_anchors": copy.deepcopy(input_anchors),
            "derived": True,
            "doc": doc,
        }
        self._artifacts[key] = art
        return art

    def copy_from_artifact(self, pointer: str, artifact_key: str, artifact_pointer: str) -> Any:
        """Copy a value (scalar or subtree) out of a registered results artifact
        (record-anchored or derived — the source descriptor follows the kind the
        artifact was registered under)."""
        if artifact_key not in self._artifacts:
            raise KeyError(f"artifact {artifact_key!r} not registered")
        art = self._artifacts[artifact_key]
        value = copy.deepcopy(resolve_pointer(art["doc"], artifact_pointer))
        self._set(pointer, value)
        for leaf_ptr, leaf_val in iter_leaves(value):
            if art.get("derived"):
                source = source_derived_artifact(
                    source_file=art["path"],
                    sha256=art["sha256"],
                    pointer=artifact_pointer + leaf_ptr,
                    artifact_kind=art["artifact_kind"],
                    git_sha=art["git_sha"],
                    config_hash=art["config_hash"],
                    input_anchors=art["input_anchors"],
                )
            else:
                source = source_results_artifact(
                    source_file=art["path"],
                    sha256=art["sha256"],
                    pointer=artifact_pointer + leaf_ptr,
                    run_id=art["run_id"],
                    anchor_pointer=art["anchor_pointer"],
                )
            self._entries.append(
                {
                    "file": self.file_name,
                    "pointer": pointer + leaf_ptr,
                    "value": leaf_val,
                    "source": source,
                }
            )
        return value

    # -- output --

    @property
    def document(self) -> dict:
        return self._doc

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    @property
    def descriptive(self) -> list[dict]:
        return list(self._descriptive)

    def untraced_numeric_leaves(self) -> list[str]:
        """Self-check: numeric leaves with neither a trace entry nor a
        descriptive declaration. Always empty by construction — asserted before
        writing so a future ``_set`` caller cannot bypass the invariant."""
        traced = {e["pointer"] for e in self._entries}
        exact = {d["pointer"] for d in self._descriptive if not d["subtree"]}
        prefixes = [d["pointer"] for d in self._descriptive if d["subtree"]]
        out = []
        for ptr, val in iter_leaves(self._doc):
            if not is_numeric(val) or ptr in traced or ptr in exact:
                continue
            if any(ptr == p or ptr.startswith(p + "/") for p in prefixes):
                continue
            out.append(ptr)
        return out


# --- cumulative trace manifest ------------------------------------------------


class TraceManifest:
    """``demo/data/trace_manifest.json``: entries for every exported document.

    Read-modify-write per file so exporters can run as separate processes.
    ``runs_jsonl_sha256`` is the staleness guard — a manifest exported against
    an older log fails verification instead of quietly disagreeing with it.
    """

    def __init__(self, path: str | Path, runs_log: str | Path) -> None:
        self.path = Path(path)
        self.runs_log = Path(runs_log)
        self.files: dict[str, dict] = {}
        self.entries: list[dict] = []
        self.descriptive: list[dict] = []
        if self.path.exists():
            prev = json.loads(self.path.read_text())
            self.files = {f["name"]: f for f in prev.get("files", [])}
            self.entries = list(prev.get("entries", []))
            self.descriptive = list(prev.get("descriptive", []))

    def replace_file(self, name: str, *, sha256: str, entries: list[dict], descriptive: list[dict]) -> None:
        self.entries = [e for e in self.entries if e["file"] != name]
        self.descriptive = [d for d in self.descriptive if d["file"] != name]
        self.entries.extend(entries)
        self.descriptive.extend(descriptive)
        self.files[name] = {"name": name, "sha256": sha256}

    def drop_missing_files(self, data_dir: str | Path) -> list[str]:
        """Forget files that no longer exist on disk (so a deleted exhibit does
        not leave the manifest describing a phantom)."""
        data_dir = Path(data_dir)
        gone = [n for n in self.files if not (data_dir / n).exists()]
        for n in gone:
            del self.files[n]
            self.entries = [e for e in self.entries if e["file"] != n]
            self.descriptive = [d for d in self.descriptive if d["file"] != n]
        return gone

    @staticmethod
    def _cited(source: dict) -> list[tuple[str, dict | None]]:
        """``(run_id, record_selector)`` pairs one source descriptor depends on."""
        out: list[tuple[str, dict | None]] = []
        if "run_id" in source:
            out.append((source["run_id"], source.get("record_selector")))
        for anchor in source.get("input_anchors", []):
            out.append((anchor["run_id"], anchor.get("record_selector")))
        return out

    def run_ids(self) -> set[str]:
        """Every run_id any entry depends on (direct, as a record-anchored
        artifact's anchor, or as a derived artifact's input anchor)."""
        return {rid for e in self.entries for rid, _ in self._cited(e["source"])}

    def record_selectors(self) -> dict[str, dict]:
        """``run_id -> record_selector`` for every cited, collided run_id.

        Two entries citing one run_id with DIFFERENT selectors would mean the
        demo shows two different records under one id — receipts.json is keyed
        by run_id and could not represent that, so it is an error here rather
        than a silently wrong card.
        """
        out: dict[str, dict] = {}
        for e in self.entries:
            for rid, sel in self._cited(e["source"]):
                if not sel:
                    continue
                if rid in out and out[rid] != sel:
                    raise ValueError(
                        f"trace manifest cites run_id {rid!r} under two different "
                        f"record_selectors: {out[rid]!r} and {sel!r}"
                    )
                out[rid] = sel
        return out

    def write(self) -> Path:
        doc = {
            "schema_version": manifest_schema_version,
            "generated_at": datetime.now(UTC).isoformat(),
            "runs_jsonl_sha256": sha256_file(self.runs_log),
            "runs_log": str(self.runs_log),
            "files": [self.files[n] for n in sorted(self.files)],
            "entries": self.entries,
            "descriptive": self.descriptive,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        return self.path


def dump_json(doc: Any) -> str:
    """Canonical demo-JSON serialization: 2-space indent, insertion order,
    UTF-8 literals, full float precision, trailing newline."""
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def write_document(writer: TracedWriter, out_dir: str | Path, manifest: TraceManifest) -> Path:
    """Write ``writer``'s document and fold its entries into the manifest."""
    stray = writer.untraced_numeric_leaves()
    if stray:
        raise AssertionError(
            f"{writer.file_name}: {len(stray)} numeric leaf/leaves written without a trace "
            f"entry or descriptive declaration: {stray[:10]}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / writer.file_name
    out_path.write_text(dump_json(writer.document))
    manifest.replace_file(
        writer.file_name,
        sha256=sha256_file(out_path),
        entries=writer.entries,
        descriptive=writer.descriptive,
    )
    return out_path


# --- shared config ------------------------------------------------------------

_CONFIG_REQUIRED = ("runs_log", "out_dir", "manifest", "headline_run_id", "split", "segments", "models")


def load_export_config(path: str | Path) -> dict:
    """Load and validate ``configs/demo_export.yaml`` (shared by every exporter)."""
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in _CONFIG_REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"demo export config {path}: missing required keys {missing}")
    keys = [m.get("key") for m in cfg["models"]]
    if None in keys:
        raise ValueError(f"demo export config {path}: every models[] entry needs a 'key'")
    if len(set(keys)) != len(keys):
        raise ValueError(f"demo export config {path}: duplicate model keys {keys}")
    for m in cfg["models"]:
        for k in ("label", "run_id"):
            if k not in m:
                raise ValueError(f"demo export config {path}: model {m['key']!r} missing {k!r}")
    if cfg["headline_run_id"] not in {m["run_id"] for m in cfg["models"]}:
        raise ValueError(
            f"demo export config {path}: headline_run_id {cfg['headline_run_id']!r} is not one of the models"
        )
    return cfg
