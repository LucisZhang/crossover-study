"""Download + manifest tooling for MovieLens 32M (Phase 9, T9-3a; docs/engineering-log/UPGRADE_PLAN.md §8c).

    python -m batch_recsys_lab.ingest.download_ml32m fetch
    python -m batch_recsys_lab.ingest.download_ml32m manifest

Same posture as the Amazon flow in ``ingest/download.py``: the published archive
carries no official checksum, so **our locally computed SHA-256 is ground truth**
— computed once here and written to ``data/MANIFEST_ML32M.md``. Research license:
the raw data is never redistributed (``data/`` is gitignored; only the manifests
are committed). Cite Harper & Konstan 2015.

**The ML-32M manifest is its own committed file, ``data/MANIFEST_ML32M.md`` —
NEVER ``data/MANIFEST.md``.** This is load-bearing, not cosmetic:
``eval/runlog.dataset_manifest_hash`` hashes the WHOLE manifest file, and
``eval/reproduce.py`` compares ``dataset_manifest_hash`` field-for-field against
the pinned Amazon headline record. Appending an ML-32M section to
``data/MANIFEST.md`` changes that hash and turns ``make reproduce-headline``'s
verdict from ``byte_exact`` into a mismatch — a second dataset's paperwork must
not be able to falsify the first dataset's reproduction. One file per dataset
keeps the two hashes independent. (If ``data/MANIFEST.md`` has already been
polluted, restore it: ``git show 5fabb21:data/MANIFEST.md > data/MANIFEST.md``.)

Two deliberate differences from the Amazon flow, both because the source differs:

* **No hardcoded expected size.** The Amazon files are pinned by an exact byte
  count published on the release site; ml-32m.zip is not (≈239MB is documentation,
  not a contract). Completeness is instead proven by a *stronger* check — the zip's
  own CRCs (:meth:`zipfile.ZipFile.testzip`), which a truncated or corrupted
  download cannot pass.
* **The download date is preserved across regenerations.** The rendered document
  is otherwise a pure function of the bytes on disk, so re-running ``manifest``
  after a run has been recorded does not move ``dataset_manifest_hash``.

``ratings.csv``, ``movies.csv`` and ``tags.csv`` are extracted — the three files
the pipeline ingests (tags because §8c T9-3b's content arm is title+genres+tags).
``links.csv`` is deliberately NOT extracted: it holds only IMDb/TMDb foreign keys
that nothing here reads; the manifest says so in writing. ``README.txt`` likewise
stays inside the archive, which remains the hashed artifact of record.

Downstream consumers of the rendered file (``eval/churn_contrast``,
``ingest/reconcile_ml32m``) parse it with :func:`parse_manifest` — the writer and
the reader of this format live in one module on purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW_ML32M = REPO_ROOT / "data" / "raw" / "ml32m"
# The ML-32M manifest: its own committed file. Never data/MANIFEST.md — see the
# module docstring (that file's hash is compared by `make reproduce-headline`).
MANIFEST_ML32M_PATH = REPO_ROOT / "data" / "MANIFEST_ML32M.md"

ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
ZIP_FILENAME = "ml-32m.zip"
# Documentation only (the release page states ~239MB) — used for the pre-flight
# space message and the "observed vs advertised" line, never as a gate.
APPROX_ZIP_BYTES = 239_000_000

CITATION = (
    "Harper, F. Maxwell, and Joseph A. Konstan. \"The MovieLens Datasets: History "
    "and Context.\" ACM Transactions on Interactive Intelligent Systems 5, no. 4 "
    "(2015): 19:1-19:19. https://doi.org/10.1145/2827872"
)

CHUNK_SIZE = 8 * 1024 * 1024  # 8MB (matches ingest/download.py)


@dataclass(frozen=True)
class Member:
    """One archive member extracted to ``data/raw/ml32m/``."""

    arcname: str  # path inside the zip
    filename: str  # flattened destination name
    expected_header: tuple[str, ...]
    # ML-32M README headline count; RECONCILED at ingest, never asserted here.
    # None = we have not verified a published figure for this file, so no delta
    # is claimed. Inventing one would create a fake cross-check.
    published_rows: int | None


MEMBERS = [
    Member(
        arcname="ml-32m/ratings.csv",
        filename="ratings.csv",
        expected_header=("userId", "movieId", "rating", "timestamp"),
        published_rows=32_000_204,
    ),
    Member(
        arcname="ml-32m/movies.csv",
        filename="movies.csv",
        expected_header=("movieId", "title", "genres"),
        published_rows=87_585,
    ),
    Member(
        arcname="ml-32m/tags.csv",
        filename="tags.csv",
        expected_header=("userId", "movieId", "tag", "timestamp"),
        published_rows=None,
    ),
]

# Archive members we deliberately do NOT extract, with the reason. Rendered into
# the manifest so "unused" is a recorded decision rather than an omission.
UNUSED_MEMBERS = {
    "ml-32m/links.csv": "IMDb/TMDb foreign keys only; no module in this lab reads them",
    "ml-32m/README.txt": "documentation; the archive itself is the hashed artifact of record",
}

# Published headline counts (ML-32M README) for the manifest; the bronze ingest
# summary reports the observed counts and the delta.
PUBLISHED_COUNTS = {"ratings": 32_000_204, "users": 200_948, "movies": 87_585}


def _zip_path() -> Path:
    return DATA_RAW_ML32M / ZIP_FILENAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _header_of(path: Path) -> tuple[str, ...]:
    with open(path, encoding="utf-8-sig") as fh:
        first_line = fh.readline()
    return tuple(f.strip().strip('"') for f in first_line.rstrip("\r\n").split(","))


def _count_data_rows(path: Path) -> int:
    """Data rows (lines minus the header). Streamed — ratings.csv is ~1GB."""
    total = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK_SIZE)
            if not block:
                break
            total += block.count(b"\n")
    return max(total - 1, 0)


def fetch() -> int:
    """Download (resumable) + CRC-verify the archive, then extract the CSVs."""
    DATA_RAW_ML32M.mkdir(parents=True, exist_ok=True)
    dest = _zip_path()

    observed = dest.stat().st_size if dest.exists() else 0
    print(
        f"[fetch] {ZIP_FILENAME}: downloading (observed={observed}, "
        f"advertised~{APPROX_ZIP_BYTES}) from {ZIP_URL}"
    )
    cmd = [
        "curl",
        "-sS",
        "--http1.1",
        "--retry",
        "15",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "-C",
        "-",
        "-o",
        str(dest),
        ZIP_URL,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[error] {ZIP_FILENAME}: curl exited with code {result.returncode}")
        return result.returncode
    return _extract(dest)


def extract() -> int:
    """CRC-verify the already-downloaded archive and (re-)extract every member.

    Separate from :func:`fetch` because adding a member to :data:`MEMBERS` must
    not force a 239MB re-download on a machine that already holds the zip (the
    tags.csv case). ``fetch`` is download-then-extract; this is the second half.
    """
    dest = _zip_path()
    if not dest.exists():
        print(f"[error] {ZIP_FILENAME}: not found at {dest}; run `fetch` first")
        return 1
    return _extract(dest)


def _extract(dest: Path) -> int:
    """CRC-verify ``dest`` and write every :data:`MEMBERS` file next to it."""
    size = dest.stat().st_size
    print(f"[done] {ZIP_FILENAME}: {size} bytes; verifying archive CRCs...")
    with zipfile.ZipFile(dest) as zf:
        bad = zf.testzip()
        if bad is not None:
            print(f"[error] {ZIP_FILENAME}: CRC failure on member {bad!r} — download is corrupt")
            return 1
        names = set(zf.namelist())
        missing = [m.arcname for m in MEMBERS if m.arcname not in names]
        if missing:
            print(f"[error] {ZIP_FILENAME}: expected member(s) {missing} not in archive")
            return 1
        print(f"[ok] {ZIP_FILENAME}: CRCs verified ({len(names)} members)")

        for member in MEMBERS:
            out = DATA_RAW_ML32M / member.filename
            with zf.open(member.arcname) as src, open(out, "wb") as dst:
                while True:
                    block = src.read(CHUNK_SIZE)
                    if not block:
                        break
                    dst.write(block)
            header = _header_of(out)
            if header != member.expected_header:
                print(
                    f"[error] {member.filename}: header {list(header)} != expected "
                    f"{list(member.expected_header)}; the published schema changed — "
                    "stop and revisit the bronze schemas before ingesting."
                )
                return 1
            rows = _count_data_rows(out)
            if member.published_rows is None:
                published = "no verified published count; observed is ground truth"
            else:
                published = (
                    f"published {member.published_rows}, "
                    f"delta {rows - member.published_rows:+d}"
                )
            print(
                f"[ok] {member.filename}: {out.stat().st_size} bytes, header verified, "
                f"{rows} data rows ({published})"
            )
    for arcname, reason in UNUSED_MEMBERS.items():
        print(f"[note] {arcname} NOT extracted ({reason}); it remains inside {dest}")
    return 0


def _document(entries: list[dict], row_counts: dict[str, int], download_date: str) -> str:
    """Render the whole ``data/MANIFEST_ML32M.md`` document.

    Section shape mirrors ``data/MANIFEST.md`` (``## Files`` → ``### <filename>``
    → ``- Size (bytes):`` / ``- Data rows (excl. header):`` / ``- SHA-256 ...``)
    so one parser shape serves both, but it is a SEPARATE file: see the module
    docstring on why the two hashes must stay independent.
    """
    lines: list[str] = []
    lines.append("# MovieLens 32M (ML-32M) — Data Manifest")
    lines.append("")
    lines.append(
        "Regime-contrast dataset for docs/engineering-log/UPGRADE_PLAN.md §8c (Phase 9, T9-3a/T9-3b). "
        "Separate from `data/MANIFEST.md` on purpose: run records hash their whole "
        "dataset manifest, and `make reproduce-headline` compares that hash for the "
        "pinned Amazon headline. One manifest per dataset keeps the two independent."
    )
    lines.append("")
    lines.append("## Source URL")
    lines.append("")
    lines.append(f"- {ZIP_URL}")
    lines.append("")
    lines.append("## Citation")
    lines.append("")
    lines.append(CITATION)
    lines.append("")
    lines.append("## License")
    lines.append("")
    lines.append(
        "Research use only per the GroupLens usage license; raw data is never "
        "redistributed and data/ is gitignored (only this manifest is committed)."
    )
    lines.append("")
    lines.append("## Download date")
    lines.append("")
    lines.append(download_date)
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for entry in entries:
        lines.append(f"### {entry['filename']}")
        lines.append("")
        if entry.get("url"):
            lines.append(f"- URL: {entry['url']}")
        else:
            lines.append(f"- Extracted from: {ZIP_FILENAME} ({entry['arcname']})")
        lines.append(f"- Size (bytes): {entry['size']}")
        if entry.get("data_rows") is not None:
            lines.append(f"- Data rows (excl. header): {entry['data_rows']}")
        lines.append(
            f"- SHA-256 (computed locally — ours is ground truth): {entry['sha256']}"
        )
        lines.append("")
    lines.append("## Not extracted")
    lines.append("")
    for arcname, reason in UNUSED_MEMBERS.items():
        lines.append(f"- {arcname}: {reason}")
    lines.append("")
    lines.append("## Published counts")
    lines.append("")
    lines.append(
        f"{PUBLISHED_COUNTS['ratings']} ratings / {PUBLISHED_COUNTS['users']} users / "
        f"{PUBLISHED_COUNTS['movies']} movies, timestamps 1995 → Oct 2023 (ML-32M "
        "README); no verified published count for tag applications, so the observed "
        "tags.csv row count above is ground truth. Observed extracted row counts: "
        + ", ".join(f"{name}={count}" for name, count in sorted(row_counts.items()))
        + ". These per-file row counts are enforced against the live bronze tables by "
        "`make bronze-verify-ml32m`."
    )
    lines.append("")
    lines.append("## Timestamp caveat (docs/engineering-log/UPGRADE_PLAN.md §8b)")
    lines.append("")
    lines.append(
        "MovieLens timestamps are rating-ENTRY times, not consumption times, and the "
        "catalog is backfilled — an item's first rating is only a proxy for its "
        "release. Disclosed wherever the regime contrast is reported (cite Sun et al., "
        "arXiv:2307.09985)."
    )
    lines.append("")
    return "\n".join(lines)


# --- parsing (the reader side of the format written above) ----------------------

_FILE_HEADING_RE = re.compile(r"^###\s+(?P<name>\S+)\s*$")
_SIZE_RE = re.compile(r"^-\s+Size\s+\(bytes\):\s*(?P<size>\d+)\s*$")
_ROWS_RE = re.compile(r"^-\s+Data\s+rows\s+\(excl\.\s+header\):\s*(?P<rows>\d+)\s*$")
_SHA_RE = re.compile(r"^-\s+SHA-256[^:]*:\s*(?P<sha>[0-9a-f]{64})\s*$")
_DOWNLOAD_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_manifest(path: str | Path) -> dict[str, dict]:
    """:func:`parse_manifest_text` for a file on disk."""
    return parse_manifest_text(Path(path).read_text())


def parse_manifest_text(text: str) -> dict[str, dict]:
    """``{filename: {"sha256", "size", "data_rows"}}`` from a manifest document.

    Only entries under ``## Files`` are returned; ``data_rows`` is ``None`` for
    files that declare none (the zip). Deliberately strict — a 64-hex SHA-256 and
    an integer byte size are the whole point of the file, so a malformed line
    leaves the field ``None`` rather than half-parsed, and the caller reports
    precisely which fact is missing.
    """
    entries: dict[str, dict] = {}
    current: str | None = None
    in_files = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            in_files = line[3:].strip().lower() == "files"
            current = None
            continue
        if not in_files:
            continue
        heading = _FILE_HEADING_RE.match(line)
        if heading:
            current = heading.group("name")
            entries[current] = {"sha256": None, "size": None, "data_rows": None}
            continue
        if current is None:
            continue
        size = _SIZE_RE.match(line)
        if size:
            entries[current]["size"] = int(size.group("size"))
            continue
        rows = _ROWS_RE.match(line)
        if rows:
            entries[current]["data_rows"] = int(rows.group("rows"))
            continue
        sha = _SHA_RE.match(line)
        if sha:
            entries[current]["sha256"] = sha.group("sha")
    return entries


def _existing_download_date(path: Path) -> str | None:
    """The ``## Download date`` already recorded, if the file is there.

    Preserved across regenerations so re-running ``manifest`` on unchanged bytes
    does not move the file hash that recorded runs attest to.
    """
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text().splitlines()]
    for i, line in enumerate(lines):
        if line.lower() == "## download date":
            for candidate in lines[i + 1 : i + 4]:
                if _DOWNLOAD_DATE_RE.match(candidate):
                    return candidate
    return None


def manifest(manifest_path: Path | None = None, write: bool = True) -> int:
    """Hash the archive + extracted CSVs and write ``data/MANIFEST_ML32M.md``."""
    out_path = Path(manifest_path) if manifest_path else MANIFEST_ML32M_PATH
    if out_path.name == "MANIFEST.md":
        print(
            "[error] refusing to write the ML-32M manifest into data/MANIFEST.md: the "
            "Amazon headline record's dataset_manifest_hash attests to that file and "
            "`make reproduce-headline` compares it. Use data/MANIFEST_ML32M.md."
        )
        return 1
    dest = _zip_path()
    if not dest.exists():
        print(f"[error] {ZIP_FILENAME}: not found at {dest}; run `fetch` first")
        return 1

    entries: list[dict] = []
    row_counts: dict[str, int] = {}

    print(f"[ok] {ZIP_FILENAME}: computing SHA-256...")
    entries.append(
        {
            "filename": ZIP_FILENAME,
            "url": ZIP_URL,
            "size": dest.stat().st_size,
            "sha256": _sha256(dest),
            "data_rows": None,
        }
    )
    print(f"[ok] {ZIP_FILENAME}: sha256={entries[-1]['sha256']}")

    for member in MEMBERS:
        path = DATA_RAW_ML32M / member.filename
        if not path.exists():
            print(f"[error] {member.filename}: not found at {path}; run `fetch` first")
            return 1
        header = _header_of(path)
        if header != member.expected_header:
            print(
                f"[error] {member.filename}: header {list(header)} != expected "
                f"{list(member.expected_header)}"
            )
            return 1
        rows = _count_data_rows(path)
        row_counts[member.filename] = rows
        sha = _sha256(path)
        print(f"[ok] {member.filename}: sha256={sha} rows={rows}")
        entries.append(
            {
                "filename": member.filename,
                "arcname": member.arcname,
                "size": path.stat().st_size,
                "sha256": sha,
                "data_rows": rows,
            }
        )

    # Preserve the recorded download date so an unchanged dataset re-renders to
    # byte-identical text (and therefore an unchanged dataset_manifest_hash).
    download_date = _existing_download_date(out_path) or date.today().isoformat()
    document = _document(entries, row_counts, download_date)

    print("\n" + "=" * 72)
    print(f"{out_path} (ML-32M ONLY — never data/MANIFEST.md):")
    print("=" * 72 + "\n")
    print(document)
    print("=" * 72)
    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        unchanged = out_path.exists() and out_path.read_text() == document
        out_path.write_text(document)
        print(f"[done] wrote {out_path}" + (" (unchanged)" if unchanged else ""))
        print("[note] commit data/MANIFEST_ML32M.md; data/MANIFEST.md must NOT be touched.")
    else:
        print(f"[note] --print-only: {out_path} not written")

    # Summary JSON MUST be the last stdout line (repo convention).
    print(
        json.dumps(
            {
                "dataset": "ml32m",
                "url": ZIP_URL,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "manifest_path": str(out_path),
                "written": bool(write),
                "download_date": download_date,
                "files": entries,
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.ingest.download_ml32m")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="Download ml-32m.zip (resumable), CRC-verify, extract the 3 CSVs.")
    sub.add_parser(
        "extract",
        help="CRC-verify the already-downloaded ml-32m.zip and re-extract the CSVs.",
    )
    man = sub.add_parser(
        "manifest",
        help="Compute SHA-256s and write data/MANIFEST_ML32M.md (never data/MANIFEST.md).",
    )
    man.add_argument("--out", default=None, help="Override the manifest path (tests).")
    man.add_argument(
        "--print-only",
        action="store_true",
        help="Print the document without writing it.",
    )
    args = parser.parse_args(argv)

    if args.command == "fetch":
        return fetch()
    elif args.command == "extract":
        return extract()
    elif args.command == "manifest":
        return manifest(
            manifest_path=Path(args.out) if args.out else None, write=not args.print_only
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
