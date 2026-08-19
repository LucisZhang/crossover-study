"""Download + manifest tooling for MovieLens 32M (Phase 9, T9-3a; UPGRADE_PLAN.md §8c).

    python -m batch_recsys_lab.ingest.download_ml32m fetch
    python -m batch_recsys_lab.ingest.download_ml32m manifest

Same posture as the Amazon flow in ``ingest/download.py``: the published archive
carries no official checksum, so **our locally computed SHA-256 is ground truth**
— computed once here, printed, and recorded in ``data/MANIFEST.md`` by the owner.
Research license: the raw data is never redistributed (``data/`` is gitignored;
only the manifest is committed). Cite Harper & Konstan 2015.

Two deliberate differences from the Amazon flow, both because the source differs:

* **No hardcoded expected size.** The Amazon files are pinned by an exact byte
  count published on the release site; ml-32m.zip is not (≈239MB is documentation,
  not a contract). Completeness is instead proven by a *stronger* check — the zip's
  own CRCs (:meth:`zipfile.ZipFile.testzip`), which a truncated or corrupted
  download cannot pass.
* **This module never writes ``data/MANIFEST.md``.** ``download.py manifest``
  regenerates that file for the Amazon dataset; regenerating it here would drop
  the Amazon sections. We print a manifest *fragment* (and drop a copy next to the
  raw files, inside gitignored ``data/``) for the owner to paste.

Only ``ratings.csv`` and ``movies.csv`` are extracted — they are the only files
the pipeline ingests. ``tags.csv``/``links.csv``/``README.txt`` stay inside the
archive (kept, since the archive is the hashed artifact of record).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW_ML32M = REPO_ROOT / "data" / "raw" / "ml32m"
FRAGMENT_FILENAME = "manifest_fragment_ml32m.md"

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
    published_rows: int  # ML-32M README; RECONCILED at ingest, never asserted here


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
]

# Published headline counts (ML-32M README) for the manifest fragment; the bronze
# ingest summary reports the observed counts and the delta.
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
    """Download (resumable) + CRC-verify the archive, then extract the two CSVs."""
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
            delta = rows - member.published_rows
            print(
                f"[ok] {member.filename}: {out.stat().st_size} bytes, header verified, "
                f"{rows} data rows (published {member.published_rows}, delta {delta:+d})"
            )
    print(
        f"[note] tags.csv / links.csv are NOT extracted (nothing ingests them); they "
        f"remain inside {dest}"
    )
    return 0


def _fragment(entries: list[dict], row_counts: dict[str, int]) -> str:
    """The markdown block for data/MANIFEST.md (owner pastes it; we never write it)."""
    lines: list[str] = []
    lines.append("## MovieLens 32M (ML-32M) — regime-contrast dataset (Phase 9, T9-3a)")
    lines.append("")
    lines.append("Dataset: MovieLens 32M (GroupLens). Used as the T8-4/T9-3 regime contrast.")
    lines.append("")
    lines.append("### Source URL")
    lines.append("")
    lines.append(f"- {ZIP_URL}")
    lines.append("")
    lines.append("### Citation")
    lines.append("")
    lines.append(CITATION)
    lines.append("")
    lines.append("### License")
    lines.append("")
    lines.append(
        "Research use only per the GroupLens usage license; raw data is never "
        "redistributed and data/ is gitignored (only this manifest is committed)."
    )
    lines.append("")
    lines.append("### Download date")
    lines.append("")
    lines.append(date.today().isoformat())
    lines.append("")
    lines.append("### Files")
    lines.append("")
    for entry in entries:
        lines.append(f"#### {entry['filename']}")
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
    lines.append("### Published counts")
    lines.append("")
    lines.append(
        f"{PUBLISHED_COUNTS['ratings']} ratings / {PUBLISHED_COUNTS['users']} users / "
        f"{PUBLISHED_COUNTS['movies']} movies, timestamps 1995 → Oct 2023 (ML-32M "
        "README). Observed extracted row counts: "
        + ", ".join(f"{name}={count}" for name, count in sorted(row_counts.items()))
        + ". Observed bronze counts are printed by "
        "`make data-ml32m` (ingest_summary.jsonl, table_name ml32m_*)."
    )
    lines.append("")
    lines.append("### Timestamp caveat (UPGRADE_PLAN.md §8b)")
    lines.append("")
    lines.append(
        "MovieLens timestamps are rating-ENTRY times, not consumption times, and the "
        "catalog is backfilled — an item's first rating is only a proxy for its "
        "release. Disclosed wherever the regime contrast is reported (cite Sun et al., "
        "arXiv:2307.09985)."
    )
    lines.append("")
    return "\n".join(lines)


def manifest() -> int:
    """Hash the archive + extracted CSVs and print the manifest fragment."""
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

    fragment = _fragment(entries, row_counts)
    fragment_path = DATA_RAW_ML32M / FRAGMENT_FILENAME
    fragment_path.write_text(fragment)

    print("\n" + "=" * 72)
    print("PASTE THE BLOCK BELOW INTO data/MANIFEST.md (this tool never edits it):")
    print("=" * 72 + "\n")
    print(fragment)
    print("=" * 72)
    print(f"[done] fragment also written to {fragment_path}")

    # Summary JSON MUST be the last stdout line (repo convention).
    print(
        json.dumps(
            {
                "dataset": "ml32m",
                "url": ZIP_URL,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "fragment_path": str(fragment_path),
                "files": entries,
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.ingest.download_ml32m")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="Download ml-32m.zip (resumable), CRC-verify, extract the 2 CSVs.")
    sub.add_parser(
        "manifest",
        help="Compute SHA-256s and print the data/MANIFEST.md fragment (never writes it).",
    )
    args = parser.parse_args(argv)

    if args.command == "fetch":
        return fetch()
    elif args.command == "manifest":
        return manifest()
    return 1


if __name__ == "__main__":
    sys.exit(main())
