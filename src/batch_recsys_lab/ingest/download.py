"""Download and manifest tooling for the Amazon Reviews 2023 raw files.

Usage:
    python -m batch_recsys_lab.ingest.download fetch
    python -m batch_recsys_lab.ingest.download manifest
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.md"

ARXIV_ID = "2403.03952"
DOWNLOAD_DATE = date(2026, 8, 5)

CHUNK_SIZE = 8 * 1024 * 1024  # 8MB


@dataclass(frozen=True)
class RawFile:
    filename: str
    url: str
    expected_size: int
    last_modified: str


RAW_FILES = [
    RawFile(
        filename="Electronics.jsonl.gz",
        url="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz",
        expected_size=6474438619,
        last_modified="Thu, 16 Jan 2025 23:28:26 GMT",
    ),
    RawFile(
        filename="meta_Electronics.jsonl.gz",
        url="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz",
        expected_size=1312900427,
        last_modified="Thu, 16 Jan 2025 22:22:45 GMT",
    ),
]


def _dest_path(raw_file: RawFile) -> Path:
    return DATA_RAW / raw_file.filename


def fetch() -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for raw_file in RAW_FILES:
        dest = _dest_path(raw_file)
        if dest.exists() and dest.stat().st_size == raw_file.expected_size:
            print(f"[skip] {raw_file.filename}: already at expected size ({raw_file.expected_size} bytes)")
            continue
        observed = dest.stat().st_size if dest.exists() else 0
        print(
            f"[fetch] {raw_file.filename}: downloading (observed={observed}, "
            f"expected={raw_file.expected_size}) from {raw_file.url}"
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
            raw_file.url,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[error] {raw_file.filename}: curl exited with code {result.returncode}")
            return result.returncode
        final_size = dest.stat().st_size if dest.exists() else 0
        if final_size != raw_file.expected_size:
            print(
                f"[warn] {raw_file.filename}: download finished but size mismatch "
                f"(observed={final_size}, expected={raw_file.expected_size})"
            )
        else:
            print(f"[done] {raw_file.filename}: complete ({final_size} bytes)")
    return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _extract_existing_reconciliation(text: str) -> str | None:
    """Return the existing '## Bronze reconciliation' section body if present and filled."""
    match = re.search(
        r"## Bronze reconciliation\n(.*?)(\n## |\Z)", text, flags=re.DOTALL
    )
    if not match:
        return None
    body = match.group(1)
    if body.strip() == "<!-- filled by make bronze-verify -->":
        return None
    return body.rstrip("\n")


def manifest() -> int:
    # Verify size + gather sizes/hashes.
    file_infos = []
    for raw_file in RAW_FILES:
        dest = _dest_path(raw_file)
        if not dest.exists():
            print(
                f"[error] {raw_file.filename}: not found at {dest} "
                f"(observed=0, expected={raw_file.expected_size})"
            )
            return 1
        observed_size = dest.stat().st_size
        if observed_size != raw_file.expected_size:
            print(
                f"[error] {raw_file.filename}: incomplete download "
                f"(observed={observed_size}, expected={raw_file.expected_size}). "
                "Wait for the background download to finish before running manifest."
            )
            return 1
        print(f"[ok] {raw_file.filename}: size matches expected ({observed_size} bytes); computing SHA-256...")
        sha256 = _sha256(dest)
        print(f"[ok] {raw_file.filename}: sha256={sha256}")
        file_infos.append((raw_file, observed_size, sha256))

    # Preserve existing reconciliation section if present and filled.
    existing_reconciliation = None
    if MANIFEST_PATH.exists():
        existing_text = MANIFEST_PATH.read_text()
        existing_reconciliation = _extract_existing_reconciliation(existing_text)

    reconciliation_body = (
        existing_reconciliation
        if existing_reconciliation is not None
        else "<!-- filled by make bronze-verify -->"
    )

    lines = []
    lines.append("# Amazon Reviews 2023 — Electronics — Data Manifest")
    lines.append("")
    lines.append(
        "Dataset: Amazon Reviews 2023 (McAuley-Lab), Electronics review category "
        "and metadata."
    )
    lines.append("")
    lines.append("## Source URLs")
    lines.append("")
    for raw_file, _, _ in file_infos:
        lines.append(f"- {raw_file.url}")
    lines.append("")
    lines.append("## Citation")
    lines.append("")
    lines.append(
        f"Hou, Yupeng, et al. \"Bridging Language and Items for Retrieval and "
        f"Recommendation.\" arXiv preprint arXiv:{ARXIV_ID} (2024)."
    )
    lines.append("")
    lines.append("## License")
    lines.append("")
    lines.append(
        "Research use only per the dataset's release terms; raw data is never "
        "redistributed and data/ is gitignored (only this manifest is committed)."
    )
    lines.append("")
    lines.append("## Download date")
    lines.append("")
    lines.append(DOWNLOAD_DATE.isoformat())
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for raw_file, size, sha256 in file_infos:
        lines.append(f"### {raw_file.filename}")
        lines.append("")
        lines.append(f"- URL: {raw_file.url}")
        lines.append(f"- Size (bytes): {size}")
        lines.append(f"- Server Last-Modified: {raw_file.last_modified}")
        lines.append(f"- SHA-256 (computed locally — ours is ground truth): {sha256}")
        lines.append("")
    lines.append("## Published counts")
    lines.append("")
    lines.append(
        "43.9M reviews / 1.61M items (per the Amazon Reviews 2023 / Hou et al. 2024 "
        "release site). Observed bronze counts and the delta against these published "
        "counts are recorded in the \"Bronze reconciliation\" section below."
    )
    lines.append("")
    lines.append("## Bronze layer notes")
    lines.append("")
    lines.append(
        "bronze.reviews projects out the `text` and `images` columns per "
        "docs/engineering-log/UPGRADE_PLAN.md §5 (the lab never uses review text; item text comes from "
        "metadata)."
    )
    lines.append("")
    lines.append("## Bronze reconciliation")
    lines.append("")
    lines.append(reconciliation_body)
    lines.append("")

    content = "\n".join(lines)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(content)
    print(f"[done] wrote {MANIFEST_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="batch_recsys_lab.ingest.download")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="Download raw files (resumable), skipping ones already at expected size.")
    sub.add_parser("manifest", help="Verify sizes, compute SHA-256, and write data/MANIFEST.md.")
    args = parser.parse_args(argv)

    if args.command == "fetch":
        return fetch()
    elif args.command == "manifest":
        return manifest()
    return 1


if __name__ == "__main__":
    sys.exit(main())
