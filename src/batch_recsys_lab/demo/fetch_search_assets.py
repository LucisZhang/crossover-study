"""Vendored in-browser model fetcher — Phase 6, T35 (plan §9 exhibit 3).

Downloads the two things the live semantic-search exhibit needs to run with
``env.allowRemoteModels = false``:

* ``@huggingface/transformers`` **3.8.1** (npm tarball) → ``demo/vendor/transformers/``.
  3.8.1 is pinned because its ``dist/`` still ships
  ``ort-wasm-simd-threaded.jsep.wasm`` in-package; 4.x dropped it and would need
  a second registry dependency to run offline.
* ``Xenova/all-MiniLM-L6-v2`` at a pinned commit — ``config.json``,
  ``tokenizer.json``, ``tokenizer_config.json``, ``onnx/model_quantized.onnx`` →
  ``demo/vendor/models/Xenova/all-MiniLM-L6-v2/``.

**The hash, not the URL, is ground truth.** Both the expected SHA-256 *and* the
source URL of every downloaded file are read from the machine-parseable table in
``demo/README.md`` under "## Search assets". Any mismatch is a hard failure:
non-zero exit, an explicit
``hash mismatch — URL content has drifted; the README hash is ground truth``
message, and **no partial install** — everything is staged and hashed before a
single byte of ``demo/vendor/`` is touched.

Two modes:

``--record-hashes``
    Bootstrap. Downloads from the URLs in ``configs/search_export.yaml``,
    computes the hashes, and prints the markdown table to paste into
    ``demo/README.md``. Installs nothing. Run once; then run the normal mode to
    prove the loop closes.

(default)
    Verified install. Reads the README table, refuses to fetch anything the
    table does not name (and anything the config does not require), verifies
    every byte, then installs atomically and writes
    ``demo/vendor/vendor_manifest.json`` — which is also how ``demo/js/search.js``
    discovers the real megabyte counts for its size warning.

The tarball is treated as one artifact with one hash: that hash covers every
extracted byte, which is a strictly stronger guarantee than per-file hashes over
a hand-listed subset (a file nobody thought to list cannot slip in).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import yaml

DEFAULT_CONFIG = "configs/search_export.yaml"
MISMATCH_MESSAGE = "hash mismatch — URL content has drifted; the README hash is ground truth"
TABLE_HEADER = ["File", "SHA-256", "Approx size", "Source URL"]
USER_AGENT = "batch-recsys-lab/demo-assets (+local, pinned)"


class FetchError(Exception):
    """Any condition that must abort the install with a non-zero exit."""


# --- README table --------------------------------------------------------------

_CELL_STRIP = re.compile(r"^`|`$")


def _clean_cell(cell: str) -> str:
    c = cell.strip()
    c = _CELL_STRIP.sub("", c.strip())
    c = c.strip()
    if c.startswith("<") and c.endswith(">"):
        c = c[1:-1]
    m = re.fullmatch(r"\[([^\]]*)\]\(([^)]*)\)", c)  # [text](url) -> url
    if m:
        c = m.group(2)
    return c.strip()


def parse_readme_table(text: str, section: str) -> dict[str, dict]:
    """Parse the "## <section>" markdown table into ``{file: {...}}``.

    Recognised by its header row containing a ``SHA-256`` column. Placeholder
    rows (any cell wrapped in ``_(...)_``) are ignored so the un-filled template
    parses to an empty table rather than to garbage.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.fullmatch(rf"#{{1,6}}\s+{re.escape(section)}\s*", line.strip()):
            start = i + 1
            break
    if start is None:
        raise FetchError(f"README has no '{section}' section")

    end = len(lines)
    for i in range(start, len(lines)):
        if re.match(r"^#{1,6}\s+", lines[i]) and not lines[i].strip().startswith("#" * 7):
            end = i
            break

    header_at = None
    for i in range(start, end):
        if lines[i].lstrip().startswith("|") and "sha-256" in lines[i].lower():
            header_at = i
            break
    if header_at is None:
        raise FetchError(f"README '{section}' section has no table with a SHA-256 column")

    def cells(line: str) -> list[str]:
        body = line.strip()
        if body.startswith("|"):
            body = body[1:]
        if body.endswith("|"):
            body = body[:-1]
        return [c for c in body.split("|")]

    header = [_clean_cell(c).lower() for c in cells(lines[header_at])]
    try:
        i_file = header.index("file")
        i_sha = header.index("sha-256")
        i_size = header.index("approx size")
        i_url = header.index("source url")
    except ValueError as exc:
        raise FetchError(
            f"README '{section}' table header is {header}; expected columns {TABLE_HEADER}"
        ) from exc

    out: dict[str, dict] = {}
    for i in range(header_at + 1, end):
        line = lines[i]
        if not line.lstrip().startswith("|"):
            break
        row = cells(line)
        if all(re.fullmatch(r"[:\-\s]*", c) for c in row):  # separator
            continue
        vals = [_clean_cell(c) for c in row]
        if len(vals) < len(header):
            continue
        if any(v.startswith("_(") for v in vals):  # template placeholder
            continue
        name = vals[i_file]
        if not name:
            continue
        sha = vals[i_sha].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise FetchError(f"README row {name!r}: {vals[i_sha]!r} is not a sha256 hex digest")
        if name in out:
            raise FetchError(f"README table lists {name!r} twice")
        out[name] = {"file": name, "sha256": sha, "size": vals[i_size], "url": vals[i_url]}
    return out


def render_readme_table(rows: Iterable[dict]) -> str:
    lines = ["| " + " | ".join(TABLE_HEADER) + " |", "|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| `{r['file']}` | `{r['sha256']}` | {r['size']} | `{r['url']}` |"
        )
    return "\n".join(lines)


def human_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} MB"
    if n >= 1_000:
        return f"{n / 1e3:.0f} kB"
    return f"{n} B"


# --- download ------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_url_allowed(url: str, allowed_hosts: list[str], *, allow_file: bool) -> None:
    from urllib.parse import urlparse

    p = urlparse(url)
    if p.scheme == "file":
        if not allow_file:
            raise FetchError(f"refusing file:// URL {url} (use --allow-file-urls; tests only)")
        return
    if p.scheme != "https":
        raise FetchError(f"refusing non-https URL {url}")
    if p.hostname not in allowed_hosts:
        raise FetchError(
            f"refusing to fetch from host {p.hostname!r}: not in allowed_url_hosts {allowed_hosts}. "
            "This script downloads the pinned transformers.js tarball and the pinned "
            "Xenova/all-MiniLM-L6-v2 files and nothing else."
        )


def download(url: str, dest: Path, *, timeout: int = 180) -> tuple[int, str]:
    """Fetch ``url`` to ``dest``. Returns ``(bytes, final_url_after_redirects)``.

    Redirects are followed (huggingface.co hands LFS blobs off to a CDN host), and
    the final URL is returned so the caller can *show* where the bytes actually
    came from. Redirect targets are deliberately not allow-listed: in verified
    mode the SHA-256 gate makes the delivery path irrelevant, and in
    ``--record-hashes`` mode the operator is looking at the printed URL.

    **Content-Length is enforced.** A dropped connection ends ``read()`` without
    raising, so an unchecked copy turns a truncated transfer into a perfectly
    well-formed short file. In verified mode the hash would catch that; in
    ``--record-hashes`` mode nothing would, and the truncated body's hash would be
    pasted into the README as ground truth. (Not hypothetical: the first bootstrap
    run of this script recorded a 9,491,392-byte prefix of the 10,482,401-byte
    transformers tarball.) A short read is a hard failure here.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
            final_url = resp.geturl()
            declared_raw = resp.headers.get("Content-Length")
            shutil.copyfileobj(resp, out, length=1 << 20)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise FetchError(f"download failed for {url}: {exc}") from exc

    got = dest.stat().st_size
    declared = None
    if declared_raw is not None:
        try:
            declared = int(declared_raw)
        except ValueError:
            declared = None
    if declared is not None and got != declared:
        dest.unlink(missing_ok=True)
        raise FetchError(
            f"truncated download for {url}: got {got:,} bytes, server declared {declared:,}. "
            "Nothing was kept. Re-run; if it repeats, the source or the connection is at fault."
        )
    if declared is None:
        print(
            f"  warning: {url} sent no Content-Length; a truncated body cannot be detected by size",
            file=sys.stderr,
        )
    return got, final_url


# --- plan ----------------------------------------------------------------------


def required_artifacts(cfg: dict) -> list[dict]:
    """The artifacts this install needs, in table order. Config is the *set*;
    the README is the *hash and the URL*."""
    try:
        return _required_artifacts(cfg)
    except (KeyError, TypeError) as exc:
        raise FetchError(f"config is malformed: {exc}") from exc


def _required_artifacts(cfg: dict) -> list[dict]:
    tj = cfg["transformers_js"]
    out = [
        {
            "file": tj["tarball"],
            "url": tj["url"],
            "kind": "tarball",
            "install_subdir": tj["install_subdir"],
            "extract": dict(tj["extract"]),
        }
    ]
    model = cfg["model"]
    for rel in model["files"]:
        out.append(
            {
                "file": f"{model['install_subdir']}/{rel}",
                "url": f"{model['base_url']}/{rel}",
                "kind": "file",
                "install_subdir": model["install_subdir"],
                "rel": rel,
            }
        )
    return out


def extract_members(tar_path: Path, mapping: dict[str, str], dest: Path) -> list[tuple[str, int]]:
    """Extract exactly ``mapping`` (member -> relative path). No symlinks, no
    absolute paths, no members outside the mapping."""
    written: list[tuple[str, int]] = []
    with tarfile.open(tar_path, "r:gz") as tf:
        names = set(tf.getnames())
        missing = [m for m in mapping if m not in names]
        if missing:
            raise FetchError(f"{tar_path.name}: tarball is missing expected members {missing}")
        for member, rel in mapping.items():
            info = tf.getmember(member)
            if not info.isfile():
                raise FetchError(f"{tar_path.name}: member {member} is not a regular file")
            target = dest / rel
            try:  # component-wise containment; a "dest-evil" sibling is not inside dest
                target.resolve().relative_to(dest.resolve())
            except ValueError as exc:
                raise FetchError(
                    f"{tar_path.name}: member {member} would write outside the destination"
                ) from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(info)
            if src is None:
                raise FetchError(f"{tar_path.name}: member {member} has no content")
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)
            written.append((rel, target.stat().st_size))
    return written


# --- modes ---------------------------------------------------------------------


def record_hashes(cfg: dict, *, allow_file: bool) -> int:
    """Bootstrap: download from the config URLs, print the README table."""
    allowed = cfg.get("allowed_url_hosts", [])
    # Staging lives OUTSIDE vendor_dir: a mode that installs nothing must not
    # leave a demo/vendor/ behind for search.js to trip over.
    stage = Path(tempfile.mkdtemp(prefix="record-hashes-"))
    rows = []
    try:
        for art in required_artifacts(cfg):
            check_url_allowed(art["url"], allowed, allow_file=allow_file)
            dest = stage / art["file"].replace("/", "__")
            print(f"fetching {art['url']} …", flush=True)
            size, final_url = download(art["url"], dest)
            digest = sha256_file(dest)
            rows.append(
                {"file": art["file"], "sha256": digest, "size": human_size(size), "url": art["url"]}
            )
            print(f"  {size:,} bytes  sha256:{digest}")
            if final_url != art["url"]:
                print(f"  (served from {final_url})")
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print("")
    print(f"--- paste under '## {cfg['readme_section']}' in {cfg['readme']} ---")
    print(render_readme_table(rows))
    print("--- end ---")
    print("")
    print("Nothing was installed. Paste the table, then re-run without --record-hashes.")
    return 0


def verified_install(cfg: dict, *, allow_file: bool, keep_downloads: bool = False) -> int:
    vendor = Path(cfg["vendor_dir"])
    readme = Path(cfg["readme"])
    section = cfg["readme_section"]
    allowed = cfg.get("allowed_url_hosts", [])

    if not readme.exists():
        raise FetchError(f"{readme} not found — it is where the expected hashes live")
    table = parse_readme_table(readme.read_text(), section)
    wanted = required_artifacts(cfg)
    wanted_names = [a["file"] for a in wanted]

    if not table:
        raise FetchError(
            f"{readme} '{section}' table is empty. Bootstrap it once with:\n"
            f"    uv run python -m batch_recsys_lab.demo.fetch_search_assets --record-hashes\n"
            "then paste the printed table into the README."
        )
    missing = [n for n in wanted_names if n not in table]
    extra = [n for n in table if n not in wanted_names]
    if missing:
        raise FetchError(f"{readme} '{section}' table has no row for {missing}")
    if extra:
        raise FetchError(
            f"{readme} '{section}' table names files this install does not want: {extra}. "
            "Only the pinned transformers.js tarball and the pinned MiniLM files are fetched."
        )

    vendor_created = not vendor.exists()
    stage = vendor / ".staging"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    downloads = stage / "_downloads"
    tree = stage / "_tree"
    tree.mkdir(parents=True, exist_ok=True)

    installed: list[dict] = []
    t0 = time.perf_counter()
    try:
        # ---- fetch + verify EVERYTHING before touching the install tree ------
        for art in wanted:
            row = table[art["file"]]
            url = row["url"]
            if url != art["url"]:
                print(
                    f"note: README URL for {art['file']} differs from configs/search_export.yaml\n"
                    f"      README: {url}\n"
                    f"      config: {art['url']}\n"
                    "      fetching the README's URL — the README is the manifest of record.",
                    file=sys.stderr,
                )
            check_url_allowed(url, allowed, allow_file=allow_file)
            blob = downloads / art["file"].replace("/", "__")
            print(f"fetching {art['file']} …", flush=True)
            size, final_url = download(url, blob)
            if final_url != url:
                print(f"  (served from {final_url})")
            digest = sha256_file(blob)
            if digest != row["sha256"]:
                raise FetchError(
                    f"{MISMATCH_MESSAGE}\n"
                    f"  file:     {art['file']}\n"
                    f"  url:      {url}\n"
                    f"  expected: {row['sha256']}   (recorded in {readme})\n"
                    f"  actual:   {digest}\n"
                    "  nothing was installed. Investigate the drift; do not edit the README hash "
                    "to make this pass."
                )
            print(f"  ok  {size:,} bytes  sha256:{digest}")
            art["_blob"] = blob
            art["_size"] = size
            art["_sha256"] = digest

        # ---- materialise the install tree in staging --------------------------
        for art in wanted:
            sub = tree / art["install_subdir"]
            sub.mkdir(parents=True, exist_ok=True)
            if art["kind"] == "tarball":
                written = extract_members(art["_blob"], art["extract"], sub)
                for rel, nbytes in written:
                    installed.append(
                        {
                            "path": f"{art['install_subdir']}/{rel}",
                            "bytes": nbytes,
                            "from": art["file"],
                            "url": table[art["file"]]["url"],
                            "tarball_sha256": f"sha256:{art['_sha256']}",
                        }
                    )
            else:
                target = tree / art["file"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(art["_blob"], target)
                installed.append(
                    {
                        "path": art["file"],
                        "bytes": art["_size"],
                        "from": art["file"],
                        "url": table[art["file"]]["url"],
                        "sha256": f"sha256:{art['_sha256']}",
                    }
                )

        vendor_manifest = build_vendor_manifest(cfg, wanted, table, installed)
        (tree / "vendor_manifest.json").write_text(
            json.dumps(vendor_manifest, ensure_ascii=False, indent=2) + "\n"
        )

        # ---- atomic-ish swap: only now does vendor/ change --------------------
        swap_in(tree, vendor)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        if vendor_created and vendor.exists() and not any(vendor.iterdir()):
            vendor.rmdir()
        raise
    finally:
        if not keep_downloads:
            shutil.rmtree(stage, ignore_errors=True)

    total = sum(f["bytes"] for f in installed)
    print("")
    print(f"installed {len(installed)} files into {vendor}/  ({total:,} bytes, {total / 1e6:.1f} MB)")
    for f in installed:
        print(f"  {f['bytes']:>12,}  {f['path']}")
    print(f"  wrote {vendor}/vendor_manifest.json")
    print(f"  {time.perf_counter() - t0:.1f}s")
    return 0


def build_vendor_manifest(cfg: dict, wanted: list[dict], table: dict, installed: list[dict]) -> dict:
    tj = cfg["transformers_js"]
    model = cfg["model"]
    return {
        "schema_version": 1,
        "generated_by": "batch_recsys_lab.demo.fetch_search_assets",
        "note": (
            "Local vendored assets for the semantic-search exhibit. Every byte here was "
            "verified against the SHA-256 recorded in demo/README.md before install. "
            "demo/js/search.js reads this file to state real download sizes in its warning "
            "and to locate the runtime."
        ),
        "transformers_js": {
            "package": "@huggingface/transformers",
            "version": tj["version"],
            "module": f"{tj['install_subdir']}/transformers.min.js",
            "wasm_paths": f"{tj['install_subdir']}/",
            "tarball_sha256": f"sha256:{table[tj['tarball']]['sha256']}",
        },
        "model": {
            "repo": model["repo"],
            "revision": model["revision"],
            "local_model_path": "models",
            "model_name": model["repo"],
            "dtype": model["dtype"],
            "pooling": "mean",
            "normalize": True,
        },
        "total_bytes": sum(f["bytes"] for f in installed),
        "files": installed,
    }


def swap_in(tree: Path, vendor: Path) -> None:
    """Move each top-level entry of ``tree`` into ``vendor``, replacing what is
    there, with restore-on-failure."""
    vendor.mkdir(parents=True, exist_ok=True)
    backups: list[tuple[Path, Path]] = []
    moved: list[Path] = []
    try:
        for entry in sorted(tree.iterdir()):
            target = vendor / entry.name
            if target.exists():
                backup = vendor / f".old-{entry.name}-{os.getpid()}"
                target.rename(backup)
                backups.append((backup, target))
            entry.rename(target)
            moved.append(target)
    except BaseException:
        for target in moved:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True) if target.is_dir() else target.unlink()
        for backup, target in backups:
            if backup.exists():
                backup.rename(target)
        raise
    for backup, _ in backups:
        if backup.is_dir():
            shutil.rmtree(backup, ignore_errors=True)
        elif backup.exists():
            backup.unlink()


# --- CLI -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="batch_recsys_lab.demo.fetch_search_assets")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument(
        "--record-hashes",
        action="store_true",
        help="bootstrap: download, hash, print the README table; install nothing",
    )
    ap.add_argument(
        "--allow-file-urls",
        action="store_true",
        help="permit file:// sources (test fixtures only)",
    )
    ap.add_argument("--keep-downloads", action="store_true", help="keep the staging dir (debug)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    for key in ("vendor_dir", "readme", "readme_section", "transformers_js", "model"):
        if key not in cfg:
            print(f"fetch_search_assets: config {args.config}: missing key {key!r}", file=sys.stderr)
            return 2

    try:
        if args.record_hashes:
            return record_hashes(cfg, allow_file=args.allow_file_urls)
        return verified_install(
            cfg, allow_file=args.allow_file_urls, keep_downloads=args.keep_downloads
        )
    except FetchError as exc:
        print(f"\nfetch_search_assets: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
