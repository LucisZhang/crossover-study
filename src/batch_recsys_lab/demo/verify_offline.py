"""Offline/static audit scanner for the demo site (Phase 6, T36).

    make demo-offline-check              # human-readable report
    uv run python -m batch_recsys_lab.demo.verify_offline --json

What it checks (see demo/README.md "What offline means here" and
docs/engineering-log/UPGRADE_PLAN.md Phase 6 acceptance item 1):

1. Every file under ``demo/`` EXCEPT ``demo/vendor/`` is scanned for external
   URLs (``https?://`` or protocol-relative ``//host``, not localhost /
   127.0.0.1) sitting in an *executable* position: ``<script src>``,
   ``<link href>``, ``<img src>``, ``<source src>``, ``srcset``, CSS
   ``url(...)``/``@import``, and the JS literal string arguments of
   ``fetch(``, ``import(``, ``new Worker(``, ``XMLHttpRequest#open(``,
   ``new URL(``, and ``navigator.sendBeacon(``. Any hit is a VIOLATION.
2. Plain ``<a href>`` citation anchors in HTML, and any URL appearing in
   Markdown prose, load nothing on page load — they are reported as
   "citation anchors (allowed)", not flagged.
3. ``demo/data/*.json`` (including ``demo/data/search/*.json``) must contain
   zero URL strings anywhere, except an explicit, code-documented JSON
   pointer-prefix whitelist (see ``DATA_URL_WHITELIST`` below) covering
   ``embeddings_meta.json``'s provenance block / the README-documented
   source column. Whitelisted hits are reported, not flagged; anything else
   is a VIOLATION.
4. ``demo/vendor/`` (present locally, never committed) gets a documented
   EXEMPTION: if present, it is scanned too, but URL literals inside it are
   only ever REPORTED, never failed on — the authoritative offline proof for
   vendor code is the DNS-black-holed runtime run (docs/engineering-log/EXPERIMENT_LOG.md Phase 6
   T36), not static scanning of a third-party bundle. What DOES fail is any
   NON-vendor file (in practice, ``demo/js/*.js``) that mis-wires the vendor
   capability: a literal ``allowRemoteModels = true``, or a remote
   (``https?://``) literal assigned to ``wasmPaths``/``localModelPath``.

Exit code 0 on a clean scan, 2 on any VIOLATION. ``--json`` prints the full
machine-readable report instead of the human-readable text summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------

# Protocol-relative matches require a dotted, domain-shaped host
# (``//example.com/...``) so plain double-slashes inside prose (product
# titles like "7/Plus" adjacent to a line break, JS comments, etc.) are not
# mistaken for URLs.
_ABS_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_PROTO_REL_RE = re.compile(r"^//(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}")
_HOST_RE = re.compile(r"^(?:https?:)?//([^/?#]+)", re.IGNORECASE)
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _host_of(url: str) -> str:
    m = _HOST_RE.match(url.strip())
    if not m:
        return ""
    host = m.group(1)
    host = host.split("@")[-1]  # strip userinfo
    host = host.split(":")[0]  # strip port
    return host.lower()


def is_external_url(candidate: str) -> bool:
    """True if ``candidate`` is an absolute or protocol-relative URL whose
    host is not a loopback address."""
    candidate = candidate.strip()
    if not (_ABS_URL_RE.match(candidate) or _PROTO_REL_RE.match(candidate)):
        return False
    return _host_of(candidate) not in _LOOPBACK_HOSTS


def _is_url_like(candidate: str) -> bool:
    """True for anything that looks like an absolute/protocol-relative URL,
    loopback or not (used for citation-anchor reporting and vendor counts)."""
    candidate = candidate.strip()
    return bool(_ABS_URL_RE.match(candidate) or _PROTO_REL_RE.match(candidate))


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    file: str
    line: int
    kind: str
    url: str
    detail: str = ""


@dataclass
class Report:
    violations: list[Hit] = field(default_factory=list)
    citation_anchors: list[Hit] = field(default_factory=list)
    whitelisted: list[Hit] = field(default_factory=list)
    vendor_present: bool = False
    vendor_url_literal_count: int = 0
    vendor_files: dict[str, int] = field(default_factory=dict)
    vendor_justification: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def ok(self) -> bool:
        return not self.violations


VENDOR_JUSTIFICATION = (
    "vendor library contains {n} URL literals; never fetched at runtime "
    "(allowRemoteModels=false, local wasmPaths); authoritative proof is the "
    "DNS-black-holed runtime run (see docs/engineering-log/EXPERIMENT_LOG.md Phase 6 T36)."
)

# Binary/opaque assets: not meaningfully scannable as text, and not a source
# of executable-position URL literals in this repo's vendor tree.
_BINARY_EXTS = {".wasm", ".onnx"}

# ---------------------------------------------------------------------------
# Explicit whitelist for demo/data/*.json string-leaf URLs.
#
# Only ``embeddings_meta.json``'s provenance block (the "source" object,
# which mirrors the README's Search-assets "Source URL" column: the model /
# recipe / snapshot identifiers that pin what produced the payload) is
# allowed to carry a URL string. Anything else in demo/data/ is a violation.
# Keyed by path relative to demo/data/, value is a tuple of allowed JSON
# pointer prefixes.
# ---------------------------------------------------------------------------

DATA_URL_WHITELIST: dict[str, tuple[str, ...]] = {
    "search/embeddings_meta.json": ("/source",),
}

_JSON_URL_RE = re.compile(r"https?://\S+|(?<!:)//(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\S*")


def _walk_json(obj, pointer: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_json(v, f"{pointer}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_json(v, f"{pointer}/{i}")
    elif isinstance(obj, str):
        yield pointer, obj


def scan_json_data(path: Path, rel_to_data: str, rel_to_demo: str, report: Report) -> None:
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    allowed_prefixes = DATA_URL_WHITELIST.get(rel_to_data, ())
    for pointer, value in _walk_json(obj, ""):
        for m in _JSON_URL_RE.finditer(value):
            url = m.group(0)
            if not _is_url_like(url):
                continue
            hit = Hit(file=rel_to_demo, line=0, kind="data-json-string", url=url, detail=pointer)
            if any(pointer == p or pointer.startswith(p + "/") or pointer.startswith(p) for p in allowed_prefixes):
                report.whitelisted.append(hit)
            else:
                report.violations.append(hit)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_TAG_ATTR_RE = re.compile(
    r"<(script|link|img|source)\b[^>]*?\b(src|href)\s*=\s*([\"'])(.*?)\3",
    re.IGNORECASE | re.DOTALL,
)
_SRCSET_RE = re.compile(r"\bsrcset\s*=\s*([\"'])(.*?)\1", re.IGNORECASE | re.DOTALL)
_A_HREF_RE = re.compile(r"<a\b[^>]*\bhref\s*=\s*([\"'])(.*?)\1", re.IGNORECASE | re.DOTALL)


def scan_html(path: Path, rel: str, report: Report) -> None:
    text = path.read_text(errors="ignore")
    for m in _TAG_ATTR_RE.finditer(text):
        tag, _attr, _q, url = m.groups()
        if is_external_url(url):
            report.violations.append(
                Hit(file=rel, line=_line_of(text, m.start()), kind=f"<{tag.lower()}>", url=url)
            )
    for m in _SRCSET_RE.finditer(text):
        _q, value = m.groups()
        for part in value.split(","):
            candidate = part.strip().split()[0] if part.strip() else ""
            if candidate and is_external_url(candidate):
                report.violations.append(
                    Hit(file=rel, line=_line_of(text, m.start()), kind="srcset", url=candidate)
                )
    for m in _A_HREF_RE.finditer(text):
        _q, url = m.groups()
        if _is_url_like(url):
            report.citation_anchors.append(
                Hit(file=rel, line=_line_of(text, m.start()), kind="<a href>", url=url)
            )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_MD_URL_RE = re.compile(r"https?://[^\s\)\"'`]+")


def scan_markdown(path: Path, rel: str, report: Report) -> None:
    text = path.read_text(errors="ignore")
    for m in _MD_URL_RE.finditer(text):
        report.citation_anchors.append(
            Hit(file=rel, line=_line_of(text, m.start()), kind="markdown prose/citation", url=m.group(0))
        )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS_URL_RE = re.compile(r"url\(\s*([\"']?)([^)\"']+)\1\s*\)", re.IGNORECASE)
_CSS_IMPORT_RE = re.compile(r"@import\s+(?:url\()?[\"']?([^\"');\s]+)", re.IGNORECASE)


def scan_css(path: Path, rel: str, report: Report) -> None:
    text = path.read_text(errors="ignore")
    for m in _CSS_URL_RE.finditer(text):
        url = m.group(2)
        if is_external_url(url):
            report.violations.append(Hit(file=rel, line=_line_of(text, m.start()), kind="css url()", url=url))
    for m in _CSS_IMPORT_RE.finditer(text):
        url = m.group(1)
        if is_external_url(url):
            report.violations.append(Hit(file=rel, line=_line_of(text, m.start()), kind="@import", url=url))


# ---------------------------------------------------------------------------
# JS
# ---------------------------------------------------------------------------

_JS_CALL_RE = re.compile(
    r"\b(fetch|import|new\s+Worker|new\s+URL|sendBeacon)\s*\(\s*([\"'])(.*?)\2",
    re.DOTALL,
)
_JS_XHR_OPEN_RE = re.compile(
    r"\.open\s*\(\s*[\"'][A-Za-z]+[\"']\s*,\s*([\"'])(.*?)\1",
    re.DOTALL,
)
_JS_ALLOW_REMOTE_RE = re.compile(r"\ballowRemoteModels\s*=\s*true\b")
_JS_REMOTE_CAPABILITY_RE = re.compile(
    r"\b(wasmPaths|localModelPath)\s*=\s*([\"'])(https?:)?//",
)


def scan_js(path: Path, rel: str, report: Report) -> None:
    text = path.read_text(errors="ignore")
    for m in _JS_CALL_RE.finditer(text):
        callee, _q, url = m.groups()
        if is_external_url(url):
            report.violations.append(
                Hit(file=rel, line=_line_of(text, m.start()), kind=f"{callee.strip()}(...)", url=url)
            )
    for m in _JS_XHR_OPEN_RE.finditer(text):
        _q, url = m.groups()
        if is_external_url(url):
            report.violations.append(
                Hit(file=rel, line=_line_of(text, m.start()), kind="XMLHttpRequest#open(...)", url=url)
            )
    for m in _JS_ALLOW_REMOTE_RE.finditer(text):
        report.violations.append(
            Hit(
                file=rel,
                line=_line_of(text, m.start()),
                kind="capability-misuse",
                url="",
                detail="allowRemoteModels = true (must stay false outside demo/vendor/)",
            )
        )
    for m in _JS_REMOTE_CAPABILITY_RE.finditer(text):
        key = m.group(1)
        report.violations.append(
            Hit(
                file=rel,
                line=_line_of(text, m.start()),
                kind="capability-misuse",
                url="",
                detail=f"{key} assigned a remote (https?://) literal",
            )
        )


# ---------------------------------------------------------------------------
# vendor/ (documented exemption)
# ---------------------------------------------------------------------------

_VENDOR_URL_RE = re.compile(r"https?://[^\s\"'\)]+")


def scan_vendor(vendor_dir: Path, demo_dir: Path, report: Report) -> None:
    report.vendor_present = True
    total = 0
    for path in sorted(vendor_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in _BINARY_EXTS:
            continue
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        n = len(_VENDOR_URL_RE.findall(text))
        if n:
            rel = str(path.relative_to(demo_dir))
            report.vendor_files[rel] = n
            total += n
    report.vendor_url_literal_count = total
    report.vendor_justification = VENDOR_JUSTIFICATION.format(n=total)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_SKIP_NAMES = {".DS_Store"}


def scan_tree(demo_dir: Path) -> Report:
    demo_dir = demo_dir.resolve()
    report = Report()
    vendor_dir = demo_dir / "vendor"

    for path in sorted(demo_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in _SKIP_NAMES:
            continue
        rel = path.relative_to(demo_dir)
        if rel.parts and rel.parts[0] == "vendor":
            continue  # handled separately below
        rel_str = str(rel)
        ext = path.suffix.lower()
        if ext == ".html":
            scan_html(path, rel_str, report)
        elif ext == ".css":
            scan_css(path, rel_str, report)
        elif ext == ".js":
            scan_js(path, rel_str, report)
        elif ext == ".json" and rel.parts and rel.parts[0] == "data":
            rel_to_data = str(Path(*rel.parts[1:]))
            scan_json_data(path, rel_to_data, rel_str, report)
        elif ext == ".md":
            scan_markdown(path, rel_str, report)

    if vendor_dir.is_dir():
        scan_vendor(vendor_dir, demo_dir, report)
    else:
        report.vendor_present = False

    return report


def _print_human(report: Report) -> None:
    print("== demo/ offline audit ==")
    if report.violations:
        print(f"\nVIOLATIONS ({len(report.violations)}):")
        for h in report.violations:
            loc = f"{h.file}:{h.line}" if h.line else h.file
            extra = f" -- {h.detail}" if h.detail else ""
            url = f" {h.url}" if h.url else ""
            print(f"  [{h.kind}] {loc}{url}{extra}")
    else:
        print("\nVIOLATIONS: none")

    if report.citation_anchors:
        print(f"\ncitation anchors (allowed) ({len(report.citation_anchors)}):")
        for h in report.citation_anchors:
            print(f"  [{h.kind}] {h.file}:{h.line} {h.url}")

    if report.whitelisted:
        print(f"\nwhitelisted data-file source URLs (allowed) ({len(report.whitelisted)}):")
        for h in report.whitelisted:
            print(f"  {h.file} {h.detail} {h.url}")

    print("\nvendor/:")
    if report.vendor_present:
        print(f"  present, url literal count = {report.vendor_url_literal_count}")
        for f, n in report.vendor_files.items():
            print(f"    {f}: {n}")
        print(f"  {report.vendor_justification}")
    else:
        print("  absent (not fetched/built in this environment) -- skipped, not a failure")

    print(f"\nresult: {'CLEAN' if report.ok else 'VIOLATIONS FOUND'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=Path("demo"))
    parser.add_argument("--json", action="store_true", help="print the full machine-readable report")
    args = parser.parse_args(argv)

    report = scan_tree(args.demo_dir)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_human(report)

    return 0 if report.ok else 2


if __name__ == "__main__":
    sys.exit(main())
