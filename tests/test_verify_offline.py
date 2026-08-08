"""Tests for the offline/static audit scanner (Phase 6, T36).

Each test builds a small fixture tree under ``tmp_path`` shaped like
``demo/`` and calls :func:`scan_tree` directly (no subprocess).
"""

from __future__ import annotations

import json

from batch_recsys_lab.demo.verify_offline import scan_tree


def _write(root, rel, content):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_clean_page_passes(tmp_path):
    _write(
        tmp_path,
        "index.html",
        '<html><head><link rel="stylesheet" href="css/site.css"></head>'
        '<body><script type="module" src="js/app.js"></script></body></html>',
    )
    _write(tmp_path, "css/site.css", "body { color: #000; }")
    _write(tmp_path, "js/app.js", "console.log('hi');")

    report = scan_tree(tmp_path)

    assert report.ok
    assert report.violations == []


def test_cdn_script_src_fails(tmp_path):
    _write(
        tmp_path,
        "index.html",
        '<html><body><script src="https://cdn.example.com/lib.js"></script></body></html>',
    )

    report = scan_tree(tmp_path)

    assert not report.ok
    assert any("cdn.example.com" in h.url for h in report.violations)


def test_css_url_external_fails(tmp_path):
    _write(tmp_path, "css/site.css", "@font-face { src: url(https://fonts.example.com/a.woff2); }")

    report = scan_tree(tmp_path)

    assert not report.ok
    assert any(h.kind == "css url()" for h in report.violations)


def test_js_fetch_literal_fails(tmp_path):
    _write(tmp_path, "js/app.js", "fetch('https://api.example.com/data.json');")

    report = scan_tree(tmp_path)

    assert not report.ok
    assert any(h.kind.startswith("fetch") for h in report.violations)


def test_js_relative_fetch_passes(tmp_path):
    _write(tmp_path, "js/data.js", "fetch('data/crossover.json');")

    report = scan_tree(tmp_path)

    assert report.ok


def test_citation_anchor_allowed(tmp_path):
    _write(
        tmp_path,
        "index.html",
        '<html><body><a href="https://amazon-reviews-2023.github.io/">dataset</a></body></html>',
    )

    report = scan_tree(tmp_path)

    assert report.ok
    assert len(report.citation_anchors) == 1
    assert report.citation_anchors[0].url == "https://amazon-reviews-2023.github.io/"


def test_data_json_url_violates_unless_whitelisted(tmp_path):
    _write(
        tmp_path,
        "data/search/items_meta.json",
        json.dumps({"title": ["see https://www.amazon.com/dp/B000X for details"]}),
    )
    _write(
        tmp_path,
        "data/search/embeddings_meta.json",
        json.dumps({"source": {"tarball_url": "https://registry.npmjs.org/pkg.tgz"}, "rows": 1}),
    )

    report = scan_tree(tmp_path)

    assert not report.ok
    assert any("amazon.com" in h.url for h in report.violations)
    assert any("npmjs.org" in h.url for h in report.whitelisted)
    assert not any("npmjs.org" in h.url for h in report.violations)


def test_vendor_exemption_reported_not_failed(tmp_path):
    _write(
        tmp_path,
        "vendor/transformers/lib.js",
        "// see https://huggingface.co/model for details\nconst x = 1;",
    )

    report = scan_tree(tmp_path)

    assert report.ok
    assert report.vendor_present is True
    assert report.vendor_url_literal_count == 1
    assert "never fetched at runtime" in report.vendor_justification


def test_allow_remote_models_true_in_nonvendor_js_fails(tmp_path):
    _write(tmp_path, "js/search.js", "env.allowRemoteModels = true;")

    report = scan_tree(tmp_path)

    assert not report.ok
    assert any(h.kind == "capability-misuse" for h in report.violations)


def test_remote_wasm_paths_literal_in_nonvendor_js_fails(tmp_path):
    _write(tmp_path, "js/search.js", "env.backends.onnx.wasm.wasmPaths = 'https://cdn.example.com/wasm/';")

    report = scan_tree(tmp_path)

    assert not report.ok
    assert any(h.kind == "capability-misuse" for h in report.violations)


def test_missing_vendor_dir_is_fine(tmp_path):
    _write(tmp_path, "js/app.js", "console.log('hi');")

    report = scan_tree(tmp_path)

    assert report.ok
    assert report.vendor_present is False
