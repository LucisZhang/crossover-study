"""Semantic-search payload + vendored-asset fetcher (Phase 6, T35).

Pure python in tmp: no Spark, no JVM, no network, no sentence-transformers, and
nothing under the real ``data/``, ``demo/vendor/`` or ``results/`` is read or
written. The download path is exercised against ``file://`` fixtures, which is
the whole point of ``--allow-file-urls``.

What these pin:

* int8 round-trip stays inside the error bound the quantizer's own formula
  implies (half a scale step), and the degenerate all-zero row survives;
* ranking is *totally* ordered — ties broken by ascending row — so the same
  query cannot rank differently on two machines, and the JS scanner has an
  exact contract to match;
* the README hash is ground truth: a tampered byte fails loudly, exits
  non-zero, and installs nothing (verified against a pre-existing install that
  must come out untouched);
* a truncated transfer is a hard failure rather than a recorded hash. That one
  is not hypothetical — the first real bootstrap of this script recorded the
  hash of a 9,491,392-byte prefix of the 10,482,401-byte transformers tarball.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from batch_recsys_lab.demo import export_search as ex
from batch_recsys_lab.demo import fetch_search_assets as fetch

# ---------------------------------------------------------------- int8 round trip


def _unit_rows(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dim)).astype(np.float32)
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


def test_int8_round_trip_within_half_a_scale_step():
    """|deq - v| <= scale/2 for every component, by construction of round()."""
    v = _unit_rows(64, 384, seed=20260805)
    codes, scales = ex.quantize_rows(v)
    deq = ex.dequantize_rows(codes, scales)

    assert codes.dtype == np.int8 and scales.dtype == np.float32
    bound = scales[:, None] / 2.0
    # 1e-6 relative slack: the division itself rounds in float32.
    assert np.all(np.abs(deq - v) <= bound * (1 + 1e-6) + 1e-9)

    # -128 is never emitted: the codebook stays symmetric.
    assert codes.min() >= -127 and codes.max() <= 127

    # The bound is not vacuous -- some component actually approaches it.
    assert (np.abs(deq - v) > 0.4 * scales[:, None]).any()


def test_int8_round_trip_preserves_direction():
    """Cosine is what the exhibit shows, so bound the angle, not just the norm."""
    v = _unit_rows(256, 384, seed=7)
    deq = ex.dequantize_rows(*ex.quantize_rows(v))
    cos = (deq * v).sum(axis=1) / np.linalg.norm(deq, axis=1)
    assert cos.min() > 0.9999
    # Matches the order of magnitude recorded in embeddings_meta.json for the
    # real 50k payload (min cosine 0.999921).
    assert np.abs(deq - v).max() < 2e-3


def test_int8_zero_row_is_exact():
    v = np.zeros((2, 8), dtype=np.float32)
    v[1, 3] = 0.5
    codes, scales = ex.quantize_rows(v)
    assert scales[0] == 0.0
    assert not codes[0].any()
    assert np.array_equal(ex.dequantize_rows(codes, scales)[0], np.zeros(8, dtype=np.float32))


def test_quantize_rejects_non_matrix():
    with pytest.raises(ValueError, match="2-D"):
        ex.quantize_rows(np.zeros(8, dtype=np.float32))


# ------------------------------------------------------------ ordering determinism


def test_top_k_breaks_ties_by_ascending_row():
    scores = np.array([0.5, 0.9, 0.9, 0.1, 0.9], dtype=np.float64)
    assert list(ex.top_k_rows(scores, 3)) == [1, 2, 4]
    assert list(ex.top_k_rows(scores, 10)) == [1, 2, 4, 0, 3]


def test_top_k_is_permutation_invariant_on_an_all_tie_input():
    """Every score identical: the ranking must still be 0,1,2,... every time."""
    scores = np.full(50, 0.25)
    for _ in range(5):
        assert list(ex.top_k_rows(scores, 10)) == list(range(10))


def test_top_k_matches_a_naive_stable_reference():
    rng = np.random.default_rng(11)
    # Coarse rounding forces many ties, which is where argsort kinds diverge.
    scores = np.round(rng.random(500), 2)
    got = list(ex.top_k_rows(scores, 25))
    want = [i for i, _ in sorted(enumerate(scores), key=lambda p: (-p[1], p[0]))][:25]
    assert got == want


def test_overlap_at_k():
    assert ex.overlap_at_k(["a", "b", "c"], ["c", "b", "z"], 3) == 2
    assert ex.overlap_at_k(["a", "b"], ["x", "y"], 2) == 0


def test_verify_pop_ordering_accepts_the_lexsort_the_slice_job_produces():
    pop = np.array([9.0, 7.0, 7.0, 7.0, 1.0])
    idx = np.array([3, 4, 10, 11, 0])
    ex.verify_pop_ordering(pop, idx)  # must not raise


def test_verify_pop_ordering_catches_a_popularity_inversion():
    with pytest.raises(ValueError, match="not descending"):
        ex.verify_pop_ordering(np.array([1.0, 5.0]), np.array([0, 1]))


def test_verify_pop_ordering_catches_a_flipped_tie_break():
    """The failure mode that matters: right items, right popularity, wrong order."""
    with pytest.raises(ValueError, match="tie at popularity"):
        ex.verify_pop_ordering(np.array([7.0, 7.0]), np.array([11, 4]))


def test_verify_pop_ordering_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        ex.verify_pop_ordering(np.array([1.0, np.nan]), np.array([0, 1]))


# ------------------------------------------------------------------ README table


def test_readme_table_round_trips():
    rows = [
        {"file": "a.tgz", "sha256": "a" * 64, "size": "1.0 MB", "url": "https://example.test/a.tgz"},
        {"file": "m/b.onnx", "sha256": "b" * 64, "size": "2 kB", "url": "https://example.test/b.onnx"},
    ]
    text = f"## Search assets\n\n{fetch.render_readme_table(rows)}\n\n## Next section\n"
    parsed = fetch.parse_readme_table(text, "Search assets")
    assert set(parsed) == {"a.tgz", "m/b.onnx"}
    assert parsed["m/b.onnx"]["sha256"] == "b" * 64
    assert parsed["m/b.onnx"]["url"] == "https://example.test/b.onnx"


def test_readme_placeholder_table_parses_empty_not_garbage():
    text = "## Search assets\n\n| File | SHA-256 | Approx size | Source URL |\n|---|---|---|---|\n| _(T35 fills this)_ | | | |\n"
    assert fetch.parse_readme_table(text, "Search assets") == {}


def test_readme_table_rejects_a_non_digest():
    text = "## Search assets\n\n| File | SHA-256 | Approx size | Source URL |\n|---|---|---|---|\n| `a.tgz` | `deadbeef` | 1 MB | `https://x.test/a` |\n"
    with pytest.raises(fetch.FetchError, match="not a sha256"):
        fetch.parse_readme_table(text, "Search assets")


def test_readme_missing_section_is_an_error():
    with pytest.raises(fetch.FetchError, match="no 'Search assets' section"):
        fetch.parse_readme_table("# demo\n\nnothing here\n", "Search assets")


def test_the_repo_readme_table_is_parseable_and_names_exactly_what_the_config_wants():
    """The live loop: demo/README.md is the manifest configs/search_export.yaml
    is verified against. If these two drift apart, `make demo-assets` is broken
    for everyone, so pin it here rather than discovering it at deploy time."""
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs/search_export.yaml").read_text())
    table = fetch.parse_readme_table((repo / cfg["readme"]).read_text(), cfg["readme_section"])
    wanted = [a["file"] for a in fetch.required_artifacts(cfg)]
    assert sorted(table) == sorted(wanted)
    for name in wanted:
        assert table[name]["url"].startswith("https://")


# ------------------------------------------------- download / verify / no-partial


def _tarball(path: Path) -> dict[str, str]:
    """A stand-in transformers tarball with the members the config extracts."""
    members = {
        "package/dist/transformers.min.js": b"export const env = {};\n",
        "package/dist/transformers.min.js.map": b"{}\n",
        "package/dist/ort-wasm-simd-threaded.jsep.mjs": b"export default 1;\n",
        "package/dist/ort-wasm-simd-threaded.jsep.wasm": b"\0asm\x01\0\0\0",
        "package/LICENSE": b"Apache-2.0\n",
    }
    with tarfile.open(path, "w:gz") as tf:
        for name, blob in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))
    return {name.split("/")[-1]: name for name in members}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def install_fixture(tmp_path):
    """A complete file:// install: source blobs, a README with real hashes, a config."""
    src = tmp_path / "src"
    (src / "onnx").mkdir(parents=True)
    tgz = src / "transformers-9.9.9.tgz"
    _tarball(tgz)
    model_files = {
        "config.json": b'{"model_type":"bert"}',
        "tokenizer.json": b'{"version":"1.0"}',
        "tokenizer_config.json": b"{}",
        "onnx/model_quantized.onnx": b"ONNX\x00fake weights",
    }
    for rel, blob in model_files.items():
        (src / rel).write_bytes(blob)

    vendor = tmp_path / "vendor"
    readme = tmp_path / "README.md"
    cfg = {
        "vendor_dir": str(vendor),
        "readme": str(readme),
        "readme_section": "Search assets",
        "allowed_url_hosts": [],
        "transformers_js": {
            "version": "9.9.9",
            "tarball": "transformers-9.9.9.tgz",
            "url": tgz.as_uri(),
            "install_subdir": "transformers",
            "extract": {
                "package/dist/transformers.min.js": "transformers.min.js",
                "package/dist/ort-wasm-simd-threaded.jsep.wasm": "ort-wasm-simd-threaded.jsep.wasm",
                "package/LICENSE": "LICENSE",
            },
        },
        "model": {
            "repo": "Xenova/all-MiniLM-L6-v2",
            "revision": "0" * 40,
            "base_url": src.as_uri(),
            "install_subdir": "models/Xenova/all-MiniLM-L6-v2",
            "dtype": "q8",
            "files": list(model_files),
        },
    }
    rows = [
        {
            "file": "transformers-9.9.9.tgz",
            "sha256": _sha(tgz),
            "size": "1 kB",
            "url": tgz.as_uri(),
        }
    ] + [
        {
            "file": f"models/Xenova/all-MiniLM-L6-v2/{rel}",
            "sha256": _sha(src / rel),
            "size": "1 kB",
            "url": (src / rel).as_uri(),
        }
        for rel in model_files
    ]
    readme.write_text(f"## Search assets\n\n{fetch.render_readme_table(rows)}\n")
    cfg_path = tmp_path / "search_export.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return {"cfg": cfg, "cfg_path": cfg_path, "vendor": vendor, "readme": readme, "src": src}


def _tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def test_verified_install_writes_the_tree_and_a_manifest(install_fixture):
    f = install_fixture
    assert fetch.verified_install(f["cfg"], allow_file=True) == 0

    vendor = f["vendor"]
    assert (vendor / "transformers/transformers.min.js").exists()
    assert (vendor / "transformers/ort-wasm-simd-threaded.jsep.wasm").exists()
    assert (vendor / "models/Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx").exists()
    assert not (vendor / ".staging").exists()

    vm = json.loads((vendor / "vendor_manifest.json").read_text())
    assert vm["model"]["dtype"] == "q8"
    assert vm["transformers_js"]["module"] == "transformers/transformers.min.js"
    # search.js states its size warning from this number, so it must be real.
    assert vm["total_bytes"] == sum(
        p.stat().st_size for p in vendor.rglob("*") if p.is_file() and p.name != "vendor_manifest.json"
    )


def test_tampered_source_fails_loudly_and_installs_nothing(install_fixture, capsys):
    """The negative test the owner asked for: URL content drifts away from the
    README hash. Expected: the message, exit 1, and a byte-identical vendor/."""
    f = install_fixture
    assert fetch.verified_install(f["cfg"], allow_file=True) == 0
    before = _tree(f["vendor"])
    assert before

    onnx = f["src"] / "onnx/model_quantized.onnx"
    onnx.write_bytes(onnx.read_bytes() + b"drift")

    rc = fetch.main(["--config", str(f["cfg_path"]), "--allow-file-urls"])
    assert rc == 1
    err = capsys.readouterr().err
    assert fetch.MISMATCH_MESSAGE in err
    assert "do not edit the README hash" in err
    assert _tree(f["vendor"]) == before  # not one byte moved
    assert not (f["vendor"] / ".staging").exists()


def test_tampered_source_leaves_no_vendor_dir_when_there_was_none(install_fixture):
    """No install existed, the fetch fails: we must not leave a half-tree that
    search.js would read as 'the model is available'."""
    f = install_fixture
    tgz = f["src"] / "transformers-9.9.9.tgz"
    tgz.write_bytes(tgz.read_bytes() + b"drift")

    with pytest.raises(fetch.FetchError, match=fetch.MISMATCH_MESSAGE):
        fetch.verified_install(f["cfg"], allow_file=True)
    assert not f["vendor"].exists()


def test_readme_row_missing_for_a_required_file_is_an_error(install_fixture):
    f = install_fixture
    text = f["readme"].read_text()
    kept = [ln for ln in text.splitlines() if "tokenizer.json" not in ln]
    f["readme"].write_text("\n".join(kept) + "\n")
    with pytest.raises(fetch.FetchError, match="has no row for"):
        fetch.verified_install(f["cfg"], allow_file=True)
    assert not f["vendor"].exists()


def test_readme_naming_an_unwanted_file_is_an_error(install_fixture):
    """The table is a closed set: nothing gets fetched that the install does not
    need, even if someone adds a plausible-looking row."""
    f = install_fixture
    extra = {"file": "evil.bin", "sha256": "c" * 64, "size": "1 kB", "url": "https://x.test/evil.bin"}
    f["readme"].write_text(f["readme"].read_text().rstrip("\n") + "\n" + fetch.render_readme_table([extra]).splitlines()[-1] + "\n")
    with pytest.raises(fetch.FetchError, match="table names files this install does not want"):
        fetch.verified_install(f["cfg"], allow_file=True)


def test_empty_readme_table_tells_you_how_to_bootstrap(install_fixture):
    f = install_fixture
    f["readme"].write_text("## Search assets\n\n| File | SHA-256 | Approx size | Source URL |\n|---|---|---|---|\n")
    with pytest.raises(fetch.FetchError, match="--record-hashes"):
        fetch.verified_install(f["cfg"], allow_file=True)


def test_file_urls_are_refused_without_the_opt_in(install_fixture):
    with pytest.raises(fetch.FetchError, match="refusing file:// URL"):
        fetch.verified_install(install_fixture["cfg"], allow_file=False)


def test_only_the_pinned_hosts_may_be_fetched():
    fetch.check_url_allowed("https://huggingface.co/x", ["huggingface.co"], allow_file=False)
    with pytest.raises(fetch.FetchError, match="not in allowed_url_hosts"):
        fetch.check_url_allowed("https://evil.test/x", ["huggingface.co"], allow_file=False)
    with pytest.raises(fetch.FetchError, match="refusing non-https"):
        fetch.check_url_allowed("http://huggingface.co/x", ["huggingface.co"], allow_file=False)


def test_truncated_download_is_a_hard_failure(tmp_path, monkeypatch):
    """A short read must never become a recorded hash (see module docstring)."""

    class _Resp:
        headers = {"Content-Length": "1000"}

        def __init__(self):
            self._body = io.BytesIO(b"x" * 400)

        def read(self, n=-1):
            return self._body.read(n)

        def geturl(self):
            return "https://huggingface.co/short"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: _Resp())
    dest = tmp_path / "blob"
    with pytest.raises(fetch.FetchError, match="truncated download"):
        fetch.download("https://huggingface.co/short", dest)
    assert not dest.exists()


def test_tarball_member_escaping_the_destination_is_refused(tmp_path):
    tgz = tmp_path / "evil.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        info = tarfile.TarInfo("package/x")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"hi"))
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(fetch.FetchError, match="outside the destination"):
        fetch.extract_members(tgz, {"package/x": "../escaped"}, dest)


def test_missing_tarball_member_is_named(tmp_path):
    tgz = tmp_path / "t.tgz"
    _tarball(tgz)
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(fetch.FetchError, match="missing expected members"):
        fetch.extract_members(tgz, {"package/dist/not-there.js": "x.js"}, dest)


# ------------------------------------------------------------------------- config


def test_export_config_requires_its_anchors(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"runs_log": "x", "queries": ["a"]}))
    with pytest.raises(SystemExit, match="missing required keys"):
        ex.load_config(p)


def test_export_config_rejects_duplicate_queries(tmp_path):
    p = tmp_path / "c.yaml"
    cfg = {k: "x" for k in ex._REQUIRED}
    cfg["queries"] = ["a", "a"]
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="duplicate canned queries"):
        ex.load_config(p)


def test_the_repo_export_config_is_valid():
    repo = Path(__file__).resolve().parents[1]
    cfg = ex.load_config(repo / "configs/search_export.yaml")
    assert cfg["expected_rows"] == 50000
    assert len(cfg["queries"]) == 12
