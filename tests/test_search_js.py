"""demo/js/search.js under node (Phase 6, T35).

The exhibit's ranking code is JavaScript, so leaving it to a manual browser pass
would leave the two halves of one contract untested against each other. This
runs ``tests/js/search_checks.mjs``: mode detection (live / fallback / absent)
against a stubbed fetch, and the tie-break in ``topK``, which must agree with
``export_search.top_k_rows`` exactly.

When the uncommitted 50k payload happens to be on disk, a numpy reference
ranking is generated here and the harness must reproduce it row-for-row. That
is the cross-language check: same int8 bytes, same scales, same ordering.

Skipped, not failed, when node is unavailable — node is a convenience for
checking the browser half, not a dependency of the pipeline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "tests/js/search_checks.mjs"
PAYLOAD = REPO / "demo/data/search"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@node
def test_search_js_parses():
    """`node --check` on the module the browser will import."""
    proc = subprocess.run(
        ["node", "--check", str(REPO / "demo/js/search.js")], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


@node
def test_search_js_behaviour(tmp_path):
    env = None
    if (PAYLOAD / "embeddings_int8.bin").exists():
        env = {"SEARCH_REFERENCE_JSON": str(_write_numpy_reference(tmp_path))}

    proc = subprocess.run(
        ["node", str(HARNESS)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**_os_environ(), **(env or {})},
    )
    print(proc.stdout)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAIL" not in proc.stdout


def _os_environ() -> dict:
    import os

    return dict(os.environ)


def _write_numpy_reference(tmp_path: Path) -> Path:
    """Reference ranking for one deterministic query, computed with numpy.

    The query is payload row 0 dequantized and re-normalized — derived from the
    payload itself, so the fixture needs no model and no stored query vector,
    and row 0 is guaranteed to be in the answer (it is its own nearest
    neighbour), which makes a silently-empty result impossible to pass.
    """
    meta = json.loads((PAYLOAD / "embeddings_meta.json").read_text())
    rows, dim = meta["rows"], meta["dim"]
    codes = np.fromfile(PAYLOAD / "embeddings_int8.bin", dtype=np.int8).reshape(rows, dim)
    scales = np.fromfile(PAYLOAD / "scales_f32.bin", dtype="<f4")

    q = (codes[0].astype(np.float32) * scales[0]).astype(np.float32)
    q = (q / np.linalg.norm(q)).astype(np.float32)

    scores = ((codes.astype(np.float32) @ q) * scales).astype(np.float32)
    top = np.lexsort((np.arange(rows), -scores.astype(np.float64)))[:20]

    out = tmp_path / "reference.json"
    out.write_text(
        json.dumps(
            {
                "query_vector": [float(x) for x in q],
                "top_rows": [int(i) for i in top],
                "top_scores": [float(scores[i]) for i in top],
            }
        )
    )
    assert int(top[0]) == 0, "row 0 must be its own nearest neighbour"
    return out
