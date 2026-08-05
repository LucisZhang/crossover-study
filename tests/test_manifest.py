import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.md"

SHA256_RE = re.compile(r"\b[0-9a-f]{64}\b")


def test_manifest_contents():
    if not MANIFEST_PATH.exists():
        pytest.skip("manifest not built in this environment")

    text = MANIFEST_PATH.read_text()

    sha256_matches = SHA256_RE.findall(text)
    assert len(sha256_matches) >= 2, "expected at least two 64-hex SHA-256 strings"

    assert "Electronics.jsonl.gz" in text
    assert "meta_Electronics.jsonl.gz" in text

    assert "2403.03952" in text

    assert "## Bronze reconciliation" in text
