"""ML-32M manifest rendering/parsing + bronze reconciliation (Phase 9, T9-3a).

JVM-free: no Spark, no warehouse, no downloaded data. Two things are load-bearing
here and neither is cosmetic:

* **The ML-32M manifest is a SEPARATE committed file.** ``dataset_manifest_hash``
  hashes the whole manifest and ``eval/reproduce.py`` compares that field, so
  ML-32M content inside ``data/MANIFEST.md`` flips the pinned Amazon headline's
  ``byte_exact`` verdict. The tool must refuse to write there.
* **Bronze counts are reconciled EXACTLY against the manifest's row counts.** The
  first real ingest landed 87,584 of 87,585 movies (CSV escape defect) and nothing
  failed. This is the gate that makes that impossible to miss.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from batch_recsys_lab.ingest import download_ml32m, reconcile_ml32m

REPO_ROOT = Path(__file__).resolve().parents[1]

# Real observed facts from the box run (ml-32m.zip SHA-256 is ours, computed
# locally — GroupLens publishes no checksum).
ZIP_SHA = "e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee"
RATINGS_ROWS, RATINGS_BYTES = 32_000_204, 877_076_222
MOVIES_ROWS, MOVIES_BYTES = 87_585, 4_242_926


def _entries():
    """What ``manifest()`` assembles, with the real ratings/movies facts.

    tags.csv is NOT given real numbers: they are unknown until the box run fills
    them in, and inventing them here would bake a fiction into the test suite.
    """
    return [
        {
            "filename": "ml-32m.zip",
            "url": download_ml32m.ZIP_URL,
            "size": 239_000_000,
            "sha256": ZIP_SHA,
            "data_rows": None,
        },
        {
            "filename": "ratings.csv",
            "arcname": "ml-32m/ratings.csv",
            "size": RATINGS_BYTES,
            "sha256": "b" * 64,
            "data_rows": RATINGS_ROWS,
        },
        {
            "filename": "movies.csv",
            "arcname": "ml-32m/movies.csv",
            "size": MOVIES_BYTES,
            "sha256": "c" * 64,
            "data_rows": MOVIES_ROWS,
        },
        {
            "filename": "tags.csv",
            "arcname": "ml-32m/tags.csv",
            "size": 1_234_567,
            "sha256": "d" * 64,
            "data_rows": 999,
        },
    ]


def _document():
    entries = _entries()
    rows = {e["filename"]: e["data_rows"] for e in entries if e["data_rows"] is not None}
    return download_ml32m._document(entries, rows, "2026-08-20")


# --------------------------------------------------------------------------- #
# Render → parse round trip.
# --------------------------------------------------------------------------- #


def test_rendered_manifest_round_trips_through_the_parser():
    parsed = download_ml32m.parse_manifest_text(_document())
    assert set(parsed) == {"ml-32m.zip", "ratings.csv", "movies.csv", "tags.csv"}
    assert parsed["ml-32m.zip"]["sha256"] == ZIP_SHA
    assert parsed["ml-32m.zip"]["data_rows"] is None  # an archive has no rows
    assert parsed["ratings.csv"] == {
        "sha256": "b" * 64,
        "size": RATINGS_BYTES,
        "data_rows": RATINGS_ROWS,
    }
    assert parsed["movies.csv"]["data_rows"] == MOVIES_ROWS


def test_rendered_manifest_records_the_deliberately_unused_members():
    doc = _document()
    # links.csv is unextracted by DECISION; the manifest says so in writing rather
    # than leaving a reader to wonder whether it was forgotten.
    assert "## Not extracted" in doc
    assert "ml-32m/links.csv" in doc
    assert "IMDb/TMDb" in doc
    # And the separation rationale is in the document itself, not just in code.
    assert "data/MANIFEST.md" in doc


def test_parser_ignores_content_outside_the_files_section():
    doc = _document() + "\n## Notes\n\n### not-a-file.csv\n\n- Size (bytes): 5\n"
    assert "not-a-file.csv" not in download_ml32m.parse_manifest_text(doc)


def test_download_date_is_preserved_across_regeneration(tmp_path):
    # Re-hashing unchanged bytes must not move the file, and therefore must not
    # move dataset_manifest_hash under an already-recorded run.
    path = tmp_path / "MANIFEST_ML32M.md"
    path.write_text(_document())
    assert download_ml32m._existing_download_date(path) == "2026-08-20"
    assert download_ml32m._existing_download_date(tmp_path / "absent.md") is None


def test_tags_is_extracted_and_links_is_not():
    members = {m.filename: m for m in download_ml32m.MEMBERS}
    assert set(members) == {"ratings.csv", "movies.csv", "tags.csv"}
    assert members["tags.csv"].expected_header == ("userId", "movieId", "tag", "timestamp")
    # No invented published count for tag applications.
    assert members["tags.csv"].published_rows is None
    assert members["ratings.csv"].published_rows == RATINGS_ROWS
    assert members["movies.csv"].published_rows == MOVIES_ROWS
    assert "ml-32m/links.csv" in download_ml32m.UNUSED_MEMBERS


def test_manifest_target_is_the_ml32m_file_and_refuses_the_amazon_one(tmp_path, capsys):
    assert download_ml32m.MANIFEST_ML32M_PATH == REPO_ROOT / "data" / "MANIFEST_ML32M.md"
    # Guard, not convention: writing ML-32M content into data/MANIFEST.md breaks
    # `make reproduce-headline`. It exits non-zero before hashing anything.
    assert download_ml32m.manifest(manifest_path=tmp_path / "MANIFEST.md") == 1
    assert "refusing to write" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Bronze reconciliation.
# --------------------------------------------------------------------------- #


def test_expected_rows_reads_every_bronze_table_from_the_manifest(tmp_path):
    path = tmp_path / "MANIFEST_ML32M.md"
    path.write_text(_document())
    assert reconcile_ml32m._expected_rows(path) == {
        "ratings": RATINGS_ROWS,
        "movies": MOVIES_ROWS,
        "tags": 999,
    }


def test_missing_manifest_or_missing_row_count_fails_before_spark(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        reconcile_ml32m._expected_rows(tmp_path / "absent.md")

    path = tmp_path / "MANIFEST_ML32M.md"
    path.write_text(_document().replace(f"- Data rows (excl. header): {999}\n", ""))
    with pytest.raises(RuntimeError, match=r"\['tags.csv'\]"):
        reconcile_ml32m._expected_rows(path)


def test_compare_is_exact_equality_not_a_tolerance():
    expected = {"ratings": RATINGS_ROWS, "movies": MOVIES_ROWS, "tags": 999}

    _, failures = reconcile_ml32m.compare(expected, dict(expected))
    assert failures == []

    # The actual bug: one movie lost to the CSV escape defect. Off by one is a
    # failure, not a rounding difference.
    _, failures = reconcile_ml32m.compare(
        expected, {**expected, "movies": MOVIES_ROWS - 1}
    )
    assert len(failures) == 1
    assert "87584 rows != 87585" in failures[0]
    assert "delta -1" in failures[0]

    _, failures = reconcile_ml32m.compare(expected, {**expected, "tags": None})
    assert failures == ["local.bronze_ml32m.tags does not exist"]


def test_reconcile_table_map_covers_every_bronze_spec():
    from batch_recsys_lab.ingest.bronze_ml32m import TABLE_SPECS

    # A new bronze table with no manifest row count would be unverified bronze.
    assert set(reconcile_ml32m.TABLE_SOURCES) == set(TABLE_SPECS)


# --------------------------------------------------------------------------- #
# The lane boundary.
# --------------------------------------------------------------------------- #


def test_gitignore_unignores_both_manifests_and_nothing_else_changed():
    lines = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "data/*" in lines
    assert "!data/MANIFEST.md" in lines
    assert "!data/MANIFEST_ML32M.md" in lines
    # The exception must come after the data/* ignore, or git never sees it.
    assert lines.index("data/*") < lines.index("!data/MANIFEST_ML32M.md")
