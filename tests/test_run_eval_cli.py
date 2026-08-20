"""Tests for the ``run_eval`` CLI's arg-parsing (Phase 9, T9-3 ML-32M lane).

Only exercises argument parsing / pass-through to ``harness.run_eval`` (via
monkeypatch) — no Spark, no real config. Defaults for ``--splits-path`` /
``--manifest-path`` must match ``runlog.DEFAULT_SPLITS_PATH`` /
``runlog.DEFAULT_MANIFEST_PATH`` so existing Amazon-lane invocations (which
never pass these flags) are unaffected.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from batch_recsys_lab.eval import run_eval as run_eval_cli
from batch_recsys_lab.eval import runlog


def _write_min_config(path: Path) -> None:
    path.write_text(yaml.safe_dump({"protocol": {"eval_split": "val"}}))


def test_cli_defaults_splits_and_manifest_path(tmp_path, monkeypatch):
    config_path = tmp_path / "cfg.yaml"
    _write_min_config(config_path)

    captured = {}

    def fake_run_eval(config, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(run_eval_cli, "run_eval", fake_run_eval)

    rc = run_eval_cli.main(["--config", str(config_path)])
    assert rc == 0
    assert captured["splits_path"] == str(runlog.DEFAULT_SPLITS_PATH)
    assert captured["manifest_path"] == str(runlog.DEFAULT_MANIFEST_PATH)


def test_cli_accepts_explicit_splits_and_manifest_path(tmp_path, monkeypatch):
    config_path = tmp_path / "cfg.yaml"
    _write_min_config(config_path)
    splits_path = tmp_path / "splits_ml32m.yaml"
    manifest_path = tmp_path / "MANIFEST_ML32M.md"

    captured = {}

    def fake_run_eval(config, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(run_eval_cli, "run_eval", fake_run_eval)

    rc = run_eval_cli.main(
        [
            "--config",
            str(config_path),
            "--splits-path",
            str(splits_path),
            "--manifest-path",
            str(manifest_path),
        ]
    )
    assert rc == 0
    assert captured["splits_path"] == str(splits_path)
    assert captured["manifest_path"] == str(manifest_path)


def test_cli_uses_config_carried_paths_when_no_flags_given(tmp_path, monkeypatch):
    """A config may carry its own splits_path/manifest_path (e.g. an ML-32M
    config) so a run without CLI flags does not silently fall back to the
    Amazon defaults (mirrors churn_contrast's dataset_manifest_path pattern).
    """
    config_path = tmp_path / "cfg_ml32m.yaml"
    config_splits_path = tmp_path / "splits_ml32m.yaml"
    config_manifest_path = tmp_path / "MANIFEST_ML32M.md"
    config_path.write_text(
        yaml.safe_dump(
            {
                "protocol": {"eval_split": "val"},
                "splits_path": str(config_splits_path),
                "manifest_path": str(config_manifest_path),
            }
        )
    )

    captured = {}

    def fake_run_eval(config, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(run_eval_cli, "run_eval", fake_run_eval)

    rc = run_eval_cli.main(["--config", str(config_path)])
    assert rc == 0
    assert captured["splits_path"] == str(config_splits_path)
    assert captured["manifest_path"] == str(config_manifest_path)


def test_cli_flag_overrides_config_carried_paths(tmp_path, monkeypatch):
    """An explicit CLI flag still wins even when the config also carries a
    splits_path/manifest_path key."""
    config_path = tmp_path / "cfg_ml32m.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "protocol": {"eval_split": "val"},
                "splits_path": str(tmp_path / "config_splits.yaml"),
                "manifest_path": str(tmp_path / "config_manifest.md"),
            }
        )
    )
    cli_splits_path = tmp_path / "cli_splits.yaml"
    cli_manifest_path = tmp_path / "cli_manifest.md"

    captured = {}

    def fake_run_eval(config, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(run_eval_cli, "run_eval", fake_run_eval)

    rc = run_eval_cli.main(
        [
            "--config",
            str(config_path),
            "--splits-path",
            str(cli_splits_path),
            "--manifest-path",
            str(cli_manifest_path),
        ]
    )
    assert rc == 0
    assert captured["splits_path"] == str(cli_splits_path)
    assert captured["manifest_path"] == str(cli_manifest_path)
