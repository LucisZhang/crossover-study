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
