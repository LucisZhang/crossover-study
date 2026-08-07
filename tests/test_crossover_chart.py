"""Config parsing + series extraction for the T14 crossover chart (no rendering)."""

import json

import pytest

from batch_recsys_lab.eval.crossover_chart import extract_series, index_runs, load_config

SEGS = ["0", "1-4"]


def _record(run_id, split="val", kind="eval", metric="ndcg@10"):
    return {
        "kind": kind,
        "run_id": run_id,
        "git_sha": "deadbeef00",
        "protocol": {"eval_split": split},
        "metrics": {
            "per_segment": {
                seg: {"n_users": 10 * (i + 1), metric: {"value": 0.01 * (i + 1), "ci_lo": 0.009, "ci_hi": 0.011}}
                for i, seg in enumerate(SEGS)
            }
        },
    }


@pytest.fixture()
def runs_log(tmp_path):
    p = tmp_path / "runs.jsonl"
    with open(p, "w") as fh:
        for rec in (_record("r1"), _record("r2", split="test")):
            fh.write(json.dumps(rec) + "\n")
    return p


def test_index_and_extract(runs_log):
    runs = index_runs(runs_log)
    assert set(runs) == {"r1", "r2"}
    s = extract_series(runs["r1"], "r1", "val", "ndcg@10", SEGS)
    assert s["values"] == [0.01, 0.02]
    assert s["n_users"] == [10, 20]
    assert s["ci_lo"] == [0.009, 0.009] and s["ci_hi"] == [0.011, 0.011]


def test_extract_errors(runs_log):
    runs = index_runs(runs_log)
    with pytest.raises(ValueError, match="eval_split"):
        extract_series(runs["r2"], "r2", "val", "ndcg@10", SEGS)
    with pytest.raises(ValueError, match="segment '20\\+' missing"):
        extract_series(runs["r1"], "r1", "val", "ndcg@10", SEGS + ["20+"])
    with pytest.raises(ValueError, match="lacks 'recall@10'"):
        extract_series(runs["r1"], "r1", "val", "recall@10", SEGS)


def test_load_config_validates(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("split: val\nmetric: ndcg@10\n")
    with pytest.raises(ValueError, match="missing required keys"):
        load_config(p)
    p.write_text(
        "runs_log: r.jsonl\nsplit: val\nmetric: ndcg@10\noutput_stem: x\n"
        'segments: ["0"]\nlines:\n  - label: a\n'
    )
    with pytest.raises(ValueError, match="missing 'run_id'"):
        load_config(p)
