"""Crossover chart (Phase 4, T14): per-segment NDCG@10 lines with 95% CI bands.

Rendered STRICTLY from results/runs.jsonl -- no model code, no caches, no Spark.
A config (configs/crossover_val.yaml; a crossover_test.yaml twin in T15) names
the split, the metric, the output stem, and the lines to plot (label + run_id).
Each run_id must resolve to exactly the append-only log's eval record for the
configured split, and every configured segment must carry value/ci_lo/ci_hi for
the configured metric -- anything missing is a hard error, never a silent skip.

Outputs: results/figures/<output_stem>.svg and .png, with run_ids + git SHAs in
small print on the figure (receipts culture). Regenerable from the log alone:

    uv run python -m batch_recsys_lab.eval.crossover_chart --config configs/crossover_val.yaml

Design follows the dataviz skill: categorical slots in fixed order (validated
palette, worst adjacent CVD deltaE 24.2), y-axis from 0, solid hairline grid,
thin 2px lines, CI bands as low-alpha fills, legend + direct labels in ink
(relief rule for the sub-3:1 aqua/yellow slots), text never in series color.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import yaml

# --- dataviz reference palette (light mode), see skill references/palette.md ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Categorical slots in fixed order -- assignment follows config line order,
# never re-ranked or cycled. Validated: scripts/validate_palette.js PASS.
SLOTS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

REQUIRED_KEYS = ("runs_log", "split", "metric", "output_stem", "segments", "lines")


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"crossover config {path}: missing required keys {missing}")
    if len(cfg["lines"]) > len(SLOTS):
        raise ValueError(f"crossover config {path}: {len(cfg['lines'])} lines > {len(SLOTS)} palette slots")
    for i, line in enumerate(cfg["lines"]):
        for k in ("label", "run_id"):
            if k not in line:
                raise ValueError(f"crossover config {path}: lines[{i}] missing '{k}'")
    return cfg


def index_runs(runs_log: str | Path) -> dict[str, dict]:
    """run_id -> record. The log is append-only; run_ids are expected unique."""
    out: dict[str, dict] = {}
    with open(runs_log) as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            rid = rec.get("run_id")
            if rid is None:
                raise ValueError(f"{runs_log}:{lineno}: record without run_id")
            out[rid] = rec  # last occurrence wins (superseding entries carry new ids anyway)
    return out


def extract_series(rec: dict, run_id: str, split: str, metric: str, segments: list[str]) -> dict:
    """Pull per-segment metric value/ci_lo/ci_hi (+ n_users); error on anything absent."""
    if rec.get("kind") != "eval":
        raise ValueError(f"run {run_id}: kind={rec.get('kind')!r}, expected 'eval'")
    got_split = rec.get("protocol", {}).get("eval_split")
    if got_split != split:
        raise ValueError(f"run {run_id}: eval_split={got_split!r} but config wants {split!r}")
    per_seg = rec.get("metrics", {}).get("per_segment")
    if not per_seg:
        raise ValueError(f"run {run_id}: no metrics.per_segment block")
    values, ci_lo, ci_hi, n_users = [], [], [], []
    for seg in segments:
        if seg not in per_seg:
            raise ValueError(f"run {run_id}: segment {seg!r} missing (has {sorted(per_seg)})")
        m = per_seg[seg].get(metric)
        if not isinstance(m, dict) or any(k not in m for k in ("value", "ci_lo", "ci_hi")):
            raise ValueError(f"run {run_id}: segment {seg!r} lacks {metric!r} value/ci_lo/ci_hi")
        values.append(m["value"])
        ci_lo.append(m["ci_lo"])
        ci_hi.append(m["ci_hi"])
        n_users.append(per_seg[seg].get("n_users"))
    return {
        "values": values,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_users": n_users,
        "git_sha": rec.get("git_sha", ""),
    }


def _spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """De-overlap direct-label y positions: preserve order, enforce min_gap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)
    placed: list[float] = []
    out = [0.0] * len(ys)
    for i in order:
        y = ys[i]
        if placed and y > placed[-1] - min_gap:
            y = placed[-1] - min_gap
        placed.append(y)
        out[i] = y
    return out


def render(cfg: dict, series: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import transforms as mtransforms

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    segments: list[str] = cfg["segments"]
    xs = list(range(len(segments)))

    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.085, right=0.775, top=0.84, bottom=0.22)

    # Chrome: recessive solid hairline grid, baseline-weight spines only.
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)

    # Marks: CI bands under 2px lines; markers wear a 2px surface ring.
    for line, s in zip(cfg["lines"], series):
        hl = bool(line.get("highlight"))
        c = s["color"]
        ax.fill_between(xs, s["ci_lo"], s["ci_hi"], color=c, alpha=0.13, linewidth=0, zorder=1)
        ax.plot(
            xs,
            s["values"],
            color=c,
            linewidth=3.0 if hl else 2.0,
            marker="o",
            markersize=6.5 if hl else 5.0,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            zorder=4 if hl else 3,
            label=line["label"],
        )

    ymax = max(max(s["ci_hi"]) for s in series)
    ax.set_ylim(0, ymax * 1.22)  # y from 0, always
    ax.set_xlim(-0.3, len(segments) - 0.7)
    ax.set_xticks(xs, segments, fontsize=11)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.3f}")
    ax.tick_params(axis="y", labelsize=9)

    # Segment sizes as a muted sublabel row under the ticks (from the records).
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    n_users = series[0]["n_users"]
    for x, n in zip(xs, n_users):
        if n is not None:
            ax.text(x, -0.085, f"n={n:,}", transform=trans, ha="center", va="top", fontsize=8, color=MUTED)
    ax.text(
        (len(segments) - 1) / 2,
        -0.155,
        cfg.get("xlabel", "User history depth (TRAIN interactions, VAL users)"),
        transform=trans,
        ha="center",
        va="top",
        fontsize=10,
        color=INK2,
    )
    ax.set_ylabel(cfg["metric"].upper().replace("@", "@"), fontsize=10, color=INK2)

    # Direct labels at line ends, in ink (relief rule for sub-3:1 slots); the
    # colored line endpoint beside the text carries identity, not the text.
    yr = ax.get_ylim()[1]
    end_ys = _spread_labels([s["values"][-1] for s in series], min_gap=0.048 * yr)
    for line, s, y in zip(cfg["lines"], series, end_ys):
        ax.annotate(
            line["label"],
            xy=(xs[-1], s["values"][-1]),
            xytext=(xs[-1] + 0.12, y),
            va="center",
            ha="left",
            fontsize=9,
            color=INK if line.get("highlight") else INK2,
            fontweight="bold" if line.get("highlight") else "normal",
            annotation_clip=False,
        )

    # Legend always present for >=2 series; ink text, surface frame so it stays
    # legible where lines pass beneath it.
    leg = ax.legend(
        loc="center left",
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
        framealpha=0.95,
        fontsize=8.5,
        labelcolor=INK2,
        handlelength=1.6,
        borderaxespad=0.4,
    )
    leg.get_frame().set_linewidth(0.8)
    leg.set_zorder(5)

    # Title block, figure-level, left-aligned with the axes.
    fig.text(0.085, 0.955, cfg.get("title", "Crossover chart"), fontsize=15, color=INK, fontweight="bold", ha="left")
    if cfg.get("subtitle"):
        fig.text(0.085, 0.905, cfg["subtitle"], fontsize=9.5, color=INK2, ha="left")

    # Routing-outcome annotation (honest, unobtrusive): hairline box, top right.
    if cfg.get("annotation"):
        ax.text(
            0.985,
            0.975,
            cfg["annotation"],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            color=INK2,
            linespacing=1.45,
            multialignment="left",
            bbox=dict(boxstyle="round,pad=0.55", facecolor=SURFACE, edgecolor=GRID, linewidth=0.8),
            zorder=6,
        )

    # Receipts: run_ids + git SHAs in small print (every number traces to the log).
    shas = sorted({s["git_sha"][:7] for s in series if s["git_sha"]})
    receipts = "  |  ".join(f"{line['label']}: {line['run_id']}" for line, s in zip(cfg["lines"], series))
    wrapped = textwrap.wrap(receipts, width=150)
    rows = ["rendered from results/runs.jsonl (append-only)  |  record git SHAs: " + ", ".join(shas)] + wrapped
    for i, row in enumerate(rows):
        fig.text(0.085, 0.062 - 0.02 * i, row, fontsize=6.5, color=MUTED, ha="left")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("svg", "png"):
        p = out_dir / f"{cfg['output_stem']}.{ext}"
        fig.savefig(p, dpi=200, facecolor=SURFACE)
        paths.append(p)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="configs/crossover_*.yaml")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    runs = index_runs(cfg["runs_log"])
    series = []
    for i, line in enumerate(cfg["lines"]):
        rid = line["run_id"]
        if rid not in runs:
            raise SystemExit(f"ERROR: run_id {rid!r} (line {line['label']!r}) not found in {cfg['runs_log']}")
        try:
            s = extract_series(runs[rid], rid, cfg["split"], cfg["metric"], cfg["segments"])
        except ValueError as e:
            raise SystemExit(f"ERROR: {e}") from e
        s["color"] = SLOTS[i]
        series.append(s)

    paths = render(cfg, series, Path(args.out_dir))
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
