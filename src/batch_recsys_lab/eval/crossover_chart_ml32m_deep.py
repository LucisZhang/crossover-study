"""Deep-bucket confirmatory chart, ML-32M TEST (Phase 9, T9-3c).

The 5-segment ``crossover_chart.py`` machinery plots per-arm ABSOLUTE NDCG@10
lines, which is right for the Amazon-comparability exhibit
(``configs/crossover_ml32m_test.yaml``). The T9-3c confirmatory evidence file
(``results/confirmatory_ml32m_test.json``, written by
``eval.confirmatory_ml32m``) does not carry that shape: Family P's rows are
PAIRED DELTAS (M* minus P*) per populated deep bucket plus one global test,
each already BH-corrected at FDR 0.05 within the family (§5b of the T9-3b
preregistration). Reusing the 5-segment renderer on delta rows would either
silently misrepresent them as absolute arm values or require bending that
module's contract; this sibling module renders the deltas directly, which is
the more honest confirmatory exhibit: one line, delta-vs-zero, with 95%
bootstrap CI bands, and BH-significant buckets marked distinctly from
uncorrected ones.

Rendered STRICTLY from ``results/confirmatory_ml32m_test.json`` -- no model
code, no caches, no Spark, no re-computation. A config
(``configs/crossover_ml32m_deep_test.yaml``) names the JSON path, the metric,
the output stem, and cosmetic fields.

    uv run python -m batch_recsys_lab.eval.crossover_chart_ml32m_deep \\
        --config configs/crossover_ml32m_deep_test.yaml

Outputs: results/figures/<output_stem>.svg and .png.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import yaml

from batch_recsys_lab.eval.crossover_chart import BASELINE, GRID, INK, INK2, MUTED, SURFACE
from batch_recsys_lab.eval.protocol import DEEP_BUCKET_LABELS

REQUIRED_KEYS = ("confirmatory_json", "metric", "output_stem")

#: BH-significant delta: filled diamond, full ink. Non-significant: hollow
#: circle, muted -- same color family, different weight, never color-coded
#: alone (dataviz skill: never rely on hue to carry a binary state).
POS_COLOR = "#1baf7a"
NEG_COLOR = "#e34948"
ZERO_LINE = "#898781"


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"crossover_ml32m_deep config {path}: missing required keys {missing}")
    return cfg


def load_family_p_rows(confirmatory_json: str | Path, metric: str) -> dict:
    """Load Family P's rows for ``metric`` from the T9-3c evidence JSON.

    Returns ``{"bucket_rows": [...], "global_row": dict|None, "verdict": dict,
    "excluded_buckets": [...], "run_id": str, "comparator_run_id": str,
    "delta_label": str}``. Hard errors on anything missing -- a chart drawn
    from a partial evidence file is worse than no chart.
    """
    doc = json.loads(Path(confirmatory_json).read_text())
    fam_p = doc.get("families", {}).get("P")
    if fam_p is None:
        raise ValueError(f"{confirmatory_json}: no families.P block")
    blk = fam_p.get("metrics", {}).get(metric)
    if blk is None:
        raise ValueError(f"{confirmatory_json}: families.P.metrics[{metric!r}] missing")
    rows = blk.get("rows") or []
    bucket_rows = [r for r in rows if r.get("unit") == "deep_bucket"]
    global_row = next((r for r in rows if r.get("unit") == "global"), None)
    if not bucket_rows:
        raise ValueError(f"{confirmatory_json}: families.P.metrics[{metric!r}] has no deep_bucket rows")
    order = {lbl: i for i, lbl in enumerate(DEEP_BUCKET_LABELS)}
    bucket_rows.sort(key=lambda r: order.get(r["label"], 999))
    for r in bucket_rows:
        for k in ("delta", "ci_lo", "ci_hi", "bh_significant"):
            if k not in r:
                raise ValueError(f"{confirmatory_json}: row {r.get('label')!r} missing {k!r}")
    return {
        "bucket_rows": bucket_rows,
        "global_row": global_row,
        "verdict": doc.get("verdict", {}),
        "excluded_buckets": blk.get("excluded_buckets", []),
        "bh": blk.get("bh", {}),
        "run_id": fam_p.get("run_id"),
        "comparator_run_id": fam_p.get("comparator_run_id"),
        "delta_label": fam_p.get("delta_label"),
        "git_sha": doc.get("git_sha", ""),
    }


def render(cfg: dict, data: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    rows = data["bucket_rows"]
    labels = [r["label"] for r in rows]
    xs = list(range(len(labels)))
    values = [r["delta"] for r in rows]
    ci_lo = [r["ci_lo"] for r in rows]
    ci_hi = [r["ci_hi"] for r in rows]
    sig = [bool(r["bh_significant"]) for r in rows]
    n_users = [r.get("n_users") for r in rows]

    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.10, right=0.965, top=0.82, bottom=0.24)

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)

    ax.axhline(0.0, color=ZERO_LINE, linewidth=1.0, zorder=2)

    line_color = INK2
    ax.fill_between(xs, ci_lo, ci_hi, color=line_color, alpha=0.13, linewidth=0, zorder=1)
    ax.plot(xs, values, color=line_color, linewidth=2.0, zorder=3)
    for x, v, s, d in zip(xs, values, sig, values):
        c = POS_COLOR if d >= 0 else NEG_COLOR
        if s:
            ax.plot(
                x, v, marker="D", markersize=9, markerfacecolor=c, markeredgecolor=INK,
                markeredgewidth=1.1, zorder=5,
            )
        else:
            ax.plot(
                x, v, marker="o", markersize=7, markerfacecolor=SURFACE, markeredgecolor=c,
                markeredgewidth=1.6, zorder=4,
            )

    ymax = max(ci_hi + [0.0])
    ymin = min(ci_lo + [0.0])
    pad = 0.15 * max(ymax - ymin, 1e-9)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(-0.4, len(labels) - 0.6)
    ax.set_xticks(xs, labels, fontsize=11)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.4f}")
    ax.tick_params(axis="y", labelsize=9)

    for x, n in zip(xs, n_users):
        if n is not None:
            ax.text(x, ax.get_ylim()[0], f"n={n:,}", ha="center", va="top", fontsize=8, color=MUTED, transform=ax.transData, clip_on=False)

    ax.set_xlabel(cfg.get("xlabel", "User history depth (TRAIN interactions, TEST users)"), fontsize=10, color=INK2, labelpad=18)
    ax.set_ylabel(f"delta {cfg['metric']}  ({data['delta_label']})", fontsize=10, color=INK2)

    n_star = cfg.get("n_star_annotation")
    if n_star is not None:
        for x, r in zip(xs, rows):
            edges = {"0": 0, "1-4": 1, "5-9": 5, "10-19": 10, "20-49": 20, "50-99": 50, "100+": 100}
            if edges.get(r["label"]) == n_star:
                ax.axvline(x - 0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
                n_star_text = cfg.get("n_star_annotation_label") or f"n*={n_star}"
                ax.text(
                    x - 0.5, ax.get_ylim()[1], f" {n_star_text}", fontsize=8, color=MUTED,
                    ha="left", va="bottom",
                )
                break

    legend_handles = [
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=INK2, markeredgecolor=INK, markersize=9, label="BH-significant (FDR 0.05)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=7, markeredgewidth=1.6, label="not BH-significant"),
    ]
    leg = ax.legend(
        handles=legend_handles, loc="lower left", frameon=True, facecolor=SURFACE,
        edgecolor=GRID, framealpha=0.95, fontsize=8.5, labelcolor=INK2, handlelength=1.6,
        borderaxespad=0.4,
    )
    leg.get_frame().set_linewidth(0.8)

    fig.text(0.10, 0.955, cfg.get("title", "Deep-bucket confirmatory chart"), fontsize=15, color=INK, fontweight="bold", ha="left")
    if cfg.get("subtitle"):
        fig.text(0.10, 0.905, cfg["subtitle"], fontsize=9.5, color=INK2, ha="left")
    if cfg.get("annotation_label"):
        fig.text(0.10, 0.87, cfg["annotation_label"], fontsize=9, color=INK2, fontweight="bold", ha="left")

    v = data.get("verdict") or {}
    if v.get("verdict"):
        vtext = f"verdict: {v['verdict']}"
        if v.get("d4_flag"):
            vtext += f"  |  {v.get('d4_token')}"
        ax.text(
            0.985, 0.03, vtext, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
            color=INK2, bbox=dict(boxstyle="round,pad=0.45", facecolor=SURFACE, edgecolor=GRID, linewidth=0.8),
            zorder=6,
        )

    receipts = (
        f"rendered from {cfg['confirmatory_json']} (derived, appends nothing to "
        f"results/runs.jsonl)  |  M*={data['run_id']}  P*={data['comparator_run_id']}"
        f"  |  record git SHA: {data['git_sha'][:7] if data['git_sha'] else ''}"
    )
    for i, row in enumerate(textwrap.wrap(receipts, width=150)):
        fig.text(0.10, 0.045 - 0.02 * i, row, fontsize=6.5, color=MUTED, ha="left")

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
    ap.add_argument("--config", required=True, help="configs/crossover_ml32m_deep_test.yaml")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data = load_family_p_rows(cfg["confirmatory_json"], cfg["metric"])

    paths = render(cfg, data, Path(args.out_dir))
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
