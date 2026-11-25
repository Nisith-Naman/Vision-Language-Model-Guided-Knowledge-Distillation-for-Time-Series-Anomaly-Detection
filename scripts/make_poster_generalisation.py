"""Poster figure: does distillation still help against a regularized baseline? (MetroPT-3)

Left: test point AUPRC (mean over seeds) of the labels-only GRU and the distilled
student under base and regularized training, read from each run's metrics.csv.
Right: train-minus-test window AUPRC gap of the deployed/backup checkpoints, read
from experiments/benchmark_v2/GENERALISATION.md (written by compare_generalisation.py).

    .venv\\Scripts\\python.exe scripts\\make_poster_generalisation.py
Writes final_ie643_report/figures/poster_generalisation.png.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"Base training": ROOT / "experiments" / "benchmark_v2",
        "Regularized training": ROOT / "experiments" / "benchmark_v2_regularized"}
MODELS = {"gru_gt": ("GRU, labels only", "#6366F1"), "gru_kd_coop": ("GRU + CLIP KD (ours)", "#B91C1C")}
INK, MUTED, GRID, SURFACE = "#1F2937", "#6B7280", "#E5E7EB", "#FCFCFB"
DS = "metropt"


def point_auprc(run: Path, model: str) -> tuple[float, int]:
    m = pd.read_csv(run / DS / "metrics.csv", dtype={"seed": str})
    v = m[(m.model == model) & (m.level == "point")]
    return float(v.auprc.astype(float).mean()), len(v)


def gaps() -> dict[str, tuple[float, float]]:
    text = (ROOT / "experiments" / "benchmark_v2" / "GENERALISATION.md").read_text(encoding="utf-8")
    out = {}
    for key, (label, _) in MODELS.items():
        short = "GRU, labels only" if key == "gru_gt" else "GRU + KD (proposed)"
        row = next(l for l in text.splitlines() if l.startswith(f"| MetroPT-3 | {short} |"))
        g = row.split("|")[7]
        a, b = (float(x) for x in re.findall(r"[+-]?\d\.\d+", g))
        out[key] = (a, b)
    return out


def bars(ax, groups, values, labels_fmt="{:.3f}"):
    x = np.arange(len(groups))
    w = 0.36
    for j, (key, (label, color)) in enumerate(MODELS.items()):
        vals = [values[g][key] for g in groups]
        pos = x + (j - 0.5) * (w + 0.02)
        ax.bar(pos, vals, w, color=color, label=label, edgecolor=SURFACE, linewidth=2, zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.015, labels_fmt.format(v), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x, groups, fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", colors=MUTED, length=0)
    ax.set_facecolor(SURFACE)


def main():
    groups = list(RUNS)
    ap = {g: {k: point_auprc(RUNS[g], k)[0] for k in MODELS} for g in groups}
    gp = gaps()
    gap = {g: {k: gp[k][i] for k in MODELS} for i, g in enumerate(groups)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    bars(axes[0], groups, ap)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_title("Test point AUPRC (mean of 3 seeds)", fontsize=12, color=INK, loc="left")
    axes[0].axhline(0.063, color=MUTED, ls="--", lw=1, zorder=2, label="Random detector (prevalence 0.063)")
    for i, g in enumerate(groups):
        d = ap[g]["gru_kd_coop"] - ap[g]["gru_gt"]
        axes[0].text(i, 1.04, f"KD gain {d:+.3f} (3/3 seeds)", ha="center", fontsize=10, color=INK, weight="bold")

    bars(axes[1], groups, gap)
    axes[1].set_ylim(0, 0.5)
    axes[1].set_title("Overfitting: train − test window AUPRC (deployed seed)", fontsize=12, color=INK, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0.01, 0.92), labelcolor=INK)

    fig.suptitle("MetroPT-3 (one training failure): distillation vs. labels only, before and after regularization",
                 fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = ROOT / "final_ie643_report" / "figures" / "poster_generalisation.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)
    print({g: {k: round(v, 3) for k, v in d.items()} for g, d in ap.items()}, gap)


if __name__ == "__main__":
    main()
