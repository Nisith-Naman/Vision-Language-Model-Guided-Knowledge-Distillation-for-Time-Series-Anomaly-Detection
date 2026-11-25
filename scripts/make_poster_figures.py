"""Poster figures: team-progress workflow and the combined teacher-student architecture.

    .venv\\Scripts\\python.exe scripts\\make_poster_figures.py
Writes final_ie643_report/figures/poster_workflow.png and poster_architecture.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "final_ie643_report" / "figures"
INK = "#1E293B"


def box(ax, x, y, w, h, text, fc, ec=INK, fs=8.5, weight="normal", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            weight=weight, wrap=True)


def arrow(ax, x0, y0, x1, y1, color=INK, style="-|>", lw=1.3, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.text(0.15, 5.1, "Primary assessment (v1)", fontsize=12, weight="bold", color="#475569")
    ax.text(1.4, 2.3, "Intensive assessment (v2): after the review", fontsize=12, weight="bold", color="#9A3412")
    top = [("Dataset collection\n5 multivariate sets\n(TA-approved)", "#E2E8F0"),
           ("Preprocessing\nsplits, train-only\nscaling, windows", "#E2E8F0"),
           ("Frozen CLIP teacher\nzero-shot, 2 prompts\nglobal-range heatmap", "#E2E8F0"),
           ("GRU student + KD\nstep BCE, window CE,\nT²·KL", "#E2E8F0"),
           ("Streamlit app\n7 pages", "#E2E8F0"),
           ("Review: 8 points\nteacher ≈ chance\n(Daphnet AUPRC 0.029)", "#FEE2E2")]
    bot = [("Label audit\nAir Quality labels\nsynthetic → excluded", "#FFEDD5"),
           ("Teacher repair\nclipped heatmap,\nlinear probe,\ncross-fitting", "#FFEDD5"),
           ("Novelty\nlearned prompts (CoOp)\n+ spectrogram view,\nconfidence-weighted KD", "#FED7AA"),
           ("Controlled benchmark\n5 teachers × 7 students\n+ 2 baselines, 3 seeds,\nidentical splits", "#FFEDD5"),
           ("Verification\nmetric recomputation,\nv1→v2 attribution,\nMetroPT audit", "#FFEDD5"),
           ("Out-of-sample test\nlabel-free series,\nhidden anomalies\n→ page 8 interface", "#FFEDD5")]
    w, h, gap = 2.0, 1.45, 0.3
    for row, items, y in ((0, top, 3.35), (1, bot, 0.55)):
        for i, (t, c) in enumerate(items):
            x = 0.2 + i * (w + gap)
            box(ax, x, y, w, h, t, c, fs=8.6, weight="bold" if "Novelty" in t else "normal",
                ec="#9A3412" if "Novelty" in t else INK)
            if i:
                arrow(ax, x - gap + 0.02, y + h / 2, x - 0.02, y + h / 2)
    xr = 0.2 + 5 * (w + gap) + w / 2
    ax.plot([xr, xr, 1.2], [3.35, 2.85, 2.85], color="#9A3412", lw=1.6)
    arrow(ax, 1.2, 2.85, 1.2, 2.02, color="#9A3412", lw=1.6)
    ax.text(7.0, 2.93, "review feedback drives the v2 work", fontsize=9.5, color="#9A3412", style="italic",
            ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "poster_workflow.png", dpi=200)
    plt.close(fig)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.4); ax.axis("off")
    frozen = "#DBEAFE"
    train = "#FDE68A"
    # Input
    box(ax, 0.1, 3.0, 1.6, 1.4, "Multivariate\nwindow\nL × F\n(train-scaled)", "#F1F5F9", fs=9, weight="bold")
    # Teacher branch (top)
    ax.text(2.0, 7.05, "TEACHER (training targets only) — CLIP ViT-B/32, all weights frozen", fontsize=10.5,
            weight="bold", color="#1D4ED8")
    box(ax, 2.1, 5.55, 1.7, 0.9, "Heatmap\n(clip ±4σ)", "#F8FAFC", fs=8.5)
    box(ax, 2.1, 4.45, 1.7, 0.9, "Spectrogram\n(STFT per channel)", "#F8FAFC", fs=8.5)
    box(ax, 4.2, 4.45, 1.9, 2.0, "CLIP image\nencoder\n(frozen)\n→ e₁, e₂ ∈ ℝ⁵¹²", frozen, fs=8.8, weight="bold")
    box(ax, 6.55, 5.95, 3.0, 0.95, "Learned prompt (CoOp)\n[v₁ … v₁₄] normal | anomalous\n7,168 trainable numbers",
        train, fs=8.3, weight="bold")
    box(ax, 6.55, 4.45, 3.0, 1.15, "CLIP text encoder (frozen)\np(c|x) = softmax(100·cos(e, t_c)),\naveraged over the 2 views",
        frozen, fs=8.3)
    box(ax, 10.0, 4.6, 1.9, 1.6, "Teacher outputs\nsoft label p_T\n(cross-fitted)\nembedding e", "#E0E7FF", fs=8.8,
        weight="bold")
    arrow(ax, 1.7, 3.95, 2.1, 6.0); arrow(ax, 1.7, 3.9, 2.1, 4.9)
    arrow(ax, 3.8, 6.0, 4.2, 5.8); arrow(ax, 3.8, 4.9, 4.2, 5.0)
    arrow(ax, 8.05, 5.95, 8.05, 5.6)
    arrow(ax, 6.1, 5.45, 6.55, 5.1)
    arrow(ax, 9.55, 5.1, 10.0, 5.3)
    # Student branch (bottom)
    ax.text(2.0, 2.55, "STUDENT (deployed) — 95k parameters, ~1/1500 of CLIP", fontsize=10.5, weight="bold",
            color="#9A3412")
    box(ax, 2.1, 0.55, 1.7, 1.7, "Conv1d F→64\n+ strided\nConv1d 64→64", train, fs=8.5)
    box(ax, 4.2, 0.55, 1.9, 1.7, "Bidirectional\nGRU (64)\nh₁ … h_L", train, fs=8.8, weight="bold")
    box(ax, 6.55, 1.75, 1.75, 0.6, "Window head  z_w", train, fs=8.3)
    box(ax, 6.55, 1.0, 1.75, 0.6, "Step head  z_s (L)", train, fs=8.3)
    box(ax, 6.55, 0.25, 1.75, 0.6, "Embed head g (train only)", train, fs=8.0, ls="--")
    arrow(ax, 1.7, 3.3, 2.1, 1.6); arrow(ax, 3.8, 1.4, 4.2, 1.4)
    for yy in (2.05, 1.3, 0.55):
        arrow(ax, 6.1, 1.4, 6.55, yy)
    # Losses
    box(ax, 8.75, 0.15, 3.15, 2.3,
        "Loss\nCE(z_w, y_win)  ← ground truth\n+ λ_KD · w·T²·KL(p_T ‖ z_w)  ← teacher\n+ BCE_w(z_s, y_step)  ← ground truth\n"
        "+ λ_feat · (1 − cos(g, e))  ← teacher", "#FEF3C7", fs=8.2)
    arrow(ax, 8.3, 2.05, 8.75, 1.9); arrow(ax, 8.3, 1.3, 8.75, 1.3); arrow(ax, 8.3, 0.55, 8.75, 0.7)
    arrow(ax, 10.95, 4.6, 10.6, 2.45, color="#1D4ED8", ls="--", lw=1.5)
    ax.text(11.0, 3.45, "distillation\n(training only)", fontsize=8.5, color="#1D4ED8", style="italic")
    # Output
    box(ax, 12.25, 1.2, 1.65, 3.0, "Localisation\nstep × p_wᵅ\n→ smooth\n→ threshold\n→ min length\n(tuned on\nvalidation)\n\nAnomaly\ntimeline",
        "#DCFCE7", fs=8.5, weight="bold")
    ax.plot([7.42, 7.42, 11.9], [2.35, 2.75, 2.75], color=INK, lw=1.3)
    arrow(ax, 11.9, 2.75, 12.25, 2.75)
    ax.text(9.05, 2.82, "inference: window + step scores", fontsize=8.3, color=INK, ha="center")
    ax.text(0.1, 0.05, "Blue = frozen CLIP; yellow = trained; dashed = used only during training.", fontsize=8.5,
            color="#475569")
    fig.tight_layout()
    fig.savefig(OUT / "poster_architecture.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    workflow()
    architecture()
    print(OUT / "poster_workflow.png")
    print(OUT / "poster_architecture.png")
