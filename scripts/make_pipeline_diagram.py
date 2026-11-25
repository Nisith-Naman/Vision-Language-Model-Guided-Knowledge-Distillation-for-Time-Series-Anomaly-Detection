"""Static figure of the complete data / training / evaluation pipeline.

    .venv\\Scripts\\python.exe scripts\\make_pipeline_diagram.py
Writes final_ie643_report/figures/pipeline_diagram.png (and .pdf).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_ie643_report" / "figures"
KIND = {"data": ("#E0F2FE", "#0284C7"), "teacher": ("#FEF3C7", "#D97706"), "student": ("#FEE2E2", "#B91C1C"),
        "loss": ("#EDE9FE", "#6D28D9"), "eval": ("#DCFCE7", "#15803D")}
INK = "#0F172A"


def box(ax, x, y, w, h, title, body, kind):
    fill, edge = KIND[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.8",
                                facecolor=fill, edgecolor=edge, linewidth=1.4))
    ax.text(x, y + h / 2 - 1.6, title, ha="center", va="top", fontsize=10.5, weight="bold", color=INK)
    ax.text(x, y + h / 2 - 4.3, body, ha="center", va="top", fontsize=8.6, color=INK, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, a, b, label="", side="top"):
    (xa, ya, wa, ha), (xb, yb, wb, hb) = a, b
    if side == "top":       # from bottom of a to top of b
        p, q = (xa, ya - ha / 2), (xb, yb + hb / 2)
    elif side == "right":
        p, q = (xa + wa / 2, ya), (xb - wb / 2, yb)
    else:
        p, q = (xa - wa / 2, ya), (xb + wb / 2, yb)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color="#475569", linewidth=1.1,
                                 shrinkA=1, shrinkB=1))
    if label and side == "top":
        ax.text((p[0] + q[0]) / 2 + 0.8, (p[1] + q[1]) / 2, label, fontsize=8, color="#475569", va="center")
    elif label:
        ax.text((p[0] + q[0]) / 2, max(p[1], q[1]) + 1.2, label, fontsize=8, color="#475569", ha="center")


def section(ax, y, text, x=1, ha="left", rule=True):
    ax.text(x, y, text, fontsize=12.5, weight="bold", color="#334155", va="center", ha=ha)
    if rule:
        ax.plot([1, 99], [y - 1.6, y - 1.6], color="#CBD5E1", linewidth=0.8)


def main():
    fig, ax = plt.subplots(figsize=(13, 19.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.5, 150)
    ax.axis("off")
    W, H = 46, 9.5

    section(ax, 147, "A · Data preparation (identical for every model)")
    a = [
        ("1. Raw files", "CSV / text rows: timestamp, F sensor columns, label column", 138),
        ("2. Matrix per sequence", "X ∈ R^(T×F) (rows = time, columns = sensors),  y ∈ {0,1}^T", 127),
        ("3. Split whole sequences", "by time, subject, instance or failure → train / validation / test", 116),
        ("4. Fill missing values", "linear interpolation in time; edges: nearest value; empty column: train mean", 105),
        ("5. Standardise (fit on train only)", "z = (x − μ_f) / σ_f  for every feature f", 94),
        ("6. Sliding windows", "length L, stride S (+ one window aligned to the end)\n"
                               "X_w ∈ R^(L×F),  y_step ∈ {0,1}^L,  y_win = max_t y_step", 83),
    ]
    prev = None
    boxes_a = []
    for title, body, y in a:
        b = box(ax, 30, y, 52, H, title, body, "data")
        if prev:
            arrow(ax, prev, b)
        prev = b
        boxes_a.append(b)
    rows = [("dataset", "L", "stride", "F", "test prev."), ("Credit Card", "64", "32", "29", "0.074"),
            ("Daphnet FoG", "256", "64", "9", "0.048"), ("3W", "120", "30", "8", "0.620"),
            ("MetroPT-3", "120", "30", "15", "0.062")]
    table = "\n".join(f"{a:<12}{b:>5}{c:>7}{d:>4}{e:>11}" for a, b, c, d, e in rows)
    ax.text(62, 120, table, fontsize=9, family="monospace", va="top", color=INK,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#CBD5E1"))
    ax.text(62, 98, "Checks shown in the app:\n• hand z-score = model input\n• window = rows s … s+L−1\n"
                    "• no timestep in two splits", fontsize=9, va="top", color=INK)

    section(ax, 82, "B · Training: frozen teacher → student", x=99, ha="right", rule=False)
    tens = boxes_a[-1]
    t1 = box(ax, 24, 66, W, H, "Teacher 1 · render (labels never drawn)",
             "heatmap: clip z to [−4, 4] → viridis, F×L → 224×224\nspectrogram: |STFT| per feature → 224×224", "teacher")
    t2 = box(ax, 24, 55, W, H, "Teacher 2 · frozen CLIP ViT-B/32",
             "image → unit embedding e ∈ R^512 (one per view)\nall 151 M weights frozen", "teacher")
    t3 = box(ax, 24, 44, W, H, "Teacher 3 · learned prompts",
             "[SOS] v1 … v14 normal | anomalous . [EOS]\n14 × 512 = 7,168 trained numbers", "teacher")
    t4 = box(ax, 24, 33, W, H, "Teacher 4 · soft label",
             "p_T = softmax(mean_views 100 · cos(e, t_c))\ntrain windows: out-of-fold (5 folds, blocks of 32)", "teacher")
    s1 = box(ax, 76, 66, W, H, "Student 1 · convolutional front-end",
             "Conv1d(F→64, k5) → BN → GELU →\nConv1d(64→64, stride d) → BN → GELU", "student")
    s2 = box(ax, 76, 55, W, H, "Student 2 · bidirectional GRU",
             "64 units per direction → h_t ∈ R^128\npooled: [mean_t h_t, max_t h_t] ∈ R^256", "student")
    s3 = box(ax, 76, 44, W, H, "Student 3 · three heads",
             "window: 256→64→2, softmax → p_win\nstep: 128→32→1 per step, sigmoid → s_t\n"
             "embedding (training only): 256→128→512 → g", "student")
    arrow(ax, tens, t1)
    arrow(ax, tens, s1)
    ax.text(14.5, 74.6, "teacher branch", fontsize=8.5, color="#475569")
    ax.text(55, 75.6, "student branch", fontsize=8.5, color="#475569")
    for p_, q_ in ((t1, t2), (t2, t3), (t3, t4), (s1, s2), (s2, s3)):
        arrow(ax, p_, q_)
    loss = box(ax, 76, 31, W, 13, "Loss (per batch)",
               "BCE_w(s, y_step) + β·CE(p_win, y_win)\n+ λ_KD · T²·mean(w_i · KL(p_T^(T) ‖ p_win^(T)))\n"
               "+ λ_feat · (1 − cos(g, e))\nβ = 1, λ_KD = 1, λ_feat = 0.5, T = 2, w_i = p_T(true class)", "loss")
    arrow(ax, s3, loss)
    arrow(ax, t4, loss, "p_T and e", side="right")
    opt = box(ax, 76, 20.5, W, 7, "Optimise", "AdamW, ≤ 20 epochs; keep epoch with best validation AUPRC", "loss")
    arrow(ax, loss, opt)

    section(ax, 14.5, "C · Inference and evaluation (thresholds from validation only)")
    c = [("Timeline score", "average overlapping windows;\n× p_win^α; moving average of k", 17),
         ("Decision", "score ≥ τ (F1-optimal on validation;\n< 5 val events: 0.1 % false alarms);\n"
                      "runs shorter than m removed", 50),
         ("Metrics", "window: AUPRC, ROC-AUC, P/R/F1\npoint: AUPRC, P/R/F1\nevent: overlap P/R/F1", 83)]
    prev = None
    for title, body, x in c:
        b = box(ax, x, 5.0, 31, 11, title, body, "eval")
        if prev:
            arrow(ax, prev, b, side="right")
        prev = b
    fig.savefig(OUT / "pipeline_diagram.png", dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "pipeline_diagram.pdf", bbox_inches="tight", facecolor="white")
    print(OUT / "pipeline_diagram.png")


if __name__ == "__main__":
    main()
