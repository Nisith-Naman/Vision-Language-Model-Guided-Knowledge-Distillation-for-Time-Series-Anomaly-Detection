"""Poster pipeline diagram: one card per processing step, connected by arrows,
each card showing what the data looks like at that step (table, matrix,
heatmap, embedding vector, logits, probabilities, scores).

Layout (4 rows x 5 columns):
  row 1  data preparation   1 CSV -> 2 split -> 3 fill gaps -> 4 standardise -> 5 windows
  row 2  teacher            6 pictures -> 7 frozen CLIP -> 8 teacher opinion --\
                                                                                12 training loss
  row 3  student            9 conv -> 10 GRU -> 11 heads ----------------------/
  row 4  test time                            13 score per step -> 14 decide -> 15 metrics

The pictures use a small synthetic example (4 sensors, windows of 24 steps)
so every shape is visible; card footers give the general shapes.

    .venv\\Scripts\\python.exe scripts\\make_poster_dataflow.py
Writes final_ie643_report/figures/poster_dataflow.png (and .pdf).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_ie643_report" / "figures"
INK, MUTED, LINE = "#0F172A", "#475569", "#334155"
KIND = {"data": ("#EFF8FF", "#0284C7", "#BAE6FD"), "teacher": ("#FFFBEB", "#D97706", "#FDE68A"),
        "student": ("#FEF2F2", "#B91C1C", "#FECACA"), "loss": ("#F5F3FF", "#6D28D9", "#DDD6FE"),
        "eval": ("#F0FDF4", "#15803D", "#BBF7D0")}
FAULT = "#F87171"

# ------------------------------------------------------------------ synthetic example
rng = np.random.default_rng(3)
T = 160
t = np.arange(T)
RAW = np.stack([
    20 + 2 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.4, T),
    100 + 5 * np.sin(2 * np.pi * t / 32 + 1) + rng.normal(0, 1.0, T),
    50 + 3 * np.sign(np.sin(2 * np.pi * t / 40)) + rng.normal(0, 0.5, T),
    0.5 + rng.normal(0, 0.08, T)], axis=1)
Y = np.zeros(T, int)
for a_, b_ in ((58, 64), (138, 148)):
    Y[a_:b_] = 1
    RAW[a_:b_, 0] += np.linspace(4, 10, b_ - a_)
    RAW[a_:b_, 1] += 14
    RAW[a_:b_, 3] += 0.6 * np.sign(np.sin(np.arange(b_ - a_) * 2.1))
RAW[40:45, 1] = np.nan
TR, VA = 96, 128                     # train rows 0-95, validation 96-127, test 128-159
FILLED = RAW.copy()
_nan = np.isnan(FILLED[:, 1])
FILLED[_nan, 1] = np.interp(np.flatnonzero(_nan), np.flatnonzero(~_nan), FILLED[~_nan, 1])
MU, SD = FILLED[:TR].mean(0), FILLED[:TR].std(0)
Z = (FILLED - MU) / SD
L, S = 24, 12
W0 = 130                             # the window followed through both models (test part)
WIN = Z[W0:W0 + L].T                 # F x L
KR = np.random.default_rng(7)


# ------------------------------------------------------------------ drawing helpers
def card(ax, x0, y0, w, h, kind, title, op, shape):
    fill, edge, head = KIND[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill,
                                edgecolor=edge, linewidth=2.0, zorder=1))
    ax.add_patch(FancyBboxPatch((x0 + 0.3, y0 + h - 7.6), w - 0.6, 7.3, boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=head, edgecolor="none", zorder=1))
    ax.text(x0 + w / 2, y0 + h - 0.9, title, ha="center", va="top", fontsize=14.5, weight="bold", color=INK,
            zorder=3)
    ax.text(x0 + w / 2, y0 + h - 4.6, op, ha="center", va="top", fontsize=11.5, color=MUTED, zorder=3)
    ax.text(x0 + w / 2, y0 + 0.8, shape, ha="center", va="bottom", fontsize=12, color=INK, style="italic",
            zorder=3)
    return (x0 + 1.6, y0 + 4.4, w - 3.2, h - 7.6 - 4.4 - 0.9)


def inset(ax, box, frame=False):
    a = ax.inset_axes(list(box), transform=ax.transData, zorder=2)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(frame); s.set_color("#94A3B8")
    a.patch.set_alpha(0)
    return a


def strip(a, x, y, vals, w, h, cmap="Greys"):
    for i, v in enumerate(vals):
        a.add_patch(Rectangle((x + i * w, y), w, h, facecolor=plt.get_cmap(cmap)(float(np.clip(v, 0, 1))),
                              edgecolor="white", linewidth=0.5))


def arrow(ax, p, q, text="", side="above", fs=11.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=20, color=LINE, linewidth=2.0, zorder=4))
    if text:
        xm, ym = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        if side == "above":
            ax.text(xm, max(p[1], q[1]) + 0.8, text, ha="center", va="bottom", fontsize=fs, color=MUTED,
                    weight="bold")
        elif side == "right":
            ax.text(xm + 1.0, ym, text, ha="left", va="center", fontsize=fs, color=MUTED, weight="bold")


def bars(a, rows, color, x=3.2, w=5.0, h=2.0, y0=6.6, step=3.4, fs=11):
    for i, (nm, p) in enumerate(rows):
        yy = y0 - i * step
        a.text(x - 0.3, yy + h / 2, nm, ha="right", va="center", fontsize=fs)
        a.add_patch(Rectangle((x, yy), w, h, facecolor="white", edgecolor="#94A3B8"))
        a.add_patch(Rectangle((x, yy), w * p, h, facecolor=color))
        a.text(x + w + 0.3, yy + h / 2, f"{p:.2f}", va="center", fontsize=fs + 0.5, weight="bold")


# ------------------------------------------------------------------ the cards' pictures
def pic_csv(ax, box):
    a = inset(ax, box)
    a.set_xlim(0, 5); a.set_ylim(0, 5)
    hdr = ["time", "s1", "s2", "s3", "label"]
    rows = [[f"{i}", f"{RAW[i, 0]:.1f}", "NaN" if np.isnan(RAW[i, 1]) else f"{RAW[i, 1]:.0f}", f"{RAW[i, 2]:.0f}",
             f"{Y[i]}"] for i in (40, 59, 60, 61)]
    for c, hname in enumerate(hdr):
        a.add_patch(Rectangle((c, 4), 1, 1, facecolor="#BAE6FD", edgecolor="#64748B", linewidth=0.8))
        a.text(c + 0.5, 4.5, hname, ha="center", va="center", fontsize=10.5, weight="bold")
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            fc = FAULT if (c == 4 and v == "1") else ("#FEF3C7" if v == "NaN" else "white")
            a.add_patch(Rectangle((c, 3 - r), 1, 1, facecolor=fc, edgecolor="#64748B", linewidth=0.8))
            a.text(c + 0.5, 3.5 - r, v, ha="center", va="center", fontsize=10.5)


def pic_split(ax, box):
    a = inset(ax, box)
    a.plot(t, RAW[:, 0], color=INK, linewidth=1.0)
    for lo, hi, col, nm in ((0, TR, "#BFDBFE", "train"), (TR, VA, "#FDE68A", "val"), (VA, T, "#BBF7D0", "test")):
        a.axvspan(lo, hi, color=col, alpha=0.85, zorder=0)
        a.text((lo + hi) / 2, 33.5, nm, ha="center", va="top", fontsize=11, weight="bold")
    for lo, hi in ((58, 64), (138, 148)):
        a.axvspan(lo, hi, ymin=0, ymax=0.13, color=FAULT, zorder=1)
    a.set_xlim(0, T); a.set_ylim(14, 34)


def pic_fill(ax, box):
    a = inset(ax, box)
    seg = slice(28, 58)
    a.plot(t[seg], RAW[seg, 1], "o-", color=INK, markersize=2.8, linewidth=1.0)
    a.plot(t[38:47], FILLED[38:47, 1], "--", color="#D97706", linewidth=2.4)
    a.axvspan(39.5, 44.5, color="#FEF3C7", zorder=0)
    a.text(42, a.get_ylim()[1], "gap filled", ha="center", va="top", fontsize=10.5, color="#B45309")


def pic_scale(ax, box):
    a = inset(ax, box)
    a.imshow(np.clip(Z.T, -4, 4), aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4, interpolation="nearest")
    a.set_yticks(range(4)); a.set_yticklabels(["s1", "s2", "s3", "s4"], fontsize=10); a.tick_params(length=0, pad=1)


def pic_windows(ax, box):
    tx, ty, tw, th = box
    a = inset(ax, (tx, ty + th * 0.55, tw, th * 0.45))
    a.imshow(np.clip(Z[VA - 40:T].T, -4, 4), aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4, interpolation="nearest")
    cols = ["#0369A1", "#B91C1C", "#15803D"]
    for k, s0 in enumerate((24, 36, 48)):
        a.add_patch(Rectangle((s0 - 0.5, -0.45), L, 3.9, fill=False, edgecolor=cols[k], linewidth=2))
    for k in range(3):
        b = inset(ax, (tx + k * (tw / 3) + 0.6, ty, tw / 3 - 1.2, th * 0.36), frame=True)
        s0 = VA - 40 + 24 + 12 * k
        b.imshow(np.clip(Z[s0:s0 + L].T, -4, 4), aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4,
                 interpolation="nearest")
        for sp in b.spines.values():
            sp.set_color(cols[k]); sp.set_linewidth(1.6)
        yw = int(Y[s0:s0 + L].max())
        b.set_title(f"y = {yw}", fontsize=10.5, pad=1.5, color="#B91C1C" if yw else INK, weight="bold")


def pic_render(ax, box):
    tx, ty, tw, th = box
    a = inset(ax, (tx + 1, ty, tw / 2 - 2, th - 2.2), frame=True)
    a.imshow((np.clip(WIN, -4, 4) + 4) / 8, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    a.set_title("heatmap, ±4", fontsize=10.5, pad=1.5)
    nper, hop = 8, 2
    starts = list(range(0, L - nper + 1, hop))
    spec = np.array([[np.log1p(np.abs(np.fft.rfft(np.hanning(nper) * WIN[f, s:s + nper]))) for s in starts]
                     for f in range(4)])                       # F x frames x freqs
    a = inset(ax, (tx + tw / 2 + 1, ty, tw / 2 - 2, th - 2.2), frame=True)
    a.imshow(spec.transpose(0, 2, 1).reshape(-1, len(starts)), aspect="auto", cmap="viridis",
             interpolation="nearest")
    a.set_title("spectrogram", fontsize=10.5, pad=1.5)


def pic_clip(ax, box):
    a = inset(ax, box)
    a.set_xlim(0, 38); a.set_ylim(0, 12.5)
    r = np.random.default_rng(5)
    for row, (nm, cmap) in enumerate((("image e", "YlOrBr"), ("'normal'", "Purples"),
                                      ("'anomalous'", "Purples"))):
        yy = 10.2 - row * 2.9
        a.text(10.4, yy + 0.9, nm, ha="right", va="center", fontsize=10.5)
        strip(a, 11.0, yy, r.uniform(0.05, 1, 14), 1.25, 1.8, cmap=cmap)
        a.text(11.0 + 14 * 1.25 + 0.3, yy + 0.9, "… 512", fontsize=10.5, va="center")
    a.text(19, 0.9, "100 · cos  →  logits 21.5 | 22.5", ha="center", fontsize=10.5, va="center", weight="bold")


def pic_teacher(ax, box):
    a = inset(ax, box)
    a.set_xlim(0, 12); a.set_ylim(0, 10)
    bars(a, (("normal", 0.27), ("anomalous", 0.73)), "#D97706", x=4.4, w=5.2, y0=6.8, step=3.6)
    a.text(6, 0.6, "training windows: scored out-of-fold", ha="center", fontsize=10, color=MUTED)


def pic_conv(ax, box):
    a = inset(ax, box, frame=True)
    filt = KR.normal(0, 0.6, (14, 4, 5))
    fm = np.array([[np.sum(filt[c] * np.pad(WIN, ((0, 0), (2, 2)))[:, i:i + 5]) for i in range(L)]
                   for c in range(14)])
    pic_conv.fm = np.maximum(fm, 0)[:, ::2]
    a.imshow(pic_conv.fm, aspect="auto", cmap="magma", interpolation="nearest")


def pic_gru(ax, box):
    tx, ty, tw, th = box
    a = inset(ax, (tx, ty + 3.4, tw, th - 3.4), frame=True)
    h = np.tanh(np.cumsum(KR.normal(0, 0.5, (16, pic_conv.fm.shape[1])) + 0.05 * pic_conv.fm[:2].mean(0),
                          axis=1) / 2)
    a.imshow(h, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    a.axhline(7.5, color="white", linewidth=2)
    a.text(0.2, 3.5, "→", color="white", fontsize=13, va="center", weight="bold")
    a.text(0.2, 11.5, "←", color="white", fontsize=13, va="center", weight="bold")
    a2 = inset(ax, (tx, ty, tw, 2.8))
    a2.set_xlim(0, tw); a2.set_ylim(0, 2.8)
    a2.text(0, 1.4, "pooled", fontsize=10, va="center")
    strip(a2, 6.5, 0.4, np.r_[h.mean(1)[:10], h.max(1)[:10]] * 0.5 + 0.5, (tw - 7) / 20, 2.0, cmap="coolwarm")


def pic_heads(ax, box):
    a = inset(ax, box)
    a.set_xlim(0, 38); a.set_ylim(0, 13)
    a.text(0.2, 11.4, "window", fontsize=10, va="center", weight="bold")
    x0 = 9.6
    a.add_patch(Rectangle((x0, 10.4), 8.4, 2, facecolor="white", edgecolor="#94A3B8"))
    a.add_patch(Rectangle((x0, 10.4), 8.4 * 0.69, 2, facecolor="#B91C1C"))
    a.text(x0 + 9.0, 11.4, "p = 0.69", fontsize=11, va="center", weight="bold")
    if box[2] > 35:
        a.text(26.6, 11.4, "(logits 0.1, 0.9)", fontsize=9.5, va="center", color=MUTED)
    a.text(0.2, 6.8, "step", fontsize=10, va="center", weight="bold")
    st = 1 / (1 + np.exp(-(3 * Y[W0:W0 + L] - 1.5 + KR.normal(0, 0.5, L))))
    xs = x0 + np.arange(L) * 27 / L
    a.fill_between(xs, 4.6, 4.6 + 4.4 * st, color="#FCA5A5", step="mid")
    a.plot(xs, 4.6 + 4.4 * st, color="#B91C1C", linewidth=1.3, drawstyle="steps-mid")
    a.text(0.2, 1.6, "embed", fontsize=10, va="center", weight="bold")
    strip(a, x0, 0.6, KR.uniform(0.05, 1, 16), 1.3, 2.0, cmap="Greys")
    a.text(x0 + 16 * 1.3 + 0.4, 1.6, "… 512", fontsize=10, va="center")


def pic_timeline(ax, box, scores, decide=False):
    tx, ty, tw, th = box
    a = inset(ax, (tx, ty + (3.2 if decide else 0), tw, th - (3.2 if decide else 0)), frame=True)
    a.plot(np.arange(T - VA), scores, color="#15803D", linewidth=1.6)
    a.axvspan(138 - VA, 148 - VA, color="#CBD5E1", zorder=0)
    a.set_ylim(0, 1.08); a.set_xlim(0, T - VA - 1)
    if not decide:
        return
    a.axhline(0.5, color="#B91C1C", linestyle="--", linewidth=1.4)
    a.text(0.8, 0.54, "τ", color="#B91C1C", fontsize=12, va="bottom")
    flags = (scores >= 0.5).astype(int)
    flags[8] = 0
    a2 = inset(ax, (tx, ty, tw, 2.6))
    a2.set_xlim(0, T - VA); a2.set_ylim(0, 1)
    for i, f in enumerate(flags):
        a2.add_patch(Rectangle((i, 0), 1, 1, facecolor=FAULT if f else "white", edgecolor="#CBD5E1", linewidth=0.3))


def pic_metrics(ax, box):
    a = inset(ax, box)
    a.set_xlim(0, 38); a.set_ylim(0, 13)
    cm = [["TP 3", "FN 1"], ["FP 1", "TN 3"]]
    cols = [["#BBF7D0", "#FECACA"], ["#FECACA", "#F1F5F9"]]
    for r in range(2):
        for c in range(2):
            a.add_patch(Rectangle((1 + c * 6, 6.6 - r * 5.6), 5.8, 5.4, facecolor=cols[r][c], edgecolor="white"))
            a.text(3.9 + c * 6, 9.3 - r * 5.6, cm[r][c], ha="center", va="center", fontsize=11)
    a.text(14.0, 9.8, "precision = 3/4 = 0.75", fontsize=10.5, va="center")
    a.text(14.0, 7.4, "recall = 3/4 = 0.75", fontsize=10.5, va="center")
    a.text(14.0, 5.0, "F1 = 0.75  (need τ)", fontsize=10.5, va="center")
    a.text(14.0, 2.4, "AUPRC = 0.89 (no τ)", fontsize=10, va="center", weight="bold")


def main():
    fig, ax = plt.subplots(figsize=(22.5, 11.4))
    ax.set_xlim(0, 225)
    ax.set_ylim(0, 117)
    ax.set_aspect("equal")
    ax.axis("off")
    cw, gap, ch, vgap = 41.0, 3.75, 25.0, 4.6
    X = [5.0 + i * (cw + gap) for i in range(5)]
    R = [1.0 + (3 - i) * (ch + vgap) for i in range(4)]      # R[0] = top row
    mid = lambda r: R[r] + ch / 2

    # row 1: data preparation
    pic_csv(ax, card(ax, X[0], R[0], cw, ch, "data", "1  Raw CSV", "one row per time step", "T rows × (F sensors + label)"))
    pic_split(ax, card(ax, X[1], R[0], cw, ch, "data", "2  Split", "by time / patient / well / failure",
                       "train | validation | test, never mixed"))
    pic_fill(ax, card(ax, X[2], R[0], cw, ch, "data", "3  Fill gaps", "straight line between neighbours",
                      "T × F, no missing values"))
    pic_scale(ax, card(ax, X[3], R[0], cw, ch, "data", "4  Standardise", "z = (x − μ) / σ, μ and σ from train",
                       "matrix Z:  T × F"))
    pic_windows(ax, card(ax, X[4], R[0], cw, ch, "data", "5  Sliding windows",
                         "length L, step S; label = any fault", "N windows × L × F"))
    for i in range(4):
        arrow(ax, (X[i] + cw + 0.2, mid(0)), (X[i + 1] - 0.2, mid(0)))

    # row 2: teacher
    pic_render(ax, card(ax, X[0], R[1], cw, ch, "teacher", "6  Draw each window", "two pictures; labels never drawn",
                        "2 images,  224 × 224 × 3"))
    pic_clip(ax, card(ax, X[1], R[1], cw, ch, "teacher", "7  Frozen CLIP",
                      "151 M frozen; 14 prompt words learned", "embedding e ∈ ℝ⁵¹²,  logits ∈ ℝ²"))
    pic_teacher(ax, card(ax, X[2], R[1], cw, ch, "teacher", "8  Teacher opinion", "softmax of the two logits",
                         "soft label  p_T"))
    arrow(ax, (X[0] + cw + 0.2, mid(1)), (X[1] - 0.2, mid(1)))
    arrow(ax, (X[1] + cw + 0.2, mid(1)), (X[2] - 0.2, mid(1)))

    # row 3: student
    pic_conv(ax, card(ax, X[0], R[2], cw, ch, "student", "9  Convolution front-end",
                      "short shapes; time shortened ÷ d", "feature map  64 × L/d"))
    pic_gru(ax, card(ax, X[1], R[2], cw, ch, "student", "10  Two-way GRU", "reads forwards and backwards",
                     "h_t ∈ ℝ¹²⁸ per step;  pooled ∈ ℝ²⁵⁶"))
    pic_heads(ax, card(ax, X[2], R[2], cw, ch, "student", "11  Three heads", "faulty? · which steps? · CLIP's view",
                       "p_win ∈ ℝ²,  s ∈ ℝᴸ,  g ∈ ℝ⁵¹²"))
    arrow(ax, (X[0] + cw + 0.2, mid(2)), (X[1] - 0.2, mid(2)))
    arrow(ax, (X[1] + cw + 0.2, mid(2)), (X[2] - 0.2, mid(2)))

    # every window goes to both models: elbow from card 5 to cards 6 and 9
    yb = R[0] - vgap / 2
    xl = 2.0
    x5 = X[4] + cw / 2
    ax.plot([x5, x5, xl, xl], [R[0], yb, yb, mid(2)], color=LINE, linewidth=2.0, zorder=4, solid_joinstyle="miter")
    arrow(ax, (xl, mid(2)), (X[0] - 0.2, mid(2)))
    arrow(ax, (X[0] + cw / 2, yb), (X[0] + cw / 2, R[1] + ch + 0.2))
    ax.text((X[1] + X[3]) / 2 + cw / 2, yb + 0.3, "every window goes to both the teacher (training only) and the student",
            ha="center", va="bottom", fontsize=11, color=MUTED, weight="bold")

    # 12 training loss (rows 2-3, columns 4-5)
    lx0, ly0, lw, lh = X[3], R[2], 2 * cw + gap, 2 * ch + vgap
    fill, edge, head = KIND["loss"]
    ax.add_patch(FancyBboxPatch((lx0, ly0), lw, lh, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill,
                                edgecolor=edge, linewidth=2.0, zorder=1))
    ax.add_patch(FancyBboxPatch((lx0 + 0.3, ly0 + lh - 7.6), lw - 0.6, 7.3, boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=head, edgecolor="none", zorder=1))
    ax.text(lx0 + lw / 2, ly0 + lh - 0.9, "12  Training loss (training only)", ha="center", va="top", fontsize=14.5,
            weight="bold")
    ax.text(lx0 + lw / 2, ly0 + lh - 4.6, "compare outputs with labels and teacher; nudge the weights (AdamW)",
            ha="center", va="top", fontsize=11.5, color=MUTED)
    ax.text(lx0 + lw / 2, ly0 + lh - 10.6,
            r"$L=\mathrm{BCE}_w(s,y_{step})+\mathrm{CE}(p_{win},y)+w\,T^2\,\mathrm{KL}(p_T\,\|\,p_{win})"
            r"+0.5\,(1-\cos(g,e))$", ha="center", va="top", fontsize=13)
    terms = [("steps vs. labels", "rare faulty steps count more", 0.376, "#B91C1C"),
             ("window vs. label", "is the window faulty?", 0.371, "#F87171"),
             ("window vs. teacher", "× teacher's confidence, T = 2", 0.0035, "#D97706"),
             ("embedding vs. CLIP", "copy CLIP's view (× 0.5)", 0.111, "#FBBF24")]
    bx0, bw = lx0 + 4, lw - 8
    tot = sum(v * (0.5 if i == 3 else 1) for i, (_, _, v, _) in enumerate(terms))
    xx = bx0
    ytop = ly0 + lh - 19.5
    for i, (nm, why, v, col) in enumerate(terms):
        vv = v * (0.5 if i == 3 else 1)
        ax.add_patch(Rectangle((xx, ytop), bw * vv / tot, 3.0, facecolor=col, edgecolor="white", linewidth=0.8,
                               zorder=2))
        xx += bw * vv / tot
    ax.text(bx0 + bw, ytop + 3.6, f"toy window: total {tot:.3f}", ha="right", va="bottom", fontsize=11, color=INK)
    for i, (nm, why, v, col) in enumerate(terms):
        yy = ytop - 3.4 - i * 4.1
        ax.add_patch(Rectangle((bx0, yy - 1.1), 2.2, 2.2, facecolor=col, zorder=2))
        ax.text(bx0 + 3.2, yy, nm, fontsize=12.5, va="center", weight="bold")
        ax.text(bx0 + 30, yy, why, fontsize=11.5, va="center", color=MUTED)
        ax.text(bx0 + bw, yy, f"{v * (0.5 if i == 3 else 1):.3f}", fontsize=12, va="center", ha="right")
    ax.text(lx0 + lw / 2, ly0 + 4.8, "labels-only baseline = the same student with only the first two terms",
            ha="center", va="bottom", fontsize=11.5, color=INK)
    ax.text(lx0 + lw / 2, ly0 + 0.8, "128 windows per batch · ≤ 20 epochs · keep the best validation AUPRC",
            ha="center", va="bottom", fontsize=12, color=INK, style="italic")
    arrow(ax, (X[2] + cw + 0.2, mid(1)), (lx0 - 0.2, mid(1)))
    arrow(ax, (X[2] + cw + 0.2, mid(2)), (lx0 - 0.2, mid(2)))

    # row 4: test time (student only), columns 3-5
    scores = np.clip(0.08 + 0.85 * np.convolve(Y[VA:T], np.ones(5) / 5, mode="same") + KR.normal(0, 0.04, T - VA), 0, 1)
    scores[8] = 0.62
    pic_timeline(ax, card(ax, X[2], R[3], cw, ch, "eval", "13  Score every time step",
                          "average of the overlapping windows", "score per step (grey = true fault)"), scores)
    pic_timeline(ax, card(ax, X[3], R[3], cw, ch, "eval", "14  Decide", "score ≥ τ, then drop very short runs",
                          "0 / 1 per time step"), scores, decide=True)
    pic_metrics(ax, card(ax, X[4], R[3], cw, ch, "eval", "15  Metrics", "window · time step · event",
                         "P, R, F1, AUPRC"))
    arrow(ax, (X[2] + cw / 2, R[2] - 0.2), (X[2] + cw / 2, R[3] + ch + 0.2), "test time: student only", side="right")
    arrow(ax, (X[2] + cw + 0.2, mid(3)), (X[3] - 0.2, mid(3)))
    arrow(ax, (X[3] + cw + 0.2, mid(3)), (X[4] - 0.2, mid(3)))

    # legend + note (row 4, columns 1-2)
    lx, ly = X[0], R[3] + ch - 3.0
    ax.text(lx, ly + 1.6, "Colours", fontsize=13, weight="bold", va="bottom")
    for i, (kind, nm) in enumerate((("data", "data preparation (same for every model)"),
                                     ("teacher", "teacher: frozen CLIP, used only in training"),
                                     ("student", "student: small GRU, the deployed detector"),
                                     ("loss", "training"), ("eval", "test time and scoring"))):
        fill, edge, hd = KIND[kind]
        yy = ly - 1.6 - i * 3.3
        ax.add_patch(Rectangle((lx, yy - 1.2), 2.6, 2.4, facecolor=hd, edgecolor=edge, linewidth=1.6))
        ax.text(lx + 3.6, yy, nm, fontsize=12, va="center")
    ax.text(lx, R[3] + 0.6, "Pictures: a 4-sensor toy with 24-step windows.\nReal data: F = 8–29 sensors, "
                            "L = 64–256 steps.", fontsize=11, color=MUTED, va="bottom", style="italic",
            linespacing=1.3)

    fig.savefig(OUT / "poster_dataflow_wide.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "poster_dataflow_wide.pdf", bbox_inches="tight", facecolor="white")
    print(OUT / "poster_dataflow_wide.png")


def main_column():
    """Same pipeline, laid out 3 cards wide so it fits one poster column."""
    fig, ax = plt.subplots(figsize=(11.9, 16.6))
    ax.set_xlim(0, 119)
    ax.set_ylim(0, 166)
    ax.set_aspect("equal")
    ax.axis("off")
    cw, gap, ch, vg, lh = 35.0, 3.5, 23.0, 4.5, 27.0
    X = [4.0 + i * (cw + gap) for i in range(3)]
    RF = 1.0
    RE = RF + ch + vg
    RD = RE + lh + vg
    RC = RD + ch + vg
    RB = RC + ch + vg
    RA = RB + ch + vg
    mid = lambda r: r + ch / 2
    right = X[2] + cw

    # row A, B: data preparation
    pic_csv(ax, card(ax, X[0], RA, cw, ch, "data", "1  Raw CSV", "one row per time step", "T × (F + label)"))
    pic_split(ax, card(ax, X[1], RA, cw, ch, "data", "2  Split", "whole sequences or time ranges",
                       "train | val | test"))
    pic_fill(ax, card(ax, X[2], RA, cw, ch, "data", "3  Fill gaps", "straight line across the gap",
                      "T × F, no gaps"))
    pic_scale(ax, card(ax, X[0], RB, cw, ch, "data", "4  Standardise", "z = (x − μ)/σ, train μ and σ",
                       "matrix Z:  T × F"))
    pic_windows(ax, card(ax, X[1], RB, cw, ch, "data", "5  Sliding windows", "length L, step S",
                         "N × L × F, label y"))
    for i in range(2):
        arrow(ax, (X[i] + cw + 0.2, mid(RA)), (X[i + 1] - 0.2, mid(RA)))
    arrow(ax, (X[0] + cw + 0.2, mid(RB)), (X[1] - 0.2, mid(RB)))
    yab = RA - vg / 2
    ax.plot([X[2] + cw / 2, X[2] + cw / 2, X[0] + cw / 2], [RA, yab, yab], color=LINE, linewidth=2.0, zorder=4)
    arrow(ax, (X[0] + cw / 2, yab), (X[0] + cw / 2, RB + ch + 0.2))

    # legend in row B, column 3
    lx, ly = X[2] + 0.5, RB + ch - 1.0
    ax.text(lx, ly, "Colours", fontsize=12.5, weight="bold", va="top")
    for i, (kind, nm) in enumerate((("data", "data preparation"), ("teacher", "teacher (training only)"),
                                     ("student", "student (deployed)"), ("loss", "training"),
                                     ("eval", "test time"))):
        fill, edge, hd = KIND[kind]
        yy = ly - 4.2 - i * 2.8
        ax.add_patch(Rectangle((lx, yy - 1.1), 2.4, 2.2, facecolor=hd, edgecolor=edge, linewidth=1.5))
        ax.text(lx + 3.4, yy, nm, fontsize=11.5, va="center")
    ax.text(lx, RB + 0.4, "pictures: 4-sensor toy,\n24-step windows", fontsize=10.5, color=MUTED, style="italic",
            va="bottom", linespacing=1.2)

    # every window goes to both models
    ybc = RB - vg / 2
    xl = 1.6
    ax.plot([X[1] + cw / 2, X[1] + cw / 2, xl, xl], [RB, ybc, ybc, mid(RD)], color=LINE, linewidth=2.0, zorder=4)
    arrow(ax, (X[0] + cw / 2, ybc), (X[0] + cw / 2, RC + ch + 0.2))
    arrow(ax, (xl, mid(RD)), (X[0] - 0.2, mid(RD)))
    ax.text(X[1] + cw / 2 + 1.0, ybc, "every window goes to both models", fontsize=11, color=MUTED,
            weight="bold", va="center")

    # row C: teacher
    pic_render(ax, card(ax, X[0], RC, cw, ch, "teacher", "6  Draw each window", "labels are never drawn",
                        "2 images, 224 × 224 × 3"))
    pic_clip(ax, card(ax, X[1], RC, cw, ch, "teacher", "7  Frozen CLIP", "151 M frozen, 14 words learned",
                      "e ∈ ℝ⁵¹²,  logits ∈ ℝ²"))
    pic_teacher(ax, card(ax, X[2], RC, cw, ch, "teacher", "8  Teacher opinion", "softmax of the two logits",
                         "soft label  p_T"))
    for i in range(2):
        arrow(ax, (X[i] + cw + 0.2, mid(RC)), (X[i + 1] - 0.2, mid(RC)))

    # row D: student
    pic_conv(ax, card(ax, X[0], RD, cw, ch, "student", "9  Conv front-end", "short shapes; time ÷ d",
                      "feature map 64 × L/d"))
    pic_gru(ax, card(ax, X[1], RD, cw, ch, "student", "10  Two-way GRU", "reads forwards and backwards",
                     "h_t ∈ ℝ¹²⁸, pooled ∈ ℝ²⁵⁶"))
    pic_heads(ax, card(ax, X[2], RD, cw, ch, "student", "11  Three heads", "faulty? which steps? CLIP view",
                       "p_win,  s ∈ ℝᴸ,  g ∈ ℝ⁵¹²"))
    for i in range(2):
        arrow(ax, (X[i] + cw + 0.2, mid(RD)), (X[i + 1] - 0.2, mid(RD)))

    # row E: training loss (full width)
    lx0, lw = X[0], right - X[0]
    fill, edge, head = KIND["loss"]
    ax.add_patch(FancyBboxPatch((lx0, RE), lw, lh, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill,
                                edgecolor=edge, linewidth=2.0, zorder=1))
    ax.add_patch(FancyBboxPatch((lx0 + 0.3, RE + lh - 7.6), lw - 0.6, 7.3, boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=head, edgecolor="none", zorder=1))
    ax.text(lx0 + lw / 2, RE + lh - 0.9, "12  Training loss (training only)", ha="center", va="top", fontsize=14.5,
            weight="bold")
    ax.text(lx0 + lw / 2, RE + lh - 4.6, "compare with labels and teacher, then nudge the weights (AdamW)",
            ha="center", va="top", fontsize=11.5, color=MUTED)
    ax.text(lx0 + lw / 2, RE + lh - 9.0,
            r"$L=\mathrm{BCE}_w(s,y_{step})+\mathrm{CE}(p_{win},y)+w\,T^2\,\mathrm{KL}(p_T\,\|\,p_{win})"
            r"+0.5\,(1-\cos(g,e))$", ha="center", va="top", fontsize=13)
    terms = [("steps vs. labels", 0.376, "#B91C1C"), ("window vs. label", 0.371, "#F87171"),
             ("window vs. teacher", 0.0035, "#D97706"), ("0.5 × (embedding vs. CLIP)", 0.056, "#FBBF24")]
    tot = sum(v for _, v, _ in terms)
    bx0, bw, by = lx0 + 3, 46, RE + lh - 17.5
    xx = bx0
    for nm, v, col in terms:
        ax.add_patch(Rectangle((xx, by), bw * v / tot, 3.0, facecolor=col, edgecolor="white", linewidth=0.8, zorder=2))
        xx += bw * v / tot
    ax.text(bx0, by + 3.5, f"toy window: total {tot:.3f}", fontsize=11, va="bottom")
    ax.text(bx0, by - 1.4, "baseline = only the first two terms", fontsize=11, va="top", color=INK)
    ax.text(bx0, by - 4.6, "128 windows per batch, ≤ 20 epochs,\nkeep the best validation AUPRC", fontsize=10.5,
            va="top", color=MUTED, style="italic", linespacing=1.2)
    tx0 = lx0 + 55
    for i, (nm, v, col) in enumerate(terms):
        yy = RE + lh - 12.8 - i * 3.3
        ax.add_patch(Rectangle((tx0, yy - 1.0), 2.0, 2.0, facecolor=col, zorder=2))
        ax.text(tx0 + 3.0, yy, nm, fontsize=11, va="center", weight="bold")
        ax.text(right - 2.0, yy, f"{v:.3f}", fontsize=11.5, va="center", ha="right")
    ax.text(tx0, RE + 0.8, "w = teacher's confidence in the true answer;  T = 2", fontsize=10, color=MUTED,
            va="bottom")
    arrow(ax, (X[2] + cw / 2, RD - 0.2), (X[2] + cw / 2, RE + lh + 0.2))
    yr = RE + lh - 3.0
    ax.plot([right + 0.2, right + 2.4, right + 2.4], [mid(RC), mid(RC), yr], color=LINE, linewidth=2.0, zorder=4)
    arrow(ax, (right + 2.4, yr), (right + 0.2, yr))

    # row F: test time
    scores = np.clip(0.08 + 0.85 * np.convolve(Y[VA:T], np.ones(5) / 5, mode="same")
                     + np.random.default_rng(11).normal(0, 0.04, T - VA), 0, 1)
    scores[8] = 0.62
    pic_timeline(ax, card(ax, X[0], RF, cw, ch, "eval", "13  Score each step", "average of overlapping windows",
                          "score per step"), scores)
    pic_timeline(ax, card(ax, X[1], RF, cw, ch, "eval", "14  Decide", "≥ τ, then drop short runs",
                          "0 / 1 per step"), scores, decide=True)
    pic_metrics(ax, card(ax, X[2], RF, cw, ch, "eval", "15  Metrics", "window · step · event", "P, R, F1, AUPRC"))
    for i in range(2):
        arrow(ax, (X[i] + cw + 0.2, mid(RF)), (X[i + 1] - 0.2, mid(RF)))
    arrow(ax, (X[0] + cw / 2, RE - 0.2), (X[0] + cw / 2, RF + ch + 0.2), "after training: student only", side="right")

    fig.savefig(OUT / "poster_dataflow.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "poster_dataflow.pdf", bbox_inches="tight", facecolor="white")
    print(OUT / "poster_dataflow.png")


if __name__ == "__main__":
    main()
    main_column()
