"""Graphviz (DOT) diagrams of the pipeline, filled with the selected dataset's numbers.

Rendered by ``st.graphviz_chart`` in the browser; no Graphviz install needed.
"""

from __future__ import annotations

from typing import Any, Dict

FONT = 'fontname="Helvetica", fontsize=13'
STYLE = {
    "data": 'fillcolor="#E0F2FE", color="#0284C7"',
    "teacher": 'fillcolor="#FEF3C7", color="#D97706"',
    "student": 'fillcolor="#FEE2E2", color="#B91C1C"',
    "loss": 'fillcolor="#EDE9FE", color="#6D28D9"',
    "eval": 'fillcolor="#DCFCE7", color="#15803D"',
}


def _node(key: str, text: str, kind: str, width: int = 34) -> str:
    import textwrap

    lines = []
    for part in text.split("\n"):
        lines += textwrap.wrap(part, width) or [""]
    label = "\\n".join(l.replace('"', "'") for l in lines)
    return f'  {key} [label="{label}", shape=box, style="rounded,filled", {STYLE[kind]}, {FONT}];'


def full_pipeline(name: str, data: Any, pretty: str, pos_weight: float, downsample: int,
                  part: str | None = None) -> str:
    """Complete flow: CSV → matrix → split → impute → scale → windows → teacher /
    student → loss → evaluation, with the dataset's shapes on every node.

    ``part``: None for everything, or 'prep' (CSV → tensors), 'train' (teacher,
    student, loss) or 'eval' (inference → thresholds → metrics)."""
    m = data.manifest["partitions"]
    F, L, S = len(data.feature_names), data.window_length, data.stride
    n = {k: m[k]["windows"] for k in ("train", "val", "test")}
    T = sum(m[k]["timesteps"] for k in m)
    lines = ["digraph G {", '  rankdir=TB; nodesep=0.35; ranksep=0.3; bgcolor="transparent";',
             f'  node [{FONT}]; edge [{FONT}, color="#94A3B8", fontcolor="#94A3B8", fontsize=9];']
    nodes = [
        ("csv", f"1. Raw files ({pretty})\nCSV / text rows: timestamp, {F} sensor columns, label", "data"),
        ("mat", f"2. Per-sequence matrix X ∈ R^(T×F)\nF = {F} features, {T:,} timesteps in total\nlabel vector y ∈ {{0,1}}^T", "data"),
        ("split", f"3. Split whole sequences\ntrain {m['train']['timesteps']:,} | val {m['val']['timesteps']:,} | "
                  f"test {m['test']['timesteps']:,} steps\n({data.manifest['split_method']})", "data"),
        ("imp", "4. Impute missing values (per sequence)\nlinear interpolation in time, edge fill;\n"
                "all-missing column → training mean", "data"),
        ("scale", "5. Standardise each feature (fit on TRAIN only)\nz = (x − μ_f) / σ_f", "data"),
        ("win", f"6. Sliding windows: length L = {L}, stride {S}\nstarts 0, {S}, 2·{S}, … (+ one aligned to the end)\n"
                f"X_w ∈ R^(L×F),  y_step ∈ {{0,1}}^L,  y_win = max_t y_step", "data"),
        ("tens", f"7. Tensors per split  (N × {L} × {F})\ntrain {n['train']:,} | val {n['val']:,} | test {n['test']:,} windows", "data"),
        ("render", "8a. Render each window (labels never drawn)\nheatmap: clip z to [−4,4] → viridis, F×L → 224×224\n"
                   "spectrogram: |STFT| per channel → 224×224", "teacher"),
        ("clip", "8b. Frozen CLIP ViT-B/32 image encoder\nimage → unit vector e ∈ R^512 (one per view)", "teacher"),
        ("prompt", "8c. Learned prompts (14 context vectors, 7,168 numbers)\n"
                   "text features t_normal, t_anomalous (frozen text encoder)", "teacher"),
        ("tprob", "8d. Teacher probability\np_T = softmax(mean_views 100·cos(e, t_c))\n"
                  "train windows: out-of-fold (cross-fitted)", "teacher"),
        ("conv", f"9a. Student front-end\nConv1d(F→64, k5) → BN → GELU → Conv1d(64→64, stride {downsample}) → BN → GELU", "student"),
        ("gru", f"9b. Bidirectional GRU (64 per direction)\nh_t ∈ R^128 for t = 1 … L/{downsample}", "student"),
        ("whead", "Window head\nmean+max pool → Linear(256→64) → ReLU → Linear(64→2)\n→ softmax → p_win", "student"),
        ("shead", "Step head\nLinear(128→32) → ReLU → Linear(32→1) per step\n→ upsample to L → sigmoid → s_t", "student"),
        ("ehead", "Embedding head (training only)\nLinear(256→128) → GELU → Linear(128→512)", "student"),
        ("lstep", f"Step BCE vs y_step\npos_weight = {pos_weight:.2f}", "loss"),
        ("lwin", "Window CE vs y_win  (× β = 1)", "loss"),
        ("lkd", "KD: w·T²·KL(p_T^(T) ‖ p_win^(T)),  T = 2\nw = p_T(true class)  (× λ_KD = 1)", "loss"),
        ("lfeat", "Feature: 1 − cos(g, e)  (× λ_feat = 0.5)", "loss"),
        ("opt", "Total loss → AdamW; keep epoch with best\nvalidation AUPRC (early stopping)", "loss"),
        ("agg", "10. Timeline scores: average overlapping windows per timestep\n"
                "gate s_t · p_win^α, moving average k (α, k chosen on validation)", "eval"),
        ("thr", "11. Threshold from VALIDATION only\n(F1-optimal; < 5 validation events → 0.1% false-alarm rate)\n"
                "drop runs shorter than min_len", "eval"),
        ("met", "12. Test metrics\nwindow: AUPRC, ROC-AUC, P/R/F1 | point: AUPRC, P/R/F1 | event: overlap P/R/F1", "eval"),
    ]
    edges = [("csv", "mat", "parse, sort by time"), ("mat", "split", ""), ("split", "imp", ""), ("imp", "scale", ""),
             ("scale", "win", ""), ("win", "tens", ""), ("tens", "render", "teacher branch"),
             ("render", "clip", ""), ("clip", "tprob", ""), ("prompt", "tprob", ""),
             ("tens", "conv", "student branch"), ("conv", "gru", ""), ("gru", "whead", ""), ("gru", "shead", ""),
             ("gru", "ehead", ""), ("shead", "lstep", ""), ("whead", "lwin", ""), ("whead", "lkd", ""),
             ("tprob", "lkd", "soft label"), ("ehead", "lfeat", ""), ("clip", "lfeat", "CLIP embedding"),
             ("lstep", "opt", ""), ("lwin", "opt", ""), ("lkd", "opt", ""), ("lfeat", "opt", ""),
             ("whead", "agg", "inference"), ("shead", "agg", "inference"), ("agg", "thr", ""), ("thr", "met", "")]
    parts = {
        "prep": {"csv", "mat", "split", "imp", "scale", "win", "tens"},
        "train": {"tens", "render", "clip", "prompt", "tprob", "conv", "gru", "whead", "shead", "ehead",
                  "lstep", "lwin", "lkd", "lfeat", "opt"},
        "eval": {"tens", "conv", "gru", "whead", "shead", "agg", "thr", "met"},
    }
    keep = parts.get(part, {k for k, _, _ in nodes})
    width = 34 if part in (None, "prep", "eval") else 26
    lines += [_node(k, t, c, width) for k, t, c in nodes if k in keep]
    for a, b, lab in edges:
        if a in keep and b in keep:
            lines.append(f'  {a} -> {b}' + (f' [label="{lab}"]' if lab else "") + ";")
    if part in (None, "train"):
        lines += ['  { rank=same; render; conv; }', '  { rank=same; lstep; lwin; lkd; lfeat; }']
    lines.append("}")
    return "\n".join(lines)


def student_heads() -> str:
    """What each head of the student computes and which target trains it."""
    rows = [
        ("x", "Input window X_w (L × F), standardised", "data"),
        ("fe", "Conv front-end: local shape filters, length L → L/d", "student"),
        ("g", "BiGRU: h_t ∈ R^128 (forward ⊕ backward state)", "student"),
        ("pool", "Pooling: [mean_t h_t, max_t h_t] ∈ R^256", "student"),
        ("w", "Window head → z_win ∈ R^2 → p_win = softmax(z_win)\n'is this window anomalous?'", "student"),
        ("s", "Step head → z_t per step → s_t = sigmoid(z_t)\n'which timesteps are anomalous?'", "student"),
        ("e", "Embedding head → g ∈ R^512\n'what would CLIP see?' (training only)", "student"),
        ("yw", "Ground truth y_win (CE)\n+ teacher p_T (KD)", "loss"),
        ("ys", "Ground truth y_step (weighted BCE)", "loss"),
        ("ye", "CLIP image embedding e (1 − cos)", "loss"),
    ]
    lines = ["digraph H {", '  rankdir=LR; bgcolor="transparent"; nodesep=0.25;', f'  edge [{FONT}, color="#94A3B8", fontcolor="#94A3B8"];']
    lines += [_node(k, t, c) for k, t, c in rows]
    for a, b in (("x", "fe"), ("fe", "g"), ("g", "pool"), ("pool", "w"), ("g", "s"), ("pool", "e"),
                 ("w", "yw"), ("s", "ys"), ("e", "ye")):
        lines.append(f"  {a} -> {b};")
    lines.append("}")
    return "\n".join(lines)


def teacher_flow(views: Dict[str, str]) -> str:
    lines = ["digraph T {", '  rankdir=LR; bgcolor="transparent"; nodesep=0.3;', f'  edge [{FONT}, color="#94A3B8", fontcolor="#94A3B8"];']
    lines += [
        _node("w", "Window X_w (L × F)\nstandardised", "data"),
        _node("h", "Heatmap image\n224×224×3", "teacher"),
        _node("sp", "Spectrogram image\n224×224×3", "teacher"),
        _node("vit", "Frozen CLIP image encoder\n(ViT-B/32) → e_view ∈ R^512, ‖e‖ = 1", "teacher"),
        _node("txt", "Frozen CLIP text encoder\n[SOS] v1…v14 normal|anomalous . [EOS]\n→ t_normal, t_anomalous", "teacher"),
        _node("lg", "logit_c = mean_views 100·cos(e_view, t_c)", "teacher"),
        _node("p", "p_T = softmax(logit_normal, logit_anomalous)", "teacher"),
    ]
    for a, b in (("w", "h"), ("w", "sp"), ("h", "vit"), ("sp", "vit"), ("vit", "lg"), ("txt", "lg"), ("lg", "p")):
        lines.append(f"  {a} -> {b};")
    lines.append("}")
    return "\n".join(lines)
