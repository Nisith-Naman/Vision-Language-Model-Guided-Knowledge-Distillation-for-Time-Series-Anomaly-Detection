"""Page: the frozen-CLIP teacher, opened up for one window.

Rendering → CLIP image embedding → zero-shot prompts → learned prompts →
teacher probability, every number computed live and compared with the cached
value the student was trained on. Then cross-fitting and teacher quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.app import core, diagrams
from src.teacher.render import render_spectrogram, render_window

TEACHER_NAMES = {
    "zeroshot": "Zero-shot (4 + 4 hand-written prompts, heatmap)",
    "probe": "Linear probe on the heatmap embedding",
    "probe_mv": "Linear probe on heatmap + spectrogram embeddings",
    "coop": "Learned prompts, heatmap",
    "coop_mv": "Learned prompts, heatmap + spectrogram (final teacher)",
}


@st.cache_resource(show_spinner="Building the learned-prompt text encoder ...")
def _prompt_encoder(name: str):
    from src.teacher.prompt_learning import PromptEncoder

    emb = core.embedder()
    enc = PromptEncoder(emb.model, emb.tokenizer)
    enc.ctx.data.copy_(torch.as_tensor(core.teacher(name)["ctx"]))
    return enc


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    return np.exp(z) / np.exp(z).sum()


def show() -> None:
    name = core.selected_dataset()
    data = core.prepared(name)
    tcache = core.teacher(name)
    tcfg = core.config()["benchmark"]["teacher"]
    clip_v, style = float(tcfg["render_clip"]), "heatmap"
    st.header(f"Teacher: frozen CLIP ViT-B/32 with learned prompts — {core.PRETTY[name]}")
    st.markdown("The teacher never sees labels at test time and none of CLIP's 151 million weights change. It turns a "
                "window into two pictures, embeds them with CLIP's image encoder and compares the embeddings with "
                "text embeddings of a *normal* and an *anomalous* prompt. Only the 14 prompt context vectors "
                "(7,168 numbers) were learned, on training windows.")
    st.graphviz_chart(diagrams.teacher_flow({}), use_container_width=True)
    if tcache is None:
        st.error("No cached teacher outputs; run scripts/run_benchmark.py for this dataset.")
        return

    split = st.radio("Split", ["test", "val"], horizontal=True, format_func=lambda s: {"test": "test",
                     "val": "validation"}[s])
    sp = data.splits[split]
    pos = np.flatnonzero(sp.y_window == 1)
    default = int(pos[0]) if len(pos) else 0
    i = st.number_input(f"Window index (0 … {len(sp.X) - 1}); default = first anomalous window", 0, len(sp.X) - 1,
                        default)
    Xw = sp.X[i]
    st.caption(f"Window {i}: label y_win = {int(sp.y_window[i])}, {int(sp.y_step[i].sum())} anomalous steps of "
               f"{data.window_length}.")

    # ------------------------------------------------------------ rendering
    st.subheader("Step 1 · Render the window as two images (labels are never drawn)")
    L, F = Xw.shape
    nperseg = int(max(8, min(64, 2 ** int(np.log2(max(8, L // 4))))))
    hop = max(1, nperseg // 4)
    frames = len(np.arange(0, max(1, L - nperseg + 1), hop))
    c1, c2 = st.columns(2)
    heat = render_window(Xw, style, clip_v)
    spec = render_spectrogram(Xw, clip_v)
    with c1:
        st.image(heat, caption="View 1 · heatmap", width=300)
        st.markdown(f"1. clip every value to [−{clip_v:g}, {clip_v:g}]\n"
                    f"2. colour index = round((z + {clip_v:g}) / {2 * clip_v:g} × 255) → viridis colour\n"
                    f"3. image of {F} rows (features) × {L} columns (time), resized to 224 × 224 (nearest)\n\n"
                    f"Clipped values in this window: {int((np.abs(Xw) > clip_v).sum())} of {Xw.size}.")
    with c2:
        st.image(spec, caption="View 2 · spectrogram", width=300)
        st.markdown(f"1. per feature, frames of {nperseg} steps every {hop} steps ({frames} frames), Hann window\n"
                    f"2. magnitude |FFT| of each frame ({nperseg // 2 + 1} frequency bins)\n"
                    f"3. brightness = log(1 + |FFT|) / log(1 + {clip_v:g}·{nperseg}/2) (fixed scale)\n"
                    f"4. rows = feature × frequency, columns = frames, resized to 224 × 224")

    # ------------------------------------------------------------ CLIP
    st.subheader("Step 2 · CLIP image encoder → embedding e ∈ ℝ⁵¹²")
    st.markdown("Pixels are scaled to [0,1] and normalised with CLIP's channel mean/std, passed through the frozen "
                "ViT-B/32 (12 transformer layers over 7 × 7 patches of 32 × 32 px) and projected to 512 numbers, "
                "then divided by their length so ‖e‖ = 1.")
    if not st.toggle("Run CLIP live on this window (loads 600 MB once, ~15 s)", value=False):
        st.info("Turn on the switch to compute steps 2–4 live. The cached probabilities are shown in step 5.")
        _quality(name, data, tcache, split, i)
        return
    emb = core.embedder()
    e = emb.encode_images(np.stack([heat, spec]))           # (2, 512)
    cached = [tcache[f"emb_{v}"][split][i] for v in ("heatmap", "spectrogram")]
    cos_cache = [float(e[k] @ cached[k]) for k in range(2)]
    st.dataframe(pd.DataFrame({
        "view": ["heatmap", "spectrogram"], "‖e‖": np.linalg.norm(e, axis=1),
        "first 6 numbers of e": [np.round(e[k, :6], 4).tolist() for k in range(2)],
        "cos(live e, cached e used in training)": cos_cache}), use_container_width=True, hide_index=True)
    (st.success if min(cos_cache) > 0.999 else st.warning)(
        f"Check: the live embeddings match the cached ones (cosine ≥ {min(cos_cache):.5f}).")
    st.caption(f"cos(heatmap, spectrogram) for this window = {float(e[0] @ e[1]):.3f}: related but different views.")

    scale = emb.logit_scale()
    from src.teacher.clip_teacher import ANOMALY_PROMPTS, NORMAL_PROMPTS

    st.subheader("Step 3 · Zero-shot: hand-written prompts (the starting point, weak)")
    tn, ta = emb.encode_texts(NORMAL_PROMPTS), emb.encode_texts(ANOMALY_PROMPTS)
    st.dataframe(pd.DataFrame({"prompt": NORMAL_PROMPTS + ANOMALY_PROMPTS,
                               "class": ["normal"] * 4 + ["anomalous"] * 4,
                               "cos(e_heatmap, t)": np.r_[tn @ e[0], ta @ e[0]]}).style.format(precision=4),
                 use_container_width=True, hide_index=True)
    t_class = np.stack([tn.mean(0) / np.linalg.norm(tn.mean(0)), ta.mean(0) / np.linalg.norm(ta.mean(0))])
    zl = scale * t_class @ e[0]
    zp = _softmax(zl)
    st.latex(rf"t_c=\frac{{\mathrm{{mean}}(t_{{c,1..4}})}}{{\|\cdot\|}},\quad z_c = {scale:.0f}\cdot\cos(e,t_c) = "
             rf"[{zl[0]:.3f},\ {zl[1]:.3f}],\quad p = \mathrm{{softmax}}(z) = [{zp[0]:.4f},\ {zp[1]:.4f}]")
    st.caption(f"Cached zero-shot probability for this window: {tcache['zeroshot'][split][i][1]:.4f}.")

    st.subheader("Step 4 · Learned prompts (final teacher)")
    enc = _prompt_encoder(name)
    with torch.no_grad():
        t = enc.text_features().numpy()                      # (2, 512)
    words = (tcache["info"].get("coop_mv") or {}).get("nearest_words")
    st.markdown("Prompt = `[SOS] v₁ … v₁₄ normal|anomalous . [EOS]`. The 14 vectors v were initialised from "
                "*“a heatmap of a multivariate sensor time series that looks”* and trained with cross-entropy "
                "through the frozen text encoder." + (f" Closest vocabulary words to the learned vectors now: "
                                                      f"`{' '.join(words)}`." if words else ""))
    cos = e @ t.T                                            # (views, classes)
    logits_v = scale * cos
    logits = logits_v.mean(0)
    p = _softmax(logits)
    st.dataframe(pd.DataFrame({"view": ["heatmap", "spectrogram"],
                               "cos(e, t_normal)": cos[:, 0], "cos(e, t_anomalous)": cos[:, 1],
                               "z_normal = 100·cos": logits_v[:, 0], "z_anomalous = 100·cos": logits_v[:, 1]})
                 .style.format(precision=4), use_container_width=True, hide_index=True)
    st.latex(rf"z = \mathrm{{mean}}_{{views}} = [{logits[0]:.3f},\ {logits[1]:.3f}],\qquad "
             rf"p_T = \mathrm{{softmax}}(z) = [{p[0]:.4f},\ {p[1]:.4f}]")
    cached_p = float(tcache["coop_mv"][split][i][1])
    (st.success if abs(cached_p - p[1]) < 1e-3 else st.warning)(
        f"Check: live p_T(anomalous) = {p[1]:.4f}; cached value used for evaluation = {cached_p:.4f}.")
    _quality(name, data, tcache, split, i)


def _quality(name, data, tcache, split, i) -> None:
    st.subheader("Step 5 · Soft labels for training windows are cross-fitted")
    tr = data.splits["train"]
    block, folds = int(tcache["identity"]["block"]), int(tcache["identity"]["folds"])
    n_show = min(len(tcache["train_idx"]), block * folds * 3)
    fold_of = (np.arange(n_show) // block) % folds
    fig = go.Figure(go.Heatmap(z=fold_of[None, :], colorscale="Viridis", showscale=False,
                               hovertemplate="scored window %{x}: fold %{z}<extra></extra>"))
    fig.update_layout(height=120, margin=dict(t=30, b=20, l=20, r=20), yaxis=dict(showticklabels=False),
                      title=f"Fold of each scored training window: blocks of {block}, assigned round-robin to {folds} folds")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"The teacher fitted on four folds labels the fifth, so no training window's soft label comes from "
                f"prompts that saw its own label. {int(tcache['mask'].sum()):,} of {len(tr.X):,} training windows "
                "are scored; the rest get no teacher term in the loss.")

    st.subheader("Step 6 · How good is each teacher? (window AUPRC, prevalence = random)")
    y_sub = tr.y_window[tcache["train_idx"]]
    rows = []
    for key, label in TEACHER_NAMES.items():
        r = {"teacher": label,
             "train (out-of-fold)": average_precision_score(y_sub, tcache[key]["train_sub"][:, 1])}
        for s in ("val", "test"):
            y = data.splits[s].y_window
            r[s] = average_precision_score(y, tcache[key][s][:, 1])
            r[f"{s} ROC-AUC"] = roc_auc_score(y, tcache[key][s][:, 1])
        rows.append(r)
    prev = {"teacher": "Random detector (= prevalence)", "train (out-of-fold)": y_sub.mean(),
            "val": data.splits["val"].y_window.mean(), "test": data.splits["test"].y_window.mean(),
            "val ROC-AUC": 0.5, "test ROC-AUC": 0.5}
    st.dataframe(pd.DataFrame(rows + [prev]).style.format(precision=3), use_container_width=True, hide_index=True)
    st.caption("AUPRC: area under the precision–recall curve of the window probability (1 = perfect ranking of "
               "anomalous above normal windows, prevalence = random). ROC-AUC: probability that a random anomalous "
               "window scores above a random normal one (0.5 = random).")
