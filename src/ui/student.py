"""Page: the GRU student — layers, heads, the loss computed by hand on a real
batch, the recorded training runs, and a live training run."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots

from src.app import core, diagrams
from src.student.losses import TotalLoss

COMP = {"loc_losses": ("Step BCE (step head vs y_step)", "#0EA5E9"),
        "window_losses": ("Window CE (window head vs y_win)", "#B91C1C"),
        "kd_losses": ("KD term (window head vs teacher)", "#D97706"),
        "feat_losses": ("Feature term (embedding head vs CLIP)", "#15803D")}


def _layer_table(model, X1) -> pd.DataFrame:
    rows, hooks = [], []

    def hook(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            rows.append({"layer": name, "type": type(mod).__name__, "output shape": tuple(o.shape),
                         "parameters": sum(p.numel() for p in mod.parameters(recurse=False))})
        return f

    for n, m in model.named_modules():
        if n and not list(m.children()):
            hooks.append(m.register_forward_hook(hook(n)))
        elif n == "gru":
            hooks.append(m.register_forward_hook(hook(n)))
    with torch.no_grad():
        model(X1, return_embedding=True)
    for h in hooks:
        h.remove()
    return pd.DataFrame(rows)


def _curves(h: dict, title: str, best: int | None = None, height: int = 330) -> go.Figure:
    ep = list(range(1, len(h["train_losses"]) + 1))
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Loss terms (training, per epoch mean)",
                                                        "Total loss: training vs validation",
                                                        "Validation AUPRC (selection metric)"))
    for k, (lab, col) in COMP.items():
        if any(v > 0 for v in h.get(k, [])):
            fig.add_trace(go.Scatter(x=ep, y=h[k], name=lab, line=dict(color=col)), 1, 1)
    fig.add_trace(go.Scatter(x=ep, y=h["train_losses"], name="train total", line=dict(color="#334155")), 1, 2)
    fig.add_trace(go.Scatter(x=ep, y=h["val_losses"], name="validation (labels-only terms)",
                             line=dict(color="#334155", dash="dot")), 1, 2)
    fig.add_trace(go.Scatter(x=ep, y=h["val_auprc"], name="val AUPRC", line=dict(color="#6D28D9")), 1, 3)
    if best:
        for c in (1, 2, 3):
            fig.add_vline(x=best, line=dict(color="#94A3B8", dash="dash"), row=1, col=c)
    fig.update_layout(height=height, template="plotly_white", title=title, margin=dict(t=60, b=20, l=20, r=20),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title_text="epoch")
    return fig


def show() -> None:
    name = core.selected_dataset()
    data = core.prepared(name)
    tcache = core.teacher(name)
    st.header(f"Student: convolutional bidirectional GRU — {core.PRETTY[name]}")
    st.markdown("The student reads the standardised numbers of a window directly (no images) and answers two "
                "questions: *is this window anomalous?* (window head) and *which timesteps are?* (step head). "
                "A third head, used only during training, learns to predict CLIP's embedding of the window.")
    st.graphviz_chart(diagrams.student_heads(), use_container_width=True)

    ck = core.checkpoint(name, "gru_kd_coop")
    if ck is None:
        st.error("No deployed student for this dataset.")
        return
    model, meta = ck
    tr = data.splits["train"]

    # ------------------------------------------------------------ layers
    st.subheader("1 · Layers, shapes and parameters (deployed proposed student)")
    X1 = torch.as_tensor(data.splits["test"].X[:1])
    table = _layer_table(model, X1)
    st.dataframe(table, use_container_width=True, hide_index=True)
    n_all = sum(p.numel() for p in model.parameters())
    n_inf = model.parameter_count()
    st.caption(f"Input shape (1, {data.window_length}, {len(data.feature_names)}). Parameters: {n_inf:,} used at "
               f"inference + {n_all - n_inf:,} in the training-only embedding head. CLIP ViT-B/32 has ≈151 million "
               f"(≈{151e6 / n_inf:,.0f}× more). Shapes are (batch, channels or steps, …); the GRU outputs "
               "(batch, steps, 2 × 64).")

    # ------------------------------------------------------------ one window
    st.subheader("2 · What each head outputs for one window")
    te = data.splits["test"]
    pos = np.flatnonzero(te.y_window == 1)
    i = st.number_input(f"Test window (0 … {len(te.X) - 1}); default = first anomalous", 0, len(te.X) - 1,
                        int(pos[0]) if len(pos) else 0)
    with torch.no_grad():
        zw, zs, g = model(torch.as_tensor(te.X[i:i + 1]), return_embedding=True)
    pw = F.softmax(zw, -1)[0].numpy()
    s = torch.sigmoid(zs)[0].numpy()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Window head**")
        st.latex(rf"z_{{win}} = [{zw[0, 0]:.3f},\ {zw[0, 1]:.3f}]")
        st.latex(rf"p_{{win}} = \mathrm{{softmax}}(z_{{win}}) = [{pw[0]:.4f},\ {pw[1]:.4f}]")
        st.caption(f"True window label y_win = {int(te.y_window[i])}. p_win[1] is the window anomaly score.")
        if tcache is not None and "emb_heatmap" in tcache:
            e = tcache["emb_heatmap"]["test"][i] + tcache["emb_spectrogram"]["test"][i]
            e = e / np.linalg.norm(e)
            gn = g[0].numpy() / np.linalg.norm(g[0].numpy())
            st.markdown("**Embedding head** (training only)")
            st.latex(rf"\cos(g, e_{{CLIP}}) = {float(gn @ e):.3f}")
            st.caption("How close the student's guess of CLIP's view-averaged embedding is for this unseen window.")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=s, name="step head s_t = sigmoid(z_t)", line=dict(color="#B91C1C")))
        fig.add_trace(go.Scatter(y=te.y_step[i], name="true y_step", line=dict(color="#94A3B8", shape="hv")))
        fig.update_layout(height=260, template="plotly_white", margin=dict(t=30, b=20, l=20, r=20),
                          title="Step head: one score per timestep", xaxis_title="step in window",
                          legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------ loss by hand
    st.subheader("3 · The training loss, computed by hand on a real training batch")
    setup = core.training_setup(name, "gru_kd_coop")
    d = setup["distillation"]
    T, beta, lkd, lfeat, pw_ = d["temperature"], d["beta"], d["lambda_kd"], d["lambda_feat"], d["pos_weight"]
    st.latex(r"\mathcal{L} = \underbrace{\mathrm{BCE}_{w}(z_s, y_{step})}_{\text{step head}} + \beta\,"
             r"\underbrace{\mathrm{CE}(z_{win}, y_{win})}_{\text{window head}} + \lambda_{KD}\,"
             r"\underbrace{\tfrac{1}{n}\textstyle\sum_i w_i T^2 \mathrm{KL}(p_{T,i}^{(T)}\Vert q_i^{(T)})}_{\text{window head}}"
             r" + \lambda_{feat}\underbrace{(1-\cos(g,e))}_{\text{embedding head}}")
    st.markdown(f"Settings: β = {beta:g}, λ_KD = {lkd:g}, λ_feat = {lfeat:g}, T = {T:g}, "
                f"step positive weight w = clip(√(n_neg/n_pos), 1, 25) = **{pw_:.3f}** "
                f"(training step prevalence {tr.y_step.mean():.4f}); "
                "w_i = p_T(true class of window i), p^(T) = softmax(log p / T), q^(T) = softmax(z_win / T).")
    if tcache is None:
        st.info("Teacher outputs missing; cannot form the KD terms.")
    else:
        scored = np.flatnonzero(tcache["mask"])
        pos_s = scored[tr.y_window[scored] == 1][:16]
        batch = np.r_[pos_s, scored[tr.y_window[scored] == 0][:128 - len(pos_s)]]
        Xb = torch.as_tensor(tr.X[batch])
        ys = torch.as_tensor(tr.y_step[batch], dtype=torch.float32)
        yw = torch.as_tensor(tr.y_window[batch], dtype=torch.long)
        pT = torch.as_tensor(tcache["coop_mv"]["train"][batch])
        eT = torch.as_tensor(tcache["emb_train_mv"][batch])
        model.eval()
        with torch.no_grad():
            zw, zs, g = model(Xb, return_embedding=True)
            sig = torch.sigmoid(zs)
            bce = -(pw_ * ys * torch.log(sig.clamp_min(1e-12)) + (1 - ys) * torch.log((1 - sig).clamp_min(1e-12)))
            l_step = bce.mean()
            logp = F.log_softmax(zw, -1)
            ce_i = -logp.gather(1, yw[:, None])[:, 0]
            l_win = ce_i.mean()
            pT_soft = F.softmax(torch.log(pT.clamp_min(1e-8)) / T, -1)
            q_soft = F.softmax(zw / T, -1)
            kl_i = (pT_soft * (torch.log(pT_soft.clamp_min(1e-8)) - torch.log(q_soft.clamp_min(1e-12)))).sum(-1)
            w_i = pT.gather(1, yw[:, None])[:, 0]
            l_kd = T * T * (w_i * kl_i).mean()
            l_feat = (1 - (F.normalize(g, dim=-1) * F.normalize(eT, dim=-1)).sum(-1)).mean()
            total = l_step + beta * l_win + lkd * l_kd + lfeat * l_feat
            ref, parts = TotalLoss(beta=beta, lambda_kd=lkd, pos_weight=pw_, temperature=T, lambda_feat=lfeat,
                                   kd_confidence=True)(zs, zw, ys, yw, pT, return_dict=True,
                                                        student_embedding=g, teacher_embedding=eT)
        k = min(8, len(batch))
        st.markdown(f"Batch: {len(batch)} teacher-scored training windows ({len(pos_s)} anomalous). "
                    f"Per-window terms for the first {k}:")
        st.dataframe(pd.DataFrame({
            "y_win": yw[:k].numpy(), "teacher p_T(anom)": pT[:k, 1].numpy(),
            "student p_win(anom)": F.softmax(zw[:k], -1)[:, 1].numpy(),
            "CE_i = −log p_win(y)": ce_i[:k].numpy(), "w_i = p_T(y)": w_i[:k].numpy(),
            "KL_i at T": kl_i[:k].numpy(), "mean step BCE_i": bce[:k].mean(1).numpy(),
            "1 − cos(g_i, e_i)": (1 - (F.normalize(g[:k], dim=-1) * F.normalize(eT[:k], dim=-1)).sum(-1)).numpy(),
        }).style.format(precision=4), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame([
            {"term": "step BCE (mean over batch × steps)", "value": float(l_step), "weight": 1.0},
            {"term": "window CE (mean)", "value": float(l_win), "weight": beta},
            {"term": "KD = T² · mean(w_i · KL_i)", "value": float(l_kd), "weight": lkd},
            {"term": "feature = mean(1 − cos)", "value": float(l_feat), "weight": lfeat},
        ]).assign(**{"weighted": lambda df: df.value * df.weight}).style.format(precision=4),
            use_container_width=True, hide_index=True)
        ok = abs(float(total) - float(ref)) < 1e-4
        (st.success if ok else st.error)(
            f"Total by hand = {float(total):.5f}; training code (`TotalLoss`) = {float(ref):.5f} → "
            f"{'identical' if ok else 'MISMATCH'}. (Computed in evaluation mode, i.e. without dropout.)")
        st.caption("For the labels-only GRU the same formula is used with λ_KD = λ_feat = 0. The validation loss "
                   "always uses only the two label terms, so both models are selected by the same criterion.")

    # ------------------------------------------------------------ recorded runs
    st.subheader("4 · Recorded training runs (the benchmark)")
    st.markdown("Optimiser AdamW (learning rate 2·10⁻³, weight decay 10⁻⁴), batch 128, at most 20 epochs, "
                "learning rate halved when validation AUPRC stalls, gradient norm clipped at 1. After every epoch "
                "the **validation AUPRC** (mean of window and step AUPRC) is computed; the epoch with the best "
                "value is kept and training stops after 5 epochs without improvement (dashed line). "
                "Regularized runs add weight averaging (EMA 0.99), random gain/noise augmentation, learning rate "
                "10⁻³, dropout 0.3 and patience 6.")
    runs_avail = {}
    for cfg_name, base in (("base", core.BENCH), ("regularized", core.REG)):
        p = base / name / "training_histories.json"
        if p.exists():
            runs_avail[cfg_name] = json.loads(p.read_text())
    c1, c2, c3 = st.columns(3)
    cfg_name = c1.selectbox("Configuration", list(runs_avail), index=0)
    variant = c2.selectbox("Model", list(core.VARIANTS), format_func=lambda v: core.VARIANTS[v].split(" (")[0])
    seeds = sorted(int(k.rsplit("_seed", 1)[1]) for k in runs_avail[cfg_name] if k.rsplit("_seed", 1)[0] == variant)
    seed = c3.selectbox("Seed", seeds)
    h = runs_avail[cfg_name][f"{variant}_seed{seed}"]
    st.plotly_chart(_curves(h, f"{core.VARIANTS[variant].split(' (')[0]}, {cfg_name}, seed {seed}: best epoch "
                               f"{h['best_epoch']} (val AUPRC {np.nanmax(h['val_auprc']):.3f})", h["best_epoch"]),
                    use_container_width=True)
    with st.expander("Per-epoch numbers"):
        cols = {k: v for k, v in h.items() if isinstance(v, list) and len(v) == len(h["train_losses"])}
        st.dataframe(pd.DataFrame(cols, index=pd.RangeIndex(1, len(h["train_losses"]) + 1, name="epoch"))
                     .style.format(precision=4), use_container_width=True)

    # ------------------------------------------------------------ live training
    st.subheader("5 · Train a student live")
    st.markdown("Runs the same code as the benchmark on this dataset and redraws the curves after every epoch. "
                "The result is kept for this session and can be evaluated on the Evaluation page.")
    with st.form("live_train"):
        c1, c2, c3, c4 = st.columns(4)
        v = c1.selectbox("Model", list(core.VARIANTS), index=1, format_func=lambda x: core.VARIANTS[x].split(" (")[0])
        cn = c2.selectbox("Configuration", ["base", "regularized"])
        sd = c3.number_input("Seed", 0, 99, 0)
        ep = c4.number_input("Max epochs", 1, 40, 10)
        go_train = st.form_submit_button("Train", type="primary")
    if go_train:
        bar = st.progress(0.0, text="starting ...")
        chart = st.empty()
        table = st.empty()
        t0 = time.time()

        def on_epoch(epoch, res):
            hd = res.to_dict()
            bar.progress(epoch / ep, text=f"epoch {epoch}/{ep} · {time.time() - t0:.0f} s · "
                                          f"val AUPRC {hd['val_auprc'][-1]:.3f} · best epoch {res.best_epoch}")
            chart.plotly_chart(_curves(hd, "Live training", res.best_epoch, 300), use_container_width=True,
                               key=f"live_{epoch}")
            table.dataframe(pd.DataFrame({k: hd[k] for k in ("train_losses", "loc_losses", "window_losses",
                                                             "kd_losses", "feat_losses", "val_losses", "val_auprc")},
                                         index=pd.RangeIndex(1, epoch + 1, name="epoch")).style.format(precision=4),
                            use_container_width=True)

        model_l, hist, setup = core.train_live(name, v, cn, int(sd), int(ep), on_epoch)
        bar.progress(1.0, text=f"done in {time.time() - t0:.0f} s; kept epoch {hist.best_epoch} "
                               f"(val AUPRC {np.nanmax(hist.val_auprc):.3f})")
        label = f"Live: {core.VARIANTS[v].split(' (')[0]}, {cn}, seed {sd}"
        st.session_state.setdefault("live_models", {})[label] = {"dataset": name, "model": model_l,
                                                                  "history": hist.to_dict(), "setup": setup}
        st.success(f"Stored as “{label}”. Open the Evaluation page to compute its test metrics step by step.")
