"""Page: test metrics computed step by step, with every count visible.

Uses ``src.evaluation.postprocess.evaluate`` — the function that produced the
benchmark tables — and then re-derives each of its numbers in the open:
validation threshold, confusion counts, precision/recall/F1, AUPRC, the
localization settings chosen on validation, point metrics and event matching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.app import core
from src.evaluation import postprocess as pp


def _counts(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def _prf_block(y, score, pred, what: str) -> dict:
    tp, fp, fn, tn = _counts(y, pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    ap = average_precision_score(y, score)
    c = st.columns([1.1, 1.4])
    with c[0]:
        st.dataframe(pd.DataFrame([[tp, fn], [fp, tn]], index=["true anomalous", "true normal"],
                                  columns=["flagged", "not flagged"]), use_container_width=True)
        st.caption(f"{what}: {len(y):,} in total, {int(np.sum(y)):,} anomalous (prevalence {np.mean(y):.4f}).")
    with c[1]:
        st.latex(rf"\text{{precision}} = \frac{{TP}}{{TP+FP}} = \frac{{{tp}}}{{{tp}+{fp}}} = {p:.4f}")
        st.latex(rf"\text{{recall}} = \frac{{TP}}{{TP+FN}} = \frac{{{tp}}}{{{tp}+{fn}}} = {r:.4f}")
        st.latex(rf"F_1 = \frac{{2PR}}{{P+R}} = {f1:.4f}\qquad \text{{AUPRC}} = {ap:.4f}\ \ "
                 rf"(\text{{random}} = {np.mean(y):.4f})")
    return {"precision": p, "recall": r, "f1": f1, "auprc": ap}


def _pr_plot(y, score, thr, title):
    prec, rec, th = precision_recall_curve(y, score)
    fig = go.Figure(go.Scatter(x=rec, y=prec, mode="lines", name="PR curve", line=dict(color="#B91C1C")))
    fig.add_hline(y=float(np.mean(y)), line=dict(color="#94A3B8", dash="dash"),
                  annotation_text=f"random = prevalence {np.mean(y):.3f}")
    pred = score >= thr
    tp = (pred & (y == 1)).sum()
    if pred.sum():
        fig.add_trace(go.Scatter(x=[tp / max(1, y.sum())], y=[tp / pred.sum()], mode="markers",
                                 marker=dict(size=11, color="#0F172A"), name=f"operating point (τ = {thr:.4g})"))
    fig.update_layout(height=320, template="plotly_white", title=title, xaxis_title="recall", yaxis_title="precision",
                      margin=dict(t=40, b=20, l=20, r=20), xaxis=dict(range=[0, 1.02]), yaxis=dict(range=[0, 1.02]))
    return fig


def _model_choices(name: str) -> dict:
    out = {"teacher": core.MODEL_LABELS["teacher"]}
    for key in ("gru_gt", "gru_kd_coop"):
        if core.checkpoint(name, key) is not None:
            out[key] = core.MODEL_LABELS[key] + " (deployed)"
    for label, entry in st.session_state.get("live_models", {}).items():
        if entry["dataset"] == name:
            out[label] = label
    return out


@st.cache_data(show_spinner="Scoring validation and test windows ...", max_entries=16)
def _scores(name: str, key: str):
    data = core.prepared(name)
    if key == "teacher":
        t = core.teacher(name)
        return t["coop_mv"]["val"][:, 1], t["coop_mv"]["test"][:, 1], None, None
    model = core.checkpoint(name, key)[0]
    vw, vs = core.predict(model, data.splits["val"].X)
    tw, ts = core.predict(model, data.splits["test"].X)
    return vw, tw, vs, ts


def show() -> None:
    name = core.selected_dataset()
    data = core.prepared(name)
    L = data.window_length
    st.header(f"Evaluation: every metric, step by step — {core.PRETTY[name]}")
    st.markdown("Protocol: **every threshold and post-processing setting is chosen on the validation split**; the "
                "test split is then scored once with those settings. Metrics are reported at three levels: "
                "**window** (is the window anomalous?), **point** (is each timestep anomalous?) and **event** "
                "(was each anomalous interval found?). No point-adjustment is used.")
    choices = _model_choices(name)
    key = st.selectbox("Model", list(choices), format_func=lambda k: choices[k], index=min(2, len(choices) - 1),
                       key=f"eval_model_{name}")
    val, test = data.splits["val"], data.splits["test"]
    if key in st.session_state.get("live_models", {}):
        m = st.session_state["live_models"][key]["model"]
        vw, vs = core.predict(m, val.X)
        tw, ts = core.predict(m, test.X)
    else:
        vw, tw, vs, ts = _scores(name, key)
    res = pp.evaluate(val, test, L, vw, tw, vs, ts)
    has_step = vs is not None

    # ================================================================ window
    st.subheader("A · Window level")
    n_val_events = sum(len(pp.runs(np.asarray(s["labels"]))) for s in val.sequences)
    thr_w = res["window"]["threshold"]
    if n_val_events >= pp.MIN_VAL_EVENTS:
        rule = (f"validation has {n_val_events} anomalous events (≥ {pp.MIN_VAL_EVENTS}), so τ is the window score "
                "that maximises F1 on the validation windows")
    else:
        rule = (f"validation has only {n_val_events} anomalous event(s) (< {pp.MIN_VAL_EVENTS}): an F1-optimal "
                "threshold fitted to so few events is unstable, so τ is the 99.9th percentile of the scores of "
                "the *normal* validation windows (0.1 % false alarms)")
    st.markdown(f"**Score:** {'teacher probability p_T(anomalous)' if key == 'teacher' else 'window head p_win(anomalous)'}"
                f" per window. **Threshold τ = {thr_w:.5g}**: {rule}. A window is flagged when score ≥ τ.")
    yw = test.y_window.astype(int)
    _prf_block(yw, tw, tw >= thr_w, "Test windows")
    c1, c2 = st.columns(2)
    c1.plotly_chart(_pr_plot(yw, tw, thr_w, "Window precision–recall curve (test)"), use_container_width=True)
    hist = go.Figure()
    for lab, col, sel in (("normal windows", "#64748B", yw == 0), ("anomalous windows", "#B91C1C", yw == 1)):
        hist.add_trace(go.Histogram(x=tw[sel], name=lab, marker_color=col, opacity=0.6, nbinsx=50,
                                    histnorm="probability"))
    hist.add_vline(x=thr_w, line=dict(color="#0F172A", dash="dash"), annotation_text="τ")
    hist.update_layout(barmode="overlay", height=320, template="plotly_white", margin=dict(t=40, b=20, l=20, r=20),
                       title="Test window scores by true class", xaxis_title="score")
    c2.plotly_chart(hist, use_container_width=True)
    st.caption(f"ROC-AUC = {roc_auc_score(yw, tw):.4f}. AUPRC = Σₖ (Rₖ − Rₖ₋₁)·Pₖ over all score cut-offs "
               "(scikit-learn's average precision): it ranks windows by score and needs no threshold.")

    # ================================================================ point
    st.subheader("B · Point level: from window outputs to one score per timestep")
    val_tl = pp.build_timelines(val.sequences, L, vs, vw)
    test_tl = pp.build_timelines(test.sequences, L, ts, tw)
    prm = res["params"]
    st.markdown(
        "1. **Aggregate:** every timestep is covered by several overlapping windows; its score is the mean of "
        + ("their step-head scores" if has_step else "their window scores (no step head)")
        + ", and p̄ₜ is the mean window probability.\n"
        "2. **Gate:** score = sₜ · p̄ₜ^α — the window head (which receives the teacher's knowledge) can suppress "
        "isolated step activations.\n3. **Smooth:** centred moving average of k steps.\n"
        "4. **Threshold** on validation (same rule as above, at timestep level).\n"
        "5. **Minimum length:** flagged runs shorter than m steps are removed (m maximises validation event F1).")
    yv = np.concatenate([t["labels"] for t in val_tl])
    alphas = (0.0, 0.5, 1.0) if has_step else (0.0,)
    ks = sorted({1, max(1, L // 16), max(1, L // 8), max(1, L // 4)})
    grid = pd.DataFrame([[average_precision_score(yv, np.concatenate([pp._score(t, a, k) for t in val_tl]))
                          for k in ks] for a in alphas], index=[f"α = {a:g}" for a in alphas],
                        columns=[f"k = {k}" for k in ks])
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("Validation point AUPRC for each (α, k) — the best cell is used:")
        st.dataframe(grid.style.format(precision=4).highlight_max(axis=None, color="#DCFCE7"),
                     use_container_width=True)
    val_scores = [pp._score(t, prm.alpha, prm.k) for t in val_tl]
    lens = sorted({1, 2, max(1, L // 16), max(1, L // 8), max(1, L // 4)})
    ml = pd.DataFrame({"m": lens, "validation event F1": [
        pp.event_scores([t["labels"] for t in val_tl],
                        [pp.remove_short((s >= prm.threshold).astype(np.int8), m) for s in val_scores])["event_f1"]
        for m in lens]})
    with c2:
        st.markdown(f"Chosen: **α = {prm.alpha:g}, k = {prm.k}, τ = {prm.threshold:.5g}, m = {prm.min_len}**")
        st.dataframe(ml.style.format(precision=4).highlight_max(subset=["validation event F1"], color="#DCFCE7"),
                     use_container_width=True, hide_index=True)
    y_pt = np.concatenate([t["labels"] for t in test_tl])
    s_pt = np.concatenate(res["test_scores"])
    p_pt = np.concatenate(res["test_preds"])
    _prf_block(y_pt, s_pt, p_pt, "Test timesteps")

    # ================================================================ event
    st.subheader("C · Event level: was each anomalous interval found?")
    st.markdown("An **event** is a maximal run of consecutive anomalous timesteps. A true event counts as detected "
                "if at least one of its timesteps is flagged; a predicted event is correct if it overlaps at least "
                "one true anomalous timestep.")
    true_rows, pred_rows = [], []
    for sq, t, pr in zip(test.sequences, test_tl, res["test_preds"]):
        cum_p, cum_y = np.r_[0, np.cumsum(pr)], np.r_[0, np.cumsum(t["labels"])]
        for a, b in pp.runs(t["labels"]):
            true_rows.append({"sequence": sq["group_id"], "start": a, "end": b - 1, "length": b - a,
                              "flagged steps inside": int(cum_p[b] - cum_p[a]), "detected": cum_p[b] > cum_p[a]})
        for a, b in pp.runs(pr):
            pred_rows.append({"sequence": sq["group_id"], "start": a, "end": b - 1, "length": b - a,
                              "true anomalous steps inside": int(cum_y[b] - cum_y[a]), "correct": cum_y[b] > cum_y[a]})
    ev = res["event"]
    nt, nd = len(true_rows), sum(r["detected"] for r in true_rows)
    npd, nc = len(pred_rows), sum(r["correct"] for r in pred_rows)
    st.latex(rf"\text{{event recall}} = \frac{{\text{{detected true events}}}}{{\text{{true events}}}} = "
             rf"\frac{{{nd}}}{{{nt}}} = {ev['event_recall']:.4f},\quad "
             rf"\text{{event precision}} = \frac{{\text{{correct predicted events}}}}{{\text{{predicted events}}}} = "
             rf"\frac{{{nc}}}{{{npd}}} = {ev['event_precision']:.4f},\quad F_1 = {ev['event_f1']:.4f}")
    c1, c2 = st.columns(2)
    c1.markdown("True events (test)")
    c1.dataframe(pd.DataFrame(true_rows), use_container_width=True, hide_index=True, height=260)
    c2.markdown("Predicted events (test)")
    c2.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True, height=260)

    # ================================================================ timeline
    st.subheader("D · Look at one test sequence")
    j = st.selectbox("Sequence", range(len(test.sequences)),
                     format_func=lambda q: f"{test.sequences[q]['group_id']} "
                                           f"({int(np.sum(test.sequences[q]['labels']))} anomalous steps)",
                     index=int(np.argmax([np.sum(s["labels"]) for s in test.sequences])))
    t, pr, sc = test_tl[j], res["test_preds"][j], res["test_scores"][j]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sc, name="final timestep score", line=dict(color="#B91C1C", width=1)))
    fig.add_trace(go.Scatter(y=t["labels"] * float(max(sc.max(), prm.threshold)), name="true anomaly",
                             line=dict(color="#94A3B8", shape="hv"), fill="tozeroy", opacity=0.3))
    fig.add_trace(go.Scatter(y=np.where(pr > 0, -0.05 * float(max(sc.max(), prm.threshold)), np.nan),
                             name="flagged", mode="lines", line=dict(color="#0F172A", width=6)))
    fig.add_hline(y=prm.threshold, line=dict(color="#0F172A", dash="dot"), annotation_text="τ")
    fig.update_layout(height=320, template="plotly_white", margin=dict(t=30, b=20, l=20, r=20),
                      xaxis_title="timestep", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

    # ================================================================ consistency
    st.subheader("E · Check against the benchmark files")
    if key in ("teacher", "gru_gt", "gru_kd_coop"):
        if key == "teacher":
            rec, src = core.benchmark_record(name, "clip_coop_mv", "-"), "base"
        else:
            meta = core.checkpoint(name, key)[1]
            seed = (meta.training_result or {}).get("seed")
            src = core.deployed_config(meta)
            rec = core.benchmark_record(name, key, seed, src)
        if rec is None:
            st.info("No matching benchmark row.")
        else:
            rows = []
            for level, metric, mine in (("window", "auprc", res["window"]["auprc"]),
                                        ("window", "f1", res["window"]["f1"]),
                                        ("point", "auprc", res["point"]["auprc"]),
                                        ("point", "f1", res["point"]["f1"]),
                                        ("event", "event_f1", res["event"]["event_f1"])):
                bench = float(rec[rec.level == level][metric].iloc[0])
                rows.append({"level": level, "metric": metric, "this page": mine, "benchmark file": bench,
                             "match": "✓" if abs(mine - bench) < 1e-6 else "✗"})
            st.dataframe(pd.DataFrame(rows).style.format(precision=6, subset=["this page", "benchmark file"]),
                         use_container_width=True, hide_index=True)
            st.caption(f"Source: {'experiments/benchmark_v2_regularized' if src == 'regularized' else 'experiments/benchmark_v2'}"
                       f"/{name}/metrics.csv" + ("" if key == "teacher" else f", seed of the deployed checkpoint"))
    else:
        st.caption("Live-trained model: no benchmark row to compare with.")
