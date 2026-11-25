"""Page 8: teacher vs. distilled student on an uploaded multivariate CSV.

Uses the frozen-CLIP teacher (learned prompts, heatmap + spectrogram views) and the GRU students produced by the v2
benchmark. The user uploads a CSV (or picks a held-out demo sequence), sees the
input plotted, and sees the anomalies found by the teacher and by the student
with colour codes, together with performance metrics for both.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.evaluation.postprocess import runs
from src.inference import dual

COLORS = {"truth": "#94A3B8", "teacher": "#F59E0B", "student": "#EF4444", "baseline": "#6366F1",
          "gru_kd_full": "#EF4444", "gru_kd_coop": "#EF4444", "gru_gt": "#6366F1"}
MODEL_NAMES = {
    "teacher": "Teacher: frozen CLIP + learned prompts",
    "gru_kd_coop": "Student: GRU distilled from CLIP",
    "gru_kd_full": "Student: GRU distilled from CLIP",
    "gru_gt": "Reference: GRU trained on labels only",
}


@st.cache_resource(show_spinner=False)
def _bundle(name: str):
    return dual.load_bundle(name)


@st.cache_resource(show_spinner="Loading frozen CLIP ViT-B/32 ...")
def _embedder():
    from src.teacher.clip_teacher import CLIPEmbedder

    return CLIPEmbedder().load()


def _results_table(name: str) -> None:
    path = dual.BENCH / "summary.csv"
    if not path.exists():
        st.info("Run `scripts/run_benchmark.py` and `scripts/finalize_benchmark.py` to populate results.")
        return
    df = pd.read_csv(path)
    df = df[df.dataset == name]
    rows = []
    for (model, label), g in df.groupby(["model", "label"], sort=False):
        def cell(level, metric):
            r = g[g.level == level]
            if r.empty or pd.isna(r[f"{metric}_mean"].iloc[0]):
                return "–"
            m, s = r[f"{metric}_mean"].iloc[0], r[f"{metric}_std"].iloc[0]
            return f"{m:.3f}" + (f" ± {s:.3f}" if pd.notna(s) and s > 0 else "")
        rows.append({"Model": label, "Window AUPRC": cell("window", "auprc"),
                     "Point AUPRC": cell("point", "auprc"), "Point F1": cell("point", "f1"),
                     "Event F1": cell("event", "event_f1")})
    prev = df[(df.level == "point")]["prevalence_mean"].dropna()
    wprev = df[(df.level == "window")]["prevalence_mean"].dropna()
    st.caption(
        f"Held-out test split. Anomaly prevalence: windows {wprev.iloc[0]:.3f}, timesteps {prev.iloc[0]:.4f} "
        "(the AUPRC of a random detector equals the prevalence). Students: mean ± std over 3 seeds."
        if len(prev) and len(wprev) else "Held-out test split."
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _training_curves(name: str) -> None:
    path = dual.BENCH / name / "training_histories.json"
    if not path.exists():
        return
    hist = json.loads(path.read_text())
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Training loss", "Validation AUPRC", "KD / feature loss (KD student)"))
    student_hist = "gru_kd_coop_seed0" if "gru_kd_coop_seed0" in hist else "gru_kd_full_seed0"
    for key, color in (("gru_gt_seed0", COLORS["baseline"]), (student_hist, COLORS["student"])):
        if key not in hist:
            continue
        h = hist[key]
        ep = list(range(1, len(h["train_losses"]) + 1))
        lab = "GRU labels only" if key.startswith("gru_gt") else "GRU + CLIP KD"
        fig.add_trace(go.Scatter(x=ep, y=h["train_losses"], name=lab, line=dict(color=color)), 1, 1)
        fig.add_trace(go.Scatter(x=ep, y=h["val_auprc"], name=lab, line=dict(color=color), showlegend=False), 1, 2)
        if key.startswith("gru_kd"):
            fig.add_trace(go.Scatter(x=ep, y=h["kd_losses"], name="KD loss", line=dict(color=COLORS["teacher"])), 1, 3)
            fig.add_trace(go.Scatter(x=ep, y=h["feat_losses"], name="feature loss", line=dict(color="#10B981")), 1, 3)
    fig.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def _md_section(path: Path, heading: str) -> str:
    """Return the markdown section that starts with ``## heading`` (until the next ``## ``)."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    start = text.find(f"## {heading}")
    if start < 0:
        return ""
    end = text.find("\n## ", start + 3)
    return text[start:end if end > 0 else None]


def _generalisation_checks(name: str) -> None:
    pretty = {"creditcard": "Credit Card", "daphnet": "Daphnet FoG", "oil3w": "3W Oil", "metropt": "MetroPT-3"}[name]
    section = _md_section(dual.BENCH / "OVERFITTING.md", pretty)
    if section:
        st.markdown(section.replace("## ", "#### Overfitting audit: ", 1))
    oos = dual.BENCH / "out_of_sample.csv"
    if oos.exists():
        df = pd.read_csv(oos)
        df = df[df.dataset == name]
        if len(df):
            st.markdown("#### Label-free out-of-sample test (mean over seeds)")
            st.caption("New series built from held-out normal data (bootstrap) or a fitted AR(1) process (synthetic), "
                       "with hidden anomalies; models see only the CSV, an answer key grades them afterwards.")
            show_cols = ["mode", "model", "prevalence", "point AUPRC", "event precision", "event recall", "event F1"]
            agg = df.groupby(["mode", "model"], sort=False)[show_cols[2:]].mean().reset_index()
            found = df.groupby(["mode", "model"], sort=False)["found:real_event"].apply(
                lambda v: f"{sum(int(x.split('/')[0]) for x in v)}/{sum(int(x.split('/')[1]) for x in v)}")
            agg["hidden real anomalies found"] = found.values
            st.dataframe(agg.style.format(precision=3), use_container_width=True, hide_index=True)
    if name == "metropt":
        audit = dual.BENCH / "metropt" / "AUDIT.md"
        if audit.exists():
            st.warning("MetroPT-3 caveat: the test set has only two, visually obvious failures. A one-sensor rule "
                       "chosen on training data reaches window AUPRC 0.917, and the teacher's 95% interval is "
                       "0.50–1.00. No leakage was found.")
            st.markdown(_md_section(audit, "3. How easy") or "")


def _timeline(df_values: pd.DataFrame, x, labels, results: dict, features: list) -> go.Figure:
    tracks = [("truth", "Ground truth", labels)] if labels is not None else []
    tracks += [(k, MODEL_NAMES[k], r["pred"]) for k, r in results.items()]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.27, 0.18],
                        vertical_spacing=0.04,
                        subplot_titles=("Input signals (standardised)", "Anomaly scores", "Detected anomalies"))
    for f in features:
        fig.add_trace(go.Scatter(x=x, y=df_values[f], name=f, mode="lines", line=dict(width=1)), 1, 1)
    if labels is not None:
        for s, e in runs(labels):
            fig.add_vrect(x0=x[s], x1=x[min(e, len(x) - 1)], fillcolor=COLORS["truth"], opacity=0.25,
                          line_width=0, row=1, col=1)
    for key, r in results.items():
        color = COLORS.get(key, COLORS["baseline"])
        fig.add_trace(go.Scatter(x=x, y=r["score"], name=MODEL_NAMES[key], line=dict(color=color, width=1.5)), 2, 1)
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[r["threshold"]] * 2, mode="lines", showlegend=False,
                                 line=dict(color=color, dash="dot", width=1)), 2, 1)
    for i, (key, label, pred) in enumerate(tracks):
        color = COLORS.get(key, COLORS["baseline"])
        # One thick line per track; gaps (None) where nothing is flagged.
        y = np.where(np.asarray(pred) > 0, i + 0.4, np.nan)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", connectgaps=False, showlegend=False,
                                 name=label, line=dict(color=color, width=14)), 3, 1)
    fig.update_yaxes(tickvals=[i + 0.4 for i in range(len(tracks))], ticktext=[t[1].split(":")[0] for t in tracks],
                     range=[0, max(1, len(tracks))], row=3, col=1)
    fig.update_layout(height=720, template="plotly_white", margin=dict(t=40, b=20, l=20, r=20),
                      legend=dict(orientation="h", y=-0.08))
    return fig


def show() -> None:
    st.header("Teacher vs. Distilled Student")
    st.caption(
        "Upload a multivariate CSV and compare the anomalies found by the frozen-CLIP teacher with those found "
        "by the GRU student distilled from it. Colour codes: grey = ground truth, orange = teacher, "
        "red = distilled student, indigo = GRU trained on labels only."
    )
    names = dual.available_datasets()
    if not names:
        st.warning("No v2 models found. Run `scripts/run_benchmark.py` then `scripts/finalize_benchmark.py`.")
        return
    name = st.selectbox("Trained model family (dataset schema)", names,
                        format_func=lambda n: {"creditcard": "Credit Card Fraud", "daphnet": "Daphnet FoG",
                                               "airquality": "UCI Air Quality (synthetic labels)",
                                               "oil3w": "Petrobras 3W", "metropt": "MetroPT-3"}.get(n, n))
    bundle = _bundle(name)

    with st.expander("Held-out benchmark metrics for this dataset", expanded=True):
        _results_table(name)
        _training_curves(name)
    with st.expander("Generalisation checks: overfitting audit and label-free out-of-sample test"):
        _generalisation_checks(name)

    st.subheader("1. Input")
    demos = sorted((dual.BENCH / name).glob("demo_*.csv"))
    source = st.radio("Source", ["Held-out demo sequence", "Upload CSV"], horizontal=True)
    df = None
    if source == "Upload CSV":
        up = st.file_uploader("CSV with the feature columns below (an optional 0/1 label column is used for metrics)",
                              type=["csv"])
        st.caption("Required columns: " + ", ".join(f"`{c}`" for c in bundle.feature_names))
        if up is not None:
            df = pd.read_csv(up)
    elif demos:
        demo = st.selectbox("Demo sequence (from the test split; never used for training or tuning)", demos,
                            format_func=lambda p: p.stem.replace("demo_", ""))
        df = pd.read_csv(demo)
    if df is None:
        return

    label_options = ["(none)"] + [c for c in df.columns if c not in bundle.feature_names]
    default_label = label_options.index("label") if "label" in label_options else 0
    label_col = st.selectbox("Ground-truth label column (optional)", label_options, index=default_label)
    labels = None
    if label_col != "(none)":
        labels = (pd.to_numeric(df[label_col], errors="coerce").fillna(0) > 0).astype(np.int8).to_numpy()

    try:
        scaled = dual.scale_frame(df, bundle)
    except ValueError as exc:
        st.error(str(exc))
        return
    scaled_df = pd.DataFrame(scaled, columns=bundle.feature_names)
    x = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df.columns else pd.Series(np.arange(len(df)))
    if x.isna().any():
        x = pd.Series(np.arange(len(df)))
    x = x.to_numpy()
    features = st.multiselect("Signals to plot", bundle.feature_names, default=bundle.feature_names[: min(4, len(bundle.feature_names))])
    st.plotly_chart(_timeline(scaled_df, x, labels, {}, features), use_container_width=True)

    st.subheader("2. Detect anomalies")
    include_gt = st.checkbox("Also show the GRU trained on labels only (ablation reference)", value=False)
    if st.button("Run teacher and student", type="primary"):
        results = {}
        with st.spinner("Scoring with the frozen CLIP teacher (renders every window) ..."):
            try:
                results["teacher"] = dual.score_teacher(bundle, scaled, embedder=_embedder())
            except ValueError as exc:
                st.warning(f"Teacher skipped: {exc}")
        with st.spinner("Scoring with the GRU student ..."):
            results[bundle.student_key] = dual.score_student(bundle, bundle.student_key, scaled)
            if include_gt and "gru_gt" in bundle.students:
                results["gru_gt"] = dual.score_student(bundle, "gru_gt", scaled)
        st.session_state["p8_results"] = (name, len(df), results)

    stored = st.session_state.get("p8_results")
    if not stored or stored[0] != name or stored[1] != len(df):
        return
    results = stored[2]
    st.plotly_chart(_timeline(scaled_df, x, labels, results, features), use_container_width=True)

    st.subheader("3. Performance")
    summary = pd.DataFrame({MODEL_NAMES[k]: dual.summarize(labels, r) for k, r in results.items()}).T
    st.dataframe(summary.style.format(precision=3), use_container_width=True)
    if labels is None:
        st.caption("Select a label column to compute AUPRC, precision, recall and F1.")
    speed = {k: r["windows"] / max(r["seconds"], 1e-9) for k, r in results.items()}
    st.caption("Throughput (windows/s): " + ", ".join(f"{MODEL_NAMES[k].split(':')[0]} {v:,.0f}" for k, v in speed.items()))

    st.subheader("4. Anomaly intervals")
    rows = []
    for k, r in results.items():
        for s, e in runs(r["pred"]):
            rows.append({"model": MODEL_NAMES[k], "start": x[s], "end": x[e - 1], "length": int(e - s),
                         "peak score": float(np.max(r["score"][s:e]))})
    if rows:
        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True, hide_index=True)
        export = df.copy()
        for k, r in results.items():
            export[f"{k}_score"] = r["score"]
            export[f"{k}_anomaly"] = r["pred"]
        st.download_button("Download scored CSV", export.to_csv(index=False).encode(), file_name=f"{name}_scored.csv")
    else:
        st.info("No anomalies were flagged at the validation-selected thresholds.")
