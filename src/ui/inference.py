"""Page: run the teacher and the deployed students on a CSV (upload or demo).

The CSV goes through the same steps as the training data: the dataset's
feature columns are taken in order, missing values are interpolated, each
feature is standardised with the *training* μ/σ stored in the checkpoint, the
series is cut into windows, and each model's validation-tuned localization
(gating, smoothing, threshold, minimum event length) turns window outputs into
one decision per timestep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.app import core
from src.evaluation.postprocess import runs
from src.inference import dual

COLORS = {"truth": "#94A3B8", "teacher": "#D97706", "gru_kd_coop": "#B91C1C", "gru_kd_full": "#B91C1C",
          "gru_gt": "#6366F1"}
NAMES = {"teacher": "Teacher: frozen CLIP + learned prompts", "gru_kd_coop": "Student: GRU distilled from CLIP",
         "gru_kd_full": "Student: GRU distilled from CLIP", "gru_gt": "Reference: GRU trained on labels only"}


@st.cache_resource(show_spinner=False)
def _bundle(name: str):
    return dual.load_bundle(name)


def _timeline(df_values: pd.DataFrame, x, labels, results: dict, features: list) -> go.Figure:
    tracks = [("truth", "Ground truth", labels)] if labels is not None else []
    tracks += [(k, NAMES[k], r["pred"]) for k, r in results.items()]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.27, 0.18], vertical_spacing=0.04,
                        subplot_titles=("Input signals (standardised with training μ, σ)",
                                        "Anomaly score per timestep (dotted = validation threshold)",
                                        "Detected anomalies"))
    for f in features:
        fig.add_trace(go.Scatter(x=x, y=df_values[f], name=f, mode="lines", line=dict(width=1)), 1, 1)
    if labels is not None:
        for s, e in runs(labels):
            fig.add_vrect(x0=x[s], x1=x[min(e, len(x) - 1)], fillcolor=COLORS["truth"], opacity=0.25,
                          line_width=0, row=1, col=1)
    for key, r in results.items():
        fig.add_trace(go.Scatter(x=x, y=r["score"], name=NAMES[key], line=dict(color=COLORS[key], width=1.5)), 2, 1)
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[r["threshold"]] * 2, mode="lines", showlegend=False,
                                 line=dict(color=COLORS[key], dash="dot", width=1)), 2, 1)
    for i, (key, label, pred) in enumerate(tracks):
        y = np.where(np.asarray(pred) > 0, i + 0.4, np.nan)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", connectgaps=False, showlegend=False, name=label,
                                 line=dict(color=COLORS[key], width=14)), 3, 1)
    fig.update_yaxes(tickvals=[i + 0.4 for i in range(len(tracks))], ticktext=[t[1].split(":")[0] for t in tracks],
                     range=[0, max(1, len(tracks))], row=3, col=1)
    fig.update_layout(height=720, template="plotly_white", margin=dict(t=40, b=20, l=20, r=20),
                      legend=dict(orientation="h", y=-0.08))
    return fig


def show() -> None:
    name = core.selected_dataset()
    st.header(f"Teacher vs. student on a CSV — {core.PRETTY[name]}")
    st.caption("Grey = ground truth (if the CSV has a label column), orange = teacher, red = distilled student, "
               "indigo = GRU trained on labels only.")
    bundle = _bundle(name)
    with st.expander("What happens to the CSV (same steps as the training data)"):
        st.markdown(
            f"1. take the {len(bundle.feature_names)} feature columns in training order; other columns are ignored\n"
            "2. non-numeric/infinite values → missing; linear interpolation in time; remaining gaps → training mean\n"
            "3. z = (x − μ) / σ with the **training** μ, σ stored in the checkpoint\n"
            f"4. windows of {bundle.window_length} steps every {bundle.stride} steps (last window aligned to the end)\n"
            "5. **student:** step scores averaged per timestep → × window probability^α → moving average k → "
            "threshold → drop runs shorter than m (α, k, threshold, m from validation)\n"
            "6. **teacher:** both renderings of every window → CLIP → learned prompts → window probability, "
            "spread over the window's steps → moving average → threshold (from validation)")

    st.subheader("1 · Input")
    demos = sorted((dual.BENCH / name).glob("demo_*.csv"))
    source = st.radio("Source", ["Held-out demo sequence", "Upload CSV"], horizontal=True)
    df = None
    if source == "Upload CSV":
        up = st.file_uploader("CSV with the feature columns below (an optional 0/1 label column enables metrics)",
                              type=["csv"])
        st.caption("Required columns: " + ", ".join(f"`{c}`" for c in bundle.feature_names))
        if up is not None:
            df = pd.read_csv(up)
    elif demos:
        demo = st.selectbox("Demo (test split or label-free out-of-sample series; never used for training or tuning)",
                            demos, format_func=lambda p: p.stem.replace("demo_", ""))
        df = pd.read_csv(demo)
    if df is None:
        return
    label_options = ["(none)"] + [c for c in df.columns if c not in bundle.feature_names]
    label_col = st.selectbox("Ground-truth label column (optional)", label_options,
                             index=label_options.index("label") if "label" in label_options else 0)
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
    st.caption(f"{len(df):,} rows → {len(dual.window_starts(len(df), bundle.window_length, bundle.stride))} windows.")
    features = st.multiselect("Signals to plot", bundle.feature_names,
                              default=bundle.feature_names[: min(4, len(bundle.feature_names))])

    st.subheader("2 · Detect anomalies")
    include_gt = st.checkbox("Also run the GRU trained on labels only", value=True)
    if st.button("Run teacher and students", type="primary"):
        results = {}
        with st.spinner("Teacher: rendering and embedding every window with frozen CLIP ..."):
            try:
                results["teacher"] = dual.score_teacher(bundle, scaled, embedder=core.embedder())
            except ValueError as exc:
                st.warning(f"Teacher skipped: {exc}")
        results[bundle.student_key] = dual.score_student(bundle, bundle.student_key, scaled)
        if include_gt and "gru_gt" in bundle.students:
            results["gru_gt"] = dual.score_student(bundle, "gru_gt", scaled)
        st.session_state["inference_results"] = (name, len(df), label_col, results)
    stored = st.session_state.get("inference_results")
    if not stored or stored[0] != name or stored[1] != len(df):
        st.plotly_chart(_timeline(scaled_df, x, labels, {}, features), use_container_width=True)
        return
    results = stored[3]
    st.plotly_chart(_timeline(scaled_df, x, labels, results, features), use_container_width=True)

    st.subheader("3 · Performance on this CSV")
    summary = pd.DataFrame({NAMES[k]: dual.summarize(labels, r) for k, r in results.items()}).T
    st.dataframe(summary.style.format(precision=3), use_container_width=True)
    st.caption("Point level, over the timesteps of this CSV: AUPRC ranks timesteps by score (random = share of "
               "anomalous steps); precision = flagged steps that are truly anomalous; recall = anomalous steps "
               "that were flagged; F1 = their harmonic mean; event F1 = the same idea for whole anomalous "
               "intervals (an interval counts as found if any of its steps is flagged)."
               if labels is not None else "Select a label column to compute metrics.")
    speed = {k: r["windows"] / max(r["seconds"], 1e-9) for k, r in results.items()}
    st.caption("Throughput (windows per second, this CPU): " +
               ", ".join(f"{NAMES[k].split(':')[0]} {v:,.0f}" for k, v in speed.items()))

    st.subheader("4 · Anomaly intervals")
    rows = [{"model": NAMES[k], "start": x[s], "end": x[e - 1], "length (steps)": int(e - s),
             "peak score": float(np.max(r["score"][s:e])), "threshold": r["threshold"]}
            for k, r in results.items() for s, e in runs(r["pred"])]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        export = df.copy()
        for k, r in results.items():
            export[f"{k}_score"], export[f"{k}_anomaly"] = r["score"], r["pred"]
        st.download_button("Download scored CSV", export.to_csv(index=False).encode(), file_name=f"{name}_scored.csv")
    else:
        st.info("No anomalies were flagged at the validation-selected thresholds.")
