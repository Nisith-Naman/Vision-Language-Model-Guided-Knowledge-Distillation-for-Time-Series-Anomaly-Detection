"""Page: how a CSV becomes the model's input tensors, step by step, with the
selected dataset's real numbers and checks that each step was applied."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app import core, diagrams
from src.evaluation.postprocess import runs


def _raw_frame(seq, n: int = 8) -> pd.DataFrame:
    df = pd.DataFrame(seq.values[:n], columns=seq.feature_names)
    if seq.timestamps is not None:
        df.insert(0, "timestamp", np.asarray(seq.timestamps)[:n])
    df["label"] = (seq.y_step[:n] if seq.y_step is not None else 0)
    return df


def _match_raw(raw, processed_seq, mu, sd):
    """Find the raw sequence and row offset a processed (split) sequence came from.

    Candidates share the sequence id (a chronological split appends _train/_val/
    _test) and the first timestamp; the one whose first rows standardise to the
    stored values is returned."""
    gid = processed_seq["group_id"]
    for suffix in ("_train", "_val", "_test"):
        if gid.endswith(suffix):
            gid = gid[: -len(suffix)]
    ts = processed_seq.get("timestamps")
    vals = processed_seq["values"]
    for s in raw:
        if str(s.group_id) != gid and str(s.group_id) != processed_seq["group_id"]:
            continue
        cands = ([0] if ts is None or s.timestamps is None
                 else np.flatnonzero(np.asarray(s.timestamps) == np.asarray(ts)[0]))
        for off in cands:
            seg = s.values[off:off + 3].astype(np.float64)
            if len(seg) == 3 and np.all(np.isfinite(seg)) and np.allclose((seg - mu) / sd, vals[:3], atol=1e-3):
                return s, int(off)
    return None, 0


def show() -> None:
    name = core.selected_dataset()
    pretty, info = core.PRETTY[name], core.INFO[name]
    data = core.prepared(name)
    settings = core.dataset_settings(name)
    st.header(f"Data pipeline: from CSV to model input — {pretty}")
    st.caption("Every number on this page is computed from the data the models were trained and tested on "
               "(`src/experiments/data.py → prepare_dataset`). Each step ends with a check that it was applied.")

    st.subheader("The complete pipeline")
    from src.ui.overview import show_pipeline

    show_pipeline(name, data)
    st.markdown("This page opens the blue block (A). Pages 3–5 open the teacher, student and evaluation blocks.")

    raw = core.raw_series(name)
    F = len(data.feature_names)

    # ---------------------------------------------------------------- step 1
    st.subheader("Step 1 · Raw files → a matrix per sequence")
    st.markdown(f"- **Source:** {info['raw']}.\n- **Sequence:** {info['sequence']}.\n- **Label:** {info['label']}.")
    lengths = np.array([s.n_timesteps for s in raw])
    c = st.columns(4)
    c[0].metric("Sequences", f"{len(raw):,}")
    c[1].metric("Timesteps (rows)", f"{lengths.sum():,}")
    c[2].metric("Features F", F)
    c[3].metric("Anomalous rows", f"{int(sum(int(np.sum(s.y_step)) for s in raw if s.y_step is not None)):,}")
    st.markdown(f"Each sequence becomes a matrix **X ∈ ℝ^(T×{F})** (rows = time, columns = sensors) and a label "
                "vector **y ∈ {0,1}^T**. First rows of the first sequence, before any processing:")
    st.dataframe(_raw_frame(raw[0]), use_container_width=True, hide_index=True)
    seq_rows = [{"sequence": s.group_id, "rows T": s.n_timesteps,
                 "anomalous rows": int(np.sum(s.y_step)) if s.y_step is not None else 0,
                 "anomalous events": len(runs(s.y_step)) if s.y_step is not None else 0} for s in raw[:200]]
    with st.expander(f"All sequences ({len(raw)})"):
        st.dataframe(pd.DataFrame(seq_rows), use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------- step 2
    st.subheader("Step 2 · Split whole sequences into train / validation / test")
    st.markdown(f"Rule: **{info['split']}** (`{data.manifest['split_method']}`). A sequence (or a time range of "
                "one) belongs to exactly one split, so no test timestep is ever seen in training.")
    parts = data.manifest["partitions"]
    st.dataframe(pd.DataFrame([{
        "split": k, "sequences / groups": v["n_groups"], "timesteps": v["timesteps"],
        "anomalous timesteps": v["anomalous_timesteps"], "point prevalence": round(v["point_prevalence"], 5),
        "windows": v["windows"], "anomalous windows": v["anomalous_windows"],
        "window prevalence": round(v["window_prevalence"], 4)} for k, v in parts.items()]),
        use_container_width=True, hide_index=True)
    st.caption("Prevalence = share of anomalous items. It is also the AUPRC of a detector that guesses at random, "
               "which is why every AUPRC in this app is shown next to it.")

    # ---------------------------------------------------------------- step 3
    st.subheader("Step 3 · Fill missing values")
    miss = np.zeros(F, dtype=int)
    for s in raw:
        miss += (~np.isfinite(s.values)).sum(axis=0)
    st.markdown("Inside each sequence, a missing value is **linearly interpolated** between its neighbours in time; "
                "gaps at the start or end take the nearest valid value; a column that is missing for a whole "
                "sequence gets the **training-split mean** (never a validation or test statistic).")
    if miss.sum():
        st.dataframe(pd.DataFrame({"feature": data.feature_names, "missing values": miss})
                     .query("`missing values` > 0"), use_container_width=True, hide_index=True)
    else:
        st.success(f"Check: no missing values in {pretty}; this step changes nothing.")

    # ---------------------------------------------------------------- step 4
    st.subheader("Step 4 · Standardise every feature with training statistics")
    mu, sd = np.asarray(data.scaler.mean_), np.asarray(data.scaler.scale_)
    st.latex(r"z_{t,f} = \frac{x_{t,f} - \mu_f}{\sigma_f},\qquad \mu_f,\ \sigma_f \text{ computed on the training split only}")
    st.dataframe(pd.DataFrame({"feature": data.feature_names, "μ (train mean)": mu, "σ (train std)": sd})
                 .style.format({"μ (train mean)": "{:.4g}", "σ (train std)": "{:.4g}"}),
                 use_container_width=True, hide_index=True, height=min(400, 38 + 35 * F))

    test = data.splits["test"]
    seq_i = st.selectbox("Test sequence used for the worked example below", range(len(test.sequences)),
                         format_func=lambda i: f"{test.sequences[i]['group_id']}  "
                                               f"({len(test.sequences[i]['labels']):,} steps, "
                                               f"{int(np.sum(test.sequences[i]['labels'])):,} anomalous)")
    tseq = test.sequences[seq_i]
    raw_seq, offset = _match_raw(raw, tseq, mu, sd)
    values = tseq["values"]
    if raw_seq is None:
        st.info("This sequence starts with imputed values, so the worked example is skipped; pick another sequence.")
    else:
        row = next((t for t in range(len(values)) if np.isfinite(raw_seq.values[offset + t]).all()), 0)
        x = raw_seq.values[offset + row].astype(np.float64)
        z = (x - mu) / sd
        ex = pd.DataFrame({"feature": data.feature_names, "raw x": x, "μ": mu, "σ": sd, "(x − μ)/σ": z,
                           "value in model input": values[row]})
        st.markdown(f"Worked example, row {row} of test sequence `{tseq['group_id']}`:")
        st.dataframe(ex.head(min(F, 10)).style.format(precision=4), use_container_width=True, hide_index=True)
        ok = np.allclose(z, values[row], atol=1e-3)
        (st.success if ok else st.error)(
            f"Check: the hand-computed z-scores {'equal' if ok else 'DO NOT equal'} the stored model input "
            f"(max difference {np.max(np.abs(z - values[row])):.2e}).")

    # ---------------------------------------------------------------- step 5
    L, S = data.window_length, data.stride
    st.subheader(f"Step 5 · Cut sliding windows (L = {L}, stride = {S})")
    st.latex(r"\text{starts} = 0, S, 2S, \dots \le T-L\ \ (+\,T-L \text{ if the tail is not covered}),\quad "
             r"X_w = Z[s:s+L,\ :] \in \mathbb{R}^{L\times F},\quad y^{\text{step}}_w = y[s:s+L],\quad "
             r"y^{\text{win}}_w = \max_t y^{\text{step}}_{w,t}")
    starts = np.asarray(tseq["window_starts"])
    idx = np.asarray(tseq["window_indices"])
    st.markdown(f"Sequence `{tseq['group_id']}` has T = {len(values):,} steps → **{len(starts)} windows**; "
                f"consecutive windows overlap by {L - S} steps, so every timestep is covered by up to "
                f"{int(np.ceil(L / S))} windows. First starts: {', '.join(map(str, starts[:8]))}"
                f"{' …' if len(starts) > 8 else ''}.")
    j = st.slider("Window within this sequence", 0, len(starts) - 1,
                  int(np.argmax(test.y_window[idx])) if test.y_window[idx].any() else 0)
    s0 = int(starts[j])
    feat = st.selectbox("Feature to plot", data.feature_names, index=0)
    fi = data.feature_names.index(feat)
    lo, hi = max(0, s0 - 2 * L), min(len(values), s0 + 3 * L)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(lo, hi), y=values[lo:hi, fi], name=f"{feat} (z)", line=dict(width=1)))
    for a, b in runs(np.asarray(tseq["labels"])[lo:hi]):
        fig.add_vrect(x0=lo + a, x1=lo + b, fillcolor="#94A3B8", opacity=0.3, line_width=0)
    for k in range(max(0, j - 2), min(len(starts), j + 3)):
        fig.add_vrect(x0=starts[k], x1=starts[k] + L, line=dict(color="#B91C1C" if k == j else "#CBD5E1",
                      width=2 if k == j else 1), fillcolor="rgba(0,0,0,0)")
    fig.update_layout(height=280, margin=dict(t=30, b=20, l=20, r=20), template="plotly_white",
                      title="Grey = labelled anomaly; red box = selected window; light boxes = neighbouring windows")
    st.plotly_chart(fig, use_container_width=True)

    Xw = test.X[idx[j]]
    c1, c2 = st.columns([3, 1])
    with c1:
        hm = go.Figure(go.Heatmap(z=Xw.T, x=np.arange(s0, s0 + L), y=data.feature_names, colorscale="Viridis",
                                  zmin=-4, zmax=4, colorbar=dict(title="z")))
        hm.update_layout(height=max(260, 18 * F + 80), margin=dict(t=30, b=20, l=20, r=20),
                         title=f"Model input X_w for window {j} (steps {s0}–{s0 + L - 1}), shape {L}×{F}")
        st.plotly_chart(hm, use_container_width=True)
    with c2:
        ys = test.y_step[idx[j]]
        st.metric("anomalous steps in window", f"{int(ys.sum())} / {L}")
        st.metric("window label y_win = max(y_step)", int(test.y_window[idx[j]]))
        same = np.allclose(Xw, values[s0:s0 + L])
        (st.success if same else st.error)(f"Check: X_w {'equals' if same else 'differs from'} rows "
                                           f"{s0}…{s0 + L - 1} of the standardised sequence.")

    # ---------------------------------------------------------------- step 6
    st.subheader("Step 6 · Model-ready tensors")
    rows = []
    for k, sp in data.splits.items():
        rows.append({"split": k, "X": f"{sp.X.shape[0]:,} × {L} × {F}", "y_step": f"{sp.y_step.shape[0]:,} × {L}",
                     "y_win": f"{sp.y_window.shape[0]:,}", "anomalous windows": int(sp.y_window.sum()),
                     "window prevalence": round(float(sp.y_window.mean()), 4)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    cap = int(settings.get("max_train_windows", 0) or 0)
    if cap and len(data.splits["train"].X) >= cap:
        st.caption(f"Training windows were capped at {cap:,} by a time-distributed, label-independent selection.")
    st.markdown("These exact tensors feed both branches: the **teacher** renders each window as images "
                "(page 3) and the **student** reads the numbers directly (page 4).")
