"""Page: benchmark results and the checks behind them, for the selected dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.app import core


SHORT = {"creditcard": "Credit Card", "daphnet": "Daphnet FoG", "oil3w": "3W Oil", "metropt": "MetroPT-3"}


def _md_section(path: Path, heading: str) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    start = text.find(f"## {heading}")
    if start < 0:
        return ""
    end = text.find("\n## ", start + 3)
    return text[start:end if end > 0 else None]


def _table(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
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
        rows.append({"model": label, "window AUPRC": cell("window", "auprc"), "window F1": cell("window", "f1"),
                     "point AUPRC": cell("point", "auprc"), "point F1": cell("point", "f1"),
                     "event F1": cell("event", "event_f1")})
    return pd.DataFrame(rows)


def show() -> None:
    name = core.selected_dataset()
    pretty = core.PRETTY[name]
    data = core.prepared(name)
    st.header(f"Results and checks — {pretty}")
    wp, pp_ = data.splits["test"].y_window.mean(), data.splits["test"].y_step.mean()
    st.markdown(f"Test prevalence: **{wp:.4f}** of windows, **{data.splits['test'].point_labels.mean():.5f}** of "
                "timesteps — the AUPRC a random detector would get. Teachers and tree baselines are deterministic; "
                "students show mean ± std over seeds 0, 1, 2 (identical splits and initial weights).")

    tabs = st.tabs(["Benchmark table", "Distillation vs labels only", "Overfitting", "Out-of-sample",
                    "Novelty checks", "Figures"])
    with tabs[0]:
        for title, path in (("Base training (all variants)", core.BENCH / "summary.csv"),
                            ("Regularized training (labels-only vs proposed)", core.REG / "summary.csv")):
            t = _table(path, name)
            if t is not None and len(t):
                st.markdown(f"**{title}**")
                st.dataframe(t, use_container_width=True, hide_index=True)
        st.caption("Window = per window; point = per timestep after localization; event = anomalous intervals "
                   "found. AUPRC is threshold-free; F1 uses the validation-chosen threshold. See the Metrics guide "
                   "page for definitions.")
    short = SHORT[name]
    with tabs[1]:
        g = core.BENCH / "GENERALISATION.md"
        sec = _md_section(g, "Does distillation still help")
        if sec:
            st.markdown(sec.replace("## ", "#### ", 1))
        if g.exists():
            lines = g.read_text(encoding="utf-8").split("## Does distillation")[0].splitlines()
            table = [l for l in lines if l.startswith("|")]
            mine = [l for l in table[2:] if l.startswith(f"| {short} |")]
            st.markdown("#### Base vs. regularized training (this dataset)\n" + "\n".join(table[:2] + mine))
            st.caption("The configuration with the higher mean validation AUPRC is deployed; differences below 0.005 "
                       "are near-ties. Regularization mainly lowers seed variance and the train − test gap.")
    with tabs[2]:
        sec = _md_section(core.BENCH / "OVERFITTING.md", short)
        st.markdown("A large train − test gap with a low test score means memorisation. Teacher train scores are "
                    "out-of-fold.")
        st.markdown(sec.replace("## ", "#### ", 1) if sec else "Run scripts/audit_overfitting.py.")
    with tabs[3]:
        oos = core.BENCH / "out_of_sample.csv"
        st.markdown("New label-free series are generated in this dataset's format — from held-out normal data "
                    "(*bootstrap*) or an AR(1) process fitted to normal training windows (*synthetic*) — with 9 "
                    "hidden anomalies each: 3 real anomalies transplanted from the test split and 6 generic shapes "
                    "(spike, level shift, drift, noise burst, oscillation, stuck sensor). Models see only the CSV; "
                    "an answer key grades them afterwards.")
        if oos.exists():
            df = pd.read_csv(oos)
            df = df[df.dataset == name]
            cols = ["prevalence", "point AUPRC", "event precision", "event recall", "event F1"]
            agg = df.groupby(["mode", "model"], sort=False)[cols].mean().reset_index()
            agg["hidden real anomalies found"] = df.groupby(["mode", "model"], sort=False)["found:real_event"].apply(
                lambda v: f"{sum(int(x.split('/')[0]) for x in v)}/{sum(int(x.split('/')[1]) for x in v)}").values
            st.dataframe(agg.style.format(precision=3), use_container_width=True, hide_index=True)
        fig = core.BENCH / "figures" / f"oos_{name}_bootstrap.png"
        if fig.exists():
            st.image(str(fig), use_container_width=True)
    with tabs[4]:
        p = core.BENCH / "NOVELTY_CHECKS.md"
        st.markdown(p.read_text(encoding="utf-8").replace("# ", "#### ", 1) if p.exists() else "")
    with tabs[5]:
        for f, cap in (("auprc_window.png", "Test window AUPRC of every model (dashed = prevalence)"),
                       ("poster_generalisation.png", "MetroPT: distillation vs labels only, base and regularized"),
                       ("daphnet_before_after.png", "Daphnet: primary assessment vs final system")):
            if (core.FIGS / f).exists():
                st.image(str(core.FIGS / f), caption=cap, use_container_width=True)
    if name == "metropt":
        st.warning("MetroPT-3 caveat: the test set holds only two, visually obvious failures. A one-sensor rule chosen "
                   "on training data reaches window AUPRC 0.917 and the teacher's 95 % interval is 0.50–1.00. No "
                   "leakage was found (experiments/benchmark_v2/metropt/AUDIT.md).")
