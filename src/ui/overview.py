"""Page: what the system is, what each model and head does, and where to look."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app import core, diagrams


def show_pipeline(name: str, data) -> None:
    """The pipeline diagram in three readable parts plus the all-in-one view."""
    args = (name, data, core.PRETTY[name], core.pos_weight(data),
            int(core.dataset_settings(name).get("downsample", 1)))
    tabs = st.tabs(["A · Data preparation", "B · Training (teacher → student)", "C · Inference & evaluation",
                    "All in one"])
    for tab, part in zip(tabs, ("prep", "train", "eval", None)):
        with tab:
            st.graphviz_chart(diagrams.full_pipeline(*args, part=part), use_container_width=part != "prep")
    st.caption("Blue: data preparation · orange: teacher (frozen CLIP) · red: student (GRU) · purple: training "
               "loss · green: evaluation. Hover a diagram and use its full-screen button to enlarge it. "
               "A static version is in final_ie643_report/figures/pipeline_diagram.png.")


def show() -> None:
    st.title("VLM-guided knowledge distillation for time-series anomaly detection")
    st.caption("IE 643 course project · final system")
    st.markdown(
        "**Problem.** Find *whether* and *where* a multivariate sensor stream behaves abnormally (card fraud, "
        "freezing of gait, oil-well faults, compressor failures). Labelled anomalies are rare, so a detector "
        "trained on labels alone learns a narrow rule.\n\n"
        "**Idea.** A large vision-language model (CLIP) has seen millions of charts. We keep it **frozen**, show it "
        "each window as two pictures, adapt only its prompts, and **distil** its judgement into a small recurrent "
        "network that reads the raw numbers and runs ~330× faster.")

    st.subheader("The two models")
    st.dataframe(pd.DataFrame([
        {"model": "Teacher: frozen CLIP ViT-B/32 + learned prompts",
         "input": "window rendered as a heatmap and a spectrogram (224×224 images)",
         "output": "p_T(anomalous) per window",
         "trained parameters": "7,168 (14 prompt vectors); CLIP's 151 M weights frozen",
         "used": "training only (soft labels + embeddings); shown for comparison"},
        {"model": "Student: Conv1d → bidirectional GRU (proposed)",
         "input": "window of standardised numbers (L × F)",
         "output": "p_win per window + s_t per timestep",
         "trained parameters": "≈95–101 k",
         "used": "deployed detector"},
        {"model": "Reference: the same GRU trained on labels only",
         "input": "same", "output": "same", "trained parameters": "same",
         "used": "ablation: what the teacher adds"},
    ]), use_container_width=True, hide_index=True)

    st.subheader("What each student head does")
    st.dataframe(pd.DataFrame([
        {"head": "Window head", "computes": "mean+max pooled GRU states → 2 logits → softmax",
         "answers": "is this window anomalous?", "trained by": "ground-truth window label (CE) + teacher soft label (KD)"},
        {"head": "Step head", "computes": "per-timestep GRU state → 1 logit → sigmoid (upsampled to L)",
         "answers": "which timesteps are anomalous?", "trained by": "ground-truth timestep labels (weighted BCE)"},
        {"head": "Embedding head", "computes": "pooled states → 512 numbers",
         "answers": "what would CLIP see? (training only)", "trained by": "CLIP image embedding (1 − cosine)"},
    ]), use_container_width=True, hide_index=True)

    name = core.selected_dataset()
    data = core.prepared(name)
    st.subheader(f"Complete pipeline (numbers for {core.PRETTY[name]}; change the dataset in the sidebar)")
    show_pipeline(name, data)

    st.subheader("Where to look")
    st.markdown(
        "| Page | What you see |\n|---|---|\n"
        "| 2 · Data pipeline | CSV → matrix → split → imputation → standardisation → windows, with checks |\n"
        "| 3 · Teacher | rendering, CLIP embedding, zero-shot vs learned prompts for one window; cross-fitting; teacher quality |\n"
        "| 4 · Student & training | layers and shapes, each head's output, the loss computed by hand, recorded and live training |\n"
        "| 5 · Evaluation | window / point / event metrics derived step by step, checked against the benchmark files |\n"
        "| 6 · Teacher vs student | run all models on a demo or uploaded CSV |\n"
        "| 7 · Results & checks | benchmark tables, distillation vs labels only, overfitting, out-of-sample, novelty checks |\n"
        "| 8 · Metrics guide | what every metric means and how to read it |")

    summary = core.BENCH / "summary.csv"
    reg = core.REG / "summary.csv"
    if summary.exists():
        s = pd.read_csv(summary)
        r = pd.read_csv(reg) if reg.exists() else None
        rows = []
        for ds in core.DATASETS:
            g = s[(s.dataset == ds) & (s.level == "window")]
            if g.empty:
                continue
            get = lambda df, m: (df[(df.dataset == ds) & (df.level == "window") & (df.model == m)]["auprc_mean"]
                                 .iloc[0] if df is not None and ((df.dataset == ds) & (df.model == m)).any() else float("nan"))
            rows.append({"dataset": core.PRETTY[ds], "random (= prevalence)": g["prevalence_mean"].iloc[0],
                         "CLIP zero-shot": get(s, "clip_zeroshot"), "teacher (learned prompts, 2 views)": get(s, "clip_coop_mv"),
                         "GRU labels only": get(r, "gru_gt"), "GRU + CLIP KD (proposed)": get(r, "gru_kd_coop")})
        st.subheader("Headline: test window AUPRC")
        st.dataframe(pd.DataFrame(rows).style.format(precision=3), use_container_width=True, hide_index=True)
        st.caption("Students: regularized training, mean of 3 seeds. Distillation helps where labelled anomalies "
                   "are scarcest (MetroPT-3: one training failure, +0.08 window AUPRC on every seed) and is neutral "
                   "elsewhere. MetroPT-3's test set has only two, visually obvious failures, so its numbers carry "
                   "wide intervals. Air Quality is excluded: its labels are synthetic.")
