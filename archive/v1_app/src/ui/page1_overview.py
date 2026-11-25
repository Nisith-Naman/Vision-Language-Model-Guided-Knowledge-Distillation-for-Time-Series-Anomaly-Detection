"""Page 1: Overview & Pipeline Architecture.

Provides a scientific dashboard overview of the VLM Teacher-RNN Student
Knowledge Distillation system for time-series anomaly detection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "final_ie643_report" / "figures"
BENCH = ROOT / "experiments" / "benchmark_v2"


def _final_system_overview() -> None:
    """Summary of the final (v2) system, its datasets and verified results."""
    st.subheader("Final system (v2): frozen CLIP teacher → distilled GRU student")
    st.markdown(
        "- **Teacher:** CLIP ViT-B/32 with every weight frozen. Each window is rendered as a clipped heatmap and a "
        "spectrogram, and scored with **learned prompts** (CoOp, 7,168 trainable numbers through the frozen text "
        "encoder). Soft labels for training windows are cross-fitted.\n"
        "- **Student:** Conv1d → bidirectional GRU, about 95k parameters, with window, step and (training-only) "
        "embedding heads; trained with ground-truth losses plus confidence-weighted KD and CLIP feature distillation.\n"
        "- **Evaluation:** identical splits and initial weights, 3 seeds, 5 teachers × 7 students + 2 classical "
        "baselines, AUPRC always next to prevalence, thresholds tuned on validation only."
    )
    for name, caption in (("poster_architecture.png", "Full teacher–student architecture"),
                          ("poster_workflow.png", "Project progress: primary assessment → intensive assessment")):
        if (FIGS / name).exists():
            st.image(str(FIGS / name), caption=caption, use_container_width=True)

    summary = BENCH / "summary.csv"
    if summary.exists():
        s = pd.read_csv(summary)
        rows = []
        for ds, pretty in (("creditcard", "Credit Card"), ("daphnet", "Daphnet FoG"), ("oil3w", "3W Oil"),
                           ("metropt", "MetroPT-3")):
            g = s[(s.dataset == ds) & (s.level == "window")]
            get = lambda m: g[g.model == m]["auprc_mean"].iloc[0] if (g.model == m).any() else float("nan")
            if g.empty:
                continue
            rows.append({"Dataset": pretty, "Prevalence (= random)": g["prevalence_mean"].iloc[0],
                         "CLIP zero-shot": get("clip_zeroshot"), "CLIP learned prompts": get("clip_coop"),
                         "GRU labels only": get("gru_gt"), "GRU + VLM KD (proposed)": get("gru_kd_coop")})
        if rows:
            st.markdown("**Held-out test window AUPRC** (students: mean of 3 seeds)")
            st.dataframe(pd.DataFrame(rows).style.format(precision=3), use_container_width=True, hide_index=True)
    st.info(
        "How to read these numbers: distillation helps where labelled anomalies are scarce. On MetroPT-3 (one "
        "training failure) it beats the same GRU trained on labels only on every seed: +0.12 window / +0.37 point "
        "AUPRC with base training, and still +0.08 / +0.06 when both models are strongly regularized (the base "
        "labels-only GRU overfits). It is neutral on Credit Card, 3W and Daphnet. MetroPT's test set has only two, "
        "visually obvious failures (a one-sensor rule reaches 0.917), so its scores carry wide intervals. Air "
        "Quality is excluded because its labels are synthetic. The deployed models are the validation-selected "
        "configuration (see GENERALISATION.md). Details: experiments/benchmark_v2/RESULTS.md, VERIFICATION.md, "
        "OVERFITTING.md, GENERALISATION.md, metropt/AUDIT.md."
    )
    st.caption("Open page 8 to run the teacher and the student side by side on a held-out or label-free series.")


def show() -> None:
    """Render the Overview page."""
    st.title("VLM Teacher–RNN Student: Time-Series Anomaly Detection")
    st.caption("IE643 Course Project | Deep Learning & Multimodal Distillation")

    _final_system_overview()
    st.markdown("---")
    st.subheader("Interactive v1 workflow (pages 2–7)")

    # Workflow Progress Tracker
    s1_done = "selected_features" in st.session_state and "raw_df" in st.session_state
    s2_done = "X_train_windows" in st.session_state and st.session_state.get("preprocessing_complete", False)
    s3_done = "teacher_probs_train" in st.session_state or st.session_state.get("teacher_complete", False)
    s4_done = bool(st.session_state.get("training_complete", False))
    s5_done = "selected_threshold" in st.session_state or st.session_state.get("eval_complete", False)
    s6_done = st.session_state.get("inference_complete", False)

    steps = [
        ("1. Upload & Inspect", s1_done, "Load tabular or time-series CSV, profile data quality, configure sensors and target labels."),
        ("2. Preprocessing", s2_done, "Temporal sorting, missing value imputation, non-leaking scaling, and sliding window generation."),
        ("3. Teacher Inference", s3_done, "Transform time-series windows into visual heatmaps; score zero-shot with CLIP ViT-B/32."),
        ("4. Student Training", s4_done, "Train lightweight GRU recurrent student with multi-task loss (localization BCE + classification CE + KD)."),
        ("5. Evaluation", s5_done, "Validation-based threshold optimization (F1/Balanced), PR/ROC analysis, and model comparison."),
        ("6. Inference", s6_done, "Continuous timeline anomaly scoring, interval clustering, interactive visualization, and artifact export."),
    ]

    st.subheader("Workflow Pipeline Status")
    cols = st.columns(6)
    for col, (name, is_done, desc) in zip(cols, steps):
        with col:
            status_color = "#10B981" if is_done else "#64748B"
            status_text = "COMPLETED" if is_done else "PENDING"
            icon = "✓" if is_done else "○"
            st.markdown(
                f"""
                <div style="background: #181926; border: 1px solid #2e3148; border-radius: 8px; padding: 12px; height: 100%;">
                    <div style="color: {status_color}; font-weight: 700; font-size: 13px; text-transform: uppercase;">
                        {icon} {status_text}
                    </div>
                    <div style="font-weight: 600; font-size: 14px; margin-top: 4px; color: #F8FAFC;">
                        {name}
                    </div>
                    <div style="font-size: 11px; color: #94A3B8; margin-top: 6px; line-height: 1.3;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column layout: Left = Pipeline description, Right = Architecture Diagram
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.subheader("System Architecture & Methodology")
        st.markdown(
            """
            This framework tackles a fundamental bottleneck in time-series anomaly detection:
            **high-capacity foundation vision-language models (VLMs)** excel at zero-shot macroscopic
            reasoning across modalities, but are too computationally heavy and lack temporal recurrence
            for high-frequency edge sensor deployment.

            ### Distillation Mechanism
            1. **Visual Encoding**: Sliding windows of multidimensional time series $\\mathbf{X}_w \\in \\mathbb{R}^{L \\times F}$
               are rendered into 2D heatmap representations without label leakage.
            2. **Frozen VLM Teacher**: OpenAI CLIP (`ViT-B/32`) evaluates image-text cosine similarity
               against contrastive semantic prompts (*"normal regular signal"* vs *"anomalous irregular spikes"*),
               producing continuous soft anomaly priors $p_T \\in [0, 1]$.
            3. **Multi-Task GRU Student**: A lightweight 2-layer Gated Recurrent Unit (GRU) learns to:
               - Classify window-level anomaly probability via Hinton temperature-scaled KL divergence ($L_{\\text{KD}}$).
               - Predict fine-grained time-step anomaly localization masks via weighted BCE ($L_{\\text{loc}}$).
               - Supervise window binary classification via cross-entropy ($L_{\\text{window}}$).
            4. **Edge Deployment**: The distilled GRU processes streaming data in microseconds,
               requiring zero GPU overhead while preserving foundation model zero-shot sensitivity.
            """
        )

        st.markdown("#### Key Innovation Features")
        st.markdown(
            """
            - **Strict Leakage Prevention**: Chronological dataset splitting guarantees that scalers,
              imputers, and decision thresholds are learned solely on past observations.
            - **Dual-Resolution Predictions**: Outputs both sequence-level window alerts and exact point-wise
              anomaly localization intervals.
            - **Robust Imbalance Handling**: Weighted loss terms ($pos\\_weight \\ge 10.0$) and AUPRC-focused
              benchmarks tailored for real-world low anomaly rates ($< 5\\%$).
            """
        )

    with col_right:
        st.subheader("End-to-End Pipeline Diagram")
        mermaid_code = """
        flowchart TD
            subgraph Data["1. Data Pipeline"]
                RAW[("Raw Sensor CSV")] --> PRE["Preprocessing & Dedup"]
                PRE --> CHRONO["Chronological Split<br/>(Train / Val / Test)"]
                CHRONO --> SCALE["Train-Only Scaler<br/>(Standard / MinMax)"]
                SCALE --> WIN["Sliding Window Slicer<br/>(Length L, Stride S)"]
            end

            subgraph Teacher["2. Teacher Branch"]
                WIN --> IMGEN["Window-to-Image<br/>(Heatmap / Line Plot)"]
                IMGEN --> CLIP["Frozen CLIP VLM<br/>(ViT-B/32)"]
                PROMPTS["Semantic Prompts<br/>(Normal vs Anomaly)"] --> CLIP
                CLIP --> T_PROBS["Soft Probability<br/>p_T(anomaly)"]
            end

            subgraph Student["3. Student Model"]
                WIN --> GRU["GRU Student RNN<br/>(2 Layers + Dropout)"]
                GRU --> W_HEAD["Window Head<br/>(Classification)"]
                GRU --> S_HEAD["Step Head<br/>(Localization)"]
            end

            subgraph Loss["4. Joint Multi-Task Loss"]
                T_PROBS & W_HEAD --> L_KD["KL Distillation Loss<br/>(T = 2.0)"]
                W_HEAD --> L_WIN["Window Cross-Entropy"]
                S_HEAD --> L_LOC["Weighted Step BCE<br/>(pos_weight = 10)"]
                L_KD & L_WIN & L_LOC --> TOTAL_LOSS["Total Loss = L_loc + β*L_win + λ*L_kd"]
            end

            subgraph Inference["5. Inference & Evaluation"]
                GRU --> AGG["Score Aggregation<br/>(Overlapping Windows)"]
                AGG --> THRESH["Validation-Tuned<br/>Threshold"]
                THRESH --> ANOM_IV["Discrete Anomaly<br/>Intervals"]
            end
        """
        st.markdown(f"```mermaid\n{mermaid_code}\n```")

    st.markdown("---")

    # Supported Datasets Section
    st.subheader("Supported Datasets Benchmark Suite")
    st.markdown(
        """
        The platform features native support for five diverse industrial and benchmark time-series datasets,
        covering medical wearables, financial transactions, environmental monitoring, and heavy industry machinery:
        """
    )

    d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns(5)

    with d_col1:
        st.markdown(
            """
            **💳 CreditCard**
            - **Domain**: Fraud Detection
            - **Dimensions**: 29 features (V1-V28, Amount)
            - **Nature**: Highly imbalanced (<0.2% anomalies)
            - **Metric**: AUPRC & Precision@Recall
            """
        )

    with d_col2:
        st.markdown(
            """
            **🏃 Daphnet FoG**
            - **Domain**: Wearable Healthcare
            - **Dimensions**: 9 tri-axial accelerometers
            - **Sampling**: 64 Hz Parkinson's Gait
            - **Task**: Freezing of Gait (FoG) detection
            """
        )

    with d_col3:
        st.markdown(
            """
            **💨 AirQualityUCI**
            - **Domain**: Environmental IoT
            - **Dimensions**: 13 chemical sensor channels
            - **Characteristics**: Missingness (-200 codes)
            - **Anomalies**: Injected spikes & drift shifts
            """
        )

    with d_col4:
        st.markdown(
            """
            **🛢️ 3W Oil & Gas**
            - **Domain**: Offshore Wells
            - **Dimensions**: 8 pressure & temp sensors
            - **Events**: Severe transient slugging
            - **Data**: Real and simulated run instances
            """
        )

    with d_col5:
        st.markdown(
            """
            **🚇 MetroPT3**
            - **Domain**: Rail Compressor
            - **Dimensions**: 15 analog & digital signals
            - **Failures**: Real catastrophic air leaks
            - **Evaluation**: Event interval overlap
            """
        )

    st.markdown("---")

    # Technical Note on CLIP Limitations
    st.info(
        """
        **Technical Note on Foundation VLM Calibration:**
        Vision-Language Models such as OpenAI CLIP were pretrained on massive web crawls of natural photographic
        imagery and paired textual captions. When applied to synthetic 2D representations (such as Gramian angular
        fields, recurrence plots, or feature heatmaps), zero-shot cosine similarities reflect perceptual and texture
        heuristics rather than calibrated Bayesian posterior probabilities.

        In this pipeline, the VLM Teacher acts as a **soft heuristic regularizer** ($L_{\\text{KD}}$). The recurrent
        Student network integrates temporal sequence dynamics and is fine-tuned against ground-truth step anomalies,
        preventing the student from overfitting to raw noise while retaining the teacher's generalized anomaly prior.
        """,
        icon="ℹ️",
    )
