"""Page 4: VLM Teacher Inference & Visual Representation.

Converts multi-sensor time-series windows into visual images (heatmaps/lineplots)
and computes zero-shot soft anomaly priors using OpenAI CLIP ViT-B/32.
"""

from __future__ import annotations

import logging
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.window_sampling import select_time_distributed_windows
from src.teacher.clip_teacher import CLIPTeacher
from src.teacher.image_gen import batch_to_images, window_to_image
from src.visualization.plots import plot_teacher_score_distribution

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _load_frozen_teacher(model_name: str, device: str) -> CLIPTeacher:
    """Load one shared, frozen CLIP resource for this local Streamlit process."""
    teacher = CLIPTeacher(model_name=model_name, device=device)
    teacher.load_model()
    return teacher


def show() -> None:
    """Render the VLM Teacher Inference page."""
    st.header("VLM Teacher: Visual Encoding & Zero-Shot Scoring")
    st.caption("Render time-series windows into visual representations and compute zero-shot anomaly probabilities.")
    st.info(
        "CLIP is already pretrained and stays frozen. It is loaded once per app process; "
        "its soft scores are generated once for the selected data/configuration and cached for student training."
    )

    # Guard: require preprocessing from page 3
    if "X_train_windows" not in st.session_state or not st.session_state.get("preprocessing_complete", False):
        st.warning("Please complete Step 3 (Preprocessing & Windowing) first to generate sliding windows.")
        return

    X_train_windows: np.ndarray = st.session_state["X_train_windows"]
    y_window_train: np.ndarray = st.session_state["y_window_train"]
    X_val_windows: np.ndarray = st.session_state["X_val_windows"]
    y_window_val: np.ndarray = st.session_state["y_window_val"]
    n_train_windows = len(X_train_windows)
    if n_train_windows == 0:
        st.error("The training split has no windows to score. Return to Step 3 and rerun preprocessing.")
        return
    selected_features: List[str] = st.session_state.get("selected_features", [])
    data_filename: str = st.session_state.get("data_filename", "dataset")
    # Render every train/validation chart on the same train-derived scale so a
    # color or amplitude change means the same thing across splits.
    visual_vmin = float(np.nanmin(X_train_windows))
    visual_vmax = float(np.nanmax(X_train_windows))
    if visual_vmin == visual_vmax:
        visual_vmax = visual_vmin + 1.0

    st.markdown("---")
    st.subheader("Teacher Configuration & Contrastive Prompts")

    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        model_name = st.selectbox(
            "VLM Foundation Architecture:",
            [
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-large-patch14",
            ],
            index=0,
            help="Pretrained Hugging Face Vision-Language Model used as zero-shot teacher.",
        )
        image_mode = st.selectbox(
            "Visual Encoding Mode:",
            ["heatmap", "lineplot"],
            index=0,
            help="'heatmap': matrix representation of sensors over time. 'lineplot': multi-channel line curves.",
        )
        use_cache = st.checkbox(
            "Enable On-Disk Score Caching",
            value=True,
            help="Avoids recomputing CLIP forward passes when parameters have not changed.",
        )

    with col_t2:
        max_budget = min(2000, n_train_windows)
        if max_budget == 1:
            budget_cap = 1
            st.caption("Only one training window is available, so the teacher will score that window.")
        else:
            budget_cap = st.slider(
                "Maximum Train Windows to Score (Budget Cap):",
                min_value=1,
                max_value=max_budget,
                value=min(1000, max_budget),
                step=1,
                help=(
                    f"Caps CLIP scoring at {max_budget:,} available training windows. "
                    "KD training and its paired supervised baseline use the same scored training subset."
                ),
            )
        batch_size = st.select_slider(
            "Inference Batch Size:",
            options=[8, 16, 32, 64],
            value=16,
            help="Mini-batch size passed through the CLIP vision and text processors.",
        )

    st.markdown("##### Contrastive Semantic Prompts")
    prompt_col1, prompt_col2 = st.columns(2)

    default_normal = "A regular, smooth, stable multidimensional time-series sensor pattern without anomalies or spikes"
    default_anom = "An anomalous, irregular, erratic multidimensional time-series signal with sudden spikes or distortions"

    with prompt_col1:
        normal_prompt = st.text_area(
            "Nominal / Regular Signal Prompt:",
            value=default_normal,
            height=90,
            help="Semantic text description of normal behavior.",
        )

    with prompt_col2:
        anomaly_prompt = st.text_area(
            "Anomalous / Distorted Signal Prompt:",
            value=default_anom,
            height=90,
            help="Semantic text description of irregular or anomalous behavior.",
        )

    st.markdown("---")
    st.subheader("Visual Representation Preview")

    col_btn_prev, _ = st.columns([1, 2])
    with col_btn_prev:
        gen_sample_images = st.button("Generate & Preview Sample Window Images", use_container_width=True)

    if gen_sample_images or "preview_images" in st.session_state:
        # Find 2 normal and 2 anomalous windows if possible
        if "preview_images" not in st.session_state or gen_sample_images:
            normal_indices = np.where(y_window_train == 0)[0]
            anom_indices = np.where(y_window_train == 1)[0]

            selected_indices: List[int] = []
            if len(normal_indices) >= 2:
                selected_indices.extend(normal_indices[:2].tolist())
            else:
                selected_indices.extend(list(range(min(2, len(X_train_windows)))) )

            if len(anom_indices) >= 2:
                selected_indices.extend(anom_indices[:2].tolist())
            else:
                remaining = [i for i in range(len(X_train_windows)) if i not in selected_indices]
                selected_indices.extend(remaining[: 4 - len(selected_indices)])

            preview_imgs = []
            preview_captions = []
            for idx in selected_indices[:4]:
                win_data = X_train_windows[idx]
                win_lbl = y_window_train[idx]
                img = window_to_image(
                    win_data,
                    feature_names=selected_features,
                    mode=image_mode,
                    image_size=(224, 224),
                    vmin=visual_vmin,
                    vmax=visual_vmax,
                )
                lbl_text = "Ground Truth: ANOMALY" if win_lbl == 1 else "Ground Truth: NORMAL"
                preview_imgs.append(img)
                preview_captions.append(f"Window #{idx} ({lbl_text})")

            st.session_state["preview_images"] = preview_imgs
            st.session_state["preview_captions"] = preview_captions

        imgs = st.session_state.get("preview_images", [])
        caps = st.session_state.get("preview_captions", [])

        if imgs:
            g_col1, g_col2, g_col3, g_col4 = st.columns(4)
            cols = [g_col1, g_col2, g_col3, g_col4]
            for c, img, cap in zip(cols, imgs, caps):
                with c:
                    st.image(img, caption=cap, use_container_width=True)

    st.markdown("---")
    st.subheader("Teacher Model Inference Execution")

    # Important VLM Calibration Notice
    st.warning(
        f"**Teacher Calibration Notice:** {CLIPTeacher.WARNING}",
        icon="⚠️",
    )

    col_run_t, _ = st.columns([1, 2])
    with col_run_t:
        run_scoring_btn = st.button("Score Windows with CLIP Teacher", type="primary", use_container_width=True)

    cache_path = None
    if use_cache:
        clean_name = data_filename.replace(".csv", "").replace(" ", "_")
        cache_context = json.dumps({
            "dataset": clean_name,
            "model": model_name,
            "normal_prompt": normal_prompt,
            "anomaly_prompt": anomaly_prompt,
            "image_mode": image_mode,
            "budget": budget_cap,
        }, sort_keys=True)
        cache_tag = hashlib.sha256(cache_context.encode("utf-8")).hexdigest()[:12]
        cache_path = f"./cache/teacher_scores_{clean_name}_{cache_tag}.pkl"

    if run_scoring_btn:
        progress_bar = st.progress(0, text="Initializing CLIP Teacher...")
        status_box = st.empty()

        try:
            status_box.info(f"Loading or reusing frozen '{model_name}' on the auto-detected device...")
            model_load_start = time.perf_counter()
            teacher = _load_frozen_teacher(model_name=model_name, device="auto")
            model_load_seconds = time.perf_counter() - model_load_start
            train_indices = select_time_distributed_windows(
                len(X_train_windows),
                st.session_state.get("train_sequence_meta"),
                budget_cap,
            )
            windows_subset = X_train_windows[train_indices]
            val_budget = min(2000, len(X_val_windows))
            val_indices = select_time_distributed_windows(
                len(X_val_windows),
                st.session_state.get("val_sequence_meta"),
                val_budget,
            )
            data_digest = hashlib.sha256(
                windows_subset.tobytes() + X_val_windows[val_indices].tobytes()
            ).hexdigest()
            cache_metadata = {
                "model": model_name,
                "normal_prompt": normal_prompt,
                "anomaly_prompt": anomaly_prompt,
                "image_mode": image_mode,
                "features": list(selected_features),
                "train_indices": train_indices.tolist(),
                "val_indices": val_indices.tolist(),
                "data_digest": data_digest,
            }

            scores = val_scores = None
            train_score_seconds = val_score_seconds = None
            if cache_path and Path(cache_path).exists():
                try:
                    with open(cache_path, "rb") as f:
                        cached = pickle.load(f)
                    if cached.get("metadata") == cache_metadata:
                        scores = np.asarray(cached.get("scores"), dtype=np.float32)
                        val_scores = np.asarray(cached.get("val_scores"), dtype=np.float32)
                        train_score_seconds = cached.get("train_score_seconds")
                        val_score_seconds = cached.get("val_score_seconds")
                        if scores.shape != (len(train_indices), 2) or val_scores.shape != (len(val_indices), 2):
                            scores = val_scores = None
                except Exception as ce:
                    logger.warning("Could not load teacher score cache: %s", ce)

            if scores is None or val_scores is None:
                progress_bar.progress(25, text="Model loaded. Rendering selected train and validation windows...")
                status_box.info(f"Rendering {len(windows_subset)} training windows to {image_mode} representations...")
                images = batch_to_images(
                    windows=windows_subset,
                    feature_names=selected_features,
                    mode=image_mode,
                    image_size=(224, 224),
                    vmin=visual_vmin,
                    vmax=visual_vmax,
                )
                val_images = batch_to_images(
                    windows=X_val_windows[val_indices],
                    feature_names=selected_features,
                    mode=image_mode,
                    image_size=(224, 224),
                    vmin=visual_vmin,
                    vmax=visual_vmax,
                )
                progress_bar.progress(55, text="Scoring train and validation images with frozen CLIP...")
                status_box.info(f"Computing contrastive logits in batches of {batch_size}...")
                train_score_start = time.perf_counter()
                scores = teacher.score_batch(images, normal_prompt, anomaly_prompt, batch_size=batch_size)
                train_score_seconds = time.perf_counter() - train_score_start
                val_score_start = time.perf_counter()
                val_scores = teacher.score_batch(val_images, normal_prompt, anomaly_prompt, batch_size=batch_size)
                val_score_seconds = time.perf_counter() - val_score_start

                if cache_path:
                    try:
                        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_path, "wb") as f:
                            pickle.dump({
                                "scores": scores,
                                "val_scores": val_scores,
                                "train_score_seconds": train_score_seconds,
                                "val_score_seconds": val_score_seconds,
                                "metadata": cache_metadata,
                            }, f, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception as ce:
                        logger.warning("Could not cache teacher scores: %s", ce)
            else:
                status_box.info("Loaded matching train and validation CLIP scores from the on-disk cache.")

            progress_bar.progress(90, text="Evaluating held-out teacher calibration metrics...")

            # Store in session state
            st.session_state["teacher_probs_train"] = scores
            st.session_state["teacher_train_indices"] = train_indices
            st.session_state["teacher_probs_val"] = val_scores
            st.session_state["teacher_val_indices"] = val_indices
            st.session_state["teacher_budget_cap"] = budget_cap
            st.session_state["teacher_model_name"] = model_name
            st.session_state["teacher_image_mode"] = image_mode
            st.session_state["teacher_normal_prompt"] = normal_prompt
            st.session_state["teacher_anomaly_prompt"] = anomaly_prompt
            st.session_state["teacher_model_parameters"] = sum(
                parameter.numel() for parameter in teacher.model.parameters()
            ) if teacher.model is not None else 0
            st.session_state["teacher_model_load_seconds"] = model_load_seconds
            st.session_state["teacher_train_score_seconds"] = train_score_seconds
            st.session_state["teacher_val_score_seconds"] = val_score_seconds
            st.session_state["teacher_scoring_config"] = (
                model_name, image_mode, normal_prompt, anomaly_prompt, int(budget_cap),
                tuple(selected_features), data_digest,
            )
            st.session_state["teacher_complete"] = True

            # Evaluate on held-out validation labels only.
            eval_dict = teacher.evaluate_on_labels(val_scores, y_window_val[val_indices])
            st.session_state["teacher_eval_dict"] = eval_dict

            progress_bar.progress(100, text="Teacher scoring complete!")
            status_box.success(f"Successfully scored {len(scores):,} windows with CLIP Teacher!")

        except Exception as e:
            status_box.error(f"Teacher inference encountered an error: {e}")
            logger.exception("CLIP Teacher scoring failure")
            return

    current_teacher_config = (
        model_name, image_mode, normal_prompt, anomaly_prompt, int(budget_cap),
        tuple(selected_features),
    )
    prior_teacher_config = st.session_state.get("teacher_scoring_config")
    if prior_teacher_config and tuple(prior_teacher_config[:6]) != current_teacher_config:
        for key in (
            "teacher_complete", "teacher_probs_train", "teacher_train_indices",
            "teacher_probs_val", "teacher_val_indices", "teacher_eval_dict",
            "teacher_scoring_config",
        ):
            st.session_state.pop(key, None)

    # Results & Inspection Section
    if st.session_state.get("teacher_complete", False) and "teacher_probs_train" in st.session_state:
        st.markdown("---")
        st.subheader("Teacher Score Diagnostics & Quantitative Profiling")

        scores = st.session_state["teacher_probs_train"]
        budget_cap = st.session_state.get("teacher_budget_cap", len(scores))
        val_indices = np.asarray(st.session_state.get("teacher_val_indices", []), dtype=int)
        y_subset = y_window_val[val_indices] if len(val_indices) else np.zeros(0, dtype=int)
        scores = st.session_state.get("teacher_probs_val", np.empty((0, 2), dtype=np.float32))

        eval_dict = st.session_state.get("teacher_eval_dict", {})
        auprc_val = eval_dict.get("auprc", float("nan"))
        roc_auc_val = eval_dict.get("roc_auc", float("nan"))

        # Quantitative Metrics Banner
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        with m_c1:
            st.metric("Validation Windows Scored", f"{len(scores):,}")
        with m_c2:
            st.metric("Mean Anomaly Prior", f"{np.mean(scores[:, 1]):.3f}")
        with m_c3:
            auprc_str = f"{auprc_val:.4f}" if not np.isnan(auprc_val) else "N/A"
            st.metric("Teacher Zero-Shot AUPRC", auprc_str)
        with m_c4:
            roc_str = f"{roc_auc_val:.4f}" if not np.isnan(roc_auc_val) else "N/A"
            st.metric("Teacher Zero-Shot ROC-AUC", roc_str)
        with m_c5:
            st.metric("CLIP Parameters", f"{st.session_state.get('teacher_model_parameters', 0):,}")
        train_score_seconds = st.session_state.get("teacher_train_score_seconds")
        val_score_seconds = st.session_state.get("teacher_val_score_seconds")
        model_load_seconds = st.session_state.get("teacher_model_load_seconds")
        if train_score_seconds is not None and val_score_seconds is not None:
            st.caption(
                f"CLIP load: {model_load_seconds:.2f} s · scoring only: "
                f"{float(train_score_seconds) + float(val_score_seconds):.2f} s "
                f"({float(train_score_seconds):.2f} s train, {float(val_score_seconds):.2f} s validation). "
                "Cached runs reuse the measured scoring times when available; chart rendering is excluded."
            )

        # Plot Score Distribution
        st.markdown("##### Soft Anomaly Target Distribution")
        dist_fig = plot_teacher_score_distribution(
            teacher_scores=scores[:, 1],
            y_true=y_subset,
        )
        st.plotly_chart(dist_fig, use_container_width=True)

        # Score sample table
        with st.expander("Inspect Individual Window Teacher Probabilities (First 15)", expanded=False):
            table_rows = []
            for i in range(min(15, len(scores))):
                p_norm = float(scores[i, 0])
                p_anom = float(scores[i, 1])
                gt_lbl = int(y_subset[i])
                table_rows.append({
                    "Window Index": i,
                    "P(Normal)": f"{p_norm:.4f}",
                    "P(Anomaly)": f"{p_anom:.4f}",
                    "Ground Truth": "Anomaly (1)" if gt_lbl == 1 else "Normal (0)",
                    "Teacher Prediction": "Anomaly" if p_anom >= 0.5 else "Normal",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.success(
            f"Teacher soft targets for {len(st.session_state.get('teacher_probs_train', [])):,} sampled training windows "
            "are prepared. Validation metrics above are held out from student training."
        )


