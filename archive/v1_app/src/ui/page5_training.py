"""Page 5: Student Model Training & Knowledge Distillation.

Trains a dual-head GRU with VLM soft targets or with ground-truth supervision
alone, and saves checkpoints plus epoch-level training logs.
"""

from __future__ import annotations

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.student.model import GRUStudent, ModelConfig
from src.training.checkpoint import (
    CheckpointMetadata,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from src.training.trainer import DistillationLoss, Trainer, TrainingResult
from src.visualization.plots import plot_training_history

logger = logging.getLogger(__name__)


def _representative_validation_indices(labels: np.ndarray, budget: int) -> np.ndarray:
    """Select a deterministic class-aware subset for faster training-time validation."""
    labels = np.asarray(labels, dtype=np.int8).ravel()
    if budget <= 0 or len(labels) <= budget:
        return np.arange(len(labels), dtype=np.int64)
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    if not len(normal) or not len(anomaly):
        return np.linspace(0, len(labels) - 1, budget, dtype=np.int64)
    n_anomaly = int(round(budget * len(anomaly) / len(labels)))
    n_anomaly = min(budget - 1, max(1, n_anomaly))
    n_normal = budget - n_anomaly
    selected = np.r_[
        np.linspace(normal[0], normal[-1], n_normal, dtype=np.int64),
        np.linspace(anomaly[0], anomaly[-1], n_anomaly, dtype=np.int64),
    ]
    return np.sort(np.unique(selected))


def _render_checkpoint_manager() -> None:
    """Allow model restoration before uploading or preprocessing another dataset."""
    st.markdown("---")
    st.subheader("Checkpoint Manager & Experiment Restoration")
    checkpoints = list_checkpoints("./checkpoints")
    if not checkpoints:
        st.info("No saved checkpoints found in `./checkpoints`. Train a labeled dataset with the KD teacher first.")
        return

    ckpt_options = [
        f"{c.get('dataset_name', 'unknown')} | {c.get('training_mode', 'mode')} | {c.get('created_at', '')[:19]}"
        for c in checkpoints
    ]
    sel_idx = st.selectbox(
        "Select a trained student checkpoint:",
        range(len(checkpoints)),
        format_func=lambda i: ckpt_options[i],
        key="checkpoint_restore_selection",
    )
    selected_ckpt = checkpoints[sel_idx]
    if st.button("Load & Activate Checkpoint", key="activate_checkpoint", use_container_width=True):
        try:
            ckpt_dir = selected_ckpt.get("checkpoint_dir")
            restored_model, restored_scaler, restored_meta = load_checkpoint(ckpt_dir)
            if restored_model is None:
                raise ValueError("This checkpoint does not contain a student model.")
            st.session_state["trained_model"] = restored_model
            st.session_state["training_complete"] = True
            st.session_state["scaler"] = restored_scaler
            st.session_state["selected_features"] = restored_meta.feature_names
            st.session_state["training_mode"] = restored_meta.training_mode
            st.session_state["window_length"] = restored_meta.window_length
            st.session_state["stride"] = restored_meta.stride
            st.session_state["last_checkpoint_path"] = ckpt_dir
            st.session_state["selected_threshold"] = float(restored_meta.threshold)
            st.session_state["contextual_threshold_robust_z"] = float(
                restored_meta.contextual_threshold_robust_z
            )
            st.session_state["contextual_calibration_normal_rows"] = int(
                restored_meta.contextual_calibration_normal_rows
            )
            st.session_state["context_robust_z_threshold"] = min(
                20.0,
                max(2.0, float(restored_meta.contextual_threshold_robust_z)),
            )
            st.session_state["selected_window_threshold"] = float(
                selected_ckpt.get("window_threshold", restored_meta.threshold)
            )
            st.session_state["missing_strategy"] = restored_meta.missing_strategy
            st.session_state["train_impute_stats"] = restored_meta.imputation_stats
            st.session_state["missing_codes"] = restored_meta.missing_codes
            st.session_state["time_col"] = restored_meta.time_column
            # v2 checkpoints carry validation-tuned localization settings.
            st.session_state["localization_params"] = (
                restored_meta.training_result or {}
            ).get("localization")
            st.success(
                f"Loaded the trained student, scaler, feature schema, and thresholds from "
                f"`{Path(ckpt_dir).name}`. Open Inference to score a CSV."
            )
        except Exception as exc:
            st.error(f"Failed to load checkpoint: {exc}")
            logger.exception("Checkpoint restoration failure")


def show() -> None:
    """Render the Student Training page."""
    st.header("Student RNN Model Training & Knowledge Distillation")
    st.caption("Train a lightweight multi-task GRU student model with localization BCE and VLM teacher soft KD loss.")

    # Guard: require preprocessing from page 3
    if "X_train_windows" not in st.session_state or not st.session_state.get("preprocessing_complete", False):
        st.warning("Please complete Step 3 (Preprocessing & Windowing) first to generate train and validation windows.")
        _render_checkpoint_manager()
        return

    X_train: np.ndarray = st.session_state["X_train_windows"]
    y_step_train: np.ndarray = st.session_state["y_step_train"]
    y_window_train: np.ndarray = st.session_state["y_window_train"]

    X_val: np.ndarray = st.session_state["X_val_windows"]
    y_step_val: np.ndarray = st.session_state["y_step_val"]
    y_window_val: np.ndarray = st.session_state["y_window_val"]

    selected_features: List[str] = st.session_state.get("selected_features", [])
    data_filename: str = st.session_state.get("data_filename", "dataset")
    window_length: int = int(st.session_state.get("window_length", X_train.shape[1]))
    stride: int = int(st.session_state.get("stride", 32))
    input_dim: int = len(selected_features) if selected_features else X_train.shape[-1]

    st.markdown("---")
    st.subheader("Training Paradigm Configuration")

    mode_options = [
        "Student with KD (Teacher supervision)",
        "Student Baseline (ground truth only)",
    ]
    mode_map = {
        "Student with KD (Teacher supervision)": "student_kd",
        "Student Baseline (ground truth only)": "student_baseline",
    }

    selected_mode_label = st.radio(
        "Select Training Mode:",
        mode_options,
        index=0,
        horizontal=True,
    )
    training_mode = mode_map[selected_mode_label]

    # Check for teacher scores if KD is selected
    teacher_scores_train = st.session_state.get("teacher_probs_train")
    has_ground_truth = bool(st.session_state.get("has_ground_truth", False))
    if not has_ground_truth:
        st.warning(
            "This student uses supervised window and timestep labels in its loss. "
            "An unlabeled CSV cannot train the advertised localization task without a separate unsupervised objective or validated pseudo-labels."
        )
    if training_mode == "student_kd":
        if teacher_scores_train is None:
            st.warning(
                "Teacher soft probabilities have not yet been generated in Step 4. "
                "You can switch to 'Student Baseline' to train without KD, or visit Step 4 to compute teacher scores.",
                icon="⚠️",
            )
        else:
            st.info(
                f"Found {len(teacher_scores_train):,} teacher scores. "
                "Ground-truth labels will train the student on every training window; "
                "the KL loss will apply only where a teacher score is available.",
                icon="ℹ️",
            )

    st.markdown("---")
    st.subheader("Student Architecture & Hyperparameter Tuning")

    col_h1, col_h2, col_h3 = st.columns(3)

    with col_h1:
        hidden_size = st.selectbox("GRU Hidden Units:", [32, 64, 128, 256], index=1)
        num_layers = st.selectbox("Stacked Recurrent Layers:", [1, 2, 3], index=0)
        bidirectional = st.checkbox("Bidirectional GRU", value=True)
        use_conv = st.checkbox(
            "Convolutional front-end + mean/max pooling (v2)", value=True,
            help="Two 1-D convolutions before the GRU (downsampling long windows), as in the v2 benchmark. "
                 "Untick for the v1 GRU-only student.",
        )
        select_by_auprc = st.checkbox(
            "Select epoch by validation AUPRC (v2)", value=True,
            help="Early stopping on the mean of validation window and step AUPRC instead of validation loss.",
        )
        dropout = st.slider("Dropout Probability:", min_value=0.0, max_value=0.5, value=0.3, step=0.05)

    with col_h2:
        learning_rate = st.number_input("Learning Rate (AdamW):", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
        batch_size = st.selectbox("Mini-Batch Size:", [16, 32, 64, 128], index=1)
        max_epochs = st.slider("Max Training Epochs:", min_value=5, max_value=50, value=20, step=5)
        early_stopping_patience = st.slider("Early Stopping Patience:", min_value=2, max_value=10, value=5, step=1)

    with col_h3:
        beta = st.slider(
            "Beta (Window CE Loss Weight):",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Weight assigned to sequence-level binary cross entropy.",
        )
        lambda_kd = st.slider(
            "Lambda KD (Distillation Weight):",
            min_value=0.0,
            max_value=3.0,
            value=0.5 if training_mode == "student_kd" else 0.0,
            step=0.1,
            help="Weight assigned to Hinton KL divergence against teacher probabilities.",
            disabled=(training_mode != "student_kd"),
        )
        temperature = st.slider(
            "Distillation Temperature (T):",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Softens probability distributions before computing KL divergence.",
            disabled=(training_mode != "student_kd"),
        )
        pos_weight = st.slider(
            "Positive Step Class Weight (pos_weight):",
            min_value=1.0,
            max_value=25.0,
            value=10.0,
            step=1.0,
            help="Multiplier weight on anomalous time-step BCE loss to counter class imbalance.",
        )

    st.markdown("---")
    st.subheader("Model Execution")

    col_btn_train, _ = st.columns([1, 2])
    with col_btn_train:
        train_btn = st.button(
            "Start Training Pipeline",
            type="primary",
            use_container_width=True,
            disabled=(
                not has_ground_truth
                or (training_mode == "student_kd" and teacher_scores_train is None)
            ),
        )

    if train_btn:
        progress_bar = st.progress(0, text="Initializing model architecture and loss criteria...")
        status_box = st.empty()

        try:
            # Paired runs start from the same initialization and data order.
            seed = int(st.session_state.get("app_config", {}).get("seed", 42))
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # 1. Model Configuration
            model_cfg = ModelConfig(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                seq_len=window_length,
                conv_channels=64 if use_conv else 0,
                downsample=(4 if window_length >= 200 else 2 if window_length >= 100 else 1) if use_conv else 1,
                pooling="meanmax" if use_conv else "mean",
            )

            student_model = GRUStudent(**model_cfg.to_dict())
            n_params = student_model.parameter_count()
            status_box.info(f"Instantiated GRUStudent with {n_params:,} trainable parameters.")

            # 2. Build Trainer Configuration
            trainer_cfg = {
                "mode": training_mode,
                "device": "auto",
                "training": {
                    "epochs": max_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "early_stopping_patience": early_stopping_patience,
                    "optimizer": "adamw",
                    "weight_decay": 1e-4,
                    "gradient_clip": 1.0,
                    "selection_metric": "val_auprc" if select_by_auprc else "val_loss",
                },
                "distillation": {
                    "temperature": temperature,
                    "beta": beta,
                    "lambda_kd": lambda_kd if training_mode == "student_kd" else 0.0,
                    "pos_weight": pos_weight,
                },
            }

            trainer = Trainer(model=student_model, config=trainer_cfg)

            benchmark_config = st.session_state.get("benchmark_config", {})
            validation_budget = int(benchmark_config.get("max_validation_windows", 0))
            if validation_budget > 0 and len(X_val) > validation_budget:
                validation_indices = _representative_validation_indices(
                    y_window_val, validation_budget
                )
                X_val_fit = X_val[validation_indices]
                y_step_val_fit = y_step_val[validation_indices]
                y_window_val_fit = y_window_val[validation_indices]
                st.info(
                    f"Training-time validation uses {len(validation_indices):,} of "
                    f"{len(X_val):,} validation windows. The full validation and test "
                    "partitions remain available for threshold selection and final metrics."
                )
            else:
                validation_indices = np.arange(len(X_val), dtype=np.int64)
                X_val_fit = X_val
                y_step_val_fit = y_step_val
                y_window_val_fit = y_window_val

            # 3. Every available labeled training window remains in both modes.
            # Sparse teacher scores are aligned to the full array and masked so
            # unscored windows still contribute supervised losses.
            X_tr_exec = X_train
            y_step_tr_exec = y_step_train
            y_win_tr_exec = y_window_train
            t_scores_tr_exec = None
            t_mask_tr_exec = None

            teacher_indices = np.asarray(
                st.session_state.get("teacher_train_indices", []), dtype=int
            )
            if training_mode == "student_kd" and teacher_scores_train is not None:
                if len(teacher_indices) == 0:
                    if len(teacher_scores_train) == len(X_train):
                        teacher_indices = np.arange(len(X_train), dtype=int)
                    else:
                        raise ValueError(
                            "Teacher scores are missing their training-window indices. "
                            "Recompute teacher scores on the current preprocessing split."
                        )
                if len(teacher_indices) != len(teacher_scores_train):
                    raise ValueError("Teacher score indices do not align with the stored training scores.")
                if np.any((teacher_indices < 0) | (teacher_indices >= len(X_train))):
                    raise ValueError("Teacher score indices point outside the current training windows.")
                if len(np.unique(teacher_indices)) != len(teacher_indices):
                    raise ValueError("Teacher score indices contain duplicates.")

                t_scores_tr_exec = np.zeros((len(X_train), 2), dtype=np.float32)
                t_scores_tr_exec[teacher_indices] = np.asarray(teacher_scores_train, dtype=np.float32)
                t_mask_tr_exec = np.zeros(len(X_train), dtype=bool)
                t_mask_tr_exec[teacher_indices] = True
                st.info(
                    f"Using all {len(X_train):,} training windows for ground-truth learning; "
                    f"the teacher KL loss is applied to {len(teacher_indices):,} scored windows."
                )

            # Progress callback for Streamlit UI
            def _st_callback(epoch: int, tot_epochs: int, tr_loss: float, v_loss: float) -> None:
                pct = int((epoch / tot_epochs) * 100)
                progress_bar.progress(
                    pct,
                    text=f"Epoch {epoch}/{tot_epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f}",
                )

            # 4. Execute Trainer Fit
            training_result = trainer.fit(
                X_train=X_tr_exec,
                y_step_train=y_step_tr_exec,
                y_window_train=y_win_tr_exec,
                teacher_probs_train=t_scores_tr_exec,
                X_val=X_val_fit,
                y_step_val=y_step_val_fit,
                y_window_val=y_window_val_fit,
                teacher_probs_val=None,
                progress_callback=_st_callback,
                teacher_mask_train=t_mask_tr_exec,
            )

            progress_bar.progress(100, text="Training completed!")
            status_box.success(
                f"Training finished in {training_result.training_time_seconds:.2f}s! "
                f"Best Val Loss: {training_result.best_val_loss:.4f} (Epoch {training_result.best_epoch})"
            )

            # 5. Checkpoint Auto-Save
            benchmark_config = st.session_state.get("benchmark_config", {})
            configured_missing_code = benchmark_config.get("missing_code")
            missing_codes = [configured_missing_code] if configured_missing_code is not None else []
            meta = CheckpointMetadata(
                model_config=model_cfg.to_dict(),
                feature_names=selected_features,
                scaler_type=str(st.session_state.get("scaler_type", "standard")),
                window_length=window_length,
                stride=stride,
                threshold=0.5,
                normal_label=st.session_state.get("normal_label_val", "0"),
                anomaly_label=st.session_state.get("anomaly_label_val", "1"),
                training_mode=training_mode,
                dataset_name=data_filename.replace(".csv", "").replace(" ", "_"),
                training_result=training_result.to_dict(),
                created_at=datetime.now().isoformat(),
                missing_strategy=str(st.session_state.get("missing_strategy", "interpolate")),
                imputation_stats={
                    key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in st.session_state.get("train_impute_stats", {}).items()
                },
                time_column=st.session_state.get("time_col"),
                missing_codes=missing_codes,
                contextual_threshold_robust_z=float(
                    st.session_state.get("contextual_threshold_robust_z", 4.5)
                ),
                contextual_calibration_normal_rows=int(
                    st.session_state.get("contextual_calibration_normal_rows", 0)
                ),
            )

            ckpt_path = save_checkpoint(
                model=student_model,
                scaler=st.session_state.get("scaler"),
                metadata=meta,
                save_dir="./checkpoints",
            )

            # Keep epoch-by-epoch losses beside the checkpoint for reproducible
            # plots and cross-dataset reporting outside the interactive session.
            history_columns = {
                "epoch": np.arange(1, len(training_result.train_losses) + 1),
                "train_loss": training_result.train_losses,
                "validation_loss": training_result.val_losses,
                "localization_loss": training_result.loc_losses,
                "window_loss": training_result.window_losses,
                "distillation_loss": training_result.kd_losses,
            }
            history_df = pd.DataFrame({
                key: pd.Series(values) for key, values in history_columns.items()
            })
            history_df.to_csv(Path(ckpt_path) / "training_history.csv", index=False)
            with open(Path(ckpt_path) / "run_config.json", "w", encoding="utf-8") as config_file:
                json.dump({
                    "model": model_cfg.to_dict(),
                    "trainer": trainer_cfg,
                    "dataset": data_filename,
                    "feature_names": selected_features,
                    "time_column": st.session_state.get("time_col"),
                    "missing_strategy": st.session_state.get("missing_strategy", "interpolate"),
                    "imputation_stats": meta.imputation_stats,
                    "missing_codes": missing_codes,
                    "window_length": window_length,
                    "stride": stride,
                    "contextual_threshold_robust_z": meta.contextual_threshold_robust_z,
                    "teacher_model": st.session_state.get("teacher_model_name") if training_mode == "student_kd" else None,
                    "teacher_image_mode": st.session_state.get("teacher_image_mode") if training_mode == "student_kd" else None,
                    "teacher_normal_prompt": st.session_state.get("teacher_normal_prompt") if training_mode == "student_kd" else None,
                    "teacher_anomaly_prompt": st.session_state.get("teacher_anomaly_prompt") if training_mode == "student_kd" else None,
                    "seed": seed,
                    "training_windows_used": int(len(X_tr_exec)),
                    "teacher_scored_training_windows": int(t_mask_tr_exec.sum()) if t_mask_tr_exec is not None else 0,
                    "teacher_training_window_indices": (
                        teacher_indices.tolist()
                        if training_mode == "student_kd" else []
                    ),
                    "split_manifest": st.session_state.get("split_manifest", {}),
                    "training_validation_windows_used": int(len(validation_indices)),
                    "training_validation_windows_available": int(len(X_val)),
                }, config_file, indent=2)

            # 6. Save in session_state
            st.session_state["trained_model"] = student_model
            st.session_state["localization_params"] = None  # UI-trained models use raw step scores
            st.session_state["training_complete"] = True
            st.session_state["training_result"] = training_result
            st.session_state["training_mode"] = training_mode
            st.session_state["model_config"] = model_cfg
            st.session_state["last_checkpoint_path"] = ckpt_path
            for key in (
                "selected_threshold", "selected_window_threshold", "thresh_info",
                "window_thresh_info", "thresh_curve_data", "test_metrics",
                "test_step_metrics", "test_window_probs", "test_step_scores",
                "val_window_probs", "val_step_scores", "eval_complete",
            ):
                st.session_state.pop(key, None)

        except Exception as e:
            status_box.error(f"Training encountered an error: {e}")
            logger.exception("Student training failure")
            return

    # Post-Training History and Convergence Analysis
    if st.session_state.get("training_result") is not None:
        st.markdown("---")
        st.subheader("Convergence Diagnostics & Loss Curves")

        res: TrainingResult = st.session_state["training_result"]
        ckpt_path_saved = st.session_state.get("last_checkpoint_path", "N/A")

        # Metric summary cards
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric("Total Epochs", len(res.train_losses))
        with c_m2:
            st.metric("Best Val Loss", f"{res.best_val_loss:.4f}")
        with c_m3:
            st.metric("Best Epoch", res.best_epoch)
        with c_m4:
            st.metric("Elapsed Time", f"{res.training_time_seconds:.1f}s")

        if ckpt_path_saved != "N/A":
            st.info(f"Checkpoint safely archived to: `{ckpt_path_saved}`", icon="💾")

        # Plot Loss Curves
        if res.train_losses and res.val_losses:
            loss_fig = plot_training_history(
                train_losses=res.train_losses,
                val_losses=res.val_losses,
                loc_losses=res.loc_losses,
                window_losses=res.window_losses,
                kd_losses=res.kd_losses,
            )
            st.plotly_chart(loss_fig, use_container_width=True)

    _render_checkpoint_manager()
