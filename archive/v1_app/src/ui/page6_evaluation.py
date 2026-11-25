"""Page 6: Evaluation & Threshold Optimization.

Performs validation-based threshold optimization (strictly without test leakage),
computes comprehensive window- and event-level metrics, and plots PR/ROC curves,
confusion matrices, and multi-model benchmark comparisons.
"""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from src.evaluation.localization import aggregate_step_scores, event_overlap_score, group_intervals
from src.evaluation.metrics import MetricsResult, compare_models, compute_metrics
from src.evaluation.threshold import select_threshold, threshold_curve_data
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_threshold_curve,
    plot_time_series_with_anomalies,
)

logger = logging.getLogger(__name__)


def _timeline_points(
    step_scores: np.ndarray,
    sequence_meta: Optional[List[dict]],
    window_length: int,
    fallback_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Aggregate overlapping window scores back to each original sequence timeline."""
    if not sequence_meta:
        return (
            np.asarray(fallback_labels, dtype=int).ravel(),
            np.asarray(step_scores, dtype=float).ravel(),
            [],
        )
    labels_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []
    per_sequence: List[Tuple[np.ndarray, np.ndarray]] = []
    for meta in sequence_meta:
        indices = np.asarray(meta.get("window_indices", []), dtype=int)
        starts = np.asarray(meta.get("window_starts", []), dtype=int)
        labels = np.asarray(meta.get("y_step", []), dtype=int).ravel()
        if not len(indices) or not len(labels):
            continue
        timeline_scores = aggregate_step_scores(
            step_scores_per_window=step_scores[indices],
            window_starts=starts,
            window_length=window_length,
            total_length=len(labels),
        )
        labels_all.append(labels)
        scores_all.append(timeline_scores)
        per_sequence.append((labels, timeline_scores))
    if not scores_all:
        return np.asarray(fallback_labels, dtype=int).ravel(), np.asarray(step_scores, dtype=float).ravel(), []
    return np.concatenate(labels_all), np.concatenate(scores_all), per_sequence


def _event_metrics(per_sequence: List[Tuple[np.ndarray, np.ndarray]], threshold: float) -> dict[str, float]:
    """Compute event precision/recall/F1 without joining separate recordings."""
    true_count = pred_count = detected_true = valid_pred = 0
    for labels, scores in per_sequence:
        true_intervals = group_intervals(labels)
        pred_intervals = group_intervals((scores >= threshold).astype(int))
        overlap = event_overlap_score(true_intervals, pred_intervals)
        n_true = int(overlap["n_true_events"])
        n_pred = int(overlap["n_pred_events"])
        true_count += n_true
        pred_count += n_pred
        detected_true += round(float(overlap["event_recall"]) * n_true)
        valid_pred += round(float(overlap["event_precision"]) * n_pred)
    recall = detected_true / true_count if true_count else (1.0 if pred_count == 0 else 0.0)
    precision = valid_pred / pred_count if pred_count else (1.0 if true_count == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"event_precision": precision, "event_recall": recall, "event_f1": f1,
            "true_events": float(true_count), "predicted_events": float(pred_count)}


NO_POSITIVE_FAR = 0.01


def _tune_one(y_val: np.ndarray, scores: np.ndarray, objective: str) -> Tuple[float, str]:
    """Threshold for one level (window or step), chosen on validation only.

    With validation anomalies: the threshold that maximises the objective.
    Without any: the 99th percentile of the (all-normal) validation scores, i.e.
    a 1% false-alarm rate. The old 0.5 fallback silently flagged nothing for
    models whose anomaly probabilities sit well below 0.5.
    """
    y = np.asarray(y_val, dtype=int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if not np.any(y == 1):
        q = float(np.quantile(s, 1.0 - NO_POSITIVE_FAR))
        return float(np.nextafter(q, np.inf)), "1% false-alarm rate on validation (no validation anomalies)"
    th, _ = select_threshold(y_val=y, scores_val=s, objective=objective, n_thresholds=100)
    return float(th), f"validation-optimal ({objective})"


def _threshold_fingerprint(model: Any, val_scores: np.ndarray, objective: str) -> Tuple[Any, ...]:
    """Identifies the model + validation scores the current thresholds were tuned for."""
    return (id(model), int(len(val_scores)), round(float(np.sum(val_scores)), 6), objective)


def _run_student_inference(
    model: torch.nn.Module,
    X_windows: np.ndarray,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run batched inference through the student model.

    Returns:
        (window_probs, step_scores, total_inference_time_sec)
    """
    model.eval()
    device = next(model.parameters()).device
    window_probs_list: List[np.ndarray] = []
    step_scores_list: List[np.ndarray] = []

    start_time = time.perf_counter()
    n_total = len(X_windows)

    with torch.no_grad():
        for start_idx in range(0, n_total, batch_size):
            batch_np = X_windows[start_idx : start_idx + batch_size]
            batch_t = torch.tensor(batch_np, dtype=torch.float32, device=device)
            if callable(getattr(model, "predict_proba", None)):
                w_proba, s_scores = model.predict_proba(batch_t)
            else:
                # Support restored or alternate torch models exposing only the
                # standard two-head forward() API.
                window_logits, step_logits = model(batch_t)
                w_proba = torch.softmax(window_logits, dim=-1)
                s_scores = torch.sigmoid(step_logits)

            window_probs_list.append(
                w_proba.detach().cpu().numpy() if torch.is_tensor(w_proba) else np.asarray(w_proba)
            )
            step_scores_list.append(
                s_scores.detach().cpu().numpy() if torch.is_tensor(s_scores) else np.asarray(s_scores)
            )

    elapsed_time = time.perf_counter() - start_time
    all_w_probs = np.vstack(window_probs_list) if window_probs_list else np.empty((0, 2))
    all_s_scores = np.vstack(step_scores_list) if step_scores_list else np.empty((0, X_windows.shape[1]))

    return all_w_probs, all_s_scores, elapsed_time


def show() -> None:
    """Render the Evaluation page."""
    st.header("Evaluation & Threshold Optimization")
    st.caption("Optimize decision thresholds on validation data, evaluate on unseen test data, and benchmark models.")

    # Guard: require a trained student model. Teacher diagnostics are reported on Page 4.
    has_model = "trained_model" in st.session_state and st.session_state["trained_model"] is not None

    if not has_model:
        st.warning("Please train a student model in Step 5 or restore an archived checkpoint first.")
        return

    # Retrieve data splits
    X_val = st.session_state.get("X_val_windows")
    y_win_val = st.session_state.get("y_window_val")
    y_step_val = st.session_state.get("y_step_val")

    X_test = st.session_state.get("X_test_windows")
    y_win_test = st.session_state.get("y_window_test")
    y_step_test = st.session_state.get("y_step_test")
    ts_test_starts = st.session_state.get("ts_test_starts")
    val_sequence_meta = st.session_state.get("val_sequence_meta")
    test_sequence_meta = st.session_state.get("test_sequence_meta")

    window_length = int(st.session_state.get("window_length", 128))
    stride = int(st.session_state.get("stride", 32))
    model = st.session_state.get("trained_model")
    mode_name = st.session_state.get("training_mode", "student_kd")

    if X_val is None or X_test is None:
        st.error("Validation and Test partitions not found. Please re-run Step 3 (Preprocessing).")
        return

    # Always recompute predictions. Cached scores can become stale after a model,
    # dataset, or preprocessing setting changes.
    if len(X_val) == 0 or len(X_test) == 0:
        st.error("Validation and test splits must each contain at least one complete window.")
        return
    val_w_probs, val_step_scores, _ = _run_student_inference(model, X_val)
    st.session_state["val_window_probs"] = val_w_probs
    st.session_state["val_step_scores"] = val_step_scores

    val_anom_scores = val_w_probs[:, 1] if val_w_probs.shape[-1] == 2 else val_w_probs.ravel()
    val_step_targets, val_step_flat_scores, _ = _timeline_points(
        val_step_scores, val_sequence_meta, window_length, y_step_val
    )

    st.markdown("---")
    st.subheader("1. Validation Threshold Tuning")
    st.markdown(
        """
        Thresholds are selected on validation labels and then frozen before test evaluation.
        Window classification and step localization have separate thresholds because their score distributions differ.
        """
    )

    t_col1, t_col2 = st.columns([1, 1])

    with t_col1:
        objective = st.selectbox(
            "Optimization Objective:",
            ["f1", "balanced", "precision", "recall"],
            index=0,
            help="'f1': maximize F1 score. 'balanced': maximize min(Precision, Recall).",
        )
        tune_btn = st.button("Find Optimal Validation Thresholds", type="primary", use_container_width=True)

    # Page-6 thresholds apply to the raw window probabilities and the raw
    # aggregated step scores shown here. They are re-tuned automatically whenever
    # the model, the validation data or the objective changes, so a threshold
    # restored from a checkpoint (tuned for the gated/smoothed scores of page 7,
    # or a default 0.5) is never applied to these scores.
    fingerprint = _threshold_fingerprint(model, val_anom_scores, objective)
    if tune_btn or st.session_state.get("eval_threshold_fingerprint") != fingerprint:
        best_window_th, window_source = _tune_one(y_win_val, val_anom_scores, objective)
        best_step_th, step_source = _tune_one(val_step_targets, val_step_flat_scores, objective)
        st.session_state["selected_window_threshold"] = best_window_th
        st.session_state["eval_step_threshold"] = best_step_th
        st.session_state["eval_threshold_source"] = {"window": window_source, "step": step_source}
        st.session_state["eval_threshold_fingerprint"] = fingerprint
        # New slider keys after every tuning, so the sliders are re-seeded with
        # the tuned values instead of keeping an older widget state.
        st.session_state["eval_tune_version"] = int(st.session_state.get("eval_tune_version", 0)) + 1
        if np.any(np.asarray(val_step_targets) == 1):
            st.session_state["thresh_curve_data"] = threshold_curve_data(
                val_step_targets, val_step_flat_scores, n_thresholds=100
            )
        else:
            st.session_state.pop("thresh_curve_data", None)

    if not np.any(y_step_val == 1) or not np.any(y_win_val == 1):
        missing_levels = []
        if not np.any(y_step_val == 1):
            missing_levels.append("timestep")
        if not np.any(y_win_val == 1):
            missing_levels.append("window")
        st.warning(
            "The validation split has no positive " + " and ".join(missing_levels)
            + " labels, so that threshold is set at a 1% false-alarm rate on validation instead of being "
            "tuned for anomaly recall. Use a validation split containing known anomalies before relying on its cutoff."
        )

    with t_col2:
        current_step_th = float(st.session_state.get("eval_step_threshold", 0.5))
        step_threshold = st.slider(
            "Step Localization Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=float(np.clip(current_step_th, 0.0, 1.0)),
            step=0.0001,
            format="%.4f",
            help="Applied to per-timestep sigmoid scores for anomaly localization.",
            key=f"eval_step_slider_{st.session_state.get('eval_tune_version', 0)}",
        )
        st.session_state["eval_step_threshold"] = step_threshold
        # Models trained in this app have no checkpoint localization settings,
        # so page 7 uses this step threshold directly. v2 checkpoints keep their
        # own threshold for the gated/smoothed scores page 7 computes.
        if not st.session_state.get("localization_params"):
            st.session_state["selected_threshold"] = step_threshold
        current_window_th = float(st.session_state.get("selected_window_threshold", 0.5))
        window_threshold = st.slider(
            "Window Classification Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=float(np.clip(current_window_th, 0.0, 1.0)),
            step=0.0001,
            format="%.4f",
            help="Applied to the student's window-level anomaly probability.",
            key=f"eval_window_slider_{st.session_state.get('eval_tune_version', 0)}",
        )
        st.session_state["selected_window_threshold"] = window_threshold

    # Display Threshold Sweep Curve
    if "thresh_curve_data" in st.session_state:
        curve_fig = plot_threshold_curve(st.session_state["thresh_curve_data"])
        st.plotly_chart(curve_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Test Set Evaluation & Benchmarking")

    eval_btn = st.button("Evaluate on Unseen Test Set", type="primary", use_container_width=True)

    if eval_btn:
        w_test, s_test, eval_time = _run_student_inference(model, X_test)
        window_metrics = compute_metrics(
            y_true=y_win_test,
            y_scores=w_test[:, 1],
            threshold=window_threshold,
        )
        test_step_targets, test_step_scores, per_sequence_test = _timeline_points(
            s_test, test_sequence_meta, window_length, y_step_test
        )
        step_metrics = compute_metrics(
            y_true=test_step_targets,
            y_scores=test_step_scores,
            threshold=step_threshold,
        )
        event_metrics = _event_metrics(per_sequence_test, step_threshold)
        st.session_state["test_metrics"] = window_metrics
        st.session_state["test_step_metrics"] = step_metrics
        st.session_state["test_event_metrics"] = event_metrics
        st.session_state["test_window_probs"] = w_test
        st.session_state["test_step_scores"] = s_test
        st.session_state["test_inference_time_seconds"] = eval_time
        st.session_state["eval_complete"] = True

        # Store separate rows so the KD student and supervised baseline can be compared.
        if "benchmark_history" not in st.session_state:
            st.session_state["benchmark_history"] = {}
        dataset_label = str(st.session_state.get("data_filename", "dataset"))
        benchmark_name = f"{dataset_label} / {mode_name}"
        st.session_state["benchmark_history"][benchmark_name] = window_metrics

        # Persist the current holdout result beside its checkpoint. The test set
        # is evaluated only after validation thresholds are fixed.
        checkpoint_dir = st.session_state.get("last_checkpoint_path")
        if checkpoint_dir and Path(checkpoint_dir).is_dir():
            result_rows = []
            for level, metric_result in (("window", window_metrics), ("step", step_metrics)):
                for metric_name, metric_value in metric_result.to_dict().items():
                    if metric_name == "confusion_matrix":
                        metric_value = json.dumps(metric_value)
                    result_rows.append({"dataset": dataset_label, "mode": mode_name,
                                        "level": level, "metric": metric_name, "value": metric_value})
            for metric_name, metric_value in event_metrics.items():
                result_rows.append({"dataset": dataset_label, "mode": mode_name,
                                    "level": "event", "metric": metric_name, "value": metric_value})
            result_rows.extend([
                {"dataset": dataset_label, "mode": mode_name, "level": "threshold",
                 "metric": "window_threshold", "value": window_threshold},
                {"dataset": dataset_label, "mode": mode_name, "level": "threshold",
                 "metric": "step_threshold", "value": step_threshold},
            ])
            pd.DataFrame(result_rows).to_csv(Path(checkpoint_dir) / "evaluation_metrics.csv", index=False)
            metadata_path = Path(checkpoint_dir) / "metadata.json"
            # Checkpoints with saved localization settings keep their own
            # threshold (it belongs to the gated/smoothed scores of page 7).
            if metadata_path.exists() and not st.session_state.get("localization_params"):
                with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                    checkpoint_metadata = json.load(metadata_file)
                checkpoint_metadata["threshold"] = float(step_threshold)
                checkpoint_metadata["window_threshold"] = float(window_threshold)
                with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                    json.dump(checkpoint_metadata, metadata_file, indent=2)

    # Display Test Results if available
    if "test_metrics" in st.session_state:
        res: MetricsResult = st.session_state["test_metrics"]
        step_res: MetricsResult = st.session_state["test_step_metrics"]
        if (
            not np.isclose(res.threshold, window_threshold)
            or not np.isclose(step_res.threshold, step_threshold)
        ):
            st.warning(
                "The validation thresholds changed after this test run. Click **Evaluate on Unseen Test Set** "
                "again to refresh the displayed metrics at the selected thresholds."
            )

        # The test metrics below use the fixed validation thresholds. Window scores
        # and reconstructed per-sequence step scores are reported separately.
        st.markdown("##### Test Benchmark Summary — Window Classification")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("AUPRC (Primary)", f"{res.auprc:.4f}")
        with m2:
            st.metric("ROC-AUC", f"{res.roc_auc:.4f}")
        with m3:
            st.metric("F1 Score", f"{res.f1:.4f}")
        with m4:
            st.metric("Precision", f"{res.precision:.4f}")
        with m5:
            st.metric("Recall", f"{res.recall:.4f}")
        source = st.session_state.get("eval_threshold_source", {})
        st.caption(
            f"Fixed validation threshold: {res.threshold:.4f} (window: {source.get('window', 'manual')}; "
            f"step {step_res.threshold:.4f}: {source.get('step', 'manual')})."
        )

        st.markdown("##### Test Localization Summary — Window-Time Pairs")
        st.caption("Overlapping window scores are averaged back onto each original sequence before point and event metrics are computed.")
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Step AUPRC", f"{step_res.auprc:.4f}")
        sm2.metric("Step ROC-AUC", f"{step_res.roc_auc:.4f}")
        sm3.metric("Step F1", f"{step_res.f1:.4f}")
        sm4.metric("Step Precision / Recall", f"{step_res.precision:.3f} / {step_res.recall:.3f}")
        event_res = st.session_state.get("test_event_metrics", {})
        ev1, ev2, ev3 = st.columns(3)
        ev1.metric("Event Precision", f"{event_res.get('event_precision', 0.0):.3f}")
        ev2.metric("Event Recall", f"{event_res.get('event_recall', 0.0):.3f}")
        ev3.metric("Event F1", f"{event_res.get('event_f1', 0.0):.3f}")
        elapsed = float(st.session_state.get("test_inference_time_seconds", 0.0))
        st.caption(f"Student inference: {elapsed:.3f} s for {len(X_test):,} windows ({len(X_test) / max(elapsed, 1e-9):,.1f} windows/s).")

        if not st.session_state.get("has_ground_truth", True):
            st.warning("This dataset has no ground-truth anomaly labels. The displayed classification scores are not meaningful.")

        # Confusion matrix at the chosen window threshold
        sec_col1, sec_col2 = st.columns(2)
        sec_col1.dataframe(pd.DataFrame([
            {"Metric": "Window threshold", "Value": f"{res.threshold:.4f}"},
            {"Metric": "True anomalous windows", "Value": res.n_true_anomalies},
            {"Metric": "Predicted anomalous windows", "Value": res.n_predicted_anomalies},
        ]), use_container_width=True, hide_index=True)
        cm_fig = plot_confusion_matrix(res.confusion_matrix, labels=["Normal", "Anomaly"])
        sec_col2.plotly_chart(cm_fig, use_container_width=True)

        # Precision-Recall and ROC curves for window classification
        st.markdown("##### Precision-Recall and ROC Curves")
        c_pr, c_roc = st.columns(2)
        y_curve = np.asarray(y_win_test, dtype=int).ravel()
        score_curve = np.asarray(st.session_state["test_window_probs"][:, 1], dtype=float)

        with c_pr:
            pr_fig = go.Figure()
            if len(np.unique(y_curve)) == 2:
                pr_precision, pr_recall, _ = precision_recall_curve(y_curve, score_curve)
            else:
                pr_precision, pr_recall = np.array([0.0, 1.0]), np.array([0.0, 1.0])
            pr_fig.add_trace(
                go.Scatter(
                    x=pr_recall,
                    y=pr_precision,
                    mode="lines",
                    name="Student PR Curve",
                    line={"color": "#38BDF8", "width": 2.5},
                    fill="tozeroy",
                    fillcolor="rgba(56, 189, 248, 0.15)",
                )
            )
            pr_fig.update_layout(
                title=f"<b>Precision-Recall Curve (AUPRC = {res.auprc:.4f})</b>",
                template="plotly_dark",
                paper_bgcolor="#181926",
                plot_bgcolor="#13141f",
                font={"family": "Inter, sans-serif", "color": "#E2E8F0"},
                xaxis={"title": "Recall", "range": [0.0, 1.05]},
                yaxis={"title": "Precision", "range": [0.0, 1.05]},
                height=380,
                margin={"l": 40, "r": 20, "t": 50, "b": 40},
            )
            pr_fig.update_xaxes(showgrid=True, gridcolor="#2e3148")
            pr_fig.update_yaxes(showgrid=True, gridcolor="#2e3148")
            st.plotly_chart(pr_fig, use_container_width=True)

        with c_roc:
            roc_fig = go.Figure()
            if len(np.unique(y_curve)) == 2:
                roc_fpr, roc_tpr, _ = roc_curve(y_curve, score_curve)
            else:
                roc_fpr, roc_tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
            roc_fig.add_trace(
                go.Scatter(
                    x=roc_fpr,
                    y=roc_tpr,
                    mode="lines",
                    name="Student ROC",
                    line={"color": "#A855F7", "width": 2.5},
                )
            )
            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random Chance",
                    line={"color": "#64748B", "dash": "dash"},
                )
            )
            roc_fig.update_layout(
                title=f"<b>ROC Curve (AUC = {res.roc_auc:.4f})</b>",
                template="plotly_dark",
                paper_bgcolor="#181926",
                plot_bgcolor="#13141f",
                font={"family": "Inter, sans-serif", "color": "#E2E8F0"},
                xaxis={"title": "False Positive Rate (FPR)", "range": [0.0, 1.05]},
                yaxis={"title": "True Positive Rate (TPR)", "range": [0.0, 1.05]},
                height=380,
                margin={"l": 40, "r": 20, "t": 50, "b": 40},
            )
            roc_fig.update_xaxes(showgrid=True, gridcolor="#2e3148")
            roc_fig.update_yaxes(showgrid=True, gridcolor="#2e3148")
            st.plotly_chart(roc_fig, use_container_width=True)

        # 4. Multi-Model Benchmark Comparison Table & Grouped Bar Chart
        benchmarks: Dict[str, MetricsResult] = st.session_state.get("benchmark_history", {})
        if len(benchmarks) > 0:
            st.markdown("---")
            st.subheader("3. Multi-Paradigm Benchmark Comparison")

            comp_df = compare_models(benchmarks)
            st.dataframe(comp_df, use_container_width=True)

            bar_fig = plot_metrics_comparison(comp_df)
            st.plotly_chart(bar_fig, use_container_width=True)

        st.success("Test evaluation successfully completed! Proceed to Step 7 (Inference & Deployment).")
