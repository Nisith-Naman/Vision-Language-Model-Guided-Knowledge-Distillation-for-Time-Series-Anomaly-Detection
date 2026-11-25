"""
Plotly visualization module for the IE643 VLM-KD Anomaly Detection system.

Generates interactive, publication-quality figures with a dark modern theme.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.evaluation.localization import group_intervals

logger = logging.getLogger(__name__)

# Dark theme visual constants
THEME_TEMPLATE = "plotly_dark"
COLOR_BG_DARK = "#13141f"
COLOR_PAPER_DARK = "#181926"
COLOR_GRID = "#2e3148"
COLOR_PRED_ANOMALY = "rgba(239, 68, 68, 0.35)"      # Red
COLOR_TRUE_ANOMALY = "rgba(245, 158, 11, 0.30)"      # Amber / Orange
COLOR_LINE_PRIMARY = "#38BDF8"                       # Sky blue
COLOR_LINE_SECONDARY = "#A855F7"                     # Purple
COLOR_LINE_SUCCESS = "#10B981"                       # Emerald green
COLOR_LINE_WARNING = "#FACC15"                       # Yellow
COLOR_LINE_DANGER = "#EF4444"                        # Red


def _apply_dark_styling(fig: go.Figure, title: str = "") -> go.Figure:
    """Applies standardized sleek dark aesthetic to any plotly figure."""
    fig.update_layout(
        template=THEME_TEMPLATE,
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 17, "family": "Inter, Segoe UI, sans-serif"},
        },
        paper_bgcolor=COLOR_PAPER_DARK,
        plot_bgcolor=COLOR_BG_DARK,
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#E2E8F0"},
        legend={
            "bgcolor": "rgba(24, 25, 38, 0.8)",
            "bordercolor": COLOR_GRID,
            "borderwidth": 1,
            "font": {"size": 12},
        },
        margin={"l": 55, "r": 40, "t": 60, "b": 50},
        hoverlabel={"bgcolor": "#1E1E2E", "font_size": 12, "font_family": "Inter, Segoe UI, sans-serif"},
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLOR_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID, zeroline=False)
    return fig


def plot_time_series_with_anomalies(
    values: np.ndarray,
    timestamps: Optional[Union[np.ndarray, list, pd.Series]] = None,
    feature_names: Optional[list[str]] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    anomaly_intervals: Optional[list[dict[str, Any]]] = None,
    title: str = "Time Series Anomaly Detection",
    ground_truth_intervals: Optional[list[dict[str, Any]]] = None,
) -> go.Figure:
    """
    Creates a multi-panel interactive plot with one subplot per feature.

    Shades predicted anomaly intervals in red and ground-truth anomaly intervals in orange.

    Args:
        values: 2D array of time series values (N, D) or 1D array (N,).
        timestamps: Sequence of timestamps or indices.
        feature_names: Names for each feature column.
        y_true: Optional ground truth binary step labels (N,).
        y_pred: Optional predicted binary step labels (N,).
        anomaly_intervals: Optional pre-grouped predicted intervals.
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)

    n_samples, n_features = vals.shape

    # Cap to max 8 features to ensure UI responsiveness
    max_feats_to_plot = min(8, n_features)

    # Resolve timestamps
    if timestamps is not None:
        if isinstance(timestamps, (pd.Series, pd.Index)):
            x_vals = timestamps.values
        else:
            x_vals = np.asarray(timestamps)
    else:
        x_vals = np.arange(n_samples)

    # Resolve feature names
    if feature_names is None:
        feat_names = [f"Feature {i+1}" for i in range(max_feats_to_plot)]
    else:
        feat_names = list(feature_names[:max_feats_to_plot])
        while len(feat_names) < max_feats_to_plot:
            feat_names.append(f"Feature {len(feat_names)+1}")

    fig = make_subplots(
        rows=max_feats_to_plot,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[f"<b>{name}</b>" for name in feat_names],
    )

    palette = [
        "#38BDF8", "#818CF8", "#A78BFA", "#F472B6",
        "#34D399", "#FBBF24", "#FB923C", "#94A3B8"
    ]

    for f_idx in range(max_feats_to_plot):
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=vals[:, f_idx],
                mode="lines",
                name=feat_names[f_idx],
                line={"color": palette[f_idx % len(palette)], "width": 1.5},
                showlegend=(f_idx == 0 or max_feats_to_plot <= 3),
            ),
            row=f_idx + 1,
            col=1,
        )

    # 1. Shade Ground Truth Anomaly Intervals (Orange/Amber)
    gt_intervals = ground_truth_intervals
    if gt_intervals is None and y_true is not None:
        gt_intervals = group_intervals(y_true, timestamps=x_vals)
    if gt_intervals is not None:
        for i, iv in enumerate(gt_intervals):
            x0 = iv["start_ts"] if iv["start_ts"] is not None else iv["start_idx"]
            x1 = iv["end_ts"] if iv["end_ts"] is not None else iv["end_idx"]
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=COLOR_TRUE_ANOMALY,
                line_width=1,
                line_color="rgba(245, 158, 11, 0.7)",
                layer="below",
                annotation_text="Ground Truth" if i == 0 else None,
                annotation_position="top left",
                annotation_font={"size": 10, "color": "#F59E0B"},
            )

    # 2. Shade Predicted Anomaly Intervals (Red)
    pred_ivs = anomaly_intervals
    if pred_ivs is None and y_pred is not None:
        pred_ivs = group_intervals(y_pred, timestamps=x_vals)

    if pred_ivs is not None:
        for i, iv in enumerate(pred_ivs):
            x0 = iv["start_ts"] if iv["start_ts"] is not None else iv["start_idx"]
            x1 = iv["end_ts"] if iv["end_ts"] is not None else iv["end_idx"]
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=COLOR_PRED_ANOMALY,
                line_width=1,
                line_color="rgba(239, 68, 68, 0.8)",
                layer="below",
                annotation_text="Predicted Anomaly" if i == 0 else None,
                annotation_position="bottom left",
                annotation_font={"size": 10, "color": "#EF4444"},
            )

    _apply_dark_styling(fig, title)
    fig.update_layout(
        height=max(450, 180 * max_feats_to_plot),
        hovermode="x unified",
    )
    return fig


def plot_anomaly_scores(
    step_scores: np.ndarray,
    timestamps: Optional[Union[np.ndarray, list, pd.Series]] = None,
    threshold: float = 0.5,
    y_true: Optional[np.ndarray] = None,
    title: str = "Anomaly Scores Over Time",
    ground_truth_intervals: Optional[list[dict[str, Any]]] = None,
    anomaly_intervals: Optional[list[dict[str, Any]]] = None,
) -> go.Figure:
    """
    Plots continuous step-level anomaly scores with threshold and shaded regions.

    Args:
        step_scores: 1D array of anomaly probabilities in [0.0, 1.0].
        timestamps: Timestamps or integer step indices.
        threshold: Decision threshold cutoff.
        y_true: Optional ground truth binary step labels.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    scores = np.asarray(step_scores, dtype=float).ravel()
    n_samples = len(scores)

    if timestamps is not None:
        if isinstance(timestamps, (pd.Series, pd.Index)):
            x_vals = timestamps.values
        else:
            x_vals = np.asarray(timestamps)
    else:
        x_vals = np.arange(n_samples)

    fig = go.Figure()

    # Shade ground truth anomalies if provided
    gt_intervals = ground_truth_intervals
    if gt_intervals is None and y_true is not None:
        gt_intervals = group_intervals(y_true, timestamps=x_vals)
    if gt_intervals is not None:
        for i, iv in enumerate(gt_intervals):
            x0 = iv["start_ts"] if iv["start_ts"] is not None else iv["start_idx"]
            x1 = iv["end_ts"] if iv["end_ts"] is not None else iv["end_idx"]
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=COLOR_TRUE_ANOMALY,
                line_width=0,
                layer="below",
                annotation_text="True Anomaly" if i == 0 else None,
                annotation_position="top left",
                annotation_font={"size": 10, "color": "#F59E0B"},
            )

    if anomaly_intervals is not None:
        for i, iv in enumerate(anomaly_intervals):
            x0 = iv["start_ts"] if iv["start_ts"] is not None else iv["start_idx"]
            x1 = iv["end_ts"] if iv["end_ts"] is not None else iv["end_idx"]
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=COLOR_PRED_ANOMALY,
                line_width=0,
                layer="below",
                annotation_text="Predicted Anomaly" if i == 0 else None,
                annotation_position="bottom left",
                annotation_font={"size": 10, "color": "#EF4444"},
            )

    # Anomaly score trace
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=scores,
            mode="lines",
            name="Anomaly Score",
            line={"color": COLOR_LINE_PRIMARY, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.08)",
            hovertemplate="Time: %{x}<br>Score: %{y:.4f}<extra></extra>",
        )
    )

    # Highlight points above threshold
    above_mask = scores >= threshold
    if np.any(above_mask):
        fig.add_trace(
            go.Scatter(
                x=x_vals[above_mask],
                y=scores[above_mask],
                mode="markers",
                name="Anomalous Steps (Score >= Th)",
                marker={"color": COLOR_LINE_DANGER, "size": 4.5, "symbol": "circle"},
                hovertemplate="Time: %{x}<br>Anomaly Score: %{y:.4f}<extra></extra>",
            )
        )

    # Decision threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=COLOR_LINE_WARNING,
        line_width=2,
        annotation_text=f"Decision Threshold: {threshold:.3f}",
        annotation_position="top right",
        annotation_font={"size": 12, "color": COLOR_LINE_WARNING},
    )

    _apply_dark_styling(fig, title)
    fig.update_layout(
        yaxis={"title": "Anomaly Probability", "range": [0.0, max(1.05, float(np.max(scores) + 0.05))]},
        xaxis={"title": "Time / Step"},
        height=420,
        hovermode="x unified",
    )
    return fig


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    loc_losses: Optional[list[float]] = None,
    window_losses: Optional[list[float]] = None,
    kd_losses: Optional[list[float]] = None,
) -> go.Figure:
    """
    Plots interactive training and validation loss curves and constituent losses.

    Args:
        train_losses: List of epoch total training losses.
        val_losses: List of epoch total validation losses.
        loc_losses: List of localization BCE losses per epoch.
        window_losses: List of window classification CE losses per epoch.
        kd_losses: List of knowledge distillation KL losses per epoch.

    Returns:
        Plotly Figure.
    """
    epochs = list(range(1, len(train_losses) + 1))
    has_components = (
        loc_losses is not None
        and len(loc_losses) > 0
        and (any(l > 0 for l in loc_losses) or any(l > 0 for l in (window_losses or [])))
    )

    if has_components:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            subplot_titles=["<b>Total Loss (Train vs Val)</b>", "<b>Loss Components Breakdown</b>"],
        )
    else:
        fig = go.Figure()

    # Panel 1: Total Train vs Val
    target_row = 1 if has_components else None
    target_col = 1 if has_components else None

    tr_trace = go.Scatter(
        x=epochs,
        y=train_losses,
        mode="lines+markers",
        name="Train Total Loss",
        line={"color": COLOR_LINE_PRIMARY, "width": 2.5},
        marker={"size": 6},
    )
    val_trace = go.Scatter(
        x=epochs,
        y=val_losses,
        mode="lines+markers",
        name="Validation Loss",
        line={"color": "#F43F5E", "width": 2.5, "dash": "solid"},
        marker={"size": 6},
    )

    if has_components:
        fig.add_trace(tr_trace, row=1, col=1)
        fig.add_trace(val_trace, row=1, col=1)
    else:
        fig.add_trace(tr_trace)
        fig.add_trace(val_trace)

    # Panel 2: Components
    if has_components:
        if loc_losses:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=loc_losses,
                    mode="lines+markers",
                    name="Localization Loss (BCE)",
                    line={"color": COLOR_LINE_SUCCESS, "width": 2},
                    marker={"size": 5},
                ),
                row=2,
                col=1,
            )
        if window_losses:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=window_losses,
                    mode="lines+markers",
                    name="Window Loss (CE)",
                    line={"color": "#FB923C", "width": 2},
                    marker={"size": 5},
                ),
                row=2,
                col=1,
            )
        if kd_losses:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=kd_losses,
                    mode="lines+markers",
                    name="Distillation Loss (KD)",
                    line={"color": COLOR_LINE_SECONDARY, "width": 2},
                    marker={"size": 5},
                ),
                row=2,
                col=1,
            )

    _apply_dark_styling(fig, "Training & Validation Convergence History")
    fig.update_layout(
        height=550 if has_components else 400,
        hovermode="x unified",
        xaxis={"title": "Epoch", "dtick": 1 if len(epochs) <= 30 else 5},
    )
    return fig


def plot_metrics_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Creates a grouped bar chart comparing models across standard anomaly metrics.

    Args:
        comparison_df: DataFrame output from src.evaluation.metrics.compare_models.

    Returns:
        Plotly Figure.
    """
    metric_cols = ["Precision", "Recall", "F1 Score", "AUPRC", "ROC-AUC"]
    available_metrics = [c for c in metric_cols if c in comparison_df.columns]

    models = comparison_df["Model"].tolist()
    palette = ["#60A5FA", "#C084FC", "#34D399", "#FBBF24"]

    fig = go.Figure()

    for idx, model in enumerate(models):
        row = comparison_df[comparison_df["Model"] == model].iloc[0]
        y_values = [float(row[m]) for m in available_metrics]
        color = palette[idx % len(palette)]

        fig.add_trace(
            go.Bar(
                name=model,
                x=available_metrics,
                y=y_values,
                text=[f"{v:.3f}" for v in y_values],
                textposition="outside",
                textfont={"size": 11, "color": "#F1F5F9"},
                marker_color=color,
            )
        )

    _apply_dark_styling(fig, "Model Performance Comparison (Evaluation Metrics)")
    fig.update_layout(
        barmode="group",
        yaxis={"title": "Score (0.0 – 1.0)", "range": [0.0, 1.15]},
        xaxis={"title": "Evaluation Metric"},
        height=450,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] = ["Normal", "Anomaly"],
) -> go.Figure:
    """
    Renders an annotated confusion matrix heatmap with counts and percentage breakdowns.

    Args:
        cm: 2x2 confusion matrix array [[TN, FP], [FN, TP]].
        labels: Class display names.

    Returns:
        Plotly Figure.
    """
    cm_arr = np.asarray(cm, dtype=int)
    total = np.sum(cm_arr)
    row_sums = np.sum(cm_arr, axis=1, keepdims=True)

    text_matrix: list[list[str]] = []
    for i in range(len(labels)):
        row_txt: list[str] = []
        for j in range(len(labels)):
            val = cm_arr[i, j]
            tot_pct = (val / total * 100.0) if total > 0 else 0.0
            row_pct = (val / row_sums[i, 0] * 100.0) if row_sums[i, 0] > 0 else 0.0
            row_txt.append(f"<b>{val:,}</b><br>({tot_pct:.1f}% overall)<br>[{row_pct:.1f}% of row]")
        text_matrix.append(row_txt)

    colorscale = [
        [0.0, "#1E1E2E"],
        [0.5, "#2563EB"],
        [1.0, "#10B981"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=[f"Pred: {lbl}" for lbl in labels],
            y=[f"True: {lbl}" for lbl in labels],
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#FFFFFF", "family": "Inter, Segoe UI"},
            colorscale=colorscale,
            showscale=True,
            colorbar={"title": "Sample Count"},
        )
    )

    _apply_dark_styling(fig, "Confusion Matrix (Classification Breakdown)")
    fig.update_layout(
        xaxis={"title": "<b>Predicted Label</b>", "side": "bottom"},
        yaxis={"title": "<b>Ground Truth Label</b>", "autorange": "reversed"},
        height=430,
        width=480,
    )
    return fig


def plot_threshold_curve(curve_data: dict[str, Any]) -> go.Figure:
    """
    Plots Precision, Recall, and F1 curves across the decision threshold spectrum.

    Args:
        curve_data: Dict with keys: thresholds, f1, precision, recall, best_threshold, best_f1.

    Returns:
        Plotly Figure.
    """
    thresholds = curve_data["thresholds"]
    f1 = curve_data["f1"]
    precision = curve_data["precision"]
    recall = curve_data["recall"]
    best_th = float(curve_data.get("best_threshold", 0.5))
    best_f1 = float(curve_data.get("best_f1", 0.0))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=precision,
            mode="lines",
            name="Precision",
            line={"color": COLOR_LINE_PRIMARY, "width": 2},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=recall,
            mode="lines",
            name="Recall",
            line={"color": "#FB923C", "width": 2},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=f1,
            mode="lines",
            name="F1 Score",
            line={"color": COLOR_LINE_SUCCESS, "width": 3},
        )
    )

    # Vertical line highlighting optimal threshold
    fig.add_vline(
        x=best_th,
        line_dash="dash",
        line_color=COLOR_LINE_WARNING,
        line_width=2,
        annotation_text=f"Optimal Threshold: {best_th:.3f} (F1 = {best_f1:.3f})",
        annotation_position="top right",
        annotation_font={"size": 12, "color": COLOR_LINE_WARNING},
    )

    _apply_dark_styling(fig, "Precision, Recall & F1 Curve vs Decision Threshold")
    fig.update_layout(
        xaxis={"title": "Decision Threshold", "range": [0.0, 1.0]},
        yaxis={"title": "Metric Value (0.0 – 1.0)", "range": [0.0, 1.05]},
        height=430,
        hovermode="x unified",
    )
    return fig


def plot_teacher_score_distribution(
    teacher_scores: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    Plots a histogram distribution of teacher anomaly probability scores.

    Optionally stratifies the distribution by Normal vs Anomaly ground truth classes.

    Args:
        teacher_scores: Continuous anomaly probability predictions from VLM Teacher.
        y_true: Optional binary ground truth labels (0 = normal, 1 = anomaly).

    Returns:
        Plotly Figure.
    """
    scores = np.asarray(teacher_scores, dtype=float).ravel()

    fig = go.Figure()

    if y_true is not None:
        targets = np.asarray(y_true, dtype=int).ravel()
        normal_scores = scores[targets == 0]
        anomaly_scores = scores[targets == 1]

        fig.add_trace(
            go.Histogram(
                x=normal_scores,
                name="Normal Windows (Label 0)",
                marker_color="#38BDF8",
                opacity=0.65,
                xbins={"start": 0.0, "end": 1.0, "size": 0.025},
            )
        )

        fig.add_trace(
            go.Histogram(
                x=anomaly_scores,
                name="Anomalous Windows (Label 1)",
                marker_color="#EF4444",
                opacity=0.70,
                xbins={"start": 0.0, "end": 1.0, "size": 0.025},
            )
        )

        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(
            go.Histogram(
                x=scores,
                name="Teacher Anomaly Probabilities",
                marker_color="#818CF8",
                opacity=0.8,
                xbins={"start": 0.0, "end": 1.0, "size": 0.025},
            )
        )

    _apply_dark_styling(fig, "Teacher Soft Target Score Distribution")
    fig.update_layout(
        xaxis={"title": "Teacher Anomaly Probability (P_anomaly)", "range": [0.0, 1.0]},
        yaxis={"title": "Window Count"},
        height=400,
    )
    return fig
