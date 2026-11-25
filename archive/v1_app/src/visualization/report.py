"""Reporting and export utilities for the IE643 VLM-KD Anomaly Detection system.

Handles generating downloadable CSV reports for predictions, evaluation metrics,
and packaging full experiment archives (models, metrics, figures) into ZIP files.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.evaluation.localization import group_intervals

logger = logging.getLogger(__name__)


def predictions_to_csv(
    timestamps: Optional[Union[np.ndarray, List[Any], pd.Series]],
    scores: Union[np.ndarray, List[float]],
    predictions: Union[np.ndarray, List[int]],
    feature_df: Optional[pd.DataFrame] = None,
    original_df: Optional[pd.DataFrame] = None,
    student_predictions: Optional[Union[np.ndarray, List[int]]] = None,
    contextual_scores: Optional[Union[np.ndarray, List[float]]] = None,
    contextual_predictions: Optional[Union[np.ndarray, List[int]]] = None,
    contextual_drivers: Optional[Union[np.ndarray, List[str]]] = None,
    contextual_available: Optional[Union[np.ndarray, List[bool]]] = None,
    student_threshold: Optional[float] = None,
    contextual_threshold: Optional[float] = None,
    context_entity_column: Optional[str] = None,
    independent_observations: bool = False,
) -> str:
    """Export time-step anomaly predictions and continuous scores to CSV format.

    Args:
        timestamps: Timestamps or integer step indices.
        scores: Continuous anomaly probability scores in [0.0, 1.0].
        predictions: Binary prediction flags (0 = normal, 1 = anomaly).
        feature_df: Optional DataFrame of original raw or scaled feature values.
        original_df: Optional uploaded CSV, preserved in the output alongside predictions.
        student_predictions: The student's thresholded decision before any contextual alerts.
        contextual_scores: Optional per-entity robust-z values from past-only context.
        contextual_predictions: Optional contextual flags, kept separate from student flags.
        contextual_drivers: Optional feature names with the largest contextual deviation.
        contextual_available: Whether enough prior observations existed for comparison.
        student_threshold: Probability cutoff applied to the student's anomaly score.
        contextual_threshold: Robust-z cutoff used for contextual flags.
        context_entity_column: Input column defining independent context groups.
        independent_observations: If true, adjacent flagged rows remain separate events.

    Returns:
        CSV string representation.
    """
    scores_arr = np.asarray(scores, dtype=float).ravel()
    preds_arr = np.asarray(predictions, dtype=int).ravel()
    n_samples = len(scores_arr)

    if timestamps is not None:
        if isinstance(timestamps, (pd.Series, pd.Index)):
            ts_vals = timestamps.values
        else:
            ts_vals = np.asarray(timestamps)
        if len(ts_vals) != n_samples:
            ts_vals = np.arange(n_samples)
    else:
        ts_vals = np.arange(n_samples)

    if original_df is not None:
        if len(original_df) != n_samples:
            raise ValueError("original_df must have the same number of rows as predictions.")
        df = original_df.reset_index(drop=True).copy()
    else:
        df = pd.DataFrame()

    if "timestamp" not in df.columns:
        df["timestamp"] = ts_vals
    df["timestep_index"] = np.arange(n_samples, dtype=np.int64)
    df["anomaly_score"] = np.round(scores_arr, 6)
    df["anomaly_score_method"] = "GRU_student_probability"
    df["student_threshold"] = (
        float(student_threshold) if student_threshold is not None else np.nan
    )
    df["is_anomaly"] = preds_arr

    student_arr = (
        preds_arr.copy()
        if student_predictions is None
        else np.asarray(student_predictions, dtype=int).ravel()
    )
    if len(student_arr) != n_samples:
        raise ValueError("student_predictions must have the same length as scores.")
    context_scores_arr = (
        np.full(n_samples, np.nan, dtype=float)
        if contextual_scores is None
        else np.asarray(contextual_scores, dtype=float).ravel()
    )
    context_flags_arr = (
        np.zeros(n_samples, dtype=int)
        if contextual_predictions is None
        else np.asarray(contextual_predictions, dtype=int).ravel()
    )
    context_drivers_arr = (
        np.full(n_samples, "", dtype=object)
        if contextual_drivers is None
        else np.asarray(contextual_drivers, dtype=object).ravel()
    )
    context_available_arr = (
        np.zeros(n_samples, dtype=bool)
        if contextual_available is None
        else np.asarray(contextual_available, dtype=bool).ravel()
    )
    for name, arr in (
        ("contextual_scores", context_scores_arr),
        ("contextual_predictions", context_flags_arr),
        ("contextual_drivers", context_drivers_arr),
        ("contextual_available", context_available_arr),
    ):
        if len(arr) != n_samples:
            raise ValueError(f"{name} must have the same length as scores.")

    df["student_anomaly_score"] = np.round(scores_arr, 6)
    df["is_student_anomaly"] = student_arr
    df["contextual_robust_z"] = np.round(context_scores_arr, 4)
    df["is_contextual_anomaly"] = context_flags_arr
    df["contextual_baseline_available"] = context_available_arr.astype(np.int8)
    df["contextual_driver_feature"] = context_drivers_arr
    df["context_entity_column"] = context_entity_column or ""
    df["contextual_threshold_robust_z"] = (
        float(contextual_threshold) if contextual_threshold is not None else np.nan
    )

    # Preserve domain semantics: row adjacency is only a collective event when
    # observations belong to a true temporal sequence.
    event_ids = pd.array([pd.NA] * n_samples, dtype="Int64")
    event_types = np.full(n_samples, "normal", dtype=object)
    anomaly_taxonomy = np.full(n_samples, "normal", dtype=object)
    detection_basis = np.full(n_samples, "none", dtype=object)
    event_starts = np.full(n_samples, "", dtype=object)
    event_ends = np.full(n_samples, "", dtype=object)
    event_lengths = pd.array([pd.NA] * n_samples, dtype="Int64")
    for event_id, interval in enumerate(
        group_intervals(
            preds_arr,
            timestamps=ts_vals,
            independent_observations=independent_observations,
        ),
        start=1,
    ):
        start_idx = interval["start_idx"]
        end_idx = interval["end_idx"]
        event_slice = slice(start_idx, end_idx + 1)
        event_ids[event_slice] = event_id
        if not independent_observations and interval["length"] > 1:
            event_types[event_slice] = "group_anomaly"
            anomaly_taxonomy[event_slice] = "collective_anomaly"
        elif bool(np.any(context_flags_arr[event_slice])):
            event_types[event_slice] = "contextual_anomaly"
            anomaly_taxonomy[event_slice] = "contextual_anomaly"
        else:
            event_types[event_slice] = "point_anomaly"
            anomaly_taxonomy[event_slice] = "point_anomaly"
        for row_idx in range(start_idx, end_idx + 1):
            if student_arr[row_idx] and context_flags_arr[row_idx]:
                detection_basis[row_idx] = "student+context"
            elif context_flags_arr[row_idx]:
                detection_basis[row_idx] = "contextual"
            elif student_arr[row_idx]:
                detection_basis[row_idx] = "student"
        event_starts[event_slice] = str(ts_vals[start_idx])
        event_ends[event_slice] = str(ts_vals[end_idx])
        event_lengths[event_slice] = int(interval["length"])

    df["detection_basis"] = detection_basis
    df["anomaly_type"] = event_types
    df["anomaly_taxonomy"] = anomaly_taxonomy
    df["anomaly_event_id"] = event_ids
    df["event_start_timestamp"] = event_starts
    df["event_end_timestamp"] = event_ends
    df["event_duration_steps"] = event_lengths
    df["event_scope"] = np.where(
        df["is_anomaly"].to_numpy(dtype=int) == 1,
        np.where(
            (event_lengths.fillna(1).to_numpy(dtype=int) > 1), "group", "point"
        ),
        "normal",
    )

    if feature_df is not None and len(feature_df) == n_samples:
        feat_cols = [c for c in feature_df.columns if c not in df.columns]
        for col in feat_cols:
            df[col] = feature_df[col].values

    return df.to_csv(index=False)


def metrics_to_csv(metrics_input: Union[Dict[str, Any], pd.DataFrame]) -> str:
    """Export evaluation metrics to CSV format.

    Args:
        metrics_input: Either a metrics dictionary (single model) or a DataFrame
            (e.g., from compare_models).

    Returns:
        CSV string representation.
    """
    if isinstance(metrics_input, pd.DataFrame):
        return metrics_input.to_csv(index=False)

    if isinstance(metrics_input, dict):
        clean_dict: Dict[str, Any] = {}
        for k, v in metrics_input.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                clean_dict[k] = json.dumps(np.asarray(v).tolist())
            else:
                clean_dict[k] = v
        df = pd.DataFrame([clean_dict])
        return df.to_csv(index=False)

    return pd.DataFrame().to_csv(index=False)


def create_report_zip(
    checkpoint_path: Optional[str] = None,
    metrics_csv: Optional[str] = None,
    predictions_csv: Optional[str] = None,
    metadata_json: Optional[Union[str, Dict[str, Any]]] = None,
    additional_files: Optional[Dict[str, Union[str, bytes]]] = None,
) -> bytes:
    """Package experiment artifacts into an in-memory ZIP archive for download.

    Args:
        checkpoint_path: Optional path to a checkpoint directory or .pt file.
        metrics_csv: Optional CSV string containing evaluation metrics.
        predictions_csv: Optional CSV string containing time-series predictions.
        metadata_json: Optional JSON string or dictionary containing configuration.
        additional_files: Optional dictionary mapping internal archive paths
            to string or byte content.

    Returns:
        Bytes of the constructed ZIP archive.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Predictions CSV
        if predictions_csv is not None:
            zf.writestr("reports/predictions.csv", predictions_csv.encode("utf-8"))

        # 2. Metrics CSV
        if metrics_csv is not None:
            zf.writestr("reports/metrics.csv", metrics_csv.encode("utf-8"))

        # 3. Metadata JSON
        if metadata_json is not None:
            if isinstance(metadata_json, dict):
                meta_str = json.dumps(metadata_json, indent=2)
            else:
                meta_str = str(metadata_json)
            zf.writestr("reports/metadata.json", meta_str.encode("utf-8"))

        # 4. Checkpoint files
        if checkpoint_path is not None:
            ckpt_p = Path(checkpoint_path)
            if ckpt_p.is_file():
                zf.write(ckpt_p, arcname=f"checkpoint/{ckpt_p.name}")
            elif ckpt_p.is_dir():
                for f_path in ckpt_p.rglob("*"):
                    if f_path.is_file():
                        rel_path = f_path.relative_to(ckpt_p)
                        zf.write(f_path, arcname=f"checkpoint/{rel_path}")

        # 5. Additional custom files
        if additional_files is not None:
            for arc_name, content in additional_files.items():
                if isinstance(content, str):
                    zf.writestr(arc_name, content.encode("utf-8"))
                elif isinstance(content, bytes):
                    zf.writestr(arc_name, content)

        # 6. Readme
        readme_content = (
            "IE643 VLM Teacher–RNN Student Anomaly Detection Report Archive\n"
            "===============================================================\n"
            "This archive contains:\n"
            "- reports/predictions.csv: Time-step scores and predicted anomaly labels\n"
            "- reports/metrics.csv: Quantitative benchmarking evaluation metrics\n"
            "- reports/metadata.json: Checkpoint hyperparameters and training config\n"
            "- checkpoint/: Saved model weights and preprocessing scaler\n"
        )
        zf.writestr("README.txt", readme_content.encode("utf-8"))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
