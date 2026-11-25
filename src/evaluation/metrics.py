"""
Metrics computation and comparison module for time series anomaly detection.

Handles class imbalance warnings, missing ground truth, and edge cases.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Standardized container for anomaly detection evaluation metrics."""

    precision: float
    recall: float
    f1: float
    auprc: float
    roc_auc: float
    confusion_matrix: np.ndarray
    threshold: float
    n_true_anomalies: int
    n_predicted_anomalies: int
    has_ground_truth: bool
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Converts result object to a dictionary."""
        d = asdict(self)
        if isinstance(d["confusion_matrix"], np.ndarray):
            d["confusion_matrix"] = d["confusion_matrix"].tolist()
        return d


def compute_metrics(
    y_true: Optional[Union[np.ndarray, list]],
    y_scores: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> MetricsResult:
    """
    Computes comprehensive anomaly detection metrics.

    Handles edge cases:
    - Missing ground truth labels (returns zeroed metrics with warning)
    - All-zero ground truth (no true anomalies)
    - All-zero or constant predictions
    - Severe class imbalance (ratio > 100:1)

    Args:
        y_true: Ground truth binary array (0 = normal, 1 = anomaly), or None.
        y_scores: Continuous anomaly probability scores in [0.0, 1.0].
        threshold: Decision threshold to convert scores to binary predictions.

    Returns:
        MetricsResult dataclass instance.
    """
    scores = np.asarray(y_scores, dtype=float).ravel()
    threshold = float(threshold)
    y_pred = (scores >= threshold).astype(int)
    n_pred_anomalies = int(np.sum(y_pred == 1))

    # Case 1: No ground truth available
    if y_true is None or len(y_true) == 0:
        return MetricsResult(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            auprc=0.0,
            roc_auc=0.0,
            confusion_matrix=np.zeros((2, 2), dtype=int),
            threshold=threshold,
            n_true_anomalies=0,
            n_predicted_anomalies=n_pred_anomalies,
            has_ground_truth=False,
            warning="No ground truth labels provided; classification metrics unavailable.",
        )

    targets = np.asarray(y_true, dtype=int).ravel()
    if len(targets) != len(scores):
        raise ValueError(f"Shape mismatch: y_true has {len(targets)} elements, y_scores has {len(scores)}")

    n_true_anomalies = int(np.sum(targets == 1))
    n_normal = int(np.sum(targets == 0))
    unique_targets = np.unique(targets)

    # Compute confusion matrix with fixed labels [0, 1]
    cm = confusion_matrix(targets, y_pred, labels=[0, 1])

    # Precision, Recall, F1
    prec = float(precision_score(targets, y_pred, pos_label=1, zero_division=0.0))
    rec = float(recall_score(targets, y_pred, pos_label=1, zero_division=0.0))
    f1 = float(f1_score(targets, y_pred, pos_label=1, zero_division=0.0))

    # ROC-AUC & AUPRC
    warning_msgs: list[str] = []

    if len(unique_targets) < 2:
        roc_auc = 0.0
        auprc = 0.0
        warning_msgs.append("Ground truth contains only one class. ROC-AUC and AUPRC are undefined.")
    else:
        try:
            roc_auc = float(roc_auc_score(targets, scores))
        except Exception as e:
            logger.debug("ROC-AUC computation failed: %s", e)
            roc_auc = 0.0

        try:
            auprc = float(average_precision_score(targets, scores))
        except Exception as e:
            logger.debug("AUPRC computation failed: %s", e)
            auprc = 0.0

    # Class imbalance check
    if n_true_anomalies > 0 and n_normal > 0:
        imbalance_ratio = n_normal / n_true_anomalies
        if imbalance_ratio > 100.0:
            warning_msgs.append(
                f"Severe class imbalance detected ({imbalance_ratio:.1f}:1 normal-to-anomaly). "
                f"AUPRC ({auprc:.4f}) and F1 ({f1:.4f}) are more reliable than ROC-AUC ({roc_auc:.4f})."
            )

    if n_pred_anomalies == 0:
        warning_msgs.append("Zero anomalies predicted at current threshold.")

    warning_text = " ".join(warning_msgs)

    return MetricsResult(
        precision=prec,
        recall=rec,
        f1=f1,
        auprc=auprc,
        roc_auc=roc_auc,
        confusion_matrix=cm,
        threshold=threshold,
        n_true_anomalies=n_true_anomalies,
        n_predicted_anomalies=n_pred_anomalies,
        has_ground_truth=True,
        warning=warning_text,
    )


def compare_models(results: dict[str, MetricsResult]) -> pd.DataFrame:
    """
    Transforms multiple model evaluation results into a formatted comparison DataFrame.

    Args:
        results: Dictionary mapping model names to MetricsResult objects.
            e.g. {'Teacher-Only': res1, 'Student-Baseline': res2, 'Student-KD': res3}

    Returns:
        Pandas DataFrame ready for display or reporting.
    """
    rows: list[dict[str, Any]] = []

    for model_name, res in results.items():
        rows.append({
            "Model": model_name,
            "Precision": round(res.precision, 4),
            "Recall": round(res.recall, 4),
            "F1 Score": round(res.f1, 4),
            "AUPRC": round(res.auprc, 4),
            "ROC-AUC": round(res.roc_auc, 4),
            "True Anomalies": res.n_true_anomalies,
            "Pred Anomalies": res.n_predicted_anomalies,
            "Threshold": round(res.threshold, 4),
            "Imbalance / Warning": res.warning if res.warning else "Normal",
        })

    df = pd.DataFrame(rows)
    return df


def compute_window_metrics(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute precision, recall, f1, pr_auc, and roc_auc for window predictions."""
    res = compute_metrics(y_true, y_scores, threshold=threshold)
    return {
        "precision": res.precision,
        "recall": res.recall,
        "f1": res.f1,
        "pr_auc": res.auprc,
        "roc_auc": res.roc_auc,
        "threshold": res.threshold,
    }


def compute_step_metrics(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute step-level point-wise precision, recall, f1, pr_auc, and roc_auc."""
    res = compute_metrics(y_true, y_scores, threshold=threshold)
    return {
        "precision": res.precision,
        "recall": res.recall,
        "f1": res.f1,
        "pr_auc": res.auprc,
        "roc_auc": res.roc_auc,
        "threshold": res.threshold,
    }


def tune_threshold(
    y_val: Union[np.ndarray, list],
    scores_val: Union[np.ndarray, list],
    metric: str = "f1",
    n_thresholds: int = 100,
) -> float:
    """Tune decision threshold on validation data to optimize the chosen metric."""
    from src.evaluation.threshold import select_threshold

    best_th, _ = select_threshold(
        y_val=np.asarray(y_val),
        scores_val=np.asarray(scores_val),
        objective=metric,
        n_thresholds=n_thresholds,
    )
    return float(best_th)


def confusion_matrix_at_threshold(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute confusion matrix at specified threshold."""
    res = compute_metrics(y_true, y_scores, threshold=threshold)
    return res.confusion_matrix


def format_metrics_table(results: Dict[str, Any]) -> str:
    """Format dictionary or MetricsResult into a human-readable ASCII table."""
    lines = [
        "=" * 50,
        f"{'Metric':<25} | {'Value':<20}",
        "-" * 50,
    ]
    for k, v in results.items():
        if isinstance(v, float):
            val_str = f"{v:.4f}"
        else:
            val_str = str(v)
        lines.append(f"{k:<25} | {val_str:<20}")
    lines.append("=" * 50)
    return "\n".join(lines)

