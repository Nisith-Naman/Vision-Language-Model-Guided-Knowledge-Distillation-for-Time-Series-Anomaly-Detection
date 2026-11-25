"""
Threshold selection and tuning module for anomaly detection.

Optimizes decision thresholds on validation splits to prevent test-set data leakage.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def select_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    objective: str = "f1",
    n_thresholds: int = 100,
) -> Tuple[float, dict]:
    """
    Sweeps thresholds over validation scores to identify the optimal cutoff.

    CRITICAL: Must strictly use validation data only, NEVER test labels.

    Args:
        y_val: Validation ground truth binary labels (0 or 1).
        scores_val: Validation predicted anomaly probability scores.
        objective: Optimization criterion ('f1', 'precision', 'recall', or 'balanced').
        n_thresholds: Approximate number of threshold candidates. Candidates adapt to
            validation-score quantiles, including both low-probability and rare high-score tails.

    Returns:
        Tuple of (best_threshold, info_dict)
        where info_dict contains:
            - 'threshold_array': np.ndarray of tested thresholds
            - 'metric_array': np.ndarray of objective metric values
            - 'best_idx': index of optimal threshold
            - 'precision_array': np.ndarray of precision scores
            - 'recall_array': np.ndarray of recall scores
            - 'f1_array': np.ndarray of F1 scores
            - 'objective': string name of the objective used
    """
    y_true = np.asarray(y_val, dtype=int).ravel()
    scores = np.asarray(scores_val, dtype=float).ravel()

    if len(y_true) != len(scores):
        raise ValueError(f"Length mismatch: y_val ({len(y_true)}) != scores_val ({len(scores)})")
    if len(scores) == 0:
        raise ValueError("Cannot tune a threshold on an empty validation score array.")
    if not np.isfinite(scores).all():
        raise ValueError("Validation scores contain NaN or infinite values.")

    # A fixed 0.01--0.99 grid can miss the useful operating range when a model's
    # probabilities are compressed below 0.01, or when positives occupy only a
    # tiny high-score tail. Candidate cutoffs therefore follow score quantiles,
    # with extra resolution near the top tail and midpoints between candidates.
    n_candidates = max(10, int(n_thresholds))
    uniform_quantiles = np.linspace(0.0, 1.0, n_candidates)
    tail_quantiles = 1.0 - np.geomspace(1e-6, 1e-1, num=max(10, n_candidates // 2))
    quantile_levels = np.unique(np.clip(np.r_[uniform_quantiles, tail_quantiles], 0.0, 1.0))
    quantile_scores = np.quantile(scores, quantile_levels)
    midpoints = (quantile_scores[:-1] + quantile_scores[1:]) / 2.0
    above_max = np.nextafter(float(np.max(scores)), np.inf)
    thresholds = np.unique(
        np.clip(np.r_[0.0, 0.5, quantile_scores, midpoints, above_max, 1.0], 0.0, 1.0)
    )

    # Vectorised counts (identical to sklearn's precision/recall/F1 with
    # zero_division=0, but O(n log n) instead of one full pass per threshold).
    pos_sorted = np.sort(scores[y_true == 1])
    neg_sorted = np.sort(scores[y_true != 1])
    n_pos = len(pos_sorted)
    tp = (n_pos - np.searchsorted(pos_sorted, thresholds, side="left")).astype(float)
    fp = (len(neg_sorted) - np.searchsorted(neg_sorted, thresholds, side="left")).astype(float)
    fn = n_pos - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precisions = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recalls = np.where(n_pos > 0, tp / max(n_pos, 1), 0.0)
        f1s = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)

    obj_lower = objective.lower()
    if obj_lower == "f1":
        metric_array = f1s
    elif obj_lower == "precision":
        metric_array = precisions
    elif obj_lower == "recall":
        metric_array = recalls
    elif obj_lower == "balanced":
        # Maximizes the minimum of precision and recall (balanced compromise)
        metric_array = np.minimum(precisions, recalls)
    else:
        logger.warning("Unrecognized objective '%s'. Defaulting to 'f1'.", objective)
        metric_array = f1s
        obj_lower = "f1"

    # Select best index
    max_metric = np.max(metric_array)
    if max_metric <= 1e-8:
        # If all candidates yield 0, fall back to median threshold 0.5
        best_idx = int(np.argmin(np.abs(thresholds - 0.5)))
        best_threshold = float(thresholds[best_idx])
        logger.warning("All thresholds yielded zero for objective '%s'. Defaulting to 0.5", objective)
    else:
        best_idx = int(np.argmax(metric_array))
        best_threshold = float(thresholds[best_idx])

    logger.info(
        "Optimal threshold selected: %.4f (Objective: %s = %.4f, P = %.4f, R = %.4f, F1 = %.4f)",
        best_threshold,
        obj_lower,
        metric_array[best_idx],
        precisions[best_idx],
        recalls[best_idx],
        f1s[best_idx],
    )

    info = {
        "threshold_array": thresholds,
        "metric_array": metric_array,
        "best_idx": best_idx,
        "precision_array": precisions,
        "recall_array": recalls,
        "f1_array": f1s,
        "objective": obj_lower,
        "best_metric": float(metric_array[best_idx]),
        "best_threshold": best_threshold,
    }

    return best_threshold, info


def threshold_curve_data(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    n_thresholds: int = 100,
) -> dict:
    """
    Computes Precision, Recall, and F1 curves across the full threshold spectrum.

    Args:
        y_val: Validation ground truth binary labels.
        scores_val: Validation continuous anomaly scores.
        n_thresholds: Number of evaluation points.

    Returns:
        Dictionary suitable for plotting threshold sweep curves:
            {'thresholds', 'f1', 'precision', 'recall', 'best_threshold', 'best_f1', 'best_idx'}
    """
    best_th, info = select_threshold(y_val, scores_val, objective="f1", n_thresholds=n_thresholds)

    return {
        "thresholds": info["threshold_array"],
        "f1": info["f1_array"],
        "precision": info["precision_array"],
        "recall": info["recall_array"],
        "best_threshold": best_th,
        "best_f1": float(info["f1_array"][info["best_idx"]]),
        "best_idx": info["best_idx"],
    }
