"""
Localization and event-level evaluation utilities for time series anomaly detection.

Aggregates window-level step scores into a continuous timeline, extracts contiguous
anomaly events, and calculates event-level precision, recall, and overlap F1.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_step_scores(
    step_scores_per_window: np.ndarray,
    window_starts: Optional[Union[list[int], np.ndarray]] = None,
    window_length: int = 128,
    stride: int = 32,
    total_length: Optional[int] = None,
) -> np.ndarray:
    """
    Aggregates step-level anomaly scores from overlapping sliding windows into a single timeline.

    Overlapping positions are averaged across windows, guaranteeing that no point
    is duplicated or distorted.

    Args:
        step_scores_per_window: Array of shape (N_windows, window_length)
        window_starts: Start index for each window in the original timeline.
            If None, generated using: [i * stride for i in range(N_windows)].
        window_length: Length of each window.
        stride: Stride between successive windows.
        total_length: Total number of time steps in the original sequence.
            If None, inferred as max(window_starts) + window_length.

    Returns:
        1D array of shape (total_length,) containing aggregated step scores.
    """
    scores_win = np.asarray(step_scores_per_window, dtype=float)
    n_windows = len(scores_win)

    if n_windows == 0:
        return np.zeros(total_length or 0, dtype=float)

    if window_starts is None:
        starts = np.array([i * stride for i in range(n_windows)], dtype=int)
    else:
        starts = np.asarray(window_starts, dtype=int)

    if total_length is None:
        tot_len = int(np.max(starts) + window_length)
    else:
        tot_len = int(total_length)

    sum_scores = np.zeros(tot_len, dtype=float)
    counts = np.zeros(tot_len, dtype=float)

    for i in range(n_windows):
        s = int(starts[i])
        if s >= tot_len:
            continue
        e = min(s + window_length, tot_len)
        span = e - s
        sum_scores[s:e] += scores_win[i, :span]
        counts[s:e] += 1.0

    # Average overlapping points; if a point wasn't covered, assign 0.0
    valid_mask = counts > 0
    aggregated = np.zeros(tot_len, dtype=float)
    aggregated[valid_mask] = sum_scores[valid_mask] / counts[valid_mask]

    return aggregated


def group_intervals(
    y_pred_steps: np.ndarray,
    timestamps: Optional[Union[np.ndarray, pd.Series, list]] = None,
    independent_observations: bool = False,
) -> list[dict[str, Any]]:
    """
    Groups contiguous anomalous time steps into discrete event intervals.

    Args:
        y_pred_steps: Binary step-level predictions (0 or 1), shape (N,).
        timestamps: Optional sequence of timestamps corresponding to time steps.
        independent_observations: Treat each flagged row as its own event. This
            is appropriate for transaction datasets where row adjacency does not
            imply one sustained time-series event.

    Returns:
        List of event dictionaries:
        [{'start_idx': int, 'end_idx': int, 'start_ts': Any, 'end_ts': Any, 'length': int}]
    """
    steps = np.asarray(y_pred_steps, dtype=int).ravel()
    n = len(steps)
    if n == 0:
        return []

    intervals: list[dict[str, Any]] = []
    in_event = False
    start_idx = 0

    ts_arr = None
    if timestamps is not None:
        if isinstance(timestamps, (pd.Series, pd.Index)):
            ts_arr = timestamps.values
        else:
            ts_arr = np.asarray(timestamps)

    if independent_observations:
        return [
            {
                "start_idx": int(i),
                "end_idx": int(i),
                "start_ts": ts_arr[i] if ts_arr is not None else None,
                "end_ts": ts_arr[i] if ts_arr is not None else None,
                "length": 1,
            }
            for i in np.flatnonzero(steps == 1)
        ]

    for i in range(n):
        if steps[i] == 1 and not in_event:
            in_event = True
            start_idx = i
        elif steps[i] == 0 and in_event:
            in_event = False
            end_idx = i - 1
            length = end_idx - start_idx + 1
            start_ts = ts_arr[start_idx] if ts_arr is not None else None
            end_ts = ts_arr[end_idx] if ts_arr is not None else None
            intervals.append({
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "length": int(length),
            })

    # Close any trailing open event at sequence end
    if in_event:
        end_idx = n - 1
        length = end_idx - start_idx + 1
        start_ts = ts_arr[start_idx] if ts_arr is not None else None
        end_ts = ts_arr[end_idx] if ts_arr is not None else None
        intervals.append({
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "length": int(length),
        })

    return intervals


def event_overlap_score(
    true_intervals: list[dict[str, Any]],
    pred_intervals: list[dict[str, Any]],
) -> dict[str, Union[float, int]]:
    """
    Computes event-level precision, recall, and F1 based on temporal interval overlap.

    A true anomaly event is considered successfully detected (recalled) if at least
    one predicted anomaly event overlaps with it in time.

    A predicted anomaly event is considered valid (precision true positive) if it
    overlaps with at least one true anomaly event.

    Args:
        true_intervals: List of ground truth intervals from group_intervals().
        pred_intervals: List of predicted intervals from group_intervals().

    Returns:
        Dictionary with:
            {'event_recall', 'event_precision', 'event_f1', 'n_true_events', 'n_pred_events'}
    """
    n_true = len(true_intervals)
    n_pred = len(pred_intervals)

    if n_true == 0 and n_pred == 0:
        return {
            "event_recall": 1.0,
            "event_precision": 1.0,
            "event_f1": 1.0,
            "n_true_events": 0,
            "n_pred_events": 0,
        }

    if n_true == 0 and n_pred > 0:
        return {
            "event_recall": 1.0,
            "event_precision": 0.0,
            "event_f1": 0.0,
            "n_true_events": 0,
            "n_pred_events": n_pred,
        }

    if n_true > 0 and n_pred == 0:
        return {
            "event_recall": 0.0,
            "event_precision": 0.0,
            "event_f1": 0.0,
            "n_true_events": n_true,
            "n_pred_events": 0,
        }

    def overlaps(iv1: dict[str, Any], iv2: dict[str, Any]) -> bool:
        s1, e1 = iv1["start_idx"], iv1["end_idx"]
        s2, e2 = iv2["start_idx"], iv2["end_idx"]
        return max(s1, s2) <= min(e1, e2)

    # Event Recall: fraction of true events overlapping at least one predicted event
    true_detected_count = sum(
        1 for t_iv in true_intervals if any(overlaps(t_iv, p_iv) for p_iv in pred_intervals)
    )
    event_recall = true_detected_count / n_true

    # Event Precision: fraction of predicted events overlapping at least one true event
    pred_valid_count = sum(
        1 for p_iv in pred_intervals if any(overlaps(p_iv, t_iv) for t_iv in true_intervals)
    )
    event_precision = pred_valid_count / n_pred

    # Event F1
    if (event_precision + event_recall) > 0.0:
        event_f1 = (2.0 * event_precision * event_recall) / (event_precision + event_recall)
    else:
        event_f1 = 0.0

    return {
        "event_recall": float(event_recall),
        "event_precision": float(event_precision),
        "event_f1": float(event_f1),
        "n_true_events": n_true,
        "n_pred_events": n_pred,
    }


def map_to_timestamps(
    indices: Union[np.ndarray, list[int]],
    timestamps: Union[np.ndarray, pd.Series, list],
) -> np.ndarray:
    """
    Safely maps array index positions back to their real-world timestamp representations.

    Args:
        indices: 1D array of integer indices.
        timestamps: 1D array or Series of timestamps.

    Returns:
        Array of mapped timestamps.
    """
    idx = np.asarray(indices, dtype=int).ravel()
    if isinstance(timestamps, (pd.Series, pd.Index)):
        ts_arr = timestamps.values
    else:
        ts_arr = np.asarray(timestamps)

    n_ts = len(ts_arr)
    valid_mask = (idx >= 0) & (idx < n_ts)
    safe_idx = np.clip(idx, 0, max(0, n_ts - 1))

    mapped = ts_arr[safe_idx]
    if not np.all(valid_mask):
        logger.warning("Some indices were out of bounds when mapping to timestamps.")

    return mapped


def point_adjustment(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> np.ndarray:
    """Apply point adjustment heuristic to predicted anomalies.

    If any single timestep within a true anomaly segment is detected,
    all timesteps in that true anomaly segment are marked as detected (1).
    Points outside true anomalies retain their original predictions.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Binary predicted labels (0 or 1).

    Returns:
        Adjusted binary prediction array of same length.
    """
    targets = np.asarray(y_true, dtype=int).ravel()
    preds = np.asarray(y_pred, dtype=int).copy().ravel()

    intervals = group_intervals(targets)
    for iv in intervals:
        s, e = iv["start_idx"], iv["end_idx"] + 1
        if np.any(preds[s:e] == 1):
            preds[s:e] = 1

    return preds


def iou_per_anomaly(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> list[float]:
    """Compute Intersection-over-Union (IoU) for each true anomaly segment.

    For each contiguous true anomaly segment:
    Finds intersecting predicted anomaly segments.
    Intersection = count of timesteps where both true and predicted are 1 in overlapping span.
    Union = count of timesteps where either true or predicted is 1.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        List of float IoU values, one per true anomaly event. Empty if no true anomalies.
    """
    targets = np.asarray(y_true, dtype=int).ravel()
    preds = np.asarray(y_pred, dtype=int).ravel()

    true_events = group_intervals(targets)
    pred_events = group_intervals(preds)

    if not true_events:
        return []

    iou_scores = []
    for t_ev in true_events:
        t_start, t_end = t_ev["start_idx"], t_ev["end_idx"]
        t_indices = set(range(t_start, t_end + 1))

        # Find all pred events that overlap with this true event
        overlapping_pred_indices = set()
        for p_ev in pred_events:
            p_start, p_end = p_ev["start_idx"], p_ev["end_idx"]
            p_indices = set(range(p_start, p_end + 1))
            if not t_indices.isdisjoint(p_indices):
                overlapping_pred_indices.update(p_indices)

        intersection = len(t_indices.intersection(overlapping_pred_indices))
        union = len(t_indices.union(overlapping_pred_indices))

        if union > 0:
            iou_scores.append(float(intersection / union))
        else:
            iou_scores.append(0.0)

    return iou_scores


def mean_iou(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> float:
    """Compute mean Intersection-over-Union across all true anomaly events.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Mean IoU as a float. Returns 1.0 if both true and predicted have no anomalies.
    """
    scores = iou_per_anomaly(y_true, y_pred)
    if not scores:
        preds = np.asarray(y_pred, dtype=int).ravel()
        return 1.0 if np.sum(preds) == 0 else 0.0
    return float(np.mean(scores))


def event_f1(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> float:
    """Compute event-level F1 score based on interval overlap.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Event-level F1 score in [0.0, 1.0].
    """
    targets = np.asarray(y_true, dtype=int).ravel()
    preds = np.asarray(y_pred, dtype=int).ravel()

    true_intervals = group_intervals(targets)
    pred_intervals = group_intervals(preds)

    metrics = event_overlap_score(true_intervals, pred_intervals)
    return float(metrics["event_f1"])

