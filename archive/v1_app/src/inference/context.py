"""Causal, robust per-entity context checks for multivariate telemetry.

The student model detects patterns learned from its training labels. This
module adds a separate contextual signal when an input CSV supplies a real
entity key (for example, cardholder, patient, machine, or well ID). It compares
each observation only with that entity's preceding observations, so it never
uses future rows or mixes different entities' baselines.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


def robust_context_scores(
    values: np.ndarray,
    entity_ids: Sequence[Any],
    feature_names: Optional[Sequence[str]] = None,
    lookback: int = 64,
    min_history: int = 16,
    scale_floor: float = 0.25,
    score_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return max robust-z, its feature name, and whether context was available.

    For each row, the reference is the previous ``lookback`` rows from the
    same entity. Per-feature location is the median and scale is 1.4826 * MAD.
    The scale floor prevents constant/quantized features from dividing by zero;
    a max across channels makes a one-channel contextual spike visible in a
    multivariate record.

    Rows with fewer than ``min_history`` prior observations are left unavailable
    (score 0), rather than judged against a weak baseline.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError("Context values must be a non-empty (timesteps, features) matrix.")
    if not np.isfinite(x).all():
        raise ValueError("Context values must be finite after imputation and scaling.")
    ids = np.asarray(entity_ids, dtype=object).reshape(-1)
    if len(ids) != len(x):
        raise ValueError("entity_ids must have one entry per timestep.")
    requested = (
        np.ones(len(x), dtype=bool)
        if score_mask is None
        else np.asarray(score_mask, dtype=bool).reshape(-1)
    )
    if len(requested) != len(x):
        raise ValueError("score_mask must have one entry per timestep.")
    if lookback <= 0 or min_history <= 0 or scale_floor <= 0:
        raise ValueError("lookback, min_history, and scale_floor must be positive.")

    names = list(feature_names or [f"feature_{i}" for i in range(x.shape[1])])
    if len(names) != x.shape[1]:
        raise ValueError("feature_names length must match the number of input columns.")

    scores = np.zeros(len(x), dtype=np.float32)
    drivers = np.full(len(x), "", dtype=object)
    available = np.zeros(len(x), dtype=bool)

    # Keep row order within each entity. Missing keys each form a singleton so
    # unrelated rows with unknown IDs can never share a fabricated baseline.
    groups: dict[str, list[int]] = {}
    for row_idx, raw_id in enumerate(ids):
        try:
            missing_id = bool(pd.isna(raw_id))
        except (TypeError, ValueError):
            missing_id = raw_id is None
        key = f"__missing_entity_row_{row_idx}" if missing_id else str(raw_id)
        groups.setdefault(key, []).append(row_idx)

    for row_indices in groups.values():
        for position, row_idx in enumerate(row_indices):
            if position < min_history or not requested[row_idx]:
                continue
            history_indices = row_indices[max(0, position - lookback) : position]
            history = x[history_indices]
            center = np.median(history, axis=0)
            mad = np.median(np.abs(history - center), axis=0)
            robust_scale = np.maximum(1.4826 * mad, scale_floor)
            z_scores = np.abs(x[row_idx] - center) / robust_scale
            driver_idx = int(np.argmax(z_scores))
            scores[row_idx] = float(z_scores[driver_idx])
            drivers[row_idx] = names[driver_idx]
            available[row_idx] = True

    return scores, drivers, available


def calibrate_context_threshold(
    sequence_meta: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    target_false_alarm_rate: float = 0.005,
    conservative_floor: float = 4.5,
    max_validation_samples: int = 20_000,
) -> tuple[float, int]:
    """Calibrate a contextual cutoff from normal validation rows only.

    The empirical normal quantile controls the contextual channel's validation
    false-alarm rate. The robust-z floor keeps the cutoff conservative on small
    or unusually quiet validation sets. Returns ``(threshold, n_normal_rows)``.
    """
    if not 0.0 < target_false_alarm_rate < 1.0:
        raise ValueError("target_false_alarm_rate must be between 0 and 1.")
    if conservative_floor <= 0:
        raise ValueError("conservative_floor must be positive.")
    if max_validation_samples <= 0:
        raise ValueError("max_validation_samples must be positive.")

    normal_scores: list[np.ndarray] = []
    usable_sequences = [
        item for item in sequence_meta
        if item.get("values_scaled") is not None and item.get("y_step") is not None
    ]
    per_sequence_budget = max(
        1, max_validation_samples // max(1, len(usable_sequences))
    )
    for sequence in usable_sequences:
        values = sequence.get("values_scaled")
        labels = sequence.get("y_step")
        values_arr = np.asarray(values, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=int).ravel()
        if len(labels_arr) != len(values_arr):
            continue
        group_id = sequence.get("group_id", "sequence")
        candidate_indices = np.flatnonzero(
            (labels_arr == 0) & (np.arange(len(labels_arr)) >= 16)
        )
        if not len(candidate_indices):
            continue
        if len(candidate_indices) > per_sequence_budget:
            candidate_indices = candidate_indices[
                np.linspace(
                    0,
                    len(candidate_indices) - 1,
                    num=per_sequence_budget,
                    dtype=np.int64,
                )
            ]
        score_mask = np.zeros(len(values_arr), dtype=bool)
        score_mask[candidate_indices] = True
        scores, _, available = robust_context_scores(
            values=values_arr,
            entity_ids=[group_id] * len(values_arr),
            feature_names=feature_names,
            score_mask=score_mask,
        )
        normal_mask = available & (labels_arr == 0)
        if np.any(normal_mask):
            normal_scores.append(scores[normal_mask])

    if not normal_scores:
        return float(conservative_floor), 0

    all_normal_scores = np.concatenate(normal_scores)
    quantile = float(np.quantile(
        all_normal_scores,
        1.0 - target_false_alarm_rate,
        method="higher",
    ))
    return max(float(conservative_floor), quantile), int(len(all_normal_scores))
