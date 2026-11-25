"""Synthetic anomaly generation and sensor dropout labeling module.

Used primarily for unlabeled datasets (e.g. UCI Air Quality) to create labeled
point spikes, level shifts, ramps/drifts, and variance bursts after the data has
been split. Each partition is injected independently and can use training-derived
feature scales.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Global registry to retain the most recent synthetic injection metadata for reproducibility
_LAST_INJECTION_METADATA: Dict[str, Any] = {}


def get_last_injection_metadata() -> Dict[str, Any]:
    """Retrieve metadata of the most recent anomaly injection run."""
    return dict(_LAST_INJECTION_METADATA)


def inject_anomalies(
    values: np.ndarray,
    timestamps: Optional[np.ndarray],
    config: Dict[str, Any],
    seed: int = 42,
    start_idx: Optional[int] = None,
    scale_stds: Optional[np.ndarray] = None,
    return_metadata: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Inject synthetic anomalies into numerical time-series feature matrix.

    Generates four distinct anomaly topologies:
        1. Point spikes: Sudden extreme deviations added as N(0, 1) * 5 * std.
        2. Level shifts: Sudden step change of segment by 3 * std for a random duration.
        3. Drift: Gradual linear ramp up over a temporal segment.
        4. Noise bursts: Elevated local variance over a temporal segment.

    Injection can be restricted to indices >= start_idx. In the Air Quality
    training pipeline, each split is injected independently after splitting so the
    supervised student has labeled training examples while held-out metrics remain
    explicitly synthetic.

    Args:
        values: Numerical feature matrix of shape (T, F).
        timestamps: Optional 1D timestamp array of length T.
        config: Dictionary containing injection parameters or top-level dataset config.
        seed: Random seed for deterministic reproducibility.
        start_idx: Optional integer index. Anomalies will ONLY be injected into
            the interval [start_idx, T). If None, the entire series is eligible.
        scale_stds: Optional feature standard deviations computed from training data.
            When supplied, use these scales for anomaly magnitudes across all splits.
        return_metadata: If True, returns a 3-tuple (modified_values, y_step, metadata).
            If False, returns a 2-tuple (modified_values, y_step).

    Returns:
        (modified_values, y_step) or (modified_values, y_step, metadata)
        where modified_values is float32 (T, F) and y_step is int8 (T,).
    """
    global _LAST_INJECTION_METADATA

    rng = np.random.RandomState(seed)

    orig_arr = np.asarray(values, dtype=np.float32)
    orig_ndim = orig_arr.ndim
    if orig_ndim == 1:
        orig_arr = orig_arr.reshape(-1, 1)

    T, F = orig_arr.shape
    modified_values = orig_arr.copy()
    y_step = np.zeros(T, dtype=np.int8)

    if T == 0:
        empty_meta = {"seed": seed, "injected_counts": {}, "note": "Empty series"}
        _LAST_INJECTION_METADATA = empty_meta
        if return_metadata:
            return modified_values, y_step, empty_meta
        return modified_values, y_step

    # Determine candidate injection window [idx_min, T)
    idx_min = 0 if start_idx is None else max(0, min(int(start_idx), T))
    valid_len = T - idx_min

    if valid_len <= 10:
        logger.warning(
            f"Candidate interval [{idx_min}, {T}) too short for synthetic anomaly injection."
        )
        meta = {"seed": seed, "status": "skipped_too_short", "eligible_length": valid_len}
        _LAST_INJECTION_METADATA = meta
        if return_metadata:
            return modified_values, y_step, meta
        return modified_values, y_step

    # Extract nested injection config if present
    inj_cfg = config.get("injection", config)
    point_spike_rate = float(inj_cfg.get("point_spike_rate", 0.002))
    level_shift_prob = float(inj_cfg.get("level_shift_prob", 0.05))
    drift_prob = float(inj_cfg.get("drift_prob", 0.03))
    noise_burst_prob = float(inj_cfg.get("noise_burst_prob", 0.04))
    # When true, each event perturbs a random subset of sensors (2..F/2) instead
    # of every channel at once, which is closer to a real single-sensor fault.
    feature_subset = bool(inj_cfg.get("feature_subset", False))
    events: list = []

    def _channels() -> np.ndarray:
        if not feature_subset or F <= 2:
            return np.arange(F)
        k = int(rng.randint(2, max(3, F // 2 + 1)))
        return np.sort(rng.choice(F, size=min(k, F), replace=False))

    # Feature-wise standard deviation for scaling anomaly magnitudes
    stds = np.asarray(scale_stds, dtype=np.float32) if scale_stds is not None else np.nanstd(modified_values, axis=0)
    if stds.shape != (F,):
        raise ValueError(f"scale_stds must have shape ({F},), got {stds.shape}")
    stds = np.where(stds < 1e-6, 1.0, stds)

    injected_counts = {
        "point_spikes": 0,
        "level_shifts": 0,
        "drifts": 0,
        "noise_bursts": 0,
    }

    # 1. Point Spikes (add N(0, 1) * 5 * std)
    n_spikes = max(1, int(round(valid_len * point_spike_rate)))
    spike_indices = rng.choice(np.arange(idx_min, T), size=n_spikes, replace=False)
    for idx in spike_indices:
        sign = rng.choice([-1.0, 1.0])
        magnitude = rng.normal(loc=0.0, scale=1.0) * 5.0 * stds
        # Ensure minimum noticeable magnitude
        if np.linalg.norm(magnitude) < 2.0 * np.linalg.norm(stds):
            magnitude += sign * 4.0 * stds
        ch = _channels()
        modified_values[idx, ch] += magnitude.astype(np.float32)[ch]
        y_step[idx] = 1
        injected_counts["point_spikes"] += 1
        events.append({"type": "point_spike", "start": int(idx), "end": int(idx) + 1, "channels": ch.tolist()})

    # 2. Level Shifts (shift segment by 3 * std)
    n_shifts = max(1, int(round(level_shift_prob * (valid_len / 100.0))))
    for _ in range(n_shifts):
        dur = rng.randint(8, min(48, valid_len))
        s_idx = rng.randint(idx_min, T - dur + 1)
        sign = rng.choice([-1.0, 1.0])
        shift_vector = (sign * 3.0 * stds).astype(np.float32)
        ch = _channels()
        modified_values[s_idx : s_idx + dur, ch] += shift_vector[ch]
        y_step[s_idx : s_idx + dur] = 1
        injected_counts["level_shifts"] += 1
        events.append({"type": "level_shift", "start": int(s_idx), "end": int(s_idx + dur), "channels": ch.tolist()})

    # 3. Drift (linear ramp up over a segment)
    n_drifts = max(1, int(round(drift_prob * (valid_len / 100.0))))
    for _ in range(n_drifts):
        dur = rng.randint(16, min(64, valid_len))
        s_idx = rng.randint(idx_min, T - dur + 1)
        sign = rng.choice([-1.0, 1.0])
        # Linear slope from 0 up to 3.5 * stds
        ramp = np.linspace(0.0, 1.0, dur)[:, None] * (sign * 3.5 * stds)[None, :]
        ch = _channels()
        modified_values[s_idx : s_idx + dur, ch] += ramp.astype(np.float32)[:, ch]
        y_step[s_idx : s_idx + dur] = 1
        injected_counts["drifts"] += 1
        events.append({"type": "drift", "start": int(s_idx), "end": int(s_idx + dur), "channels": ch.tolist()})

    # 4. Noise Bursts (elevated variance over a segment)
    n_bursts = max(1, int(round(noise_burst_prob * (valid_len / 100.0))))
    for _ in range(n_bursts):
        dur = rng.randint(10, min(36, valid_len))
        s_idx = rng.randint(idx_min, T - dur + 1)
        burst_noise = rng.normal(0.0, 2.5 * stds, size=(dur, F)).astype(np.float32)
        ch = _channels()
        modified_values[s_idx : s_idx + dur, ch] += burst_noise[:, ch]
        y_step[s_idx : s_idx + dur] = 1
        injected_counts["noise_bursts"] += 1
        events.append({"type": "noise_burst", "start": int(s_idx), "end": int(s_idx + dur), "channels": ch.tolist()})

    if orig_ndim == 1:
        modified_values = modified_values.squeeze(axis=-1)

    total_anom = int(np.sum(y_step == 1))
    anom_pct = float(total_anom / T) if T > 0 else 0.0

    metadata = {
        "reproducibility_seed": seed,
        "eligible_start_idx": idx_min,
        "total_timesteps": T,
        "anomalous_timesteps": total_anom,
        "anomaly_rate": anom_pct,
        "injected_event_counts": injected_counts,
        "events": events,
        "injection_config": dict(inj_cfg),
        "synthetic_anomalies": True,
        "note": "Ground truth generated via synthetic anomaly injection pipeline",
    }
    _LAST_INJECTION_METADATA = metadata

    logger.info(
        f"Synthetic anomaly injection completed (seed={seed}, start_idx={idx_min}): "
        f"{total_anom}/{T} anomalous steps ({anom_pct:.2%}). Events: {injected_counts}"
    )

    if return_metadata:
        return modified_values, y_step, metadata
    return modified_values, y_step


def dropout_labels(
    values_raw: np.ndarray,
    missing_code: float = -200.0,
    min_run: int = 48,
) -> np.ndarray:
    """Label sustained sensor dropouts (consecutive runs of missing_code) as anomalies.

    Identifies sequences where sensors produce repeated invalid or dropout values
    (e.g. -200 in UCI Air Quality). If a run equals or exceeds min_run consecutive
    timesteps, all timesteps within that run are labeled as anomalous (y_step = 1).

    Args:
        values_raw: Raw sensor array of shape (T, F) or (T,).
        missing_code: Numerical value indicating a sensor dropout or missing reading.
        min_run: Minimum consecutive timesteps of missing_code required to qualify
            as an anomalous dropout event.

    Returns:
        1D int8 array of shape (T,) with 1 for dropout anomalies and 0 for normal steps.
    """
    arr = np.asarray(values_raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    T, F = arr.shape
    y_step = np.zeros(T, dtype=np.int8)

    if T == 0 or min_run <= 0:
        return y_step

    # A timestep is marked as a candidate dropout if any sensor matches missing_code
    is_missing = np.isclose(arr, missing_code, atol=1e-3).any(axis=1)

    # Scan for contiguous runs of missing steps
    run_start = None
    for t in range(T):
        if is_missing[t]:
            if run_start is None:
                run_start = t
        else:
            if run_start is not None:
                run_len = t - run_start
                if run_len >= min_run:
                    y_step[run_start:t] = 1
                run_start = None

    # Handle run continuing to the final timestep
    if run_start is not None:
        run_len = T - run_start
        if run_len >= min_run:
            y_step[run_start:T] = 1

    total_drops = int(np.sum(y_step == 1))
    logger.info(
        f"Dropout labeling: detected {total_drops}/{T} timesteps "
        f"({(total_drops/T):.2%}) with missing runs >= {min_run}"
    )

    return y_step
