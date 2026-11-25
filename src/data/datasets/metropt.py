"""Dataset loader for the MetroPT-3 Urban Train Air Compressor Dataset.

The MetroPT-3 dataset consists of sensor readings from an Air Production Unit (APU)
installed on an urban passenger train in Porto, Portugal.
The system is monitored by 15 sensors:
  - 7 analog signals: TP2, TP3, H1, DV_pressure, Reservoirs, Motor_current, Oil_temperature
  - 8 digital signals: COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulse

NOTE ON UCI DOCUMENTATION DISCREPANCIES:
1. Sampling Rate Discrepancy:
   The official UCI repository page claims the data was recorded at 1 Hz. However,
   empirical measurement of the raw timestamp delta reveals a median sampling spacing
   of dt = 10.0 seconds (~0.1 Hz). Resampling to 1-minute intervals standardizes the grid.
2. UCI Event Numbering Typo:
   The UCI dataset description and associated technical report contain a known typographic
   discrepancy regarding failure event indexing and maintenance report numbers. This module
   uses the verified ground-truth failure time windows corresponding to the documented APU
   failure events from April to July 2020.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data.loader import Series

logger = logging.getLogger(__name__)

DEFAULT_ANALOG_COLS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Motor_current", "Oil_temperature"
]
DEFAULT_DIGITAL_COLS = [
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulse"
]
DEFAULT_FAILURE_WINDOWS = [
    {"start": "2020-04-18 00:00:00", "end": "2020-04-18 23:59:59", "event_id": 1},
    {"start": "2020-05-29 23:30:00", "end": "2020-05-30 06:00:00", "event_id": 2},
    {"start": "2020-06-05 10:00:00", "end": "2020-06-07 14:30:00", "event_id": 3},
    {"start": "2020-07-15 14:30:00", "end": "2020-07-15 19:00:00", "event_id": 4},
]


def load(config: Dict[str, Any]) -> List[Series]:
    """Load, resample, and annotate the MetroPT-3 Air Compressor dataset.

    Args:
        config: Configuration dictionary containing:
            - 'path': Path to MetroPT CSV file (default './data/MetroPT3(AirCompressor).csv').
            - 'analog_cols': List of 7 analog sensor column names.
            - 'digital_cols': List of 8 digital sensor column names.
            - 'time_col': Timestamp column name (default 'timestamp').
            - 'resample_freq': Target pandas offset string for resampling (default '1min').
            - 'resample_agg_analog': Aggregation method for analog signals (default 'median').
            - 'resample_agg_digital': Aggregation method for digital signals (default 'max').
            - 'max_gap_seconds': Maximum raw sampling gap filled within one block (default 300).
            - 'failure_windows': List of dicts specifying failure intervals and event IDs.

    Returns:
    Continuous Series segments. Acquisition gaps and event boundaries are never
    joined. Failure events 1, 2, and 3-4 are assigned to train, validation, and
    test; normal-only acquisition blocks are chronologically divided across all
    three partitions so held-out evaluation includes normal observations.

    Raises:
        FileNotFoundError: If the MetroPT CSV file is missing.
        ValueError: If essential columns or timestamps cannot be resolved.
    """
    csv_path_raw = config.get("path", "./data/MetroPT3(AirCompressor).csv")
    csv_path = Path(csv_path_raw)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"MetroPT-3 dataset not found at '{csv_path.resolve()}'.\n"
            f"Please download 'MetroPT3(AirCompressor).csv' from the UCI Machine Learning Repository:\n"
            f"  https://archive.ics.uci.edu/dataset/791/metropt+3+dataset\n"
            f"or Zenodo repository and place it at '{csv_path.resolve()}'."
        )

    logger.info(f"Loading MetroPT-3 dataset from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    time_col = config.get("time_col", "timestamp")
    analog_cols = config.get("analog_cols", DEFAULT_ANALOG_COLS)
    digital_cols = config.get("digital_cols", DEFAULT_DIGITAL_COLS)
    feature_cols = list(analog_cols) + list(digital_cols)
    resample_freq = config.get("resample_freq", "1min")
    agg_analog = config.get("resample_agg_analog", "median")
    agg_digital = config.get("resample_agg_digital", "max")
    failure_windows = config.get("failure_windows", DEFAULT_FAILURE_WINDOWS)

    if time_col not in df.columns:
        # Check if first unnamed column contains timestamps
        first_col = df.columns[0]
        if "time" in str(first_col).lower() or "unnamed" in str(first_col).lower():
            time_col = first_col
        else:
            raise ValueError(f"Timestamp column '{time_col}' missing from MetroPT CSV.")

    # Parse timestamps and sort
    df["_dt"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["_dt"]).sort_values(by="_dt").reset_index(drop=True)

    # 1. Measure empirical raw timestamp spacing and compare with UCI claims
    dt_deltas = df["_dt"].diff().dt.total_seconds().dropna()
    median_dt = float(dt_deltas.median()) if not dt_deltas.empty else 0.0
    empirical_hz = (1.0 / median_dt) if median_dt > 0 else 0.0

    print(
        f"[MetroPT] Measured raw timestamp spacing: median dt = {median_dt:.1f}s (~{empirical_hz:.2f} Hz).\n"
        f"          Note: UCI repository documentation claims 1.0 Hz, but actual data recording interval is ~10s (0.1 Hz)."
    )

    # Ensure numeric types
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.warning(f"Feature column '{col}' missing from MetroPT raw CSV; filling with 0.0.")
            df[col] = 0.0

    # 2. Split at long acquisition gaps before resampling. Resampling the entire
    # calendar range would synthesize rows for days without measurements, then
    # forward-fill those rows as if the compressor had stayed unchanged.
    df_indexed = df.set_index("_dt")
    analog_present = [c for c in analog_cols if c in df_indexed.columns]
    digital_present = [c for c in digital_cols if c in df_indexed.columns]
    max_gap_seconds = float(config.get("max_gap_seconds", 300.0))
    raw_gaps = df["_dt"].diff().dt.total_seconds().to_numpy()
    block_starts = np.r_[0, np.flatnonzero(raw_gaps > max_gap_seconds) + 1]
    block_ends = np.r_[block_starts[1:], len(df)]
    resampled_parts: List[pd.DataFrame] = []
    resampled_block_ids: List[np.ndarray] = []
    for block_id, (block_start, block_end) in enumerate(zip(block_starts, block_ends)):
        block = df_indexed.iloc[block_start:block_end]
        analog = block[analog_present].resample(resample_freq).agg(agg_analog)
        digital = block[digital_present].resample(resample_freq).agg(agg_digital)
        part = pd.concat([analog, digital], axis=1).ffill().bfill().fillna(0.0)
        if not part.empty:
            resampled_parts.append(part)
            resampled_block_ids.append(np.full(len(part), block_id, dtype=np.int32))

    if not resampled_parts:
        raise ValueError("MetroPT contains no valid timestamps to resample.")
    df_resampled = pd.concat(resampled_parts, axis=0)
    block_ids = np.concatenate(resampled_block_ids)

    resampled_timestamps = df_resampled.index.to_numpy()
    values = df_resampled[feature_cols].to_numpy(dtype=np.float32)

    # 3. Construct point-wise anomaly labels and event IDs from failure intervals
    T_resampled = len(df_resampled)
    y_step = np.zeros(T_resampled, dtype=np.int8)
    event_ids = np.zeros(T_resampled, dtype=np.int32)

    for win in failure_windows:
        start_ts = pd.to_datetime(win["start"])
        end_ts = pd.to_datetime(win["end"])
        ev_id = int(win.get("event_id", 1))

        mask = (df_resampled.index >= start_ts) & (df_resampled.index <= end_ts)
        y_step[mask] = 1
        event_ids[mask] = ev_id

    total_anom = int(np.sum(y_step == 1))
    anom_rate = (total_anom / T_resampled) if T_resampled > 0 else 0.0

    print(
        f"[MetroPT] Successfully resampled to '{resample_freq}' | "
        f"Resampled Timesteps T: {T_resampled:,d} | Features (F): {len(feature_cols)} "
        f"(7 analog, 8 digital) | Acquisition blocks: {len(resampled_parts):,d} | "
        f"Anomaly Timesteps: {total_anom:,d} ({anom_rate:.2%})"
    )

    # Keep each contiguous event/background segment separate so sliding windows
    # cannot bridge across long time gaps. Hold out the four failures by event
    # identity, and distribute normal-only acquisition blocks chronologically so
    # validation/test contain both normal and anomalous observations.
    block_event_sets = {
        int(block_id): set(int(x) for x in np.unique(event_ids[block_ids == block_id]) if x > 0)
        for block_id in np.unique(block_ids)
    }
    block_roles: Dict[int, str] = {}
    normal_block_ids = []
    event_role = {1: "train", 2: "val", 3: "test", 4: "test"}
    for block_id, events in block_event_sets.items():
        if len(events) > 1:
            raise ValueError(
                f"Acquisition block {block_id} contains multiple failure events: {sorted(events)}"
            )
        if events:
            block_roles[block_id] = event_role[next(iter(events))]
        else:
            normal_block_ids.append(block_id)

    normal_block_ids.sort()
    if len(normal_block_ids) < 3:
        raise ValueError(
            "MetroPT needs at least three normal-only acquisition blocks to create "
            "normal examples in train, validation, and test."
        )
    n_normal = len(normal_block_ids)
    n_normal_train = max(1, int(round(n_normal * 0.70)))
    n_normal_val = max(1, int(round(n_normal * 0.15)))
    if n_normal_train + n_normal_val >= n_normal:
        n_normal_train, n_normal_val = n_normal - 2, 1
    for block_id in normal_block_ids[:n_normal_train]:
        block_roles[block_id] = "train"
    for block_id in normal_block_ids[n_normal_train:n_normal_train + n_normal_val]:
        block_roles[block_id] = "val"
    for block_id in normal_block_ids[n_normal_train + n_normal_val:]:
        block_roles[block_id] = "test"

    changes = np.flatnonzero((np.diff(event_ids) != 0) | (np.diff(block_ids) != 0)) + 1
    boundaries = np.r_[0, changes, T_resampled]
    segments: List[Series] = []
    for segment_index, (start_idx, end_idx) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        event_id = int(event_ids[start_idx])
        block_id = int(block_ids[start_idx])
        role = block_roles[block_id]
        group_id = f"metropt-normal-block-{block_id}" if event_id == 0 else f"metropt-event-{event_id}"
        segments.append(Series(
            values=values[start_idx:end_idx],
            y_step=y_step[start_idx:end_idx],
            timestamps=resampled_timestamps[start_idx:end_idx],
            group_id=group_id,
            feature_names=list(feature_cols),
            metadata={
                "dataset": "metropt",
                "event_id": event_id,
                "split_role": role,
                "segment_index": segment_index,
                "measured_dt_seconds": median_dt,
                "max_gap_seconds": max_gap_seconds,
                "acquisition_block": block_id,
                "resample_freq": resample_freq,
                "failure_windows": failure_windows,
                "analog_cols": analog_cols,
                "digital_cols": digital_cols,
                "source": "UCI MetroPT-3 Urban Train Dataset",
                "documentation_note": (
                    "Failure-event holdout: event 1 train, event 2 validation, events 3-4 test; "
                    "normal-only acquisition blocks are chronologically split 70/15/15."
                ),
            },
        ))

    return segments
