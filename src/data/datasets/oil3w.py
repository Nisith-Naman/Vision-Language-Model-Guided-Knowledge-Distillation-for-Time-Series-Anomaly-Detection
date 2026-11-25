"""Dataset loader for the Petrobras 3W Oil Well Anomaly Dataset.

The 3W dataset contains real, simulated, and hand-drawn instances of undesirable
events (anomalies) in offshore petroleum production wells.
Sensor features (F=8):
    1. P-PDG: Pressure at Permanent Downhole Gauge
    2. P-TPT: Pressure at Temperature and Pressure Transducer
    3. T-TPT: Temperature at Temperature and Pressure Transducer
    4. P-MON-CKP: Upstream pressure of production choke
    5. T-JUS-CKP: Downstream temperature of production choke
    6. P-JUS-CKGL: Downstream pressure of gas lift choke
    7. T-JUS-CKGL: Downstream temperature of gas lift choke
    8. QGL: Gas lift flow rate

Classes:
    0: Normal
    1-8: Steady-state fault classes
    101-108: Transient fault classes preceding steady-state faults
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data.loader import Series

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_COLS = [
    "P-PDG", "P-TPT", "T-TPT", "P-MON-CKP",
    "T-JUS-CKP", "P-JUS-CKGL", "T-JUS-CKGL", "QGL",
]


def load(config: Dict[str, Any]) -> List[Series]:
    """Load and process the Petrobras 3W dataset from CSV files.

    Args:
        config: Configuration dictionary containing:
            - 'data_dir': Root directory containing 3W CSV files (default './data/3w').
            - 'feature_cols': List of 8 sensor column names.
            - 'time_col': Timestamp column name (default 'timestamp').
            - 'label_col': Class label column name (default 'class').
            - 'transient_is_anomaly': Whether transient events (101-108) are labeled as anomalies (default True).
            - 'missing_fill_strategy': Imputation strategy (default 'ffill').
            - 'include_nan_flags': Whether to include boolean NaN indicator columns in feature matrix (default False).

    Returns:
        List of Series objects, one per CSV instance file.

    Raises:
        FileNotFoundError: If data_dir does not exist or contains no CSV files.
    """
    data_dir_raw = config.get("data_dir", "./data/3w")
    data_dir = Path(data_dir_raw)

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"3W Oil Well dataset directory not found at '{data_dir.resolve()}'.\n"
            f"Please run 'python scripts/download_3w.py' or download the dataset from:\n"
            f"  https://github.com/ricardovvargas/3w_dataset\n"
            f"and place the instance CSV files into '{data_dir.resolve()}'."
        )

    csv_files = sorted(list(data_dir.rglob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in 3W dataset directory '{data_dir.resolve()}'.\n"
            f"Please run 'python scripts/download_3w.py' to fetch the instance files."
        )

    csv_files = _select_instances(csv_files, config)
    resample_rows = int(config.get("resample_seconds", 0) or 0)

    feature_cols = config.get("feature_cols", DEFAULT_FEATURE_COLS)
    time_col = config.get("time_col", "timestamp")
    label_col = config.get("label_col", "class")
    transient_is_anomaly = bool(config.get("transient_is_anomaly", True))
    include_nan_flags = bool(config.get("include_nan_flags", False))

    series_list: List[Series] = []
    anomalous_instance_count = 0
    total_timesteps = 0
    total_anom_steps = 0

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            logger.warning(f"Failed to read 3W file '{fpath}': {e}. Skipping.")
            continue

        if df.empty or label_col not in df.columns:
            continue

        # Sort chronologically if timestamp exists
        timestamps = None
        if time_col in df.columns:
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                df = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)
                timestamps = df[time_col].to_numpy()
            except Exception:
                timestamps = df[time_col].to_numpy()

        # Extract labels
        raw_classes = df[label_col].fillna(0).to_numpy(dtype=np.int32)
        if transient_is_anomaly:
            # class != 0 (including transients 101-108) is anomaly
            y_step = (raw_classes != 0).astype(np.int8)
        else:
            # Only steady-state faults (1-8) are anomalies
            y_step = ((raw_classes >= 1) & (raw_classes <= 8)).astype(np.int8)

        # Process features and handle missing values
        instance_features: Dict[str, np.ndarray] = {}
        all_nan_cols: List[str] = []
        nan_flag_cols: Dict[str, np.ndarray] = {}

        for col in feature_cols:
            if col in df.columns:
                series_vals = pd.to_numeric(df[col], errors="coerce")
            else:
                series_vals = pd.Series(np.nan, index=df.index)

            # Some 3W readings are infinite or too large for the float32 arrays
            # used by the model. Normalize them to missing before filling so
            # no non-finite values reach the scaler or model.
            float32_limit = np.finfo(np.float32).max
            valid_float32 = np.isfinite(series_vals) & (series_vals.abs() <= float32_limit)
            series_vals = series_vals.where(valid_float32, np.nan)

            # Check if column is entirely NaN
            if series_vals.isna().all():
                all_nan_cols.append(col)
                # Fill all-NaN with 0.0 and generate a boolean flag column
                instance_features[col] = np.zeros(len(df), dtype=np.float32)
                nan_flag_cols[f"{col}_all_nan"] = np.ones(len(df), dtype=np.float32)
            else:
                # Forward-fill short missing gaps, followed by bfill
                filled = series_vals.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
                instance_features[col] = filled
                nan_flag_cols[f"{col}_all_nan"] = np.zeros(len(df), dtype=np.float32)

        feat_matrix_list = [instance_features[col] for col in feature_cols]
        active_feature_names = list(feature_cols)

        if include_nan_flags:
            for flag_name, flag_arr in nan_flag_cols.items():
                feat_matrix_list.append(flag_arr)
                active_feature_names.append(flag_name)

        values = np.column_stack(feat_matrix_list).astype(np.float32)
        if resample_rows > 1 and len(values) >= resample_rows:
            # 3W is sampled at 1 Hz; average consecutive blocks (labels: any
            # anomalous second marks the block anomalous) to shorten instances.
            n_blocks = len(values) // resample_rows
            cut = n_blocks * resample_rows
            values = values[:cut].reshape(n_blocks, resample_rows, -1).mean(axis=1).astype(np.float32)
            y_step = y_step[:cut].reshape(n_blocks, resample_rows).max(axis=1).astype(np.int8)
            if timestamps is not None:
                timestamps = np.asarray(timestamps)[:cut:resample_rows]

        # Unique relative path identifier
        try:
            rel_path = str(fpath.relative_to(data_dir)).replace("\\", "/")
        except ValueError:
            rel_path = fpath.name

        has_anomaly = bool(np.any(y_step == 1))
        if has_anomaly:
            anomalous_instance_count += 1

        total_timesteps += len(y_step)
        total_anom_steps += int(np.sum(y_step == 1))

        series_obj = Series(
            values=values,
            y_step=y_step,
            timestamps=timestamps,
            group_id=rel_path,
            feature_names=active_feature_names,
            metadata={
                "dataset": "oil3w",
                "instance_file": rel_path,
                "all_nan_columns": all_nan_cols,
                "has_anomaly": has_anomaly,
                "transient_is_anomaly": transient_is_anomaly,
            },
        )
        series_list.append(series_obj)

    if not series_list:
        raise ValueError(f"No valid series could be loaded from 3W directory '{data_dir.resolve()}'.")

    total_instances = len(series_list)
    anom_step_rate = (total_anom_steps / total_timesteps) if total_timesteps > 0 else 0.0
    anom_inst_rate = (anomalous_instance_count / total_instances) if total_instances > 0 else 0.0

    print(
        f"[Oil3W] Loaded {total_instances:,d} instances | Anomalous Instances: {anomalous_instance_count:,d} "
        f"({anom_inst_rate:.2%}) | Total Timesteps T: {total_timesteps:,d} | "
        f"Anom Timesteps: {total_anom_steps:,d} ({anom_step_rate:.2%}) | "
        f"Features (F): {series_list[0].n_features}"
    )

    return series_list


def _select_instances(csv_files: List[Path], config: Dict[str, Any]) -> List[Path]:
    """Optionally keep a deterministic, class-balanced subset of instance files.

    Config keys (all optional; defaults keep every file):
        exclude_drawn: skip hand-drawn ``DRAWN_*`` instances.
        max_instances_per_class: cap per class folder (0 = no cap). Real
            ``WELL-*`` instances are taken first, then simulated ones.
        max_normal_instances: cap for class folder ``0`` (defaults to the class cap).
        subset_seed: seed for the deterministic shuffle within each class.
    """
    exclude_drawn = bool(config.get("exclude_drawn", False))
    per_class = int(config.get("max_instances_per_class", 0) or 0)
    normal_cap = int(config.get("max_normal_instances", per_class) or 0)
    if not exclude_drawn and per_class <= 0 and normal_cap <= 0:
        return csv_files
    rng = np.random.default_rng(int(config.get("subset_seed", 42)))
    by_class: Dict[str, List[Path]] = {}
    for f in csv_files:
        if exclude_drawn and f.name.upper().startswith("DRAWN"):
            continue
        by_class.setdefault(f.parent.name, []).append(f)
    selected: List[Path] = []
    for cls in sorted(by_class):
        files = by_class[cls]
        real = [f for f in files if f.name.upper().startswith("WELL")]
        other = [f for f in files if not f.name.upper().startswith("WELL")]
        ordered = [real[i] for i in rng.permutation(len(real))] + [other[i] for i in rng.permutation(len(other))]
        cap = normal_cap if cls == "0" else per_class
        selected.extend(ordered[:cap] if cap > 0 else ordered)
    return sorted(selected)
