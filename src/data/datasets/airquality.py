"""Dataset loader for UCI Air Quality dataset with synthetic label generation.

The Air Quality dataset contains hourly responses from an array of 5 metal oxide chemical
sensors embedded in an Air Quality Chemical Multisensor Device deployed in an Italian city.
Since the raw dataset does not contain labeled ground-truth anomalies, ground truth is
established via:
  1. 'semi_synthetic': Injection of point spikes, level shifts, drifts, and noise bursts
     independently into train, validation, and test after splitting. Injection magnitudes
     use scales estimated from clean training data.
  2. 'dropout': Labeling long runs of sensor dropouts (-200 missing codes) as anomalies.

ALL evaluation metrics derived from this dataset are explicitly tagged as 'synthetic anomalies'.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data.loader import Series
from src.data.synthetic_labels import dropout_labels, inject_anomalies

logger = logging.getLogger(__name__)


def load(config: Dict[str, Any]) -> List[Series]:
    """Load and process the UCI Air Quality dataset.

    Args:
        config: Configuration dictionary containing:
            - 'path': Path to AirQualityUCI.csv (default './data/AirQualityUCI.csv').
            - 'date_col': Column name for date (default 'Date').
            - 'time_col': Column name for time (default 'Time').
            - 'missing_code': Value denoting missing measurement (default -200).
            - 'missing_column_threshold': Drop columns with missing fraction > threshold (default 0.5).
            - 'label_strategy': 'semi_synthetic' or 'dropout' (default 'semi_synthetic').
            - 'dropout_min_run': Min consecutive missing steps to flag as anomaly (default 48).
            - 'synthetic_seed': Random seed for injection (default 42).

    Returns:
        List containing a single Series object with group_id='airquality'.

    Raises:
        FileNotFoundError: If AirQualityUCI.csv is missing.
        ValueError: If parsing fails or required columns cannot be found.
    """
    csv_path_raw = config.get("path", "./data/AirQualityUCI.csv")
    csv_path = Path(csv_path_raw)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"UCI Air Quality dataset not found at '{csv_path.resolve()}'.\n"
            f"Please download 'AirQualityUCI.csv' from the UCI Machine Learning Repository:\n"
            f"  https://archive.ics.uci.edu/dataset/360/air+quality\n"
            f"and place it at '{csv_path.resolve()}'."
        )

    logger.info(f"Loading UCI Air Quality dataset from: {csv_path}")

    # Semicolon-delimited file with comma decimal separators
    try:
        df = pd.read_csv(csv_path, sep=";", decimal=",", low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, sep=";", low_memory=False)
        # Convert comma strings to dots for numeric conversion
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".")

    date_col = config.get("date_col", "Date")
    time_col = config.get("time_col", "Time")
    missing_code = float(config.get("missing_code", -200))
    missing_col_threshold = float(config.get("missing_column_threshold", 0.5))
    label_strategy = config.get("label_strategy", "semi_synthetic").strip().lower()
    synthetic_seed = int(config.get("synthetic_seed", 42))

    # Drop trailing empty columns (UCI CSV frequently ends lines with semicolons)
    unnamed_cols = [c for c in df.columns if "Unnamed" in str(c) or c.strip() == ""]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    # Drop entirely empty rows at bottom of file where Date is NaN
    df = df.dropna(subset=[date_col, time_col], how="any").copy()

    # Construct unified datetime timestamp
    time_str = df[time_col].astype(str).str.strip().str.replace(".", ":", regex=False)
    date_str = df[date_col].astype(str).str.strip()
    dt_combined = date_str + " " + time_str

    try:
        df["_timestamp"] = pd.to_datetime(dt_combined, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        if df["_timestamp"].isna().sum() > len(df) * 0.1:
            df["_timestamp"] = pd.to_datetime(dt_combined, dayfirst=True, errors="coerce")
    except Exception:
        df["_timestamp"] = pd.to_datetime(dt_combined, dayfirst=True, errors="coerce")

    # Drop any remaining unparseable timestamp rows
    valid_time_mask = df["_timestamp"].notna()
    df = df[valid_time_mask].copy()

    # Sort chronologically
    df = df.sort_values(by="_timestamp", ascending=True).reset_index(drop=True)
    timestamps = df["_timestamp"].to_numpy()

    # Exclude non-feature columns
    candidate_cols = [c for c in df.columns if c not in (date_col, time_col, "_timestamp")]

    # Replace missing_code (-200) with NaN and evaluate missingness
    raw_feature_df = df[candidate_cols].copy()
    for col in candidate_cols:
        raw_feature_df[col] = pd.to_numeric(raw_feature_df[col], errors="coerce")
    raw_feature_df = raw_feature_df.replace(missing_code, np.nan)

    # Filter columns exceeding missingness threshold
    dropped_cols = []
    retained_cols = []
    for col in candidate_cols:
        nan_frac = raw_feature_df[col].isna().mean()
        if nan_frac > missing_col_threshold:
            dropped_cols.append(col)
        else:
            retained_cols.append(col)

    if not retained_cols:
        raise ValueError(
            f"All feature columns exceeded missing_column_threshold={missing_col_threshold}. "
            f"Candidate columns: {candidate_cols}"
        )

    # Preserve missing values here. The app imputes after splitting so statistics
    # and fitted transformations cannot see validation/test values.
    raw_values = raw_feature_df[retained_cols].to_numpy(dtype=np.float32)

    # Determine ground truth labeling strategy
    raw_for_dropout = df[retained_cols].to_numpy(dtype=np.float32)

    if label_strategy == "dropout":
        min_run = int(config.get("dropout_min_run", 48))
        y_step = dropout_labels(raw_for_dropout, missing_code=missing_code, min_run=min_run)
        modified_values = raw_values
    elif label_strategy == "semi_synthetic":
        # The preprocessing page injects independent labeled anomalies into each
        # split after train-only imputation/scaler fitting, so the supervised
        # student sees positive examples without contaminating held-out splits.
        modified_values = raw_values
        y_step = np.zeros(len(raw_values), dtype=np.int8)
    else:
        raise ValueError(
            f"Unknown label_strategy '{label_strategy}'. Must be 'semi_synthetic' or 'dropout'."
        )

    anomaly_count = int(np.sum(y_step == 1))
    total_steps = len(y_step)
    anomaly_rate = (anomaly_count / total_steps) if total_steps > 0 else 0.0

    print(
        f"[AirQuality] Loaded {total_steps:,d} timesteps | Retained Features (F): {len(retained_cols)} | "
        f"Dropped Columns (> {missing_col_threshold:.0%} missing): {dropped_cols} | "
        f"Strategy: '{label_strategy}' | Anomaly Rate: {anomaly_rate:.2%} (SYNTHETIC ANOMALIES)"
    )

    series = Series(
        values=modified_values,
        y_step=y_step,
        timestamps=timestamps,
        group_id="airquality",
        feature_names=list(retained_cols),
        metadata={
            "dataset": "airquality",
            "synthetic_anomalies": label_strategy == "semi_synthetic",
            "label_strategy": label_strategy,
            "synthetic_seed": synthetic_seed,
            "injection_config": dict(config.get("injection", {})),
            "dropped_columns": dropped_cols,
            "retained_columns": retained_cols,
            "metric_tag": "synthetic anomalies",
            "missing_code": missing_code,
            "source": "UCI Air Quality Multisensor",
        },
    )

    return [series]
