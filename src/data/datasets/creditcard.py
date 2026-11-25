"""Dataset loader for Credit Card Fraud Detection (Kaggle).

The dataset consists of credit card transactions made by European cardholders in
September 2013, with 492 frauds out of 284,807 transactions (0.172% anomaly rate).
Features V1-V28 are PCA components; Amount is the transaction amount.
Time is the elapsed seconds since the first transaction and serves as the ordering index.
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
    f"V{i}" for i in range(1, 29)
] + ["Amount"]


def load(config: Dict[str, Any]) -> List[Series]:
    """Load and prepare the Credit Card Fraud dataset as a Series object.

    Args:
        config: Configuration dictionary containing:
            - 'path': Path to creditcard.csv (default './data/creditcard.csv').
            - 'feature_cols': List of feature column names (default V1..V28 + Amount).
            - 'time_col': Ordering column name (default 'Time').
            - 'label_col': Binary label column name (default 'Class').

    Returns:
        List containing a single Series object with group_id='creditcard'.

    Raises:
        FileNotFoundError: If creditcard.csv does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    csv_path_raw = config.get("path", "./data/creditcard.csv")
    csv_path = Path(csv_path_raw)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Credit Card Fraud dataset not found at '{csv_path.resolve()}'.\n"
            f"Please download 'creditcard.csv' from Kaggle:\n"
            f"  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and place it at '{csv_path.resolve()}'."
        )

    logger.info(f"Loading Credit Card Fraud dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    time_col = config.get("time_col", "Time")
    label_col = config.get("label_col", "Class")
    feature_cols = config.get("feature_cols", DEFAULT_FEATURE_COLS)

    # Validate essential schema requirements
    if time_col not in df.columns:
        raise ValueError(f"Required time column '{time_col}' missing from Credit Card dataset.")
    if label_col not in df.columns:
        raise ValueError(f"Required label column '{label_col}' missing from Credit Card dataset.")

    missing_feats = [col for col in feature_cols if col not in df.columns]
    if missing_feats:
        raise ValueError(
            f"Credit Card dataset missing expected feature columns: {missing_feats}"
        )

    # Sort chronologically by Time (seconds since first transaction)
    df = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)

    # Time is solely the temporal ordering index; excluded from feature matrix
    timestamps = df[time_col].to_numpy(dtype=np.float64)
    values = df[feature_cols].to_numpy(dtype=np.float32)
    y_step = (df[label_col].to_numpy() == config.get("anomaly_value", 1)).astype(np.int8)

    anomaly_count = int(np.sum(y_step == 1))
    total_samples = len(y_step)
    anomaly_rate = (anomaly_count / total_samples) if total_samples > 0 else 0.0

    print(
        f"[CreditCard] Successfully loaded {total_samples:,d} transactions | "
        f"Features (F): {values.shape[1]} | Fraud Anomalies: {anomaly_count:,d} "
        f"({anomaly_rate:.4%})"
    )

    series = Series(
        values=values,
        y_step=y_step,
        timestamps=timestamps,
        group_id="creditcard",
        feature_names=list(feature_cols),
        metadata={
            "dataset": "creditcard",
            "time_col": time_col,
            "label_col": label_col,
            "headline_metric": config.get("headline_metric", "auprc"),
            "source": "Kaggle Credit Card Fraud Detection",
        },
    )

    return [series]
