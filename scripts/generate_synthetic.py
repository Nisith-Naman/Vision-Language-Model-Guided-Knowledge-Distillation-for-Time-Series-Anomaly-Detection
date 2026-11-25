#!/usr/bin/env python
"""Generate synthetic time-series datasets for testing the anomaly detection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def generate_univariate(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a univariate time-series with point anomalies and level shifts.

    Args:
        n: Number of timesteps to generate.
        seed: Random seed for deterministic generation.

    Returns:
        DataFrame with columns ['timestamp', 'value', 'label'].
    """
    rng = np.random.RandomState(seed)

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1min")
    t = np.linspace(0, 30 * np.pi, n)
    base = np.sin(t) + rng.normal(0, 0.15, size=n)

    label = np.zeros(n, dtype=np.int8)

    # Inject 5 point anomalies (isolated extreme spikes)
    # Avoid first and last 50 steps
    point_indices = rng.choice(np.arange(100, n - 100), size=5, replace=False)
    for idx in point_indices:
        sign = rng.choice([-1.0, 1.0])
        base[idx] += sign * (4.0 + rng.uniform(0.5, 2.0))
        label[idx] = 1

    # Inject 2 level shifts (sustained deviation)
    shift_lengths = [35, 45]
    shift_candidates = [
        int(n * 0.25) + rng.randint(-30, 30),
        int(n * 0.70) + rng.randint(-30, 30),
    ]

    for s_start, dur in zip(shift_candidates, shift_lengths):
        sign = rng.choice([-1.0, 1.0])
        shift_val = sign * (2.5 + rng.uniform(0.5, 1.5))
        base[s_start : s_start + dur] += shift_val
        label[s_start : s_start + dur] = 1

    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "value": base.astype(np.float32),
            "label": label,
        }
    )
    return df


def generate_multivariate(n: int = 2000, n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate a multivariate time-series with correlated feature bursts.

    Args:
        n: Number of timesteps to generate.
        n_features: Number of numerical sensor features.
        seed: Random seed for deterministic generation.

    Returns:
        DataFrame with columns ['timestamp', 'feat_1', ..., 'feat_N', 'label'].
    """
    rng = np.random.RandomState(seed)
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1min")

    data = np.zeros((n, n_features), dtype=np.float32)
    label = np.zeros(n, dtype=np.int8)

    # Latent base process
    t = np.linspace(0, 20 * np.pi, n)
    latent_signal = np.sin(t)

    for i in range(n_features):
        freq_factor = 0.8 + 0.3 * (i % 3)
        phase_shift = (i * np.pi) / 4.0
        feature_base = np.sin(t * freq_factor + phase_shift) + 0.3 * latent_signal
        noise = rng.normal(0, 0.12, size=n)
        data[:, i] = (feature_base + noise).astype(np.float32)

    # Inject 3 correlated anomaly bursts across multiple features
    burst_starts = [
        int(n * 0.20) + rng.randint(-20, 20),
        int(n * 0.55) + rng.randint(-20, 20),
        int(n * 0.82) + rng.randint(-20, 20),
    ]
    burst_lengths = [25, 40, 30]

    for b_start, b_len in zip(burst_starts, burst_lengths):
        affected_feats = rng.choice(n_features, size=max(2, n_features - 1), replace=False)
        for f_idx in affected_feats:
            sign = rng.choice([-1.0, 1.0])
            magnitude = sign * (3.0 + rng.uniform(0.5, 1.5))
            data[b_start : b_start + b_len, f_idx] += magnitude
        label[b_start : b_start + b_len] = 1

    # Inject 4 isolated point spikes
    point_indices = rng.choice(np.arange(50, n - 50), size=4, replace=False)
    for idx in point_indices:
        f_idx = rng.randint(0, n_features)
        sign = rng.choice([-1.0, 1.0])
        data[idx, f_idx] += sign * 5.0
        label[idx] = 1

    df_dict: Dict[str, np.ndarray] = {
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S")
    }
    for i in range(n_features):
        df_dict[f"feat_{i+1}"] = data[:, i]
    df_dict["label"] = label

    return pd.DataFrame(df_dict)


def generate_no_labels(n: int = 2000, n_features: int = 3, seed: int = 42) -> pd.DataFrame:
    """Generate multivariate time-series data without a ground-truth label column.

    Args:
        n: Number of timesteps.
        n_features: Number of feature columns.
        seed: Random seed.

    Returns:
        DataFrame with columns ['timestamp', 'feat_1', ..., 'feat_N'].
    """
    df = generate_multivariate(n=n, n_features=n_features, seed=seed)
    return df.drop(columns=["label"])


def generate_with_missing(
    n: int = 2000, n_features: int = 3, missing_rate: float = 0.05, seed: int = 42
) -> pd.DataFrame:
    """Generate multivariate time-series containing randomly injected NaN values.

    Args:
        n: Number of timesteps.
        n_features: Number of feature columns.
        missing_rate: Fraction of cells in feature columns set to NaN.
        seed: Random seed.

    Returns:
        DataFrame with columns ['timestamp', 'feat_1', ..., 'feat_N', 'label'] and missing entries.
    """
    df = generate_multivariate(n=n, n_features=n_features, seed=seed)
    rng = np.random.RandomState(seed + 100)

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    for col in feature_cols:
        mask = rng.rand(n) < missing_rate
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic time-series datasets for pipeline testing."
    )
    parser.add_argument("--output-dir", default="./data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random reproducibility seed")
    parser.add_argument("--n", type=int, default=2000, help="Number of rows per dataset")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = {
        "synthetic_univariate.csv": generate_univariate(n=args.n, seed=args.seed),
        "synthetic_multivariate.csv": generate_multivariate(
            n=args.n, n_features=5, seed=args.seed
        ),
        "synthetic_no_labels.csv": generate_no_labels(n=args.n, n_features=3, seed=args.seed),
        "synthetic_with_missing.csv": generate_with_missing(
            n=args.n, n_features=3, missing_rate=0.05, seed=args.seed
        ),
    }

    print("=" * 70)
    print("Generating Synthetic Time-Series Datasets")
    print("=" * 70)

    for filename, df in datasets.items():
        path = output_path / filename
        df.to_csv(path, index=False)
        print(f"Saved {filename:<28}: {len(df):>5d} rows, {df.shape[1]:>2d} columns")

    print("\nAll synthetic datasets generated!")
    print("IMPORTANT: These are synthetic datasets for testing only.")
    print("Do not present results on synthetic data as real experimental findings.")
    print("=" * 70)
