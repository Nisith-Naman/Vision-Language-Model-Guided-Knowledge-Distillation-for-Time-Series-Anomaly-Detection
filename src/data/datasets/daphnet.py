"""Dataset loader for Daphnet Freezing of Gait (FoG).

The Daphnet dataset contains wearable acceleration measurements from Parkinson's disease
patients experiencing Freezing of Gait (FoG) events while walking.
Recordings have 11 columns:
    Column 0: Time (in milliseconds)
    Columns 1-3: Ankle acceleration (horizontal forward, vertical, horizontal lateral)
    Columns 4-6: Thigh acceleration (horizontal forward, vertical, horizontal lateral)
    Columns 7-9: Trunk acceleration (horizontal forward, vertical, horizontal lateral)
    Column 10: Annotation (0 = not part of experiment, 1 = normal, 2 = freeze of gait)

Rows with annotation == 0 are omitted, and segments separated by dropped rows are
partitioned into separate continuous Series objects.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data.loader import Series

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_COLS = [
    "ankle_x", "ankle_y", "ankle_z",
    "thigh_x", "thigh_y", "thigh_z",
    "trunk_x", "trunk_y", "trunk_z",
]


def load(config: Dict[str, Any]) -> List[Series]:
    """Load and process the Daphnet Freezing of Gait dataset from .txt files.

    Args:
        config: Configuration dictionary containing:
            - 'data_dir': Path to directory containing Daphnet .txt files (default './data/daphnet').
            - 'feature_cols': 9 acceleration channel names.
            - 'drop_value': Annotation code to drop (default 0).
            - 'normal_value': Annotation code for normal locomotion (default 1).
            - 'anomaly_value': Annotation code for freezing of gait (default 2).

    Returns:
        List of Series objects, one per continuous segment of valid experimental data.

    Raises:
        FileNotFoundError: If data_dir does not exist or contains no .txt files.
        ValueError: If file formatting is corrupted.
    """
    data_dir_raw = config.get("data_dir", "./data/daphnet")
    data_dir = Path(data_dir_raw)

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"Daphnet dataset directory not found at '{data_dir.resolve()}'.\n"
            f"Please download the Daphnet Freezing of Gait dataset from UCI Machine Learning Repository:\n"
            f"  https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait\n"
            f"Extract all .txt recording files (e.g. S01R01.txt) into '{data_dir.resolve()}'."
        )

    # UCI's extracted archive keeps recordings under dataset/; search nested folders.
    txt_files = sorted(data_dir.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in Daphnet directory '{data_dir.resolve()}'.\n"
            f"Ensure patient recording files (e.g. S01R01.txt ... S10R01.txt) are unzipped."
        )

    feature_cols = config.get("feature_cols", DEFAULT_FEATURE_COLS)
    drop_val = config.get("drop_value", 0)
    normal_val = config.get("normal_value", 1)
    anomaly_val = config.get("anomaly_value", 2)

    all_series: List[Series] = []
    total_raw_rows = 0
    total_valid_rows = 0
    total_anom_steps = 0

    for fpath in txt_files:
        # Extract Subject ID (e.g. 'S01' from 'S01R01.txt')
        match = re.match(r"(S\d+)", fpath.stem, re.IGNORECASE)
        subject_id = match.group(1).upper() if match else fpath.stem

        try:
            # Daphnet files are whitespace-delimited with 11 columns
            data = pd.read_csv(
                fpath,
                sep=r"\s+",
                header=None,
                engine="c",
                dtype=np.float32,
            ).to_numpy()
        except Exception as e:
            logger.warning(f"Failed to read '{fpath.name}' with c-engine ({e}); retrying with python engine.")
            data = pd.read_csv(fpath, sep=r"\s+", header=None, engine="python").to_numpy(dtype=np.float32)

        if data.ndim != 2 or data.shape[1] < 11:
            logger.warning(
                f"Skipping '{fpath.name}': expected 11 columns, got shape {data.shape}"
            )
            continue

        total_raw_rows += len(data)
        time_ms = data[:, 0]
        accel_channels = data[:, 1:10]
        annotations = data[:, 10].astype(np.int32)

        # Identify continuous segments where annotation != drop_val
        is_valid = (annotations != drop_val)
        if not np.any(is_valid):
            continue

        # Find boundaries of contiguous valid blocks
        diff = np.diff(is_valid.astype(np.int32))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if is_valid[0]:
            starts = np.r_[0, starts]
        if is_valid[-1]:
            ends = np.r_[ends, len(is_valid)]

        seg_idx = 0
        for s_idx, e_idx in zip(starts, ends):
            seg_len = e_idx - s_idx
            if seg_len < 4:  # Filter out trivial 1-3 step noise blips
                continue

            seg_accel = accel_channels[s_idx:e_idx].astype(np.float32)
            seg_ann = annotations[s_idx:e_idx]
            seg_time = time_ms[s_idx:e_idx]

            # Label mapping: annotation == 2 -> 1 (freeze); annotation == 1 -> 0 (normal)
            seg_y_step = np.where(seg_ann == anomaly_val, 1, 0).astype(np.int8)

            series_obj = Series(
                values=seg_accel,
                y_step=seg_y_step,
                timestamps=seg_time,
                group_id=subject_id,
                feature_names=list(feature_cols),
                metadata={
                    "dataset": "daphnet",
                    "subject_id": subject_id,
                    "recording_file": fpath.name,
                    "segment_index": seg_idx,
                    "raw_start_idx": int(s_idx),
                    "raw_end_idx": int(e_idx),
                },
            )
            all_series.append(series_obj)
            seg_idx += 1

            total_valid_rows += seg_len
            total_anom_steps += int(np.sum(seg_y_step == 1))

    if not all_series:
        raise ValueError(f"No valid annotated segments extracted from {data_dir.resolve()}.")

    anom_rate = (total_anom_steps / total_valid_rows) if total_valid_rows > 0 else 0.0
    subjects_count = len({s.group_id for s in all_series})

    print(
        f"[Daphnet] Loaded {len(all_series)} continuous segments across {subjects_count} subjects | "
        f"Raw Rows: {total_raw_rows:,d} | Valid Timesteps T: {total_valid_rows:,d} | "
        f"Freeze Steps: {total_anom_steps:,d} ({anom_rate:.2%})"
    )

    return all_series
