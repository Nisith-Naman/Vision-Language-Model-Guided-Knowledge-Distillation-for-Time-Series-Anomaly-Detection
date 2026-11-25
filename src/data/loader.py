"""Time-series data loader, schema validator, and dataset dispatching module.

This module defines core data structures (Series, DataQualityReport) and provides
utilities for loading, validating, and inspecting time-series datasets used in the
IE643 VLM-KD anomaly detection project.
"""

from __future__ import annotations

import csv
import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Series:
    """Standardized time-series container for anomaly detection.

    Attributes:
        values: Numerical time-series matrix of shape (T, F) and dtype float32,
            where T is the number of timesteps and F is the number of features.
        y_step: Optional point-wise binary anomaly ground truth of shape (T,) and dtype int8,
            where 0 indicates normal and 1 indicates anomalous. None if unlabeled.
        timestamps: Optional 1D array of timestamps (datetime, string, or numeric
            seconds) of length T corresponding to each timestep, or None.
        group_id: Unique string identifier for the sequence, recording session,
            subject, well instance, or file.
        feature_names: List of string names for the F feature columns.
        metadata: Optional dictionary storing dataset-specific annotations,
            event windows, flags (e.g. 'synthetic anomalies'), or provenance.
    """

    values: np.ndarray
    y_step: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    group_id: str = ""
    feature_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and coerce array shapes and data types."""
        # Coerce values to 2D float32 array
        val_arr = np.asarray(self.values, dtype=np.float32)
        if val_arr.ndim == 1:
            val_arr = val_arr.reshape(-1, 1)
        elif val_arr.ndim != 2:
            raise ValueError(
                f"Expected 2D array for values (T, F), got shape {val_arr.shape}"
            )
        self.values = val_arr

        # Coerce y_step if provided
        if self.y_step is not None:
            y_arr = np.asarray(self.y_step, dtype=np.int8)
            if y_arr.ndim != 1:
                y_arr = y_arr.squeeze()
                if y_arr.ndim != 1:
                    raise ValueError(
                        f"Expected 1D array for y_step (T,), got shape {y_arr.shape}"
                    )
            self.y_step = y_arr

            # Check temporal length alignment
            if len(self.values) != len(self.y_step):
                raise ValueError(
                    f"Length mismatch: values has {len(self.values)} timesteps, "
                    f"y_step has {len(self.y_step)} labels"
                )

        # Validate timestamps if provided
        if self.timestamps is not None:
            ts_arr = np.asarray(self.timestamps)
            if len(ts_arr) != len(self.values):
                raise ValueError(
                    f"Length mismatch: timestamps has {len(ts_arr)} items, "
                    f"values has {len(self.values)} items"
                )
            self.timestamps = ts_arr

        # Populate default feature names if not supplied
        if not self.feature_names and self.values.shape[1] > 0:
            self.feature_names = [f"feat_{i}" for i in range(self.values.shape[1])]
        elif len(self.feature_names) != self.values.shape[1]:
            raise ValueError(
                f"Feature count mismatch: {len(self.feature_names)} names provided "
                f"for {self.values.shape[1]} columns"
            )

    @property
    def n_timesteps(self) -> int:
        """Return number of timesteps T."""
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        """Return number of features F."""
        return int(self.values.shape[1]) if self.values.ndim == 2 else 0

    @property
    def has_ground_truth(self) -> bool:
        """Return True if ground truth labels y_step are present."""
        return self.y_step is not None

    @property
    def anomaly_rate(self) -> float:
        """Return the fraction of timesteps labeled as anomalous."""
        if self.y_step is None or len(self.y_step) == 0:
            return 0.0
        return float(np.mean(self.y_step == 1))

    @property
    def label_rate(self) -> float:
        """Alias for anomaly_rate."""
        return self.anomaly_rate

    def __repr__(self) -> str:
        return (
            f"Series(group_id='{self.group_id}', T={self.n_timesteps}, "
            f"F={self.n_features}, anomaly_rate={self.anomaly_rate:.2%})"
        )


@dataclass
class DataQualityReport:
    """Summary of data quality checks and profiling on loaded raw data.

    Attributes:
        filename: Name or path of the evaluated file.
        n_rows: Number of rows in DataFrame.
        n_cols: Number of columns in DataFrame.
        dtypes_dict: Dictionary mapping column names to their data types.
        missing_counts_dict: Dictionary mapping column names to count of null values.
        duplicate_count: Number of duplicate rows detected.
        numeric_cols: List of column names with numeric types.
        non_numeric_cols: List of column names with non-numeric types.
        sample_rows: First 5 rows of the loaded DataFrame.
        detected_anomaly_rate: Fraction of detected anomalies, if any.
        gap_indices: Timestep indices where temporal gaps occur.
        sampling_intervals: Observed intervals between consecutive timestamps.
    """

    filename: str
    n_rows: int
    n_cols: int
    dtypes_dict: Dict[str, str]
    missing_counts_dict: Dict[str, int]
    duplicate_count: int
    numeric_cols: List[str]
    non_numeric_cols: List[str]
    sample_rows: pd.DataFrame
    detected_anomaly_rate: float = 0.0
    gap_indices: List[int] = field(default_factory=list)
    sampling_intervals: List[float] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        """Return total row count."""
        return self.n_rows

    @property
    def total_columns(self) -> int:
        """Return total column count."""
        return self.n_cols

    @property
    def missing_values_per_col(self) -> Dict[str, int]:
        """Return dictionary of missing value counts per column."""
        return self.missing_counts_dict

    @property
    def duplicate_timestamps(self) -> int:
        """Return count of duplicate timestamps/rows."""
        return self.duplicate_count

    def summary_dict(self) -> Dict[str, Any]:
        """Convert report metrics into a dictionary."""
        return {
            "filename": self.filename,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing_values_per_col": self.missing_values_per_col,
            "duplicate_timestamps": self.duplicate_timestamps,
            "detected_anomaly_rate": self.detected_anomaly_rate,
            "gap_indices": self.gap_indices,
            "sampling_intervals": self.sampling_intervals,
            "numeric_cols": self.numeric_cols,
            "non_numeric_cols": self.non_numeric_cols,
        }

    def print_summary(self) -> None:
        """Print concise summary of dataset quality."""
        print(f"Data Quality Report for: {self.filename}")
        print(f"  Rows: {self.total_rows:,d} | Columns: {self.total_columns}")
        print(f"  Duplicates: {self.duplicate_timestamps} | Anomaly Rate: {self.detected_anomaly_rate:.2%}")


def detect_separator(path: Union[str, Path], sample_lines: int = 15) -> str:
    """Auto-detect the delimiter for a tabular text or CSV file.

    Checks standard candidates comma (,), semicolon (;), and tab (\t).

    Args:
        path: Path to the input CSV file.
        sample_lines: Number of lines to inspect.

    Returns:
        Detected delimiter character (e.g. ',', ';', or '\t').
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")

    with path_obj.open("r", encoding="utf-8", errors="replace") as f:
        sample_text = "".join(f.readline() for _ in range(sample_lines))

    if not sample_text.strip():
        return ","

    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # Fallback heuristic: count occurrences across non-empty lines
        candidates = [",", ";", "\t", "|"]
        counts = {c: sample_text.count(c) for c in candidates}
        best_sep = max(counts, key=counts.get)
        return best_sep if counts[best_sep] > 0 else ","


def load_csv(
    path: Union[str, Path],
    time_col: Optional[str] = None,
    value_cols: Optional[List[str]] = None,
    label_col: Optional[str] = None,
    group_id: str = "",
) -> Union[Tuple[pd.DataFrame, DataQualityReport], Series]:
    """Load a CSV file, parse features/labels into a Series or return (DataFrame, DataQualityReport).

    Args:
        path: Filepath to the CSV file.
        time_col: Optional timestamp column name.
        value_cols: Optional list of numerical feature columns. If supplied, returns Series.
        label_col: Optional anomaly label column name.
        group_id: Unique string identifier for the loaded Series.

    Returns:
        Series object if value_cols/time_col/label_col are requested,
        otherwise Tuple of (loaded DataFrame, DataQualityReport instance).

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If requested columns are missing or parsing fails.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path_obj.resolve()}")

    sep = detect_separator(path_obj)
    try:
        # Load with pandas, handling possible comma-decimal or standard notation
        df = pd.read_csv(path_obj, sep=sep, low_memory=False)
    except Exception as err:
        raise ValueError(
            f"Failed to parse CSV file '{path_obj}' with separator {repr(sep)}: {err}"
        ) from err

    # If value_cols is specified or caller asks for structured Series
    if value_cols is not None or time_col is not None or label_col is not None:
        if value_cols is None:
            exclude = set()
            if time_col:
                exclude.add(time_col)
            if label_col:
                exclude.add(label_col)
            value_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

        # Verify that all value_cols exist in df
        missing_cols = [col for col in value_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required value column(s): {missing_cols}")

        values = df[value_cols].to_numpy(dtype=np.float32)

        timestamps = None
        if time_col is not None:
            if time_col in df.columns:
                timestamps = df[time_col].to_numpy()
            else:
                raise ValueError(f"Time column '{time_col}' not found in CSV")

        y_step = None
        if label_col is not None:
            if label_col in df.columns:
                y_step = df[label_col].to_numpy(dtype=np.int8)
            else:
                raise ValueError(f"Label column '{label_col}' not found in CSV")

        return Series(
            values=values,
            y_step=y_step,
            timestamps=timestamps,
            group_id=group_id or path_obj.stem,
            feature_names=value_cols,
        )

    # Compute profiling metrics
    n_rows, n_cols = df.shape
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_counts = df.isna().sum().to_dict()
    duplicate_count = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    sample_rows = df.head(5).copy()

    report = DataQualityReport(
        filename=str(path_obj.name),
        n_rows=n_rows,
        n_cols=n_cols,
        dtypes_dict=dtypes_dict,
        missing_counts_dict=missing_counts,
        duplicate_count=duplicate_count,
        numeric_cols=numeric_cols,
        non_numeric_cols=non_numeric_cols,
        sample_rows=sample_rows,
    )

    return df, report


def inspect_dataset(path: Union[str, Path]) -> DataQualityReport:
    """Inspect a CSV file and return a DataQualityReport.

    Args:
        path: Path to the target CSV file.

    Returns:
        DataQualityReport instance.
    """
    res = load_csv(path)
    if isinstance(res, tuple):
        return res[1]
    # If loaded as series, generate report
    return DataQualityReport(
        filename=Path(path).name,
        n_rows=res.n_timesteps,
        n_cols=res.n_features,
        dtypes_dict={f: "float32" for f in res.feature_names},
        missing_counts_dict={f: 0 for f in res.feature_names},
        duplicate_count=0,
        numeric_cols=res.feature_names,
        non_numeric_cols=[],
        sample_rows=pd.DataFrame(res.values[:5], columns=res.feature_names),
        detected_anomaly_rate=res.anomaly_rate,
    )


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Retrieve catalog metadata information for a dataset.

    Args:
        name: Name of the dataset (e.g. 'daphnet', 'airquality', 'metropt').

    Returns:
        Dictionary with metadata attributes.
    """
    name_clean = name.strip().lower()
    catalog: Dict[str, Dict[str, Any]] = {
        "creditcard": {
            "name": "Credit Card Fraud Detection",
            "domain": "Finance / Tabular Sequential",
            "ground_truth": True,
            "features": 29,
            "anomaly_type": "Point anomalies (fraud)",
        },
        "daphnet": {
            "name": "Daphnet Freezing of Gait",
            "domain": "Biomedical / Wearable accelerometers",
            "ground_truth": True,
            "features": 9,
            "anomaly_type": "Freezing of Gait episodes",
        },
        "airquality": {
            "name": "UCI Air Quality",
            "domain": "Environmental / Chemical sensor array",
            "ground_truth": False,
            "synthetic_injected": True,
            "features": 12,
            "anomaly_type": "Synthetic level shifts & spikes",
        },
        "oil3w": {
            "name": "Petrobras 3W Oil Well",
            "domain": "Industrial / Petrochemical offshore wells",
            "ground_truth": True,
            "features": 8,
            "anomaly_type": "8 real event failure classes",
        },
        "metropt": {
            "name": "MetroPT-3 Train Compressor",
            "domain": "Transportation / Heavy rail pneumatic system",
            "ground_truth": True,
            "features": 15,
            "anomaly_type": "4 documented compressor maintenance failures",
        },
    }
    return catalog.get(
        name_clean,
        {
            "name": name,
            "domain": "Custom",
            "ground_truth": False,
            "features": -1,
            "anomaly_type": "Unknown",
        },
    )



def validate_csv_schema(df: pd.DataFrame, expected_features: List[str]) -> List[str]:
    """Validate DataFrame against an expected list of feature columns.

    Args:
        df: Input pandas DataFrame to validate.
        expected_features: List of required column names.

    Returns:
        List of validation error messages. Empty list if validation succeeds.
    """
    errors: List[str] = []

    if df.empty:
        errors.append("DataFrame is empty (0 rows).")
        return errors

    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing expected feature columns: {missing_cols}")

    # Check that present feature columns are numeric
    present_cols = [col for col in expected_features if col in df.columns]
    for col in present_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(
                f"Feature column '{col}' has non-numeric dtype '{df[col].dtype}'"
            )

    return errors


def load_dataset(name: str, config: Dict[str, Any]) -> List[Series]:
    """Dispatch to the appropriate dataset loader by dataset name.

    Supported dataset names:
        - 'creditcard'
        - 'daphnet'
        - 'airquality'
        - 'oil3w'
        - 'metropt'

    Args:
        name: Name of the dataset (case-insensitive).
        config: Configuration dictionary for the dataset.

    Returns:
        List of Series objects loaded from the dataset.

    Raises:
        ValueError: If the dataset name is not supported.
        ImportError: If the loader module cannot be imported.
    """
    dataset_key = name.strip().lower()

    dispatch_table: Dict[str, Tuple[str, str]] = {
        "creditcard": ("src.data.datasets.creditcard", "load"),
        "daphnet": ("src.data.datasets.daphnet", "load"),
        "airquality": ("src.data.datasets.airquality", "load"),
        "oil3w": ("src.data.datasets.oil3w", "load"),
        "metropt": ("src.data.datasets.metropt", "load"),
    }

    if dataset_key not in dispatch_table:
        supported = list(dispatch_table.keys())
        raise ValueError(
            f"Unsupported dataset name '{name}'. Supported datasets are: {supported}"
        )

    module_name, func_name = dispatch_table[dataset_key]

    try:
        mod = importlib.import_module(module_name)
    except ImportError as err:
        raise ImportError(
            f"Failed to import dataset loader module '{module_name}' for dataset '{name}'. "
            f"Original error: {err}. Please ensure all required dependencies and dataset files are present."
        ) from err

    loader_fn = getattr(mod, func_name, None)
    if loader_fn is None or not callable(loader_fn):
        raise AttributeError(
            f"Loader module '{module_name}' has no callable function '{func_name}'"
        )

    # Support passing either the root config (containing a 'datasets' section)
    # or the dataset-specific configuration dictionary directly.
    if isinstance(config, dict) and "datasets" in config and dataset_key in config["datasets"]:
        dataset_config = config["datasets"][dataset_key]
    else:
        dataset_config = config

    logger.info(f"Dispatching dataset loader for '{name}' with config keys: {list(dataset_config.keys())}")
    series_list = loader_fn(dataset_config)

    if not isinstance(series_list, list) or not all(isinstance(s, Series) for s in series_list):
        raise TypeError(
            f"Dataset loader '{module_name}.{func_name}' must return a list[Series], "
            f"got {type(series_list)}"
        )

    return series_list


def summary_command(
    dataset_name: str,
    series_list: List[Series],
    split_labels: Optional[Dict[str, Any]] = None,
    window_length: int = 64,
    stride: int = 32,
) -> str:
    """Generate and return a formatted summary table for a dataset and its splits.

    Computes per-split count of series, total timesteps (T), number of features (F),
    point-wise anomaly rate, and estimated window anomaly rate.

    Args:
        dataset_name: Name of the dataset.
        series_list: Complete list of Series objects.
        split_labels: Optional dictionary of splits. Can be:
            1. Mapping of split name to list of Series:
               {'train': [Series, ...], 'val': [...], 'test': [...]}
            2. Mapping of group_id to split name:
               {'S01': 'train', 'S02': 'val', ...}
            If None, summarizes the entire series_list as a single group.
        window_length: Window size for window anomaly rate calculation.
        stride: Stride size for window anomaly rate calculation.

    Returns:
        Formatted multi-line string table summarizing dataset statistics.
    """
    splits_dict: Dict[str, List[Series]] = {}

    if split_labels is None:
        splits_dict["all"] = series_list
    elif isinstance(split_labels, dict):
        first_val = next(iter(split_labels.values()), None)
        if isinstance(first_val, list):
            # Form: {'train': [Series...], 'val': [Series...], 'test': [Series...]}
            splits_dict = {
                k: [s for s in v if isinstance(s, Series)]
                for k, v in split_labels.items()
            }
        else:
            # Form: {'group_id': 'train', ...}
            group_to_split = split_labels
            temp_splits: Dict[str, List[Series]] = {}
            for s in series_list:
                s_split = group_to_split.get(s.group_id, "unassigned")
                temp_splits.setdefault(s_split, []).append(s)
            splits_dict = temp_splits
    else:
        splits_dict["all"] = series_list

    header = (
        f"========================================================================================\n"
        f" Dataset Summary: {dataset_name.upper()}\n"
        f"========================================================================================\n"
        f"{'Split':<12} | {'#Series':<8} | {'Total T':<12} | {'Features (F)':<12} | {'Step Anom %':<14} | {'Win Anom %':<14}\n"
        f"----------------------------------------------------------------------------------------"
    )

    rows: List[str] = []
    total_series = 0
    total_T = 0
    global_anom_steps = 0
    global_win_anom_count = 0
    global_total_wins = 0
    features_seen = set()

    for split_name, s_list in splits_dict.items():
        n_ser = len(s_list)
        total_series += n_ser
        if n_ser == 0:
            rows.append(
                f"{split_name:<12} | {0:<8} | {0:<12} | {'-':<12} | {'0.00%':<14} | {'0.00%':<14}"
            )
            continue

        split_T = sum(s.n_timesteps for s in s_list)
        split_anom_steps = sum(int(np.sum(s.y_step == 1)) for s in s_list)
        step_anom_rate = (split_anom_steps / split_T) if split_T > 0 else 0.0

        f_dims = {s.n_features for s in s_list}
        features_seen.update(f_dims)
        f_repr = str(next(iter(f_dims))) if len(f_dims) == 1 else str(sorted(list(f_dims)))

        # Approximate window anomaly count
        split_win_anom = 0
        split_total_wins = 0
        for s in s_list:
            t_len = s.n_timesteps
            if t_len >= window_length:
                n_w = (t_len - window_length) // stride + 1
                for idx in range(0, t_len - window_length + 1, stride):
                    w_anom = np.any(s.y_step[idx : idx + window_length] == 1)
                    if w_anom:
                        split_win_anom += 1
                split_total_wins += n_w

        win_anom_rate = (split_win_anom / split_total_wins) if split_total_wins > 0 else 0.0

        total_T += split_T
        global_anom_steps += split_anom_steps
        global_win_anom_count += split_win_anom
        global_total_wins += split_total_wins

        rows.append(
            f"{split_name:<12} | {n_ser:<8} | {split_T:<12,d} | {f_repr:<12} | "
            f"{step_anom_rate:<14.2%} | {win_anom_rate:<14.2%}"
        )

    footer_step_rate = (global_anom_steps / total_T) if total_T > 0 else 0.0
    footer_win_rate = (
        (global_win_anom_count / global_total_wins) if global_total_wins > 0 else 0.0
    )
    all_f_repr = str(next(iter(features_seen))) if len(features_seen) == 1 else str(sorted(list(features_seen)))

    footer = (
        f"----------------------------------------------------------------------------------------\n"
        f"{'TOTAL':<12} | {total_series:<8} | {total_T:<12,d} | {all_f_repr:<12} | "
        f"{footer_step_rate:<14.2%} | {footer_win_rate:<14.2%}\n"
        f"========================================================================================"
    )

    table_text = "\n".join([header] + rows + [footer])
    print(table_text)
    return table_text
