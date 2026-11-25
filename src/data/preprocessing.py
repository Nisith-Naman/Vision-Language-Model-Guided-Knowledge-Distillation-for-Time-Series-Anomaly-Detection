"""Data preprocessing, window slicing, scaling, and train/val/test splitting pipeline.

This module provides production-grade routines for temporal sorting, missing value
imputation, scaling, sliding-window generation with gap guards, and split leakage validation.
"""

from __future__ import annotations

import bisect
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.data.loader import Series

logger = logging.getLogger(__name__)


class IdentityScaler(BaseEstimator, TransformerMixin):
    """Passthrough scaler that performs an identity mapping."""

    def fit(self, X: Any, y: Any = None) -> "IdentityScaler":
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)

    def inverse_transform(self, X: Any) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)


def sort_chronologically(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Sort a DataFrame in ascending chronological order by time_col.

    Args:
        df: Input pandas DataFrame.
        time_col: Name of the timestamp or temporal ordering column.

    Returns:
        New chronologically sorted DataFrame with reset integer index.

    Raises:
        KeyError: If time_col is not present in df.
    """
    if df.empty:
        return df.copy()

    if time_col not in df.columns:
        raise KeyError(f"Timestamp column '{time_col}' not found in DataFrame.")

    df_sorted = df.copy()

    # Attempt datetime conversion if string or object, otherwise preserve numerical ordering
    if pd.api.types.is_object_dtype(df_sorted[time_col]) or pd.api.types.is_string_dtype(
        df_sorted[time_col]
    ):
        converted = pd.to_datetime(df_sorted[time_col], errors="coerce")
        if converted.notna().sum() > 0:
            df_sorted[time_col] = converted

    df_sorted = df_sorted.sort_values(by=time_col, ascending=True).reset_index(drop=True)
    return df_sorted


def handle_duplicates(
    df: pd.DataFrame, time_col: str, policy: str = "first"
) -> pd.DataFrame:
    """Resolve duplicate timestamps in a DataFrame according to specified policy.

    Args:
        df: Input pandas DataFrame.
        time_col: Name of the timestamp column.
        policy: Deduplication policy: 'first', 'last', or 'mean'.
            - 'first': Keep first occurrence.
            - 'last': Keep last occurrence.
            - 'mean': Average numeric features for identical timestamps, keeping
                      first occurrence for non-numeric columns.

    Returns:
        Deduplicated DataFrame with reset integer index.

    Raises:
        ValueError: If an unsupported policy is specified.
        KeyError: If time_col is not in df.
    """
    if df.empty:
        return df.copy()

    if time_col not in df.columns:
        raise KeyError(f"Timestamp column '{time_col}' not found in DataFrame.")

    policy_norm = policy.strip().lower()

    if policy_norm == "first":
        return df.drop_duplicates(subset=[time_col], keep="first").reset_index(drop=True)
    elif policy_norm == "last":
        return df.drop_duplicates(subset=[time_col], keep="last").reset_index(drop=True)
    elif policy_norm == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != time_col]

        agg_dict: Dict[str, Any] = {}
        for col in numeric_cols:
            if col != time_col:
                agg_dict[col] = "mean"
        for col in non_numeric_cols:
            agg_dict[col] = "first"

        if not agg_dict:
            return df.drop_duplicates(subset=[time_col], keep="first").reset_index(drop=True)

        df_dedup = df.groupby(time_col, as_index=False).agg(agg_dict)
        # Restore original column ordering
        existing_cols = [c for c in df.columns if c in df_dedup.columns]
        return df_dedup[existing_cols].reset_index(drop=True)
    else:
        raise ValueError(
            f"Unsupported duplicate policy '{policy}'. Must be 'first', 'last', or 'mean'."
        )


def fill_missing(
    arr: np.ndarray,
    strategy: str = "interpolate",
    train_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Impute missing values (NaNs) in a time-series feature matrix.

    Handles edge cases including all-NaN columns and leading/trailing NaNs.

    Args:
        arr: Input array of shape (T, F) or (T,).
        strategy: Imputation strategy: 'interpolate', 'ffill', 'bfill', or 'mean'.
        train_stats: Optional dictionary containing precomputed training statistics
            (e.g. {'col_means': np.ndarray}) to prevent validation/test data leakage.

    Returns:
        Tuple of (imputed array of shape (T, F) as float32, dictionary of imputation stats).

    Raises:
        ValueError: If an unknown strategy is specified.
    """
    if arr.size == 0:
        return np.asarray(arr, dtype=np.float32), (train_stats or {})

    orig_ndim = arr.ndim
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Treat infinities and values that cannot be represented in the float32
    # arrays used by the model as missing sensor readings. These otherwise
    # overflow during conversion and break downstream scalers.
    float32_limit = np.finfo(np.float32).max
    data[~np.isfinite(data) | (np.abs(data) > float32_limit)] = np.nan

    T, F = data.shape
    strat_norm = strategy.strip().lower()
    stats: Dict[str, Any] = dict(train_stats) if train_stats is not None else {}

    # Extract or compute baseline reference means
    if "col_means" in stats:
        ref_means = np.asarray(stats["col_means"], dtype=np.float64)
        if len(ref_means) != F:
            ref_means = np.zeros(F, dtype=np.float64)
    else:
        ref_means = np.nanmean(data, axis=0)
        ref_means = np.nan_to_num(ref_means, nan=0.0)
        stats["col_means"] = ref_means.astype(np.float32)

    stats["strategy"] = strat_norm

    df = pd.DataFrame(data)

    if strat_norm == "interpolate":
        # Linear interpolation followed by bfill/ffill for boundary NaNs
        df_filled = df.interpolate(method="linear", axis=0).bfill().ffill()
    elif strat_norm == "ffill":
        # Forward fill followed by bfill for leading NaNs
        df_filled = df.ffill().bfill()
    elif strat_norm == "bfill":
        # Backward fill followed by ffill for trailing NaNs
        df_filled = df.bfill().ffill()
    elif strat_norm == "mean":
        df_filled = df.copy()
        for col_idx in range(F):
            col_mean = ref_means[col_idx]
            df_filled[col_idx] = df_filled[col_idx].fillna(col_mean)
    else:
        raise ValueError(
            f"Unsupported missing strategy '{strategy}'. "
            f"Must be 'interpolate', 'ffill', 'bfill', or 'mean'."
        )

    # Any remaining NaNs (e.g. all-NaN column during interpolate/ffill) get filled with ref_means
    out_arr = df_filled.to_numpy(dtype=np.float64)
    for col_idx in range(F):
        mask = np.isnan(out_arr[:, col_idx])
        if np.any(mask):
            out_arr[mask, col_idx] = ref_means[col_idx]

    result = out_arr.astype(np.float32)
    if orig_ndim == 1:
        result = result.squeeze(axis=-1)

    return result, stats


def fit_imputation_stats_batches(X_batches: Iterable[np.ndarray]) -> Dict[str, Any]:
    """Compute training-only fallback means without concatenating large datasets."""
    totals: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None
    limit = np.finfo(np.float32).max

    for batch in X_batches:
        values = np.asarray(batch, dtype=np.float64)
        if values.size == 0:
            continue
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if totals is None:
            totals = np.zeros(values.shape[1], dtype=np.float64)
            counts = np.zeros(values.shape[1], dtype=np.int64)
        elif values.shape[1] != len(totals):
            raise ValueError("All imputation batches must have the same feature count.")

        valid = np.isfinite(values) & (np.abs(values) <= limit)
        totals += np.where(valid, values, 0.0).sum(axis=0, dtype=np.float64)
        counts += valid.sum(axis=0)

    if totals is None or counts is None:
        raise ValueError("Cannot fit imputation statistics on empty batches.")

    means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    return {"col_means": means.astype(np.float32)}


def fit_scaler(X_train: np.ndarray, scaler_type: str = "standard") -> Any:
    """Fit a feature scaler on training data.

    Args:
        X_train: Training feature matrix of shape (N, F) or (T, F).
        scaler_type: Type of scaler: 'standard', 'minmax', 'robust', or 'none'.

    Returns:
        Fitted scikit-learn scaler object.

    Raises:
        ValueError: If X_train is empty or scaler_type is unknown.
    """
    if X_train.size == 0:
        raise ValueError("Cannot fit scaler on empty array.")

    X_mat = np.asarray(X_train, dtype=np.float32)
    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(-1, 1)

    st = (scaler_type or "standard").strip().lower()

    if st == "standard":
        scaler = StandardScaler()
    elif st == "minmax":
        scaler = MinMaxScaler()
    elif st == "robust":
        scaler = RobustScaler()
    elif st in ("none", "identity"):
        scaler = IdentityScaler()
    else:
        raise ValueError(
            f"Unsupported scaler_type '{scaler_type}'. "
            f"Must be 'standard', 'minmax', 'robust', or 'none'."
        )

    scaler.fit(X_mat)
    return scaler


def fit_scaler_batches(X_batches: Iterable[np.ndarray], scaler_type: str = "standard") -> Any:
    """Fit supported scalers incrementally over training arrays.

    This avoids concatenating tens of millions of rows (for example, all 3W well
    instances) into another full-size temporary matrix. StandardScaler and
    MinMaxScaler use their exact ``partial_fit`` implementations. RobustScaler
    keeps the existing full-matrix behavior because it has no incremental fit.
    """
    st = (scaler_type or "standard").strip().lower()
    batches = [np.asarray(batch, dtype=np.float32) for batch in X_batches if np.asarray(batch).size]
    if not batches:
        raise ValueError("Cannot fit a scaler on empty batches.")

    if st in ("none", "identity"):
        return IdentityScaler()
    if st == "robust":
        return fit_scaler(np.concatenate(batches, axis=0), scaler_type=st)
    if st == "standard":
        scaler = StandardScaler()
    elif st == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(
            f"Unsupported scaler_type '{scaler_type}'. "
            "Must be 'standard', 'minmax', 'robust', or 'none'."
        )

    for batch in batches:
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        scaler.partial_fit(batch)
    return scaler


def transform_scaler(X: np.ndarray, scaler: Any) -> np.ndarray:
    """Transform features using a fitted scaler.

    Args:
        X: Feature matrix of shape (N, F) or (T, F).
        scaler: Fitted scaler object.

    Returns:
        Scaled array of shape (N, F) or (T, F) as float32.
    """
    if X.size == 0:
        return np.asarray(X, dtype=np.float32)

    orig_ndim = X.ndim
    X_mat = np.asarray(X, dtype=np.float32)
    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(-1, 1)

    if scaler is None:
        scaled = X_mat
    else:
        scaled = scaler.transform(X_mat)

    res = np.asarray(scaled, dtype=np.float32)
    if orig_ndim == 1:
        res = res.squeeze(axis=-1)
    return res


def chronological_split(
    series_list: List[Series],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    split_by: str = "chronological",
    seed: int = 42,
) -> Tuple[List[Series], List[Series], List[Series]]:
    """Partition a list of Series into train, validation, and test splits.

    Supports chronological time slicing for continuous timelines, and subject / group
    partitioning for multi-recording or multi-subject datasets.

    Args:
        series_list: List of Series objects.
        train_frac: Fraction of data allocated to training (e.g. 0.70).
        val_frac: Fraction of data allocated to validation (e.g. 0.15).
        split_by: Splitting strategy: 'chronological', 'subject', 'group',
            'instance', or 'event'.
        seed: Random seed for reproducible group assignment.

    Returns:
        Tuple of (train_series, val_series, test_series).

    Raises:
        ValueError: If split fractions are invalid or series_list is empty.
    """
    if not series_list:
        return [], [], []

    if train_frac <= 0 or val_frac < 0 or (train_frac + val_frac >= 1.0):
        raise ValueError(
            f"Invalid split fractions: train={train_frac}, val={val_frac}. "
            f"Must satisfy train > 0, val >= 0, and train + val < 1.0"
        )

    strategy = split_by.strip().lower()

    if strategy in ("subject", "group", "instance", "event"):
        # Group-based split: preserve group integrity
        unique_groups: List[str] = []
        for s in series_list:
            if s.group_id not in unique_groups:
                unique_groups.append(s.group_id)

        n_groups = len(unique_groups)
        if n_groups < 3:
            raise ValueError(
                f"Group-based split needs at least 3 distinct {strategy} groups to create "
                "train, validation, and test partitions. Use chronological splitting for a single series."
            )

        # Stratify by whether each group contains any anomaly, then shuffle within
        # each class so sorted subject/file names do not determine the partitions.
        rng = np.random.default_rng(seed)
        group_anomaly = {
            group_id: any(
                s.group_id == group_id and s.y_step is not None and np.any(s.y_step == 1)
                for s in series_list
            )
            for group_id in unique_groups
        }
        train_groups: set[str] = set()
        val_groups: set[str] = set()
        test_groups: set[str] = set()

        for class_value in (False, True):
            class_groups = [g for g in unique_groups if group_anomaly[g] is class_value]
            if not class_groups:
                continue
            class_groups = list(np.asarray(class_groups, dtype=object)[rng.permutation(len(class_groups))])
            n_class = len(class_groups)
            if n_class >= 3:
                n_train = max(1, int(round(n_class * train_frac)))
                n_val = max(1, int(round(n_class * val_frac)))
                if n_train + n_val >= n_class:
                    n_train, n_val = n_class - 2, 1
                train_groups.update(class_groups[:n_train])
                val_groups.update(class_groups[n_train:n_train + n_val])
                test_groups.update(class_groups[n_train + n_val:])
            elif n_class == 2:
                train_groups.add(class_groups[0])
                test_groups.add(class_groups[1])
            else:
                # Rare classes cannot appear in all splits without duplicating a group.
                destinations = (train_groups, val_groups, test_groups)
                destination = min(destinations, key=len)
                destination.add(class_groups[0])

        # Ensure every split is populated when the total number of groups permits it.
        for destination in (train_groups, val_groups, test_groups):
            if destination:
                continue
            candidates = sorted(train_groups, key=lambda g: sum(s.n_timesteps for s in series_list if s.group_id == g), reverse=True)
            if len(candidates) <= 1:
                candidates = sorted(test_groups, key=lambda g: sum(s.n_timesteps for s in series_list if s.group_id == g), reverse=True)
            if candidates:
                moved = candidates[-1]
                train_groups.discard(moved)
                val_groups.discard(moved)
                test_groups.discard(moved)
                destination.add(moved)

        train_series = [s for s in series_list if s.group_id in train_groups]
        val_series = [s for s in series_list if s.group_id in val_groups]
        test_series = [s for s in series_list if s.group_id in test_groups]

        return train_series, val_series, test_series

    elif strategy == "chronological":
        if len(series_list) == 1:
            # Single continuous timeline: slice timesteps
            s = series_list[0]
            T = s.n_timesteps
            t_train = int(round(T * train_frac))
            t_val = int(round(T * (train_frac + val_frac)))

            train_s = Series(
                values=s.values[:t_train],
                y_step=s.y_step[:t_train],
                timestamps=s.timestamps[:t_train] if s.timestamps is not None else None,
                group_id=f"{s.group_id}_train",
                feature_names=list(s.feature_names),
                metadata=dict(s.metadata),
            )
            val_s = Series(
                values=s.values[t_train:t_val],
                y_step=s.y_step[t_train:t_val],
                timestamps=s.timestamps[t_train:t_val] if s.timestamps is not None else None,
                group_id=f"{s.group_id}_val",
                feature_names=list(s.feature_names),
                metadata=dict(s.metadata),
            )
            test_s = Series(
                values=s.values[t_val:],
                y_step=s.y_step[t_val:],
                timestamps=s.timestamps[t_val:] if s.timestamps is not None else None,
                group_id=f"{s.group_id}_test",
                feature_names=list(s.feature_names),
                metadata=dict(s.metadata),
            )
            return [train_s], [val_s], [test_s]
        else:
            # Multiple series chronologically partitioned
            total_T = sum(s.n_timesteps for s in series_list)
            target_train_T = total_T * train_frac
            target_val_T = total_T * (train_frac + val_frac)

            train_series = []
            val_series = []
            test_series = []
            accum_T = 0

            for s in series_list:
                mid_T = accum_T + (s.n_timesteps / 2.0)
                if mid_T < target_train_T:
                    train_series.append(s)
                elif mid_T < target_val_T:
                    val_series.append(s)
                else:
                    test_series.append(s)
                accum_T += s.n_timesteps

            # Ensure non-empty partitions if sufficient series exist
            if len(series_list) >= 3:
                if not train_series and val_series:
                    train_series.append(val_series.pop(0))
                if not val_series and test_series:
                    val_series.append(test_series.pop(0))
                if not test_series and val_series:
                    test_series.append(val_series.pop())

            return train_series, val_series, test_series
    else:
        raise ValueError(
            f"Unsupported split strategy '{split_by}'. "
            f"Choose from 'chronological', 'subject', 'group', 'instance', 'event'."
        )


def make_windows(
    values: np.ndarray,
    y_step: np.ndarray,
    timestamps: Optional[np.ndarray],
    window_length: int,
    stride: int,
    gap_indices: Optional[List[int]] = None,
    anomaly_rule: str = "any",
    cover_tail: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """Generate sliding window slices from a continuous time series.

    Strictly forbids window generation across gap boundaries specified in gap_indices.

    Args:
        values: 2D array of shape (T, F) and dtype float32.
        y_step: 1D ground truth array of shape (T,) and dtype int8.
        timestamps: Optional 1D array of timestamps of length T, or None.
        window_length: Temporal length (L) of each sliding window.
        stride: Step size between successive window starts.
        gap_indices: Optional list of integer timestep indices where temporal continuity
            is broken. Windows overlapping these indices are excluded.
        anomaly_rule: Rule to assign window-level label y_window from y_step_window:
            - 'any': 1 if at least one anomalous step is present in the window, else 0.
            - 'majority': 1 if at least 50% of the steps in the window are anomalous, else 0.
        cover_tail: Add a final end-anchored window when regular strides leave
            trailing timesteps uncovered.

    Returns:
        Tuple containing:
            - X_windows: shape (N, window_length, F), float32.
            - y_window: shape (N,), int8 binary window labels (0 or 1).
            - y_step_windows: shape (N, window_length), int8 timestep labels.
            - ts_starts: List of starting timestamps or start indices of length N.
    """
    T = len(values)
    F = values.shape[1] if values.ndim > 1 else 1
    empty_X = np.empty((0, window_length, F), dtype=np.float32)
    empty_y_win = np.empty((0,), dtype=np.int8)
    empty_y_step_win = np.empty((0, window_length), dtype=np.int8)

    if T < window_length or window_length <= 0 or stride <= 0:
        return empty_X, empty_y_win, empty_y_step_win, []

    # Sort gaps for fast bisect range queries
    sorted_gaps = sorted(gap_indices) if gap_indices else []

    X_list: List[np.ndarray] = []
    y_win_list: List[int] = []
    y_step_win_list: List[np.ndarray] = []
    ts_starts: List[Any] = []

    rule_norm = anomaly_rule.strip().lower()

    starts = list(range(0, T - window_length + 1, stride))
    if cover_tail and starts and starts[-1] + window_length < T:
        starts.append(T - window_length)

    for start in starts:
        end = start + window_length

        # Check if any gap lies strictly inside the window interval (start, end)
        if sorted_gaps:
            idx = bisect.bisect_right(sorted_gaps, start)
            if idx < len(sorted_gaps) and sorted_gaps[idx] < end:
                # Discontinuity detected inside window; skip
                continue

        x_win = values[start:end]
        y_step_win = y_step[start:end]

        if rule_norm == "any":
            win_label = 1 if np.any(y_step_win == 1) else 0
        elif rule_norm == "majority":
            win_label = 1 if (np.mean(y_step_win == 1) >= 0.5) else 0
        else:
            raise ValueError(
                f"Unsupported anomaly_rule '{anomaly_rule}'. Must be 'any' or 'majority'."
            )

        X_list.append(x_win)
        y_win_list.append(win_label)
        y_step_win_list.append(y_step_win)

        if timestamps is not None and len(timestamps) > start:
            ts_starts.append(timestamps[start])
        else:
            ts_starts.append(start)

    if not X_list:
        return empty_X, empty_y_win, empty_y_step_win, []

    X_windows = np.ascontiguousarray(np.stack(X_list, axis=0), dtype=np.float32)
    y_window = np.asarray(y_win_list, dtype=np.int8)
    y_step_windows = np.ascontiguousarray(np.stack(y_step_win_list, axis=0), dtype=np.int8)

    return X_windows, y_window, y_step_windows, ts_starts


def check_leakage(
    train_ts: Any,
    val_ts: Any,
    test_ts: Any,
    check_group_identity: bool = False,
) -> bool:
    """Assert that no temporal or identity overlap occurs between splits.

    Accepts lists of Series, timestamp arrays, or scalar collections.

    Args:
        train_ts: Timestamps or Series from training set.
        val_ts: Timestamps or Series from validation set.
        test_ts: Timestamps or Series from test set.
        check_group_identity: If True, validate that whole subject/instance IDs are
            disjoint. Timestamp overlap is then allowed across distinct entities.

    Returns:
        True if all leakage checks pass.

    Raises:
        AssertionError: If timestamp intersection or temporal sequence violation occurs.
    """
    def _extract_ts(obj: Any) -> np.ndarray:
        if obj is None:
            return np.array([])
        if isinstance(obj, list):
            collected = []
            for item in obj:
                if isinstance(item, Series):
                    if item.timestamps is not None:
                        collected.extend(item.timestamps)
                elif isinstance(item, (np.ndarray, list, pd.Series)):
                    collected.extend(item)
                else:
                    collected.append(item)
            return np.asarray(collected)
        if isinstance(obj, Series):
            return obj.timestamps if obj.timestamps is not None else np.array([])
        return np.asarray(obj)

    if check_group_identity:
        def _extract_groups(obj: Any) -> set[str]:
            items = obj if isinstance(obj, list) else [obj]
            return {str(item.group_id) for item in items if isinstance(item, Series)}

        groups_train = _extract_groups(train_ts)
        groups_val = _extract_groups(val_ts)
        groups_test = _extract_groups(test_ts)
        assert groups_train.isdisjoint(groups_val), "Group leakage detected between train and validation."
        assert groups_train.isdisjoint(groups_test), "Group leakage detected between train and test."
        assert groups_val.isdisjoint(groups_test), "Group leakage detected between validation and test."
        return True

    t_tr = _extract_ts(train_ts)
    t_va = _extract_ts(val_ts)
    t_te = _extract_ts(test_ts)

    # If timestamps are absent across splits, leakage check is vacuous
    if len(t_tr) == 0 or len(t_va) == 0 or len(t_te) == 0:
        logger.info("Timestamps missing in one or more splits; set-based leakage check skipped.")
        return True

    # If the inputs are Series objects, timestamps are only comparable within a
    # recording. Distinct subjects often share wall-clock timestamps, which is
    # not leakage. Include group_id in the identity key while retaining the raw
    # timestamps below for chronological-order diagnostics.
    def _temporal_keys(obj: Any) -> set[Any]:
        items = obj if isinstance(obj, list) else [obj]
        keys: set[Any] = set()
        saw_series = False
        for item in items:
            if isinstance(item, Series):
                saw_series = True
                if item.timestamps is not None:
                    keys.update((str(item.group_id), timestamp) for timestamp in item.timestamps)
        if saw_series:
            return keys
        return set(_extract_ts(obj))

    # 1. Exact timestamp (or recording/timestamp) intersection check
    set_tr = _temporal_keys(train_ts)
    set_va = _temporal_keys(val_ts)
    set_te = _temporal_keys(test_ts)

    tr_va_overlap = set_tr.intersection(set_va)
    assert not tr_va_overlap, (
        f"Temporal leakage detected: Train and Val share {len(tr_va_overlap)} timestamps! "
        f"Sample overlap: {list(tr_va_overlap)[:3]}"
    )

    va_te_overlap = set_va.intersection(set_te)
    assert not va_te_overlap, (
        f"Temporal leakage detected: Val and Test share {len(va_te_overlap)} timestamps! "
        f"Sample overlap: {list(va_te_overlap)[:3]}"
    )

    tr_te_overlap = set_tr.intersection(set_te)
    assert not tr_te_overlap, (
        f"Temporal leakage detected: Train and Test share {len(tr_te_overlap)} timestamps! "
        f"Sample overlap: {list(tr_te_overlap)[:3]}"
    )

    # 2. Chronological sequence check if timestamps are ordered
    try:
        max_tr = np.max(t_tr)
        min_va = np.min(t_va)
        max_va = np.max(t_va)
        min_te = np.min(t_te)

        if max_tr >= min_va:
            logger.warning(
                f"Non-chronological ordering between Train (max={max_tr}) and Val (min={min_va}). "
                f"Verify if subject/group splitting was intended."
            )
        if max_va >= min_te:
            logger.warning(
                f"Non-chronological ordering between Val (max={max_va}) and Test (min={min_te}). "
                f"Verify if subject/group splitting was intended."
            )
    except Exception as e:
        logger.debug(f"Chronological comparison skipped due to data type: {e}")

    return True


def print_split_stats(
    name: str,
    train_series: List[Series],
    val_series: List[Series],
    test_series: List[Series],
    window_length: int = 64,
    stride: int = 32,
) -> None:
    """Print split dataset statistics and issue warnings for zero-anomaly partitions.

    Args:
        name: Name of the dataset.
        train_series: List of training Series.
        val_series: List of validation Series.
        test_series: List of test Series.
        window_length: Window length used for window-level evaluation.
        stride: Stride used for window-level evaluation.
    """
    splits = {
        "Train": train_series,
        "Val": val_series,
        "Test": test_series,
    }

    print(f"\n==================== Split Statistics: {name.upper()} ====================")
    for split_name, s_list in splits.items():
        n_series = len(s_list)
        total_T = sum(s.n_timesteps for s in s_list)
        total_anom = sum(int(np.sum(s.y_step == 1)) for s in s_list)
        anom_rate = (total_anom / total_T) if total_T > 0 else 0.0

        f_dim = s_list[0].n_features if s_list else 0

        # Calculate window-level anomaly count
        total_wins = 0
        anom_wins = 0
        for s in s_list:
            if s.n_timesteps >= window_length:
                n_w = (s.n_timesteps - window_length) // stride + 1
                for idx in range(0, s.n_timesteps - window_length + 1, stride):
                    if np.any(s.y_step[idx : idx + window_length] == 1):
                        anom_wins += 1
                total_wins += n_w

        win_rate = (anom_wins / total_wins) if total_wins > 0 else 0.0

        print(
            f"[{split_name:<5}] Series: {n_series:<4} | Timesteps T: {total_T:<10,d} | "
            f"F: {f_dim:<3} | Step Anom: {total_anom:<6,d} ({anom_rate:.2%}) | "
            f"Win Anom: {anom_wins:<6,d}/{total_wins:<6,d} ({win_rate:.2%})"
        )

        if total_anom == 0:
            logger.warning(
                f"Split '{split_name}' in dataset '{name}' contains ZERO positive anomaly steps! "
                f"Evaluation metrics such as F1, Precision, and AUPRC cannot be computed properly."
            )
    print("========================================================================\n")
