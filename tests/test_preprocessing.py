"""Unit tests for the time-series preprocessing module (src.data.preprocessing)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import Series
from src.data.preprocessing import (
    check_leakage,
    chronological_split,
    fill_missing,
    fit_imputation_stats_batches,
    fit_scaler,
    fit_scaler_batches,
    handle_duplicates,
    make_windows,
    sort_chronologically,
    transform_scaler,
)


class TestSortChronologically:
    """Test suite for chronological sorting of time-series DataFrames."""

    def test_sorts_correctly(self) -> None:
        """Verify that rows are re-ordered into ascending chronological sequence."""
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2024-01-01 02:00:00",
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                ],
                "feat_1": [3.0, 1.0, 2.0],
            }
        )
        sorted_df = sort_chronologically(df, time_col="timestamp")
        assert list(sorted_df["feat_1"]) == [1.0, 2.0, 3.0]
        assert list(sorted_df.index) == [0, 1, 2]

    def test_handles_missing_time_col(self) -> None:
        """Verify that KeyError is raised when the specified time_col does not exist."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        with pytest.raises(KeyError, match="Timestamp column 'timestamp' not found"):
            sort_chronologically(df, time_col="timestamp")

    def test_empty_dataframe(self) -> None:
        """Verify that an empty DataFrame returns an empty copy without error."""
        df = pd.DataFrame(columns=["timestamp", "feat"])
        sorted_df = sort_chronologically(df, time_col="timestamp")
        assert len(sorted_df) == 0

    def test_already_sorted(self) -> None:
        """Verify that an already sorted DataFrame preserves row order and values."""
        df = pd.DataFrame(
            {
                "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
                "feat": [10.0, 20.0],
            }
        )
        sorted_df = sort_chronologically(df, time_col="timestamp")
        assert list(sorted_df["feat"]) == [10.0, 20.0]


class TestHandleDuplicates:
    """Test suite for deduplication policies (first, last, mean)."""

    def test_policy_first(self) -> None:
        """Verify policy='first' retains the first recorded observation."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        dedup = handle_duplicates(df, time_col="time", policy="first")
        assert len(dedup) == 2
        assert list(dedup["value"]) == [1.0, 3.0]

    def test_policy_last(self) -> None:
        """Verify policy='last' retains the latest observation."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        dedup = handle_duplicates(df, time_col="time", policy="last")
        assert len(dedup) == 2
        assert list(dedup["value"]) == [2.0, 3.0]

    def test_policy_mean(self) -> None:
        """Verify policy='mean' computes numerical column averages."""
        df = pd.DataFrame(
            {
                "time": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "feat": [10.0, 20.0, 50.0],
                "tag": ["A", "B", "C"],
            }
        )
        dedup = handle_duplicates(df, time_col="time", policy="mean")
        assert len(dedup) == 2
        assert dedup.loc[dedup["time"] == "2024-01-01", "feat"].iloc[0] == 15.0
        assert dedup.loc[dedup["time"] == "2024-01-02", "feat"].iloc[0] == 50.0

    def test_invalid_policy(self) -> None:
        """Verify that unsupported duplicate policies raise ValueError."""
        df = pd.DataFrame({"time": ["2024-01-01"], "val": [1]})
        with pytest.raises(ValueError, match="Unsupported duplicate policy"):
            handle_duplicates(df, time_col="time", policy="median")


class TestMakeWindows:
    """Test suite for sliding window slicing and gap-boundary guards."""

    def test_window_count(self) -> None:
        """Verify sliding window count formula (T - L) // stride + 1."""
        T = 100
        L = 20
        stride = 10
        values = np.random.randn(T, 3).astype(np.float32)
        y_step = np.zeros(T, dtype=np.int8)

        X_win, y_win, y_step_win, ts_starts = make_windows(
            values=values,
            y_step=y_step,
            timestamps=None,
            window_length=L,
            stride=stride,
        )

        expected_count = (T - L) // stride + 1  # (100 - 20) // 10 + 1 = 9
        assert len(X_win) == expected_count
        assert len(y_win) == expected_count
        assert len(y_step_win) == expected_count
        assert len(ts_starts) == expected_count

    def test_no_cross_boundary_windowing(self) -> None:
        """CRITICAL: Windows must NOT cross gap boundaries in gap_indices."""
        T = 100
        L = 20
        stride = 5
        values = np.ones((T, 2), dtype=np.float32)
        y_step = np.zeros(T, dtype=np.int8)

        # Place a gap right in the middle at index 50
        gap_indices = [50]

        X_win, y_win, _, ts_starts = make_windows(
            values=values,
            y_step=y_step,
            timestamps=None,
            window_length=L,
            stride=stride,
            gap_indices=gap_indices,
        )

        # For any generated window [start, start + L), verify gap 50 is NOT inside
        for start in ts_starts:
            end = start + L
            # Gap must not lie strictly inside window: start < gap < end
            assert not (start < 50 < end), (
                f"Window interval [{start}, {end}) illegally crossed gap boundary at 50!"
            )

    def test_window_shape(self) -> None:
        """Verify dimensional shapes of generated window tensors."""
        values = np.zeros((64, 4), dtype=np.float32)
        y_step = np.zeros(64, dtype=np.int8)

        X_win, y_win, y_step_win, _ = make_windows(
            values=values,
            y_step=y_step,
            timestamps=None,
            window_length=16,
            stride=8,
        )

        assert X_win.shape == (7, 16, 4)
        assert y_win.shape == (7,)
        assert y_step_win.shape == (7, 16)
        assert X_win.dtype == np.float32
        assert y_win.dtype == np.int8
        assert y_step_win.dtype == np.int8

    def test_y_window_derivation_any(self) -> None:
        """Verify rule 'any': window label is 1 if >=1 anomalous step is present."""
        values = np.zeros((30, 1), dtype=np.float32)
        y_step = np.zeros(30, dtype=np.int8)
        # Put an anomaly at step 15
        y_step[15] = 1

        # Window length 10, stride 10 -> window 0: [0..9], window 1: [10..19], window 2: [20..29]
        _, y_win, _, _ = make_windows(
            values=values,
            y_step=y_step,
            timestamps=None,
            window_length=10,
            stride=10,
            anomaly_rule="any",
        )

        assert list(y_win) == [0, 1, 0]

    def test_y_window_derivation_majority(self) -> None:
        """Verify rule 'majority': window label is 1 only if >=50% of steps are anomalous."""
        values = np.zeros((20, 1), dtype=np.float32)
        y_step = np.zeros(20, dtype=np.int8)
        # Window 1: 3 out of 10 anomalous -> <50% -> 0
        y_step[0:3] = 1
        # Window 2: 6 out of 10 anomalous -> >=50% -> 1
        y_step[10:16] = 1

        _, y_win, _, _ = make_windows(
            values=values,
            y_step=y_step,
            timestamps=None,
            window_length=10,
            stride=10,
            anomaly_rule="majority",
        )

        assert list(y_win) == [0, 1]

    def test_empty_series_returns_empty(self) -> None:
        """Verify that an empty time-series returns empty array slices."""
        values = np.empty((0, 3), dtype=np.float32)
        y_step = np.empty((0,), dtype=np.int8)

        X_win, y_win, y_step_win, starts = make_windows(
            values=values,
            y_step=y_step,
            timestamps=None,
            window_length=16,
            stride=8,
        )

        assert len(X_win) == 0
        assert len(y_win) == 0
        assert len(starts) == 0

    def test_window_longer_than_series_returns_empty(self) -> None:
        """Verify that when T < window_length, empty arrays are returned."""
        values = np.ones((10, 2), dtype=np.float32)
        y_step = np.zeros(10, dtype=np.int8)

        X_win, y_win, _, _ = make_windows(
            values=values,
            y_step=y_step,
            timestamps=None,
            window_length=32,
            stride=8,
        )

        assert len(X_win) == 0
        assert len(y_win) == 0


class TestSplits:
    """Test suite for dataset splitting and temporal leakage prevention."""

    def test_chronological_no_leakage(self) -> None:
        """Verify timestamps strictly do not overlap across chronological train/val/test splits."""
        T = 300
        dates = pd.date_range("2024-01-01", periods=T, freq="1h")
        series = Series(
            values=np.random.randn(T, 2),
            y_step=np.zeros(T, dtype=np.int8),
            timestamps=dates.to_numpy(),
            group_id="S01",
        )

        train_s, val_s, test_s = chronological_split(
            [series],
            train_frac=0.70,
            val_frac=0.15,
            split_by="chronological",
        )

        assert len(train_s) == 1
        assert len(val_s) == 1
        assert len(test_s) == 1

        # Check strict leakage check
        passed = check_leakage(train_s, val_s, test_s)
        assert passed is True

        # Timestamp boundaries should be monotonically strictly increasing
        assert train_s[0].timestamps[-1] < val_s[0].timestamps[0]
        assert val_s[0].timestamps[-1] < test_s[0].timestamps[0]

    def test_split_ratios_approximately_correct(self) -> None:
        """Verify partitioned split lengths approximate the requested fractions."""
        T = 1000
        series = Series(
            values=np.zeros((T, 1)),
            y_step=np.zeros(T),
            group_id="seq",
        )

        train_s, val_s, test_s = chronological_split(
            [series],
            train_frac=0.60,
            val_frac=0.20,
            split_by="chronological",
        )

        assert train_s[0].n_timesteps == pytest.approx(600, abs=2)
        assert val_s[0].n_timesteps == pytest.approx(200, abs=2)
        assert test_s[0].n_timesteps == pytest.approx(200, abs=2)

    def test_subject_split_preserves_groups(self) -> None:
        """Verify subject/group-based partitioning keeps entities in disjoint splits."""
        series_list = [
            Series(values=np.zeros((50, 1)), y_step=np.zeros(50), group_id=f"Subject_{i}")
            for i in range(10)
        ]

        train_s, val_s, test_s = chronological_split(
            series_list,
            train_frac=0.70,
            val_frac=0.15,
            split_by="subject",
        )

        train_groups = {s.group_id for s in train_s}
        val_groups = {s.group_id for s in val_s}
        test_groups = {s.group_id for s in test_s}

        assert train_groups.isdisjoint(val_groups)
        assert val_groups.isdisjoint(test_groups)
        assert train_groups.isdisjoint(test_groups)
        assert len(train_groups) + len(val_groups) + len(test_groups) == 10


class TestScaler:
    """Test suite for feature normalization and fit/transform leakage guards."""

    def test_standard_scaler_mean_zero(self) -> None:
        """Verify StandardScaler produces zero mean and unit variance on training data."""
        data = np.array([[10.0, 200.0], [20.0, 400.0], [30.0, 600.0]], dtype=np.float32)
        scaler = fit_scaler(data, scaler_type="standard")
        transformed = transform_scaler(data, scaler)

        np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(transformed.std(axis=0), [1.0, 1.0], atol=1e-5)

    def test_minmax_scaler_range(self) -> None:
        """Verify MinMaxScaler scales features within [0, 1]."""
        data = np.array([[-5.0, 50.0], [0.0, 100.0], [15.0, 150.0]], dtype=np.float32)
        scaler = fit_scaler(data, scaler_type="minmax")
        transformed = transform_scaler(data, scaler)

        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0

    def test_scaler_fitted_on_train_only(self) -> None:
        """Verify scaler fit on train transforms val without val re-fitting or leaking."""
        train_data = np.array([[0.0], [10.0]], dtype=np.float32)
        val_data = np.array([[20.0], [30.0]], dtype=np.float32)

        scaler = fit_scaler(train_data, scaler_type="minmax")
        val_transformed = transform_scaler(val_data, scaler)

        # Using train min=0, max=10:
        # 20.0 should scale to (20 - 0) / 10 = 2.0
        # 30.0 should scale to (30 - 0) / 10 = 3.0
        np.testing.assert_allclose(val_transformed.squeeze(), [2.0, 3.0], atol=1e-5)

    def test_incremental_scaler_matches_full_training_matrix(self) -> None:
        data = np.random.default_rng(12).normal(size=(41, 5)).astype(np.float32)
        batches = [data[:7], data[7:23], data[23:]]
        for scaler_type in ("standard", "minmax"):
            full = fit_scaler(data, scaler_type=scaler_type)
            incremental = fit_scaler_batches(batches, scaler_type=scaler_type)
            np.testing.assert_allclose(
                transform_scaler(data, full),
                transform_scaler(data, incremental),
                atol=1e-6,
            )

    def test_incremental_imputation_stats_ignore_nonfinite_values(self) -> None:
        batches = [
            np.array([[1.0, 10.0], [3.0, np.inf]], dtype=np.float64),
            np.array([[5.0, 30.0]], dtype=np.float64),
        ]

        stats = fit_imputation_stats_batches(batches)

        np.testing.assert_allclose(stats["col_means"], [3.0, 20.0])


class TestFillMissing:
    """Test suite for missing value (NaN) imputation routines."""

    def test_interpolate_fills_nan(self) -> None:
        """Verify linear interpolation correctly fills internal and boundary NaNs."""
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
        filled, stats = fill_missing(arr, strategy="interpolate")

        assert not np.isnan(filled).any()
        np.testing.assert_allclose(filled, [1.0, 2.0, 3.0, 4.0, 5.0], atol=1e-5)

    def test_ffill_no_leading_nan(self) -> None:
        """Verify ffill with bfill fallback handles leading NaNs cleanly."""
        arr = np.array([np.nan, np.nan, 7.0, np.nan, 9.0], dtype=np.float32)
        filled, _ = fill_missing(arr, strategy="ffill")

        assert not np.isnan(filled).any()
        assert filled[0] == 7.0
        assert filled[1] == 7.0
        assert filled[2] == 7.0
        assert filled[3] == 7.0
        assert filled[4] == 9.0

    def test_nonfinite_values_are_treated_as_missing(self) -> None:
        arr = np.array([[1.0], [np.inf], [3.0], [-np.inf], [1e300]], dtype=np.float64)

        filled, _ = fill_missing(arr, strategy="ffill")

        assert np.isfinite(filled).all()
        np.testing.assert_allclose(filled[:, 0], [1.0, 1.0, 3.0, 3.0, 3.0])

    def test_mean_with_train_stats_prevents_leakage(self) -> None:
        """Verify that passing pre-computed train_stats applies training column means."""
        val_arr = np.array([[np.nan], [100.0]], dtype=np.float32)
        train_stats = {"col_means": np.array([5.0], dtype=np.float32)}

        filled, _ = fill_missing(val_arr, strategy="mean", train_stats=train_stats)
        assert filled[0, 0] == 5.0


class TestCheckLeakage:
    """Test suite for explicit split intersection and chronology checks."""

    def test_clean_splits_pass(self) -> None:
        """Verify that disjoint timestamps pass check_leakage."""
        t1 = [1, 2, 3]
        t2 = [4, 5, 6]
        t3 = [7, 8, 9]
        assert check_leakage(t1, t2, t3) is True

    def test_overlapping_timestamps_raise_assertion(self) -> None:
        """Verify that overlapping timestamps between splits raise AssertionError."""
        t1 = [1, 2, 3]
        t2 = [3, 4, 5]  # Overlap on 3
        t3 = [6, 7, 8]
        with pytest.raises(AssertionError, match="Temporal leakage detected"):
            check_leakage(t1, t2, t3)
