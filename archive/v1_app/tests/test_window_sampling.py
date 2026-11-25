"""Tests for bounded, deterministic window sampling in the preprocessing UI."""

from __future__ import annotations

import numpy as np

from src.ui.page3_preprocessing import (
    _allocate_window_quotas,
    _representative_window_indices,
)
from src.ui.page5_training import _representative_validation_indices
from src.ui.page2_upload import _benchmark_to_frame
from scripts.validate_local_datasets import _count_windows
from src.data.loader import Series


def test_window_quotas_respect_cap_and_cover_each_sequence():
    counts = [1, 4, 100, 20]
    quotas = _allocate_window_quotas(counts, cap=75)
    assert int(quotas.sum()) == 75
    assert np.all(quotas >= 1)
    assert np.all(quotas <= counts)


def test_window_sampling_is_deterministic_and_keeps_both_labels():
    labels = np.r_[np.zeros(80, dtype=np.int8), np.ones(20, dtype=np.int8)]
    selected = _representative_window_indices(labels, quota=30)
    assert len(selected) == 30
    assert len(np.unique(selected)) == 30
    assert np.any(labels[selected] == 0)
    assert np.any(labels[selected] == 1)


def test_training_validation_subset_is_class_aware():
    labels = np.r_[np.zeros(90, dtype=np.int8), np.ones(10, dtype=np.int8)]
    selected = _representative_validation_indices(labels, budget=20)
    assert len(selected) == 20
    assert np.any(labels[selected] == 0)
    assert np.any(labels[selected] == 1)


def test_local_validation_script_counts_tail_and_positive_windows():
    sequence = Series(
        values=np.zeros((11, 2), dtype=np.float32),
        y_step=np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        group_id="test-sequence",
        feature_names=["x", "y"],
    )
    # Length 4, stride 3 yields starts 0, 3, 6, plus a tail-anchored start at 7.
    assert _count_windows([sequence], length=4, stride=3) == (4, 1)


def test_benchmark_preview_is_capped_without_losing_full_series_count():
    sequence = Series(
        values=np.arange(200, dtype=np.float32).reshape(100, 2),
        y_step=np.zeros(100, dtype=np.int8),
        group_id="large-sequence",
        feature_names=["x", "y"],
    )
    preview = _benchmark_to_frame([sequence], max_rows=10)
    assert len(preview) == 10
    assert preview.attrs["full_timesteps"] == 100
    assert preview.attrs["is_sampled_preview"] is True
