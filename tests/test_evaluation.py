"""Unit tests for evaluation metrics, threshold tuning, and temporal anomaly localization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.localization import (
    event_f1,
    iou_per_anomaly,
    mean_iou,
    point_adjustment,
)
from src.evaluation.metrics import (
    compute_step_metrics,
    compute_window_metrics,
    confusion_matrix_at_threshold,
    format_metrics_table,
    tune_threshold,
)


class TestWindowMetrics:
    """Test suite for window-level anomaly classification metrics."""

    def test_perfect_predictions(self) -> None:
        """Verify that perfect probability predictions achieve F1=1.0, PR-AUC=1.0, ROC-AUC=1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
        y_scores = np.array([0.05, 0.1, 0.0, 0.95, 0.99, 0.2, 0.9, 0.15])

        metrics = compute_window_metrics(y_true, y_scores, threshold=0.5)

        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["pr_auc"] == pytest.approx(1.0)
        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)

    def test_all_zeros_predictions(self) -> None:
        """Verify that all-zero predictions produce zero precision/recall without raising exceptions."""
        y_true = np.array([0, 0, 1, 1, 0])
        y_scores = np.zeros(5)

        metrics = compute_window_metrics(y_true, y_scores, threshold=0.5)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_pr_auc_range(self) -> None:
        """Verify that PR-AUC is bounded within [0.0, 1.0]."""
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], size=100, p=[0.8, 0.2])
        y_scores = rng.uniform(0.0, 1.0, size=100)

        metrics = compute_window_metrics(y_true, y_scores)
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_f1_score(self) -> None:
        """Verify exact calculation of F1 score at custom threshold."""
        # 2 TP, 1 FP, 1 FN, 2 TN -> P = 2/3, R = 2/3, F1 = 2/3 = 0.6667
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.8, 0.7, 0.3, 0.9, 0.1, 0.2])

        metrics = compute_window_metrics(y_true, y_scores, threshold=0.5)
        assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["recall"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["f1"] == pytest.approx(2 / 3, abs=1e-4)


class TestStepMetrics:
    """Test suite for step-level point-wise metrics."""

    def test_step_metrics_shape(self) -> None:
        """Verify step metrics dictionary returns standard performance keys."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.2, 0.1, 0.85, 0.05])

        metrics = compute_step_metrics(y_true, y_scores, threshold=0.5)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert metrics["f1"] == pytest.approx(1.0)

    def test_step_pr_auc(self) -> None:
        """Verify precision-recall AUC is calculated accurately on point-wise steps."""
        y_true = np.array([0, 0, 0, 1, 1, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.4])

        metrics = compute_step_metrics(y_true, y_scores)
        assert metrics["pr_auc"] == pytest.approx(1.0)


class TestThresholdTuning:
    """Test suite for validation-only threshold optimization."""

    def test_threshold_on_validation_only(self) -> None:
        """Verify threshold tuned on validation yields high performance on matching test distribution."""
        # Validation set with clear separation at 0.6
        y_val = np.array([0, 0, 0, 1, 1, 1])
        scores_val = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        best_threshold = tune_threshold(y_val, scores_val, metric="f1")

        # Threshold should be cleanly between 0.3 and 0.7
        assert 0.3 < best_threshold < 0.7

        # Apply threshold to test set without test tuning
        y_test = np.array([0, 0, 1, 1])
        scores_test = np.array([0.15, 0.25, 0.75, 0.85])
        cm = confusion_matrix_at_threshold(y_test, scores_test, threshold=best_threshold)

        # 2 TN, 0 FP, 0 FN, 2 TP
        assert cm[0, 0] == 2  # TN
        assert cm[0, 1] == 0  # FP
        assert cm[1, 0] == 0  # FN
        assert cm[1, 1] == 2  # TP

    def test_threshold_in_01_range(self) -> None:
        """Verify tuned threshold lies strictly between 0 and 1."""
        y_val = np.array([0, 1, 0, 1, 0, 1])
        scores_val = np.array([0.2, 0.4, 0.3, 0.6, 0.5, 0.8])

        th = tune_threshold(y_val, scores_val, metric="f1")
        assert 0.0 < th < 1.0

    def test_threshold_search_adapts_to_probabilities_below_one_percent(self) -> None:
        """Calibration must not default to 0.5 when useful scores are all below 0.01."""
        y_val = np.array([0, 0, 0, 1, 1, 1])
        scores_val = np.array([0.0001, 0.0002, 0.0003, 0.0007, 0.0008, 0.0009])

        th = tune_threshold(y_val, scores_val, metric="f1")

        assert 0.0003 < th < 0.0007

    def test_edge_case_no_positives(self) -> None:
        """Verify that validation data with zero positive anomalies defaults gracefully."""
        y_val = np.zeros(20, dtype=int)
        scores_val = np.linspace(0.1, 0.9, 20)

        th = tune_threshold(y_val, scores_val, metric="f1")
        assert 0.0 < th < 1.0


class TestLocalization:
    """Test suite for event-level evaluation, IoU, and point adjustment."""

    def test_iou_perfect_match(self) -> None:
        """Verify IoU is 1.0 when predicted interval matches ground truth interval exactly."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0])

        scores = iou_per_anomaly(y_true, y_pred)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(1.0)
        assert mean_iou(y_true, y_pred) == pytest.approx(1.0)

    def test_iou_no_overlap(self) -> None:
        """Verify IoU is 0.0 when predicted interval does not overlap ground truth."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1])

        scores = iou_per_anomaly(y_true, y_pred)
        assert len(scores) == 1
        assert scores[0] == 0.0
        assert mean_iou(y_true, y_pred) == 0.0

    def test_mean_iou(self) -> None:
        """Verify mean IoU across multiple discrete anomaly episodes."""
        # Event 1: true [2..4], pred [3..4] -> intersection 2, union 3 -> IoU = 2/3
        # Event 2: true [7..8], pred [7..8] -> intersection 2, union 2 -> IoU = 1.0
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 0])

        scores = iou_per_anomaly(y_true, y_pred)
        assert len(scores) == 2
        assert scores[0] == pytest.approx(2 / 3, abs=1e-3)
        assert scores[1] == pytest.approx(1.0, abs=1e-3)
        assert mean_iou(y_true, y_pred) == pytest.approx((2 / 3 + 1.0) / 2, abs=1e-3)

    def test_point_adjustment(self) -> None:
        """Verify point adjustment inflates sparse single-point hits to full anomaly duration."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 0])  # Only 1 step detected inside segment

        y_adj = point_adjustment(y_true, y_pred)
        assert np.array_equal(y_adj, y_true)  # All 3 steps adjusted to 1

    def test_point_adjustment_flaw_warning(self) -> None:
        """Demonstrate known flaw in Point-Adjustment: random guessing gets artificially inflated F1.

        Kim et al. (2022) 'Towards a Rigorous Evaluation of Time-Series Anomaly Detection'
        showed that sparse random positive guesses hitting a long anomaly segment result in
        near-perfect Point-Adjusted F1, masking severe real-world detection failure.
        """
        N = 1000
        y_true = np.zeros(N, dtype=int)
        # One long anomaly episode of 200 steps
        y_true[400:600] = 1

        # Deterministic sparse predictor: only predict positive at step 500 (inside anomaly)
        # and step 100 (outside, a false positive). This demonstrates the PA flaw reliably.
        y_pred = np.zeros(N, dtype=int)
        y_pred[100] = 1   # FP outside anomaly
        y_pred[500] = 1   # Single TP inside the long anomaly window

        # Standard point-wise F1: precision=0.5 (1 TP, 1 FP), recall=1/200 (very low)
        raw_p = np.sum((y_pred == 1) & (y_true == 1)) / max(1, np.sum(y_pred == 1))
        raw_r = np.sum((y_pred == 1) & (y_true == 1)) / max(1, np.sum(y_true == 1))
        raw_f1 = (2 * raw_p * raw_r) / max(1e-8, raw_p + raw_r)

        # After Point Adjustment, one hit at step 500 inflates recall to 200/200 = 1.0
        y_adj = point_adjustment(y_true, y_pred)
        adj_p = np.sum((y_adj == 1) & (y_true == 1)) / max(1, np.sum(y_adj == 1))
        adj_r = np.sum((y_adj == 1) & (y_true == 1)) / max(1, np.sum(y_true == 1))
        adj_f1 = (2 * adj_p * adj_r) / max(1e-8, adj_p + adj_r)

        # PA-F1 is drastically higher than unadjusted F1 — this demonstrates the known flaw
        assert adj_f1 > 0.80   # PA inflates score significantly
        assert raw_f1 < 0.20   # True detection performance is much lower

    def test_event_f1(self) -> None:
        """Verify event-level F1 calculation across multiple discrete segments."""
        y_true = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 0])

        f1 = event_f1(y_true, y_pred)
        assert f1 == pytest.approx(1.0)


class TestFormatMetricsTable:
    """Test suite for metrics reporting and table formatting."""

    def test_format_metrics_table(self) -> None:
        """Verify formatted ASCII table contains headers and numeric entries."""
        metrics = {"F1 Score": 0.8754, "Precision": 0.9123, "Recall": 0.8412}
        table = format_metrics_table(metrics)

        assert "Metric" in table
        assert "Value" in table
        assert "0.8754" in table
        assert "0.9123" in table
