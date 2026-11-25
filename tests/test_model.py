"""Unit tests for student RNN model architecture, configuration, and KD losses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.student.losses import (
    KnowledgeDistillationLoss,
    LocalizationLoss,
    TotalLoss,
    WindowClassificationLoss,
)
from src.student.model import GRUStudent, ModelConfig


class TestGRUStudent:
    """Test suite for GRUStudent architecture, tensor shapes, and inference APIs."""

    def test_output_shapes_univariate(self) -> None:
        """Verify output tensor shapes on univariate input (input_size=1)."""
        model = GRUStudent(
            input_size=1, hidden_size=32, num_layers=1, dropout=0.0, seq_len=64
        )
        x = torch.randn(4, 64, 1)  # (batch=4, L=64, F=1)
        wl, sl = model(x)

        assert wl.shape == (4, 2)
        assert sl.shape == (4, 64)

    def test_output_shapes_multivariate(self) -> None:
        """Verify output tensor shapes on multivariate input (input_size=29)."""
        model = GRUStudent(
            input_size=29, hidden_size=64, num_layers=2, dropout=0.1, seq_len=128
        )
        x = torch.randn(8, 128, 29)
        wl, sl = model(x)

        assert wl.shape == (8, 2)
        assert sl.shape == (8, 128)

    def test_predict_proba_sums_to_one(self) -> None:
        """Verify that predict_proba window probabilities sum to 1.0 per sample."""
        model = GRUStudent(
            input_size=4, hidden_size=32, num_layers=1, seq_len=32, dropout=0.0
        )
        x = torch.randn(5, 32, 4)
        window_proba, step_scores = model.predict_proba(x)

        assert window_proba.shape == (5, 2)
        np.testing.assert_allclose(
            window_proba.sum(axis=1), np.ones(5, dtype=np.float32), atol=1e-5
        )

    def test_step_scores_in_01_range(self) -> None:
        """Verify that per-step anomaly scores lie strictly within [0.0, 1.0]."""
        model = GRUStudent(input_size=3, hidden_size=16, num_layers=1, seq_len=40)
        x = torch.randn(3, 40, 3)
        _, step_scores = model.predict_proba(x)

        assert step_scores.shape == (3, 40)
        assert (step_scores >= 0.0).all()
        assert (step_scores <= 1.0).all()

    def test_parameter_count(self) -> None:
        """Verify parameter counting utility returns valid non-zero integer."""
        model = GRUStudent(input_size=5, hidden_size=32, num_layers=2, seq_len=64)
        param_count = model.parameter_count()

        assert isinstance(param_count, int)
        assert param_count > 0

    def test_different_input_sizes_different_models(self) -> None:
        """Verify that models with different feature dimensions instantiate differing parameter counts."""
        m1 = GRUStudent(input_size=3, hidden_size=32, num_layers=1, seq_len=64)
        m2 = GRUStudent(input_size=12, hidden_size=32, num_layers=1, seq_len=64)

        assert m1.parameter_count() < m2.parameter_count()

    def test_bidirectional_model(self) -> None:
        """Verify bidirectional GRU generates valid window and step representations."""
        model = GRUStudent(
            input_size=6,
            hidden_size=32,
            num_layers=1,
            seq_len=50,
            bidirectional=True,
        )
        x = torch.randn(2, 50, 6)
        wl, sl = model(x)

        assert wl.shape == (2, 2)
        assert sl.shape == (2, 50)

    def test_input_size_mismatch_raises_error(self) -> None:
        """Verify ValueError is raised if input tensor feature dimension does not match input_size."""
        model = GRUStudent(input_size=4, hidden_size=16, seq_len=20)
        invalid_x = torch.randn(2, 20, 8)  # Expected 4, got 8
        with pytest.raises(ValueError, match="Expected input_size=4"):
            model(invalid_x)


class TestLosses:
    """Test suite for localization, classification, and distillation loss objectives."""

    def test_kd_loss_zero_when_distributions_equal(self) -> None:
        """Verify KD loss is approximately zero when student matches teacher distribution."""
        T = 2.0
        kd = KnowledgeDistillationLoss(temperature=T)
        logits = torch.tensor([[2.5, -2.5], [-1.2, 1.2]], requires_grad=True)
        # Exactly matching teacher distribution
        # The loss applies T to both sides; provide the teacher's base probabilities.
        teacher_probs = torch.softmax(logits, dim=-1)

        loss = kd(logits, teacher_probs)
        assert loss.item() < 1e-4

    def test_localization_loss_shape(self) -> None:
        """Verify weighted BCE localization loss produces positive scalar."""
        loc_loss_fn = LocalizationLoss(pos_weight=8.0)
        step_logits = torch.randn(4, 64)
        y_step = torch.zeros(4, 64)
        y_step[0, 10:15] = 1.0  # anomalous span

        loss = loc_loss_fn(step_logits, y_step)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_window_classification_loss(self) -> None:
        """Verify categorical cross entropy produces positive scalar."""
        win_fn = WindowClassificationLoss()
        logits = torch.randn(6, 2)
        y_win = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)

        loss = win_fn(logits, y_win)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_total_loss_none_teacher(self) -> None:
        """Verify TotalLoss runs smoothly when teacher_probs is None."""
        total_loss_fn = TotalLoss(beta=1.0, lambda_kd=0.5, pos_weight=5.0)
        step_logits = torch.randn(4, 32)
        window_logits = torch.randn(4, 2)
        y_step = torch.zeros(4, 32)
        y_win = torch.tensor([0, 1, 0, 0], dtype=torch.long)

        loss, metrics = total_loss_fn(
            step_logits=step_logits,
            window_logits=window_logits,
            y_step=y_step,
            y_window=y_win,
            teacher_probs=None,
            return_dict=True,
        )

        assert loss.ndim == 0
        assert loss.item() > 0.0
        assert metrics["kd_loss"] == 0.0
        assert metrics["loc_loss"] > 0.0
        assert metrics["win_loss"] > 0.0

    def test_no_gradient_through_teacher(self) -> None:
        """Verify teacher_probs tensor is detached and receives no backpropagation gradient."""
        kd_fn = KnowledgeDistillationLoss(temperature=2.0)
        student_logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
        teacher_probs = torch.tensor([[0.7, 0.3]], requires_grad=True)

        loss = kd_fn(student_logits, teacher_probs)
        loss.backward()

        assert teacher_probs.grad is None
        assert student_logits.grad is not None


class TestModelConfig:
    """Test suite for configuration serialization and reproducibility."""

    def test_serialize_deserialize(self) -> None:
        """Verify to_dict and from_dict roundtrip preserves all hyperparameters."""
        cfg = ModelConfig(
            input_size=15,
            hidden_size=128,
            num_layers=3,
            dropout=0.25,
            seq_len=64,
            bidirectional=True,
        )
        data = cfg.to_dict()
        reconstructed = ModelConfig.from_dict(data)

        assert cfg == reconstructed
        assert reconstructed.input_size == 15
        assert reconstructed.hidden_size == 128
        assert reconstructed.num_layers == 3
        assert reconstructed.dropout == 0.25
        assert reconstructed.seq_len == 64
        assert reconstructed.bidirectional is True
