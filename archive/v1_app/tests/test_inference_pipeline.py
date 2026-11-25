"""Tests for reusable CSV inference and point/group result exports."""

from __future__ import annotations

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.pipeline import prepare_inference_features, run_student_inference
from src.student.model import GRUStudent
from src.training.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from src.ui.page6_evaluation import _run_student_inference
from src.ui.page7_inference import _read_uploaded_telemetry_csv, _timestamps_for_csv
from src.visualization.report import predictions_to_csv


def test_prepare_inference_features_reorders_sensors_and_ignores_extra_columns() -> None:
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="min"),
        "sensor_b": [1.0, -200.0, 3.0, 4.0],
        "label": [0, 0, 1, 0],
        "sensor_a": [10.0, 20.0, 30.0, 40.0],
    })

    raw, scaled = prepare_inference_features(
        dataframe=frame,
        feature_names=["sensor_a", "sensor_b"],
        missing_strategy="ffill",
        imputation_stats={"col_means": [25.0, 2.0]},
        missing_codes=[-200],
    )

    assert raw.shape == (4, 2)
    np.testing.assert_allclose(raw[:, 0], [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(raw[:, 1], [1.0, 1.0, 3.0, 4.0])
    np.testing.assert_allclose(scaled, raw)


def test_prepare_inference_features_reports_missing_required_column() -> None:
    with pytest.raises(ValueError, match="missing model feature columns: sensor_b"):
        prepare_inference_features(pd.DataFrame({"sensor_a": [1.0]}), ["sensor_a", "sensor_b"])


def test_uploaded_telemetry_csv_detects_semicolon_delimiter() -> None:
    frame = _read_uploaded_telemetry_csv(
        io.StringIO("Date;Time;CO(GT);T\n10/03/2004;18.00.00;2,6;13,6\n")
    )

    assert frame.columns.tolist() == ["Date", "Time", "CO(GT)", "T"]
    assert frame.iloc[0]["CO(GT)"] == pytest.approx(2.6)
    assert frame.iloc[0]["T"] == pytest.approx(13.6)
    timestamps = _timestamps_for_csv(
        frame,
        "Date + Time",
        "Date + Time",
        "Date",
        "Time",
    )
    assert str(timestamps[0]).startswith("2004-03-10T18:00:00")


@pytest.mark.parametrize("n_steps", [3, 11])
def test_student_inference_scores_short_and_tail_sequences(n_steps: int) -> None:
    model = GRUStudent(input_size=2, hidden_size=8, num_layers=1, dropout=0.0, seq_len=8)
    values = np.random.default_rng(7).normal(size=(n_steps, 2)).astype(np.float32)

    scores, elapsed = run_student_inference(model, values, window_length=8, stride=4, batch_size=2)

    assert scores.shape == (n_steps,)
    assert np.isfinite(scores).all()
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert elapsed >= 0.0


def test_evaluation_inference_supports_forward_only_model() -> None:
    class ForwardOnlyStudent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch_size, seq_len, _ = x.shape
            window_logits = self.bias.expand(batch_size, 2)
            step_logits = self.bias.expand(batch_size, seq_len)
            return window_logits, step_logits

    values = np.ones((5, 8, 2), dtype=np.float32)
    window_probs, step_scores, elapsed = _run_student_inference(ForwardOnlyStudent(), values)

    assert window_probs.shape == (5, 2)
    assert step_scores.shape == (5, 8)
    np.testing.assert_allclose(window_probs, 0.5)
    np.testing.assert_allclose(step_scores, 0.5)
    assert elapsed >= 0.0


def test_restored_checkpoint_uses_canonical_student_model(tmp_path: Path) -> None:
    model = GRUStudent(input_size=2, hidden_size=8, num_layers=1, dropout=0.0, seq_len=8)
    metadata = CheckpointMetadata(
        model_config=model.get_config().to_dict(),
        feature_names=["sensor_a", "sensor_b"],
        scaler_type="standard",
        window_length=8,
        stride=4,
        threshold=0.5,
        normal_label="0",
        anomaly_label="1",
        training_mode="student_kd",
        dataset_name="restore_smoke_test",
        training_result={},
        created_at="2026-11-03T00:00:00",
    )
    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(model, None, metadata, str(checkpoint_dir))

    restored, _, _ = load_checkpoint(str(checkpoint_dir))

    assert isinstance(restored, GRUStudent)
    assert callable(getattr(restored, "predict_proba", None))


def test_predictions_csv_preserves_input_and_labels_point_vs_group_events() -> None:
    timestamps = pd.date_range("2026-01-01", periods=6, freq="min")
    original = pd.DataFrame({
        "time": timestamps,
        "sensor": np.arange(6, dtype=float),
        "operator_note": ["a", "b", "c", "d", "e", "f"],
    })
    predictions = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)

    output = pd.read_csv(
        __import__("io").StringIO(
            predictions_to_csv(
                timestamps=timestamps,
                scores=np.asarray([0.1, 0.9, 0.2, 0.8, 0.95, 0.1]),
                predictions=predictions,
                original_df=original,
            )
        )
    )

    assert output["operator_note"].tolist() == original["operator_note"].tolist()
    assert output["time"].tolist() == original["time"].astype(str).tolist()
    assert output["is_anomaly"].tolist() == predictions.tolist()
    assert output["anomaly_type"].tolist() == [
        "normal", "point_anomaly", "normal", "group_anomaly", "group_anomaly", "normal"
    ]
    assert pd.isna(output.loc[0, "anomaly_event_id"])
    assert output.loc[1, "anomaly_event_id"] == 1
    assert output.loc[3, "anomaly_event_id"] == output.loc[4, "anomaly_event_id"] == 2
    assert output.loc[3, "event_duration_steps"] == output.loc[4, "event_duration_steps"] == 2
    assert "anomaly_score" in output
    assert "timestamp" in output


def test_checkpoint_metadata_round_trips_inference_preprocessing() -> None:
    metadata = CheckpointMetadata(
        model_config={"input_size": 2},
        feature_names=["sensor_a", "sensor_b"],
        scaler_type="standard",
        window_length=8,
        stride=4,
        threshold=0.37,
        normal_label="0",
        anomaly_label="1",
        training_mode="student_kd",
        dataset_name="example",
        training_result={},
        created_at="2026-11-02T00:00:00",
        missing_strategy="mean",
        imputation_stats={"col_means": [2.0, 5.0]},
        time_column="timestamp",
        missing_codes=[-200],
    )

    restored = CheckpointMetadata.from_dict(json.loads(json.dumps(metadata.to_dict())))

    assert restored.missing_strategy == "mean"
    assert restored.imputation_stats == {"col_means": [2.0, 5.0]}
    assert restored.time_column == "timestamp"
    assert restored.missing_codes == [-200]
    assert restored.threshold == pytest.approx(0.37)
