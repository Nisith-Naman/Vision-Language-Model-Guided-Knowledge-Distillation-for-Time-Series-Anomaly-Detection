"""Unit tests for dataset loaders, CSV parsers, schema validation, and Series containers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import (
    DataQualityReport,
    Series,
    get_dataset_info,
    inspect_dataset,
    load_csv,
)
from src.data.datasets.oil3w import load as load_oil3w
from scripts.download_3w import inspect_existing_dataset

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLoadCSV:
    """Test suite for loading and parsing tabular CSV files into standardized Series."""

    def test_loads_univariate(self) -> None:
        """Verify loading of a univariate time-series fixture with anomaly labels."""
        s = load_csv(
            FIXTURES_DIR / "tiny_univariate.csv",
            time_col="timestamp",
            value_cols=["value"],
            label_col="label",
        )
        assert isinstance(s, Series)
        assert s.n_timesteps == 200
        assert s.n_features == 1
        assert s.y_step is not None
        assert s.y_step.sum() == 22  # 11 steps at rows 80-90 + 11 steps at rows 150-160
        assert s.label_rate == pytest.approx(22 / 200, abs=1e-3)
        assert s.has_ground_truth is True
        assert len(s.timestamps) == 200

    def test_loads_multivariate(self) -> None:
        """Verify loading of a 3-feature multivariate time-series fixture."""
        s = load_csv(
            FIXTURES_DIR / "tiny_multivariate.csv",
            time_col="timestamp",
            value_cols=["feat_1", "feat_2", "feat_3"],
            label_col="label",
        )
        assert isinstance(s, Series)
        assert s.n_timesteps == 200
        assert s.n_features == 3
        assert s.y_step is not None
        assert s.y_step.sum() == 11  # rows 90-100 inclusive
        assert s.feature_names == ["feat_1", "feat_2", "feat_3"]

    def test_loads_no_labels(self) -> None:
        """Verify loading of time-series fixture lacking ground truth labels."""
        s = load_csv(
            FIXTURES_DIR / "tiny_no_labels.csv",
            time_col="timestamp",
            value_cols=["feat_1", "feat_2"],
            label_col=None,
        )
        assert isinstance(s, Series)
        assert s.n_timesteps == 200
        assert s.n_features == 2
        assert s.y_step is None
        assert s.has_ground_truth is False
        assert s.label_rate == 0.0

    def test_missing_file_raises(self) -> None:
        """Verify FileNotFoundError is raised when file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent_path_xyz.csv", value_cols=["a"])

    def test_missing_column_raises(self) -> None:
        """Verify ValueError is raised when requested feature column is not in CSV."""
        with pytest.raises(ValueError, match="Missing required value column"):
            load_csv(
                FIXTURES_DIR / "tiny_univariate.csv",
                value_cols=["nonexistent_col"],
            )

    def test_missing_time_col_raises(self) -> None:
        """Verify ValueError is raised when requested timestamp column is absent."""
        with pytest.raises(ValueError, match="Time column 'bogus_time' not found"):
            load_csv(
                FIXTURES_DIR / "tiny_univariate.csv",
                time_col="bogus_time",
                value_cols=["value"],
            )


class TestSeries:
    """Test suite for Series data container attributes, validation, and properties."""

    def test_series_properties(self) -> None:
        """Verify core container properties on synthetic zero-filled inputs."""
        s = Series(
            values=np.zeros((100, 3)),
            y_step=np.zeros(100),
            group_id="test_series_01",
        )
        assert s.n_timesteps == 100
        assert s.n_features == 3
        assert s.has_ground_truth is True
        assert s.label_rate == 0.0
        assert s.group_id == "test_series_01"
        assert repr(s).startswith("Series(")

    def test_series_1d_values_reshaped(self) -> None:
        """Verify 1D values array of shape (T,) is automatically reshaped to (T, 1)."""
        s = Series(values=np.ones(50), y_step=np.zeros(50))
        assert s.values.shape == (50, 1)
        assert s.n_features == 1

    def test_length_mismatch_raises(self) -> None:
        """Verify ValueError is raised when values and y_step lengths disagree."""
        with pytest.raises(ValueError, match="Length mismatch"):
            Series(values=np.zeros((100, 2)), y_step=np.zeros(50))

    def test_timestamp_length_mismatch_raises(self) -> None:
        """Verify ValueError is raised when timestamps array length does not match values."""
        with pytest.raises(ValueError, match="Length mismatch"):
            Series(
                values=np.zeros((30, 1)),
                y_step=np.zeros(30),
                timestamps=np.arange(10),
            )


class TestDataQualityReport:
    """Test suite for DataQualityReport metrics, summary serialization, and formatting."""

    def test_report_generation(self) -> None:
        """Verify DataQualityReport holds complete profiling fields and summary dictionary."""
        report = inspect_dataset(FIXTURES_DIR / "tiny_univariate.csv")
        assert isinstance(report, DataQualityReport)
        assert report.total_rows == 200
        assert report.total_columns == 3
        assert "value" in report.missing_values_per_col
        assert report.duplicate_timestamps == 0

        summary = report.summary_dict()
        assert "total_rows" in summary
        assert "missing_values_per_col" in summary
        assert summary["total_rows"] == 200

    def test_anomaly_rate_zero_when_no_labels(self) -> None:
        """Verify detected_anomaly_rate is 0.0 when dataset contains no labels."""
        report = inspect_dataset(FIXTURES_DIR / "tiny_no_labels.csv")
        assert report.detected_anomaly_rate == 0.0


class TestInspectDatasetAndCatalog:
    """Test suite for dataset catalog metadata and inspection dispatch."""

    def test_get_dataset_info_known_datasets(self) -> None:
        """Verify dataset catalog returns correct metadata for all supported benchmark datasets."""
        for ds_name in ["daphnet", "airquality", "metropt", "oil3w", "creditcard"]:
            info = get_dataset_info(ds_name)
            assert "name" in info
            assert "domain" in info
            assert "features" in info
            assert "anomaly_type" in info
            assert info["features"] > 0

    def test_get_dataset_info_unknown(self) -> None:
        """Verify get_dataset_info falls back gracefully on unknown datasets."""
        info = get_dataset_info("unknown_xyz")
        assert info["domain"] == "Custom"
        assert info["features"] == -1


def test_3w_download_helper_finds_extracted_nested_class_folders(tmp_path: Path) -> None:
    data_dir = tmp_path / "3w" / "data" / "0"
    data_dir.mkdir(parents=True)
    (data_dir / "normal.csv").write_text("timestamp,class\n2020-01-01,0\n", encoding="utf-8")
    git_dir = tmp_path / "3w" / ".git" / "data" / "1"
    git_dir.mkdir(parents=True)
    (git_dir / "repository-metadata.csv").write_text("not,a,dataset\n", encoding="utf-8")

    assert inspect_existing_dataset(tmp_path / "3w") == {"0": 1}


def test_3w_loader_fills_infinite_sensor_readings(tmp_path: Path) -> None:
    data_dir = tmp_path / "3w"
    data_dir.mkdir()
    rows = {
        "timestamp": ["2020-01-01 00:00:00", "2020-01-01 00:01:00", "2020-01-01 00:02:00"],
        "P-PDG": ["-inf", "2", "1e300"],
        "P-TPT": ["1", "1", "1"],
        "T-TPT": ["1", "1", "1"],
        "P-MON-CKP": ["1", "1", "1"],
        "T-JUS-CKP": ["1", "1", "1"],
        "P-JUS-CKGL": ["1", "1", "1"],
        "T-JUS-CKGL": ["1", "1", "1"],
        "QGL": ["1", "1", "1"],
        "class": ["0", "0", "1"],
    }
    pd.DataFrame(rows).to_csv(data_dir / "instance.csv", index=False)

    [series] = load_oil3w({"data_dir": str(data_dir)})

    assert np.isfinite(series.values).all()
    np.testing.assert_allclose(series.values[:, 0], [2.0, 2.0, 2.0])
