"""
Checkpoint management for the IE643 VLM Teacher-RNN Student system.

Handles saving, loading, listing, and schema-validating model artifacts,
scalers, training metrics, and experiment metadata.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import joblib
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

# Always restore the same implementation used for training. A duplicate fallback
# GRU can load some state dictionaries while silently omitting convenience methods
# (such as predict_proba), which makes restored models behave differently.
from src.student.model import GRUStudent


@dataclass
class CheckpointMetadata:
    """Metadata describing a saved experiment checkpoint."""

    model_config: dict
    feature_names: list[str]
    scaler_type: str
    window_length: int
    stride: int
    threshold: float
    normal_label: Any
    anomaly_label: Any
    training_mode: str  # 'teacher_only' | 'student_baseline' | 'student_kd'
    dataset_name: str
    training_result: dict
    created_at: str
    missing_strategy: str = "interpolate"
    imputation_stats: dict[str, Any] = field(default_factory=dict)
    time_column: Optional[str] = None
    missing_codes: list[Any] = field(default_factory=list)
    contextual_threshold_robust_z: float = 4.5
    contextual_calibration_normal_rows: int = 0

    def to_dict(self) -> dict:
        """Converts metadata to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointMetadata:
        """Instantiates CheckpointMetadata from a dictionary."""
        return cls(
            model_config=data.get("model_config", {}),
            feature_names=data.get("feature_names", []),
            scaler_type=str(data.get("scaler_type", "standard")),
            window_length=int(data.get("window_length", 128)),
            stride=int(data.get("stride", 32)),
            threshold=float(data.get("threshold", 0.5)),
            normal_label=data.get("normal_label", 0),
            anomaly_label=data.get("anomaly_label", 1),
            training_mode=str(data.get("training_mode", "student_kd")),
            dataset_name=str(data.get("dataset_name", "unknown")),
            training_result=data.get("training_result", {}),
            created_at=str(data.get("created_at", datetime.now().isoformat())),
            missing_strategy=str(data.get("missing_strategy", "interpolate")),
            imputation_stats=data.get("imputation_stats", {}),
            time_column=data.get("time_column"),
            missing_codes=data.get("missing_codes", []),
            contextual_threshold_robust_z=float(
                data.get("contextual_threshold_robust_z", 4.5)
            ),
            contextual_calibration_normal_rows=int(
                data.get("contextual_calibration_normal_rows", 0)
            ),
        )


def save_checkpoint(
    model: Optional[nn.Module],
    scaler: Any,
    metadata: CheckpointMetadata,
    save_dir: str,
) -> str:
    """
    Saves a checkpoint consisting of model.pt, scaler.joblib, and metadata.json.

    Args:
        model: Trained PyTorch student model (or None for teacher_only mode).
        scaler: Fitted preprocessing scaler (e.g. StandardScaler).
        metadata: CheckpointMetadata object.
        save_dir: Directory where the checkpoint should be stored.

    Returns:
        Absolute or resolved path to the checkpoint directory as a string.
    """
    target_dir = Path(save_dir)

    # If save_dir is a top-level directory like './checkpoints', create a subfolder
    if target_dir.name in ("checkpoints", ".", ""):
        timestamp_clean = metadata.created_at.replace(":", "-").replace(" ", "_").replace(".", "_")
        folder_name = f"{metadata.dataset_name}_{metadata.training_mode}_{timestamp_clean}"
        target_dir = target_dir / folder_name

    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model weights
    model_path = target_dir / "model.pt"
    if model is not None:
        torch.save(model.state_dict(), model_path)
    else:
        # In teacher_only mode, save empty state
        torch.save({}, model_path)

    # 2. Save scaler
    scaler_path = target_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # 3. Save metadata JSON
    metadata_path = target_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    logger.info("Saved checkpoint successfully to %s", target_dir)
    return str(target_dir.resolve())


def load_checkpoint(checkpoint_path: str) -> tuple[Optional[nn.Module], Any, CheckpointMetadata]:
    """
    Loads checkpoint files and reconstructs the GRUStudent model and scaler.

    Args:
        checkpoint_path: Path to checkpoint directory or metadata.json file.

    Returns:
        Tuple of (model, scaler, metadata)
    """
    path_obj = Path(checkpoint_path)
    ckpt_dir = path_obj.parent if path_obj.is_file() else path_obj

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    # 1. Load metadata
    metadata_file = ckpt_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing metadata.json in checkpoint directory: {ckpt_dir}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    metadata = CheckpointMetadata.from_dict(meta_dict)

    # 2. Load scaler
    scaler_file = ckpt_dir / "scaler.joblib"
    scaler = None
    if scaler_file.exists():
        try:
            scaler = joblib.load(scaler_file)
        except Exception as e:
            logger.warning("Could not load scaler from %s: %s", scaler_file, e)

    # 3. Reconstruct model
    model = None
    model_file = ckpt_dir / "model.pt"

    if metadata.training_mode != "teacher_only" and model_file.exists():
        try:
            cfg = dict(metadata.model_config)
            # Ensure input_dim is set properly from feature names if missing
            if "input_dim" not in cfg and "input_size" not in cfg:
                cfg["input_dim"] = len(metadata.feature_names)

            model = GRUStudent(**cfg)
            state_dict = torch.load(model_file, map_location="cpu")
            if state_dict:
                model.load_state_dict(state_dict)
            model.eval()
            logger.info("Reconstructed and loaded GRUStudent weights from %s", model_file)
        except Exception as e:
            logger.error("Failed to reconstruct model from %s: %s", model_file, e)
            raise

    return model, scaler, metadata


def list_checkpoints(checkpoint_dir: str) -> list[dict]:
    """
    Scans a directory for saved checkpoints and returns their metadata for UI display.

    Args:
        checkpoint_dir: Base directory to search.

    Returns:
        List of dictionaries with checkpoint metadata, sorted newest first.
    """
    base_dir = Path(checkpoint_dir)
    if not base_dir.exists():
        return []

    checkpoints: list[dict] = []
    for meta_path in base_dir.rglob("metadata.json"):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["checkpoint_dir"] = str(meta_path.parent.resolve())
            meta["checkpoint_path"] = str(meta_path.resolve())
            checkpoints.append(meta)
        except Exception as err:
            logger.warning("Failed to parse metadata at %s: %s", meta_path, err)

    # Sort descending by created_at timestamp
    checkpoints.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return checkpoints


def validate_input_compatibility(
    metadata: CheckpointMetadata,
    feature_names: list[str],
) -> list[str]:
    """
    Validates that incoming data features match the expected features of a checkpoint.

    Args:
        metadata: CheckpointMetadata loaded from checkpoint.
        feature_names: Ordered list of feature names from the new input dataset.

    Returns:
        List of validation error strings. If empty, the input is fully compatible.
    """
    errors: list[str] = []
    expected = metadata.feature_names

    if not feature_names:
        errors.append("Incoming feature list is empty.")
        return errors

    if len(feature_names) != len(expected):
        errors.append(
            f"Feature count mismatch: Checkpoint expects {len(expected)} features, "
            f"but provided data has {len(feature_names)} features."
        )

    missing = [f for f in expected if f not in feature_names]
    if missing:
        errors.append(f"Missing required features ({len(missing)}): {', '.join(missing)}")

    unexpected = [f for f in feature_names if f not in expected]
    if unexpected:
        errors.append(f"Encountered unexpected extra features ({len(unexpected)}): {', '.join(unexpected)}")

    # Check for order mismatches if sets of features match
    if not missing and not unexpected and feature_names != expected:
        errors.append(
            f"Feature order mismatch: Checkpoint expects column order {expected}, "
            f"but received {feature_names}. Please reorder columns."
        )

    return errors
