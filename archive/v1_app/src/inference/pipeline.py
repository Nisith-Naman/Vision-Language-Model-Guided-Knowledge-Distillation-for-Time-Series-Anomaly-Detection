"""CSV feature preparation and memory-bounded student inference."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import fill_missing, transform_scaler


def prepare_inference_features(
    dataframe: pd.DataFrame,
    feature_names: Sequence[str],
    scaler: Any = None,
    missing_strategy: str = "interpolate",
    imputation_stats: Optional[Dict[str, Any]] = None,
    missing_codes: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select, clean, impute, and scale the features expected by a checkpoint.

    Extra input columns are allowed and preserved by the caller. Required model
    features are selected in the checkpoint's stored order.
    """
    required = list(feature_names)
    missing = [name for name in required if name not in dataframe.columns]
    if missing:
        raise ValueError(f"CSV is missing model feature columns: {', '.join(missing)}")
    if not required:
        raise ValueError("The trained model has no feature names.")

    numeric = dataframe.loc[:, required].apply(pd.to_numeric, errors="coerce")
    if missing_codes:
        numeric = numeric.mask(numeric.isin(list(missing_codes)), np.nan)
    raw = numeric.to_numpy(dtype=np.float64, copy=True)
    cleaned, _ = fill_missing(
        raw,
        strategy=missing_strategy,
        train_stats=imputation_stats or None,
    )
    scaled = transform_scaler(cleaned, scaler)

    if not np.isfinite(cleaned).all():
        raise ValueError("Feature imputation left non-finite sensor values.")
    if not np.isfinite(scaled).all():
        raise ValueError("Scaling produced non-finite values; check the checkpoint scaler and CSV ranges.")
    return cleaned, scaled


def run_student_inference(
    model: torch.nn.Module,
    scaled_values: np.ndarray,
    window_length: int,
    stride: int,
    batch_size: int = 64,
    localization: Optional[dict] = None,
) -> Tuple[np.ndarray, float]:
    """Score a sequence in bounded batches and average overlapping step scores.

    When ``localization`` (saved with v2 checkpoints) is given, the averaged
    step score is multiplied by the window anomaly probability raised to
    ``alpha`` and smoothed with a centred moving average of length ``k``;
    these are the validation-tuned settings the checkpoint threshold refers to.
    """
    values = np.asarray(scaled_values, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("Inference features must be a non-empty (timesteps, features) matrix.")
    if window_length <= 0 or stride <= 0:
        raise ValueError("Window length and stride must both be positive.")
    if batch_size <= 0:
        raise ValueError("Inference batch size must be positive.")

    total_steps = len(values)
    if total_steps < window_length:
        padding = np.repeat(values[-1:, :], window_length - total_steps, axis=0)
        model_values = np.concatenate([values, padding], axis=0)
        starts = np.asarray([0], dtype=np.int64)
    else:
        model_values = values
        starts = np.arange(0, total_steps - window_length + 1, stride, dtype=np.int64)
        if starts[-1] + window_length < total_steps:
            starts = np.append(starts, total_steps - window_length)

    score_sums = np.zeros(total_steps, dtype=np.float64)
    score_counts = np.zeros(total_steps, dtype=np.int32)
    window_sums = np.zeros(total_steps, dtype=np.float64)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            batch = np.stack(
                [model_values[start : start + window_length] for start in batch_starts],
                axis=0,
            )
            batch_tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
            if hasattr(model, "predict_proba"):
                window_proba, step_scores = model.predict_proba(batch_tensor)
                window_p = np.asarray(window_proba)[:, 1]
            else:
                window_logits, step_logits = model(batch_tensor)
                step_scores = torch.sigmoid(step_logits).detach().cpu().numpy()
                window_p = torch.softmax(window_logits, dim=-1)[:, 1].detach().cpu().numpy()
            step_scores = np.asarray(
                step_scores.detach().cpu().numpy() if torch.is_tensor(step_scores) else step_scores,
                dtype=np.float64,
            )
            if step_scores.shape != (len(batch_starts), window_length):
                raise ValueError(
                    "Student step head returned shape "
                    f"{step_scores.shape}; expected {(len(batch_starts), window_length)}."
                )
            for local_idx, start in enumerate(batch_starts):
                end = min(int(start) + window_length, total_steps)
                span = end - int(start)
                score_sums[start:end] += step_scores[local_idx, :span]
                score_counts[start:end] += 1
                window_sums[start:end] += float(window_p[local_idx])

    elapsed = time.perf_counter() - started
    if np.any(score_counts == 0):
        raise RuntimeError("Inference window coverage left one or more timesteps unscored.")
    scores = score_sums / score_counts
    if localization:
        from src.evaluation.postprocess import smooth

        alpha = float(localization.get("alpha", 0.0))
        if alpha > 0:
            scores = scores * np.clip(window_sums / score_counts, 0.0, 1.0) ** alpha
        scores = smooth(scores, int(localization.get("k", 1)))
    return scores.astype(np.float32), elapsed
