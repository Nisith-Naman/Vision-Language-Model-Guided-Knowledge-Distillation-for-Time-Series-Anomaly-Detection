#!/usr/bin/env python
"""Load local project datasets and smoke-check split/window/student compatibility.

This is an opt-in check for the large datasets in ``data/``. It reports the
sequence and split sizes without materializing every sliding window, then runs one
real-data GRU + distillation-loss optimization step per dataset.
"""

from __future__ import annotations

import gc
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.loader import Series, load_dataset
from src.data.preprocessing import (
    chronological_split,
    fill_missing,
    fit_scaler,
    fit_scaler_batches,
    transform_scaler,
)
from src.data.synthetic_labels import inject_anomalies
from src.student.model import GRUStudent
from src.training.trainer import DistillationLoss
from src.ui.page3_preprocessing import _allocate_window_quotas, _window_start_indices


def _split_series(
    series_list: list[Series],
    dataset_cfg: dict[str, Any],
    global_splits: dict[str, float],
) -> tuple[list[Series], list[Series], list[Series], str]:
    roles = {"train", "val", "test"}
    if all(series.metadata.get("split_role") in roles for series in series_list):
        return (
            [s for s in series_list if s.metadata["split_role"] == "train"],
            [s for s in series_list if s.metadata["split_role"] == "val"],
            [s for s in series_list if s.metadata["split_role"] == "test"],
            "explicit event roles",
        )

    split_cfg = dataset_cfg.get("splits", global_splits)
    train_frac = float(split_cfg.get("train", 0.70))
    val_frac = float(split_cfg.get("val", 0.15))
    strategy = str(dataset_cfg.get("split_by", "chronological"))
    distinct_groups = {s.group_id for s in series_list}
    if len(distinct_groups) >= 3 and strategy in {"subject", "instance", "event", "group"}:
        train, val, test = chronological_split(
            series_list,
            train_frac=train_frac,
            val_frac=val_frac,
            split_by=strategy,
            seed=42,
        )
        return train, val, test, f"group holdout: {strategy}"
    if len(series_list) == 1:
        train, val, test = chronological_split(
            series_list,
            train_frac=train_frac,
            val_frac=val_frac,
            split_by="chronological",
        )
        return train, val, test, "chronological holdout"

    train: list[Series] = []
    val: list[Series] = []
    test: list[Series] = []
    for sequence in series_list:
        tr, va, te = chronological_split(
            [sequence], train_frac=train_frac, val_frac=val_frac, split_by="chronological"
        )
        train.extend(tr)
        val.extend(va)
        test.extend(te)
    return train, val, test, "per-sequence chronological holdout"


def _count_windows(series_list: list[Series], length: int, stride: int) -> tuple[int, int]:
    window_count = positive_count = 0
    for series in series_list:
        n = series.n_timesteps
        if n < length:
            continue
        starts = np.arange(0, n - length + 1, stride, dtype=np.int64)
        if len(starts) and starts[-1] + length < n:
            starts = np.append(starts, n - length)
        labels = series.y_step if series.y_step is not None else np.zeros(n, dtype=np.int8)
        cumulative = np.r_[0, np.cumsum(labels, dtype=np.int64)]
        positive_count += int(np.sum((cumulative[starts + length] - cumulative[starts]) > 0))
        window_count += len(starts)
    return window_count, positive_count


def _smoke_student_step(
    dataset_key: str,
    series: Series,
    window_length: int,
    stride: int,
) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
    values, _ = fill_missing(series.values[: min(series.n_timesteps, 4096)], strategy="interpolate")
    if dataset_key == "airquality" and series.metadata.get("label_strategy") == "semi_synthetic":
        scale = np.nanstd(values, axis=0).astype(np.float32)
        scale = np.where(scale < 1e-6, 1.0, scale)
        values, labels = inject_anomalies(
            values,
            series.timestamps[: len(values)] if series.timestamps is not None else None,
            config={"injection": series.metadata.get("injection_config", {})},
            seed=int(series.metadata.get("synthetic_seed", 42)),
            scale_stds=scale,
        )
    else:
        labels = (
            series.y_step[: len(values)]
            if series.y_step is not None
            else np.zeros(len(values), dtype=np.int8)
        )

    scaler = fit_scaler(values, "standard")
    values_scaled = transform_scaler(values, scaler)
    if len(values_scaled) < window_length:
        raise ValueError(
            f"No {window_length}-step sample window available in sequence {series.group_id!r}."
        )
    x = torch.tensor(values_scaled[:window_length][None, ...], dtype=torch.float32)
    y_step = torch.tensor(labels[:window_length][None, ...], dtype=torch.float32)
    y_window = torch.tensor([int(np.any(labels[:window_length] == 1))], dtype=torch.long)
    teacher_probs = torch.tensor([[0.6, 0.4]], dtype=torch.float32)

    model = GRUStudent(
        input_size=x.shape[-1], hidden_size=16, num_layers=1, dropout=0.0,
        seq_len=window_length,
    )
    loss_fn = DistillationLoss(temperature=2.0, lambda_kd=0.5, pos_weight=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    window_logits, step_logits = model(x)
    loss, _, _, _ = loss_fn(window_logits, step_logits, y_window, y_step, teacher_probs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach()), tuple(window_logits.shape), tuple(step_logits.shape)


def main(dataset_keys: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    with (root / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    print("Local dataset integration validation")
    keys = dataset_keys or list(config["datasets"])
    unknown_keys = [key for key in keys if key not in config["datasets"]]
    if unknown_keys:
        raise ValueError(f"Unknown dataset key(s): {unknown_keys}")
    for key in keys:
        dataset_cfg = config["datasets"][key]
        print(f"\n[{key}] loading...")
        series_list = load_dataset(key, dataset_cfg)
        total_steps = sum(s.n_timesteps for s in series_list)
        positive_steps = sum(int(np.sum(s.y_step == 1)) for s in series_list if s.y_step is not None)
        train, val, test, split_name = _split_series(series_list, dataset_cfg, config["splits"])
        length = int(dataset_cfg.get("window_length", config["windowing"]["window_length"]))
        stride = int(dataset_cfg.get("stride", config["windowing"]["stride"]))

        if key == "airquality" and dataset_cfg.get("label_strategy") == "semi_synthetic":
            train_concat = np.concatenate([s.values for s in train], axis=0)
            _, impute_stats = fill_missing(train_concat, strategy="interpolate")
            train_clean = [
                fill_missing(s.values, strategy="interpolate", train_stats=impute_stats)[0]
                for s in train
            ]
            train_stds = np.nanstd(np.concatenate(train_clean, axis=0), axis=0).astype(np.float32)
            train_stds = np.where(train_stds < 1e-6, 1.0, train_stds)
            for split_items, offset in ((train, 0), (val, 1000), (test, 2000)):
                for seq_idx, sequence in enumerate(split_items):
                    imputed, _ = fill_missing(
                        sequence.values, strategy="interpolate", train_stats=impute_stats
                    )
                    sequence.values, sequence.y_step = inject_anomalies(
                        imputed,
                        sequence.timestamps,
                        config={"injection": dataset_cfg.get("injection", {})},
                        seed=int(dataset_cfg.get("synthetic_seed", 42)) + offset + seq_idx,
                        scale_stds=train_stds,
                    )

        print(
            f"  sequences={len(series_list):,}; timesteps={total_steps:,}; "
            f"features={series_list[0].n_features}; labeled anomaly steps={positive_steps:,}"
        )
        print(f"  split={split_name}; sequences train/val/test={len(train):,}/{len(val):,}/{len(test):,}")
        if key == "oil3w":
            fitted_scaler = fit_scaler_batches(
                (sequence.values for sequence in train), scaler_type="standard"
            )
            if not np.isfinite(fitted_scaler.mean_).all():
                raise ValueError("The incremental training scaler produced non-finite feature means.")
            per_sequence_counts = [
                len(_window_start_indices(s.n_timesteps, length, stride)) for s in train
            ]
            cap = int(dataset_cfg.get("max_train_windows", 0))
            quotas = _allocate_window_quotas(per_sequence_counts, cap)
            print(
                f"  incremental train-only scaler: {len(fitted_scaler.mean_)} features; "
                f"training windows retained={int(quotas.sum()):,}/"
                f"{sum(per_sequence_counts):,} available (cap={cap:,})"
            )
        for name, part in (("train", train), ("val", val), ("test", test)):
            steps = sum(s.n_timesteps for s in part)
            labels = sum(int(np.sum(s.y_step == 1)) for s in part if s.y_step is not None)
            windows, positives = _count_windows(part, length, stride)
            print(
                f"  {name}: steps={steps:,}; positive_steps={labels:,}; "
                f"windows={windows:,}; positive_windows={positives:,}"
            )

        candidate = next((s for s in train if s.n_timesteps >= length), None)
        if candidate is None:
            raise ValueError(f"{key}: no training sequence is long enough for window length {length}.")
        loss, window_shape, step_shape = _smoke_student_step(key, candidate, length, stride)
        print(
            f"  student+KD step: loss={loss:.4f}; window_logits={window_shape}; "
            f"step_logits={step_shape}"
        )
        del series_list, train, val, test
        gc.collect()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", dest="dataset_keys", action="append",
        choices=("creditcard", "daphnet", "airquality", "oil3w", "metropt"),
        help="Dataset to check (repeat to check several); defaults to all five.",
    )
    raise SystemExit(main(parser.parse_args().dataset_keys))
