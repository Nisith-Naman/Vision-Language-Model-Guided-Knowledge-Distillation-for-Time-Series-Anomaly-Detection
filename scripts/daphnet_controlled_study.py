"""Run a reproducible, full-data Daphnet teacher/student comparison.

The train, validation, and test subjects are fixed by seed 42. The supervised
baseline and KD student use every training window and the same initial weights.
The teacher scores a time-distributed training subset and all validation/test
windows; validation scores set thresholds, while test scores are reported only
after training is complete.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\daphnet_controlled_study.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import average_precision_score

from src.data.datasets.daphnet import load as load_daphnet
from src.data.preprocessing import (
    chronological_split,
    fill_missing,
    fit_imputation_stats_batches,
    fit_scaler_batches,
    make_windows,
    transform_scaler,
)
from src.data.window_sampling import select_time_distributed_windows
from src.evaluation.localization import (
    aggregate_step_scores,
    event_overlap_score,
    group_intervals,
)
from src.evaluation.metrics import compute_metrics
from src.evaluation.threshold import select_threshold
from src.student.model import GRUStudent
from src.teacher.clip_teacher import CLIPTeacher
from src.training.trainer import Trainer


NORMAL_PROMPT = (
    "A regular, smooth, stable multidimensional time-series sensor pattern "
    "without anomalies or spikes"
)
ANOMALY_PROMPT = (
    "An anomalous, irregular, erratic multidimensional time-series signal "
    "with sudden spikes or distortions"
)


def _window_starts(length: int, window_length: int, stride: int) -> np.ndarray:
    if length < window_length:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, length - window_length + 1, stride, dtype=np.int64)
    if len(starts) and starts[-1] + window_length < length:
        starts = np.append(starts, length - window_length)
    return starts


def _hash_array(hasher: Any, value: np.ndarray | None) -> None:
    if value is None:
        hasher.update(b"<none>")
        return
    arr = np.ascontiguousarray(value)
    hasher.update(str(arr.dtype).encode("ascii"))
    hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    if arr.dtype.hasobject:
        hasher.update(repr(arr.tolist()).encode("utf-8"))
    else:
        hasher.update(memoryview(arr).cast("B"))


def _audit_raw_labels(data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.rglob("*.txt")):
        subject = path.stem[:3].upper()
        annotation = pd.read_csv(
            path, sep=r"\s+", header=None, usecols=[10], dtype=np.int16
        ).iloc[:, 0]
        counts = annotation.value_counts().to_dict()
        rows.append({
            "subject": subject,
            "file": path.name,
            "raw_rows": int(len(annotation)),
            "code_0_excluded": int(counts.get(0, 0)),
            "code_1_normal": int(counts.get(1, 0)),
            "code_2_freezing": int(counts.get(2, 0)),
            "other_codes": int(sum(n for code, n in counts.items() if code not in (0, 1, 2))),
        })
    return pd.DataFrame(rows)


def _preprocess(
    series: list,
    train_fraction: float,
    val_fraction: float,
    window_length: int,
    stride: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    train_series, val_series, test_series = chronological_split(
        series,
        train_frac=train_fraction,
        val_frac=val_fraction,
        split_by="subject",
        seed=seed,
    )
    parts = {"train": train_series, "validation": val_series, "test": test_series}
    train_stats = fit_imputation_stats_batches([s.values for s in train_series])

    def impute(sequences: list) -> list[np.ndarray]:
        return [
            s.values.astype(np.float32, copy=False)
            if np.isfinite(s.values).all()
            else fill_missing(s.values, "interpolate", train_stats)[0]
            for s in sequences
        ]

    imputed = {name: impute(group) for name, group in parts.items()}
    scaler = fit_scaler_batches(imputed["train"], scaler_type="standard")

    processed: dict[str, dict[str, Any]] = {}
    for split_name, sequences in parts.items():
        X_chunks: list[np.ndarray] = []
        y_window_chunks: list[np.ndarray] = []
        y_step_chunks: list[np.ndarray] = []
        sequence_meta: list[dict[str, Any]] = []
        next_index = 0
        for sequence, values in zip(sequences, imputed[split_name]):
            scaled = transform_scaler(values, scaler)
            Xw, yw, ysw, _ = make_windows(
                values=scaled,
                y_step=sequence.y_step,
                timestamps=sequence.timestamps,
                window_length=window_length,
                stride=stride,
                anomaly_rule="any",
                cover_tail=True,
            )
            if not len(Xw):
                continue
            starts = _window_starts(sequence.n_timesteps, window_length, stride)
            if len(starts) != len(Xw):
                raise RuntimeError(f"Window start mismatch for {sequence.group_id}.")
            indices = np.arange(next_index, next_index + len(Xw), dtype=np.int64)
            X_chunks.append(Xw)
            y_window_chunks.append(yw)
            y_step_chunks.append(ysw)
            sequence_meta.append({
                "group_id": str(sequence.group_id),
                "recording_file": sequence.metadata.get("recording_file", ""),
                "labels": np.asarray(sequence.y_step, dtype=np.int8),
                "window_indices": indices,
                "window_starts": starts,
            })
            next_index += len(Xw)

        if not X_chunks:
            raise ValueError(f"No complete windows in {split_name} split.")
        processed[split_name] = {
            "X": np.concatenate(X_chunks, axis=0),
            "y_window": np.concatenate(y_window_chunks),
            "y_step_windows": np.concatenate(y_step_chunks, axis=0),
            "sequence_meta": sequence_meta,
            "series": sequences,
        }

    source_hasher = hashlib.sha256()
    for sequence in series:
        source_hasher.update(str(sequence.group_id).encode("utf-8"))
        _hash_array(source_hasher, sequence.values)
        _hash_array(source_hasher, sequence.y_step)
        _hash_array(source_hasher, sequence.timestamps)
    source_hash = source_hasher.hexdigest()

    processed_hasher = hashlib.sha256()
    processed_hasher.update(source_hash.encode("ascii"))
    processed_hasher.update(str((window_length, stride, seed, "subject", "standard")).encode("utf-8"))
    split_manifest: dict[str, Any] = {}
    for split_name, split in processed.items():
        processed_hasher.update(split_name.encode("ascii"))
        _hash_array(processed_hasher, split["X"])
        _hash_array(processed_hasher, split["y_window"])
        _hash_array(processed_hasher, split["y_step_windows"])
        raw_steps = sum(s.n_timesteps for s in split["series"])
        raw_anomalies = sum(int(np.asarray(s.y_step).sum()) for s in split["series"])
        split_manifest[split_name] = {
            "subjects": sorted({str(s.group_id) for s in split["series"]}),
            "recording_segments": len(split["series"]),
            "raw_timesteps": int(raw_steps),
            "raw_anomaly_timesteps": int(raw_anomalies),
            "raw_anomaly_prevalence": float(raw_anomalies / raw_steps) if raw_steps else 0.0,
            "windows": int(len(split["X"])),
            "anomalous_windows": int(split["y_window"].sum()),
            "window_prevalence": float(split["y_window"].mean()),
            "window_point_labels": int(split["y_step_windows"].sum()),
        }

    manifest = {
        "dataset": "Daphnet FoG",
        "source_data_sha256": source_hash,
        "processed_split_sha256": processed_hasher.hexdigest(),
        "split_method": "subject-disjoint stratified group split",
        "split_seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": val_fraction,
        "test_fraction": 1.0 - train_fraction - val_fraction,
        "window_length": window_length,
        "stride": stride,
        "scaler": "standard, fitted on training subjects only",
        "imputation": "linear interpolation; train-derived fallback means",
        "window_label_rule": "positive if any point label in the window is FoG",
        "labels": "annotation 0 excluded; 1 normal; 2 freezing of gait anomaly",
        "partitions": split_manifest,
    }
    return processed, manifest


def _score_teacher(
    teacher: CLIPTeacher,
    X: np.ndarray,
    indices: np.ndarray,
    feature_names: list[str],
    vmin: float,
    vmax: float,
    image_batch_size: int = 64,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    # Reuse one Matplotlib figure and image artist. The regular helper creates
    # a new seaborn figure per window, which made scoring thousands of held-out
    # windows unnecessarily slow. This renders the same label-free viridis
    # heatmap with time on x and sensors on y, using the same fixed train range.
    fig, ax = plt.subplots(figsize=(224 / 100, 224 / 100), dpi=100)
    first_matrix = np.asarray(X[indices[0]], dtype=np.float32).T if len(indices) else None
    if first_matrix is None:
        plt.close(fig)
        return np.empty((0, 2), dtype=np.float32)
    artist = ax.imshow(
        first_matrix,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    if first_matrix.shape[0] <= 30:
        ax.set_yticks(np.arange(first_matrix.shape[0]))
        ax.set_yticklabels(feature_names if len(feature_names) == first_matrix.shape[0] else [f"F_{i}" for i in range(first_matrix.shape[0])])
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Time Series Window", fontsize=9, pad=6)
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Features", fontsize=8)
    if first_matrix.shape[0] > 10:
        ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout(pad=0.8)
    for start in range(0, len(indices), image_batch_size):
        batch_indices = indices[start : start + image_batch_size]
        images = []
        try:
            for window_idx in batch_indices:
                artist.set_data(np.asarray(X[window_idx], dtype=np.float32).T)
                fig.canvas.draw()
                rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
                images.append(Image.fromarray(rgb, mode="RGB"))
            scores.append(
                teacher.score_batch(
                    images,
                    normal_prompt=NORMAL_PROMPT,
                    anomaly_prompt=ANOMALY_PROMPT,
                    batch_size=64,
                )
            )
        finally:
            images.clear()
    plt.close(fig)
    return np.vstack(scores) if scores else np.empty((0, 2), dtype=np.float32)


def _timeline_predictions(
    step_scores: np.ndarray,
    sequence_meta: list[dict[str, Any]],
    window_length: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    per_sequence: list[tuple[np.ndarray, np.ndarray]] = []
    for meta in sequence_meta:
        indices = np.asarray(meta["window_indices"], dtype=np.int64)
        labels = np.asarray(meta["labels"], dtype=np.int8)
        scores = aggregate_step_scores(
            step_scores_per_window=step_scores[indices],
            window_starts=meta["window_starts"],
            window_length=window_length,
            total_length=len(labels),
        )
        all_labels.append(labels)
        all_scores.append(scores)
        per_sequence.append((labels, scores))
    return np.concatenate(all_labels), np.concatenate(all_scores), per_sequence


def _event_summary(
    per_sequence: list[tuple[np.ndarray, np.ndarray]], threshold: float
) -> dict[str, float]:
    true_events = predicted_events = detected_events = valid_predictions = 0
    total_hours = 0.0
    for labels, scores in per_sequence:
        true_intervals = group_intervals(labels)
        predicted_intervals = group_intervals((scores >= threshold).astype(np.int8))
        overlap = event_overlap_score(true_intervals, predicted_intervals)
        true_events += int(overlap["n_true_events"])
        predicted_events += int(overlap["n_pred_events"])
        detected_events += round(float(overlap["event_recall"]) * int(overlap["n_true_events"]))
        valid_predictions += round(float(overlap["event_precision"]) * int(overlap["n_pred_events"]))
        total_hours += len(labels) / 64.0 / 3600.0
    recall = detected_events / true_events if true_events else 0.0
    precision = valid_predictions / predicted_events if predicted_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_events = predicted_events - valid_predictions
    return {
        "true_events": float(true_events),
        "predicted_events": float(predicted_events),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "false_alarm_events_per_hour": float(false_events / total_hours) if total_hours else 0.0,
    }


def _predict(model: GRUStudent, X: np.ndarray, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    window_chunks: list[np.ndarray] = []
    step_chunks: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32)
        window_probs, step_probs = model.predict_proba(batch)
        window_chunks.append(window_probs)
        step_chunks.append(step_probs)
    return np.vstack(window_chunks), np.vstack(step_chunks)


def _train_student(
    mode: str,
    seed: int,
    train: dict[str, Any],
    validation: dict[str, Any],
    teacher_probs_aligned: np.ndarray | None,
    teacher_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[GRUStudent, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    initial_model = GRUStudent(
        input_size=train["X"].shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.3,
        seq_len=args.window_length,
        bidirectional=False,
    )
    initial_state = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}

    config = {
        "mode": mode,
        "device": "cpu",
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 1e-3,
            "early_stopping_patience": args.patience,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
        },
        "distillation": {
            "temperature": 2.0,
            "beta": 1.0,
            "lambda_kd": 0.5 if mode == "student_kd" else 0.0,
            "pos_weight": 10.0,
        },
    }
    trainer = Trainer(model=initial_model, config=config)
    fit_start = time.perf_counter()
    history = trainer.fit(
        X_train=train["X"],
        y_step_train=train["y_step_windows"],
        y_window_train=train["y_window"],
        teacher_probs_train=teacher_probs_aligned,
        X_val=validation["X"],
        y_step_val=validation["y_step_windows"],
        y_window_val=validation["y_window"],
        teacher_probs_val=None,
        teacher_mask_train=teacher_mask,
        progress_callback=lambda epoch, total, tr, va: print(
            f"  {mode} seed={seed} epoch={epoch}/{total} train_loss={tr:.4f} val_loss={va:.4f}",
            flush=True,
        ),
    )
    result = history.to_dict()
    result["wall_time_seconds"] = time.perf_counter() - fit_start
    result["training_windows"] = int(len(train["X"]))
    result["teacher_scored_training_windows"] = int(teacher_mask.sum()) if teacher_mask is not None else 0
    result["initial_weights_sha256"] = hashlib.sha256(
        b"".join(tensor.cpu().numpy().tobytes() for tensor in initial_state.values())
    ).hexdigest()
    return trainer.model, result


def _metric_row(
    model_name: str,
    level: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    result = compute_metrics(labels, scores, threshold)
    return {
        "model": model_name,
        "level": level,
        "prevalence": float(np.mean(labels)),
        "auprc": result.auprc,
        "roc_auc": result.roc_auc,
        "threshold": float(threshold),
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "true_anomalies": result.n_true_anomalies,
        "predicted_anomalies": result.n_predicted_anomalies,
        "confusion_matrix": json.dumps(result.confusion_matrix.tolist()),
    }


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    daphnet_cfg = dict(config["datasets"]["daphnet"])
    daphnet_cfg["data_dir"] = str((ROOT / daphnet_cfg["data_dir"]).resolve())
    series = load_daphnet(daphnet_cfg)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    label_audit = _audit_raw_labels(Path(daphnet_cfg["data_dir"]))
    label_audit.to_csv(output_dir / "raw_label_audit.csv", index=False)
    if int(label_audit["other_codes"].sum()) != 0:
        raise ValueError("Unexpected annotation codes were found. Inspect raw_label_audit.csv before training.")

    processed, manifest = _preprocess(
        series=series,
        train_fraction=float(config["splits"]["train"]),
        val_fraction=float(config["splits"]["val"]),
        window_length=args.window_length,
        stride=args.stride,
        seed=args.split_seed,
    )
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Split summary:")
    for name, info in manifest["partitions"].items():
        print(
            f"  {name}: subjects={info['subjects']} steps={info['raw_timesteps']:,} "
            f"positive_steps={info['raw_anomaly_timesteps']:,} "
            f"({info['raw_anomaly_prevalence']:.2%}) windows={info['windows']:,} "
            f"positive_windows={info['anomalous_windows']:,} ({info['window_prevalence']:.2%})"
        )
    train = processed["train"]
    validation = processed["validation"]
    test = processed["test"]

    train_indices = select_time_distributed_windows(
        len(train["X"]), train["sequence_meta"], args.teacher_budget
    )
    feature_names = list(series[0].feature_names)
    visual_vmin = float(np.nanmin(train["X"]))
    visual_vmax = float(np.nanmax(train["X"]))
    if visual_vmin == visual_vmax:
        visual_vmax = visual_vmin + 1.0

    teacher_indices_hash = hashlib.sha256(train_indices.tobytes()).hexdigest()
    teacher_cache_metadata = {
        "source_data_sha256": manifest["source_data_sha256"],
        "processed_split_sha256": manifest["processed_split_sha256"],
        "teacher_model": "openai/clip-vit-base-patch32",
        "normal_prompt": NORMAL_PROMPT,
        "anomaly_prompt": ANOMALY_PROMPT,
        "image_mode": "viridis heatmap; time x-axis; sensors y-axis; ground-truth labels excluded",
        "visual_vmin": visual_vmin,
        "visual_vmax": visual_vmax,
        "teacher_budget": args.teacher_budget,
        "teacher_indices_sha256": teacher_indices_hash,
        "train_windows": int(len(train["X"])),
        "validation_windows": int(len(validation["X"])),
        "test_windows": int(len(test["X"])),
    }

    print(
        f"Scoring frozen CLIP: {len(train_indices):,} training windows, "
        f"{len(validation['X']):,} validation windows, {len(test['X']):,} test windows.",
        flush=True,
    )
    teacher_cache_path = output_dir / "frozen_teacher_scores.npz"
    teacher_metadata_path = output_dir / "teacher_cache_metadata.json"
    cached_teacher_scores: dict[str, np.ndarray] | None = None
    cached_teacher_seconds = 0.0
    if teacher_cache_path.exists() and teacher_metadata_path.exists():
        stored_cache = json.loads(teacher_metadata_path.read_text(encoding="utf-8"))
        stored_metadata = stored_cache.get("identity", stored_cache)
        if stored_metadata == teacher_cache_metadata:
            cached_teacher_seconds = float(stored_cache.get("teacher_scoring_seconds", 0.0))
            with np.load(teacher_cache_path) as cached:
                cached_teacher_scores = {name: cached[name] for name in cached.files}
            if (
                np.array_equal(cached_teacher_scores.get("train_indices"), train_indices)
                and cached_teacher_scores.get("train_scores", np.empty((0, 2))).shape == (len(train_indices), 2)
                and cached_teacher_scores.get("validation_scores", np.empty((0, 2))).shape == (len(validation["X"]), 2)
                and cached_teacher_scores.get("test_scores", np.empty((0, 2))).shape == (len(test["X"]), 2)
            ):
                print("Reusing frozen-teacher scores with a matching data/prompt manifest.", flush=True)
            else:
                cached_teacher_scores = None
        if cached_teacher_scores is None:
            print("Teacher cache failed validation; scoring the frozen teacher again.", flush=True)

    if cached_teacher_scores is not None:
        teacher_train_sample = cached_teacher_scores["train_scores"]
        teacher_val = cached_teacher_scores["validation_scores"]
        teacher_test = cached_teacher_scores["test_scores"]
        teacher_seconds = cached_teacher_seconds
    else:
        teacher = CLIPTeacher(model_name="openai/clip-vit-base-patch32", device="cpu")
        teacher.load_model()
        if not teacher.is_frozen():
            raise RuntimeError("The CLIP teacher unexpectedly has trainable parameters.")
        teacher_start = time.perf_counter()
        teacher_train_sample = _score_teacher(
            teacher, train["X"], train_indices, feature_names, visual_vmin, visual_vmax
        )
        val_teacher_indices = np.arange(len(validation["X"]), dtype=np.int64)
        teacher_val = _score_teacher(
            teacher, validation["X"], val_teacher_indices, feature_names, visual_vmin, visual_vmax
        )
        test_teacher_indices = np.arange(len(test["X"]), dtype=np.int64)
        teacher_test = _score_teacher(
            teacher, test["X"], test_teacher_indices, feature_names, visual_vmin, visual_vmax
        )
        del teacher
        teacher_seconds = time.perf_counter() - teacher_start
        np.savez_compressed(
            teacher_cache_path,
            train_indices=train_indices,
            train_scores=teacher_train_sample,
            validation_scores=teacher_val,
            test_scores=teacher_test,
        )
        teacher_metadata_path.write_text(
            json.dumps({
                "identity": teacher_cache_metadata,
                "teacher_scoring_seconds": teacher_seconds,
            }, indent=2), encoding="utf-8"
        )
    teacher_train_aligned = np.zeros((len(train["X"]), 2), dtype=np.float32)
    teacher_train_aligned[train_indices] = teacher_train_sample
    teacher_mask = np.zeros(len(train["X"]), dtype=bool)
    teacher_mask[train_indices] = True
    teacher_val_threshold, _ = select_threshold(
        validation["y_window"], teacher_val[:, 1], objective="f1"
    )
    teacher_test_row = _metric_row(
        "Frozen CLIP", "window", test["y_window"], teacher_test[:, 1], teacher_val_threshold
    )
    teacher_test_row["validation_selected_threshold"] = teacher_val_threshold
    rows = [teacher_test_row]
    teacher_validation_row = _metric_row(
        "Frozen CLIP", "window", validation["y_window"], teacher_val[:, 1], teacher_val_threshold
    )
    teacher_validation_row["validation_selected_threshold"] = teacher_val_threshold
    rows.append(teacher_validation_row)
    print(
        f"Frozen CLIP complete in {teacher_seconds:.1f}s; "
        f"validation window AP={teacher_validation_row['auprc']:.4f}, "
        f"test window AP={teacher_test_row['auprc']:.4f}.",
        flush=True,
    )

    run_summaries: list[dict[str, Any]] = []
    for seed in args.training_seeds:
        print(f"\nTraining paired models for seed {seed}.", flush=True)
        baseline, baseline_history = _train_student(
            "student_baseline", seed, train, validation, None, None, args
        )
        kd, kd_history = _train_student(
            "student_kd", seed, train, validation,
            teacher_train_aligned, teacher_mask, args
        )
        if baseline_history["initial_weights_sha256"] != kd_history["initial_weights_sha256"]:
            raise RuntimeError("Paired baseline and KD models did not start from identical weights.")

        for model_name, model, history in (
            ("GRU supervised", baseline, baseline_history),
            ("GRU + CLIP KD", kd, kd_history),
        ):
            val_window, val_step = _predict(model, validation["X"])
            val_point_labels, val_point_scores, val_per_sequence = _timeline_predictions(
                val_step, validation["sequence_meta"], args.window_length
            )
            if not np.any(val_point_labels == 1):
                raise ValueError("Validation has no positive point labels; threshold selection is invalid.")
            step_threshold, _ = select_threshold(
                val_point_labels, val_point_scores, objective="f1"
            )
            window_threshold, _ = select_threshold(
                validation["y_window"], val_window[:, 1], objective="f1"
            )

            test_window, test_step = _predict(model, test["X"])
            test_point_labels, test_point_scores, test_per_sequence = _timeline_predictions(
                test_step, test["sequence_meta"], args.window_length
            )
            rows.extend([
                _metric_row(model_name, "window", test["y_window"], test_window[:, 1], window_threshold),
                _metric_row(model_name, "point", test_point_labels, test_point_scores, step_threshold),
            ])
            rows[-2]["validation_selected_threshold"] = window_threshold
            rows[-1]["validation_selected_threshold"] = step_threshold
            event = _event_summary(test_per_sequence, step_threshold)
            rows.append({"model": model_name, "level": "event", **event})

            history["validation_window_auprc"] = float(
                average_precision_score(validation["y_window"], val_window[:, 1])
            )
            history["validation_point_auprc"] = float(
                average_precision_score(val_point_labels, val_point_scores)
            )
            history["selected_step_threshold"] = float(step_threshold)
            history["selected_window_threshold"] = float(window_threshold)
            history["test_window_auprc"] = float(
                average_precision_score(test["y_window"], test_window[:, 1])
            )
            history["test_point_auprc"] = float(
                average_precision_score(test_point_labels, test_point_scores)
            )
            state_path = output_dir / f"{model_name.lower().replace(' ', '_').replace('+', 'kd')}_seed{seed}.pt"
            torch.save(model.state_dict(), state_path)
            with open(output_dir / f"{state_path.stem}_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            run_summaries.append({"model": model_name, "seed": seed, **history})

    pd.DataFrame(rows).to_csv(output_dir / "comparison_metrics.csv", index=False)
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest,
            "teacher": {
                "model": "openai/clip-vit-base-patch32",
                "frozen": True,
                "normal_prompt": NORMAL_PROMPT,
                "anomaly_prompt": ANOMALY_PROMPT,
                "training_windows_scored": int(len(train_indices)),
                "training_window_prevalence_in_sample": float(train["y_window"][train_indices].mean()),
                "training_window_prevalence_overall": float(train["y_window"].mean()),
                "scoring_seconds": teacher_seconds,
            },
            "training": {
                "epochs_max": args.epochs,
                "batch_size": args.batch_size,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "early_stopping_patience": args.patience,
                "optimizer": "AdamW",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "temperature": 2.0,
                "beta_window_loss": 1.0,
                "lambda_kd": 0.5,
                "point_pos_weight": 10.0,
                "seeds": args.training_seeds,
                "all_train_windows_used_by_both_students": int(len(train["X"])),
            },
            "runs": run_summaries,
        }, f, indent=2)

    print(f"\nSaved controlled comparison to: {output_dir}", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/daphnet_controlled_v1")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--training-seeds", type=lambda v: [int(x) for x in v.split(",")], default=[42])
    parser.add_argument("--window-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--teacher-budget", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
