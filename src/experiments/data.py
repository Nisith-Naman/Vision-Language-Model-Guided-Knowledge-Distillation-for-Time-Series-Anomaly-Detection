"""Deterministic dataset preparation shared by every model in the benchmark.

For a dataset, ``prepare_dataset`` produces one fixed train/validation/test
partition, fits imputation and scaling on training data only, injects the
synthetic Air Quality events per split (recording every event for audit), and
cuts each sequence into windows. The teacher, all student variants, and the
classical baselines are evaluated on exactly these windows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.data.loader import Series, load_dataset
from src.data.preprocessing import (
    chronological_split,
    fill_missing,
    fit_imputation_stats_batches,
    fit_scaler_batches,
    make_windows,
    transform_scaler,
)
from src.data.synthetic_labels import inject_anomalies
from src.data.window_sampling import select_time_distributed_windows


@dataclass
class Split:
    X: np.ndarray                     # (N, L, F) standardized windows
    y_window: np.ndarray              # (N,)
    y_step: np.ndarray                # (N, L)
    sequences: List[Dict[str, Any]]   # per sequence: group_id, labels, starts, window_indices, values
    n_timesteps: int = 0

    @property
    def point_labels(self) -> np.ndarray:
        return np.concatenate([s["labels"] for s in self.sequences]) if self.sequences else np.empty(0)


@dataclass
class PreparedData:
    name: str
    feature_names: List[str]
    window_length: int
    stride: int
    scaler: Any
    splits: Dict[str, Split]
    manifest: Dict[str, Any] = field(default_factory=dict)
    label_audit: List[Dict[str, Any]] = field(default_factory=list)


def _window_starts(length: int, window_length: int, stride: int) -> np.ndarray:
    if length < window_length:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, length - window_length + 1, stride, dtype=np.int64)
    if len(starts) and starts[-1] + window_length < length:
        starts = np.append(starts, length - window_length)
    return starts


def _split_series(series: List[Series], ds_cfg: Dict[str, Any], global_cfg: Dict[str, Any], seed: int):
    if all(s.metadata.get("split_role") in {"train", "val", "test"} for s in series):
        parts = {r: [s for s in series if s.metadata["split_role"] == r] for r in ("train", "val", "test")}
        return parts["train"], parts["val"], parts["test"], "dataset-provided roles (MetroPT event holdout)"
    fr = ds_cfg.get("splits") or global_cfg.get("splits", {})
    train_frac, val_frac = float(fr.get("train", 0.7)), float(fr.get("val", 0.15))
    split_by = str(ds_cfg.get("split_by", "chronological"))
    if split_by in {"subject", "instance", "event", "group"} and len({s.group_id for s in series}) >= 3:
        tr, va, te = chronological_split(series, train_frac, val_frac, split_by=split_by, seed=seed)
        return tr, va, te, f"{split_by}-disjoint stratified group split (seed {seed})"
    tr, va, te = chronological_split(series, train_frac, val_frac, split_by="chronological")
    return tr, va, te, f"chronological {train_frac:.0%}/{val_frac:.0%}/{1 - train_frac - val_frac:.0%}"


def prepare_dataset(
    name: str,
    config: Dict[str, Any],
    seed: int = 42,
    max_train_windows: int = 0,
    keep_values: bool = True,
) -> PreparedData:
    ds_cfg = dict(config["datasets"][name])
    window_length = int(ds_cfg.get("window_length", config["windowing"]["window_length"]))
    stride = int(ds_cfg.get("stride", config["windowing"]["stride"]))
    series = load_dataset(name, ds_cfg)
    train_s, val_s, test_s, split_desc = _split_series(series, ds_cfg, config, seed)
    parts = {"train": train_s, "val": val_s, "test": test_s}
    if not all(parts.values()):
        raise ValueError(f"{name}: empty partition ({ {k: len(v) for k, v in parts.items()} })")

    strategy = config.get("preprocessing", {}).get("missing_strategy", "interpolate")
    stats = fit_imputation_stats_batches([s.values for s in train_s])

    def impute(seq: Series) -> np.ndarray:
        v = seq.values.astype(np.float32, copy=False)
        return v if np.isfinite(v).all() else fill_missing(v, strategy, stats)[0]

    imputed = {k: [impute(s) for s in v] for k, v in parts.items()}
    scaler = fit_scaler_batches(imputed["train"], scaler_type="standard")

    label_audit: List[Dict[str, Any]] = []
    if any(s.metadata.get("label_strategy") == "semi_synthetic" for s in series):
        train_std = np.asarray(fit_scaler_batches(imputed["train"], "standard").scale_, dtype=np.float32)
        for offset, split_name in ((0, "train"), (1000, "val"), (2000, "test")):
            for i, seq in enumerate(parts[split_name]):
                values, labels, meta = inject_anomalies(
                    imputed[split_name][i], seq.timestamps,
                    {"injection": seq.metadata.get("injection_config", {})},
                    seed=int(seq.metadata.get("synthetic_seed", 42)) + offset + i,
                    start_idx=0, scale_stds=train_std, return_metadata=True,
                )
                imputed[split_name][i] = values
                seq.y_step = labels
                for ev in meta.get("events", []):
                    label_audit.append({"split": split_name, "sequence": seq.group_id, **ev,
                                        "channels": ",".join(seq.feature_names[c] for c in ev["channels"])})

    splits: Dict[str, Split] = {}
    for split_name, seqs in parts.items():
        Xs, yws, yss, metas = [], [], [], []
        next_idx, total_T = 0, 0
        for seq, values in zip(seqs, imputed[split_name]):
            labels = seq.y_step if seq.y_step is not None else np.zeros(seq.n_timesteps, np.int8)
            scaled = transform_scaler(values, scaler)
            Xw, yw, ysw, _ = make_windows(scaled, labels, None, window_length, stride,
                                          anomaly_rule="any", cover_tail=True)
            total_T += seq.n_timesteps
            if not len(Xw):
                continue
            starts = _window_starts(seq.n_timesteps, window_length, stride)
            idx = np.arange(next_idx, next_idx + len(Xw))
            Xs.append(Xw); yws.append(yw); yss.append(ysw)
            meta = {"group_id": str(seq.group_id), "labels": np.asarray(labels, np.int8),
                    "window_starts": starts, "window_indices": idx}
            if keep_values and split_name != "train":
                meta["values"] = scaled.astype(np.float32)
                meta["timestamps"] = seq.timestamps
            metas.append(meta)
            next_idx += len(Xw)
        if not Xs:
            raise ValueError(f"{name}: no complete windows in {split_name}")
        splits[split_name] = Split(np.concatenate(Xs), np.concatenate(yws).astype(np.int8),
                                   np.concatenate(yss).astype(np.int8), metas, total_T)

    # Optional label-independent cap on training windows (time-distributed).
    tr = splits["train"]
    if max_train_windows and len(tr.X) > max_train_windows:
        keep = select_time_distributed_windows(len(tr.X), tr.sequences, max_train_windows)
        splits["train"] = Split(tr.X[keep], tr.y_window[keep], tr.y_step[keep], [], tr.n_timesteps)

    hasher = hashlib.sha256()
    for k in ("train", "val", "test"):
        hasher.update(k.encode()); hasher.update(np.ascontiguousarray(splits[k].X).tobytes()[:1 << 20])
        hasher.update(splits[k].y_step.tobytes())
    manifest = {
        "dataset": name,
        "split_method": split_desc,
        "window_length": window_length,
        "stride": stride,
        "features": list(series[0].feature_names),
        "processed_sha256": hasher.hexdigest(),
        "partitions": {
            k: {
                "groups": sorted({str(s.group_id) for s in parts[k]})[:50],
                "n_groups": len({str(s.group_id) for s in parts[k]}),
                "timesteps": int(sum(s.n_timesteps for s in parts[k])),
                "anomalous_timesteps": int(sum(int(np.sum(s.y_step)) for s in parts[k] if s.y_step is not None)),
                "point_prevalence": float(
                    sum(int(np.sum(s.y_step)) for s in parts[k] if s.y_step is not None)
                    / max(1, sum(s.n_timesteps for s in parts[k]))
                ),
                "windows": int(len(splits[k].X)),
                "anomalous_windows": int(splits[k].y_window.sum()),
                "window_prevalence": float(splits[k].y_window.mean()),
            }
            for k in ("train", "val", "test")
        },
        "synthetic_labels": bool(any(s.metadata.get("synthetic_anomalies") for s in series)),
    }
    return PreparedData(name, list(series[0].feature_names), window_length, stride, scaler,
                        splits, manifest, label_audit)
