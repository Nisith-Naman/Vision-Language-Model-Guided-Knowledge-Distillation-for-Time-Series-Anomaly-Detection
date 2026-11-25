"""Score one multivariate series with both the frozen-CLIP teacher and the
distilled GRU student, using the artefacts written by ``scripts/run_benchmark.py``
and ``scripts/finalize_benchmark.py``.

Artefacts per dataset ``<name>``:
    checkpoints/v2_<name>_gru_kd_coop/        distilled student (learned-prompt teacher)
    checkpoints/v2_<name>_gru_kd_full/        distilled student (probe teacher), fallback
    checkpoints/v2_<name>_gru_gt/             same GRU trained on ground truth only
    experiments/benchmark_v2/<name>/clip_coop_mv.npz      learned prompt context (frozen CLIP)
    experiments/benchmark_v2/<name>/clip_probe.joblib     probe head on frozen CLIP, fallback
    experiments/benchmark_v2/<name>/teacher_params.json   teacher type, views, localization
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.evaluation.postprocess import aggregate, event_scores, remove_short, runs, smooth
from src.training.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "experiments" / "benchmark_v2"
CKPT = ROOT / "checkpoints"


@dataclass
class Bundle:
    name: str
    feature_names: List[str]
    window_length: int
    stride: int
    scaler: Any
    students: Dict[str, Any]            # key -> (model, localization params dict)
    probe: Any                          # LinearProbeTeacher or PromptLearningTeacher
    teacher_params: Dict[str, Any]
    student_key: str = "gru_kd_full"


STUDENT_KEYS = ("gru_kd_coop", "gru_kd_full")   # preferred distilled student first


def available_datasets() -> List[str]:
    out = []
    for d in sorted(BENCH.glob("*/teacher_params.json")):
        name = d.parent.name
        if any((CKPT / f"v2_{name}_{k}" / "metadata.json").exists() for k in STUDENT_KEYS):
            out.append(name)
    return out


def load_bundle(name: str) -> Bundle:
    from src.teacher.probe import LinearProbeTeacher

    students, scaler, meta_ref = {}, None, None
    for key in (*STUDENT_KEYS, "gru_gt"):
        path = CKPT / f"v2_{name}_{key}"
        if not (path / "metadata.json").exists():
            continue
        model, sc, meta = load_checkpoint(str(path))
        students[key] = (model.eval(), dict((meta.training_result or {}).get("localization") or {}))
        scaler, meta_ref = scaler or sc, meta_ref or meta
    if meta_ref is None:
        raise FileNotFoundError(f"No v2 checkpoints for {name}")
    params = json.loads((BENCH / name / "teacher_params.json").read_text())
    if params.get("teacher") == "coop_mv":
        from src.teacher.prompt_learning import PromptLearningTeacher

        teacher = PromptLearningTeacher()
        teacher.ctx_ = np.load(BENCH / name / "clip_coop_mv.npz")["ctx"]
    else:
        teacher = LinearProbeTeacher.load(str(BENCH / name / "clip_probe.joblib"))
    student_key = next(k for k in (*STUDENT_KEYS, "gru_gt") if k in students)
    return Bundle(name, list(meta_ref.feature_names), int(meta_ref.window_length),
                  int(meta_ref.stride), scaler, students, teacher, params, student_key)


def scale_frame(df: pd.DataFrame, bundle: Bundle) -> np.ndarray:
    missing = [c for c in bundle.feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {missing}")
    values = df[bundle.feature_names].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    means = getattr(bundle.scaler, "mean_", None)
    values = values.fillna(pd.Series(means, index=bundle.feature_names) if means is not None else 0.0)
    return np.asarray(bundle.scaler.transform(values.to_numpy(np.float64)), dtype=np.float32)


def window_starts(total: int, L: int, stride: int) -> np.ndarray:
    if total < L:
        return np.array([0])
    s = np.arange(0, total - L + 1, stride)
    return np.append(s, total - L) if s[-1] + L < total else s


def _windows(scaled: np.ndarray, L: int, stride: int):
    if len(scaled) < L:
        scaled = np.concatenate([scaled, np.repeat(scaled[-1:], L - len(scaled), 0)])
    starts = window_starts(len(scaled), L, stride)
    return np.stack([scaled[s:s + L] for s in starts]), starts, len(scaled)


def score_student(bundle: Bundle, key: str, scaled: np.ndarray) -> Dict[str, Any]:
    model, loc = bundle.students[key]
    X, starts, total = _windows(scaled, bundle.window_length, bundle.stride)
    t0 = time.perf_counter()
    win, step = [], []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            pw, ps = model.predict_proba(torch.as_tensor(X[i:i + 256]))
            win.append(pw[:, 1]); step.append(ps)
    elapsed = time.perf_counter() - t0
    win, step = np.concatenate(win), np.concatenate(step)
    step_tl = aggregate(step, starts, bundle.window_length, total)
    win_tl = aggregate(win, starts, bundle.window_length, total)
    alpha = float(loc.get("alpha", 0.0))
    score = step_tl * np.clip(win_tl, 0, 1) ** alpha if alpha > 0 else step_tl
    score = smooth(score, int(loc.get("k", 1)))[: len(scaled)]
    thr = float(loc.get("threshold", 0.5))
    pred = remove_short((score >= thr).astype(np.int8), int(loc.get("min_len", 1)))
    return {"score": score, "pred": pred, "threshold": thr, "seconds": elapsed, "windows": len(X)}


def score_teacher(bundle: Bundle, scaled: np.ndarray, embedder=None, max_windows: int = 3000) -> Dict[str, Any]:
    from src.teacher.clip_teacher import CLIPEmbedder
    from src.teacher.render import render_batch, render_spectrogram

    p = bundle.teacher_params
    X, starts, total = _windows(scaled, bundle.window_length, bundle.stride)
    if len(X) > max_windows:
        raise ValueError(f"{len(X)} windows exceed the teacher budget ({max_windows}); upload a shorter CSV.")
    embedder = embedder or CLIPEmbedder(p.get("model_name", "openai/clip-vit-base-patch32")).load()
    t0 = time.perf_counter()
    clip_v = float(p["render_clip"])
    heat = np.concatenate([embedder.encode_images(render_batch(X[i:i + 256], p["render_style"], clip_v))
                           for i in range(0, len(X), 256)])
    if p.get("teacher") == "coop_mv":
        spec = np.concatenate([
            embedder.encode_images(np.stack([render_spectrogram(w, clip_v) for w in X[i:i + 256]]))
            for i in range(0, len(X), 256)])
        win = bundle.probe.predict_proba(np.stack([heat, spec]), embedder=embedder)[:, 1]
    else:
        win = bundle.probe.predict_proba(heat)[:, 1]
    elapsed = time.perf_counter() - t0
    loc = p["localization"]
    score = smooth(aggregate(win, starts, bundle.window_length, total), int(loc.get("k", 1)))[: len(scaled)]
    thr = float(loc["threshold"])
    pred = remove_short((score >= thr).astype(np.int8), int(loc.get("min_len", 1)))
    return {"score": score, "pred": pred, "threshold": thr, "seconds": elapsed, "windows": len(X)}


def summarize(labels: Optional[np.ndarray], result: Dict[str, Any]) -> Dict[str, Any]:
    pred, score = result["pred"], result["score"]
    out = {"flagged_steps": int(pred.sum()), "events": len(runs(pred)),
           "seconds": round(float(result["seconds"]), 3)}
    if labels is not None and 0 < labels.sum() < len(labels):
        from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

        out.update({
            "AUPRC": float(average_precision_score(labels, score)),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "F1": float(f1_score(labels, pred, zero_division=0)),
            "event_F1": float(event_scores([labels], [pred])["event_f1"]),
        })
    return out
