"""Timeline reconstruction, localization post-processing, and evaluation.

Localization pipeline (all parameters chosen on validation data only):

1. ``step``    average the student's per-step scores over overlapping windows.
2. ``gate``    multiply by the window-level anomaly probability raised to
               ``alpha`` (0 = off). The window head is the part of the student
               that receives teacher knowledge (KD), so gating lets that
               knowledge suppress isolated step activations inside windows the
               window head considers normal.
3. ``smooth``  centred moving average of length ``k`` (1 = off).
4. ``threshold`` chosen to maximise validation point-F1. If validation holds
               fewer than ``MIN_VAL_EVENTS`` anomalous events, an F1-optimal
               threshold is unreliable (on MetroPT, one validation failure gave
               thresholds of 0.004-0.35 across seeds), so the threshold is set
               instead at the (1 - ``TARGET_FAR``) quantile of the *normal*
               validation scores, i.e. a 0.1% false-alarm rate.
5. ``min_len`` drop predicted runs shorter than ``min_len`` steps
               (chosen to maximise validation event-F1).

No point-adjustment is used: a detected event does not mark the whole true
event as detected at the point level.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.evaluation.threshold import select_threshold


MIN_VAL_EVENTS = 5
TARGET_FAR = 0.001


def calibrated_threshold(y: np.ndarray, scores: np.ndarray, n_events: int) -> float:
    """F1-optimal threshold on validation, or a false-alarm-rate threshold on the
    normal validation scores when there are too few validation events."""
    y = np.asarray(y).astype(int)
    if n_events >= MIN_VAL_EVENTS or not np.any(y == 0):
        return float(select_threshold(y, scores, objective="f1")[0])
    q = float(np.quantile(np.asarray(scores)[y == 0], 1.0 - TARGET_FAR))
    return float(np.nextafter(q, np.inf))


def aggregate(per_window: np.ndarray, starts: np.ndarray, window_length: int, total: int) -> np.ndarray:
    """Average per-window values (N, L) or (N,) over overlapping positions."""
    per_window = np.asarray(per_window, dtype=np.float64)
    if per_window.ndim == 1:
        per_window = np.repeat(per_window[:, None], window_length, axis=1)
    acc = np.zeros(total)
    cnt = np.zeros(total)
    for row, s in zip(per_window, np.asarray(starts, dtype=int)):
        e = min(s + window_length, total)
        acc[s:e] += row[: e - s]
        cnt[s:e] += 1
    out = np.zeros(total)
    m = cnt > 0
    out[m] = acc[m] / cnt[m]
    return out


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < 2:
        return x
    k = min(int(k), len(x))
    kernel = np.ones(k) / k
    pad = k // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def runs(binary: np.ndarray) -> List[tuple]:
    b = np.asarray(binary, dtype=np.int8)
    if not len(b):
        return []
    d = np.diff(np.r_[0, b, 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def remove_short(binary: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1:
        return np.asarray(binary, dtype=np.int8)
    out = np.asarray(binary, dtype=np.int8).copy()
    for s, e in runs(out):
        if e - s < min_len:
            out[s:e] = 0
    return out


def event_scores(label_seqs: Sequence[np.ndarray], pred_seqs: Sequence[np.ndarray]) -> Dict[str, float]:
    """Overlap-based event precision/recall/F1 summed over sequences."""
    n_true = n_pred = hit_true = hit_pred = 0
    for y, p in zip(label_seqs, pred_seqs):
        t_runs, p_runs = runs(y), runs(p)
        n_true += len(t_runs)
        n_pred += len(p_runs)
        cum_p = np.r_[0, np.cumsum(p)]
        cum_y = np.r_[0, np.cumsum(y)]
        hit_true += sum(1 for s, e in t_runs if cum_p[e] - cum_p[s] > 0)
        hit_pred += sum(1 for s, e in p_runs if cum_y[e] - cum_y[s] > 0)
    rec = hit_true / n_true if n_true else 0.0
    prec = hit_pred / n_pred if n_pred else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"event_precision": prec, "event_recall": rec, "event_f1": f1,
            "true_events": float(n_true), "predicted_events": float(n_pred)}


@dataclass
class LocalizationParams:
    alpha: float = 0.0
    k: int = 1
    threshold: float = 0.5
    min_len: int = 1

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def build_timelines(sequences: List[Dict], window_length: int, step_scores: Optional[np.ndarray],
                    window_scores: np.ndarray) -> List[Dict[str, np.ndarray]]:
    """Per-sequence timelines of labels, step scores and window scores."""
    out = []
    for seq in sequences:
        idx = np.asarray(seq["window_indices"], dtype=int)
        total = len(seq["labels"])
        win = aggregate(window_scores[idx], seq["window_starts"], window_length, total)
        step = aggregate(step_scores[idx], seq["window_starts"], window_length, total) if step_scores is not None else win
        out.append({"labels": np.asarray(seq["labels"], np.int8), "step": step, "win": win})
    return out


def _score(tl: Dict[str, np.ndarray], alpha: float, k: int) -> np.ndarray:
    s = tl["step"] * (np.clip(tl["win"], 0, 1) ** alpha) if alpha > 0 else tl["step"]
    return smooth(s, k)


def tune_localization(val_tl: List[Dict[str, np.ndarray]], window_length: int,
                      has_step_head: bool = True) -> LocalizationParams:
    """Choose alpha and k by validation point AUPRC, the threshold by point F1,
    and min_len by event F1."""
    y = np.concatenate([t["labels"] for t in val_tl])
    alphas = (0.0, 0.5, 1.0) if has_step_head else (0.0,)
    ks = sorted({1, max(1, window_length // 16), max(1, window_length // 8), max(1, window_length // 4)})
    best = (-1.0, 0.0, 1)
    for a in alphas:
        for k in ks:
            s = np.concatenate([_score(t, a, k) for t in val_tl])
            ap = average_precision_score(y, s) if 0 < y.sum() < len(y) else 0.0
            if ap > best[0] + 1e-6:
                best = (ap, a, k)
    _, alpha, k = best
    scores = [_score(t, alpha, k) for t in val_tl]
    n_events = sum(len(runs(t["labels"])) for t in val_tl)
    thr = calibrated_threshold(y, np.concatenate(scores), n_events)
    best_len, best_f1 = 1, -1.0
    for m in sorted({1, 2, max(1, window_length // 16), max(1, window_length // 8), max(1, window_length // 4)}):
        preds = [remove_short((s >= thr).astype(np.int8), m) for s in scores]
        f1 = event_scores([t["labels"] for t in val_tl], preds)["event_f1"]
        if f1 > best_f1 + 1e-9:
            best_f1, best_len = f1, m
    return LocalizationParams(alpha=float(alpha), k=int(k), threshold=float(thr), min_len=int(best_len))


def apply_localization(tl: List[Dict[str, np.ndarray]], p: LocalizationParams):
    scores = [_score(t, p.alpha, p.k) for t in tl]
    preds = [remove_short((s >= p.threshold).astype(np.int8), p.min_len) for s in scores]
    return scores, preds


def _binary_metrics(y: np.ndarray, s: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    has_both = 0 < y.sum() < len(y)
    return {
        "prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, s)) if has_both else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if has_both else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def evaluate(val_split, test_split, window_length: int,
             val_window: np.ndarray, test_window: np.ndarray,
             val_step: Optional[np.ndarray] = None, test_step: Optional[np.ndarray] = None) -> Dict:
    """Window-, point- and event-level test metrics with validation-only tuning.

    ``*_window`` are anomaly probabilities per window; ``*_step`` are per-step
    scores (N, L) or None for models without a step head (their window score is
    spread over the window's steps).
    """
    n_val_events = sum(len(runs(np.asarray(sq["labels"]))) for sq in val_split.sequences)
    w_thr = calibrated_threshold(val_split.y_window, val_window, n_val_events)
    window = _binary_metrics(test_split.y_window.astype(int), test_window,
                             (test_window >= w_thr).astype(int))
    window["threshold"] = float(w_thr)

    val_tl = build_timelines(val_split.sequences, window_length, val_step, val_window)
    test_tl = build_timelines(test_split.sequences, window_length, test_step, test_window)
    params = tune_localization(val_tl, window_length, has_step_head=val_step is not None)
    scores, preds = apply_localization(test_tl, params)
    y = np.concatenate([t["labels"] for t in test_tl])
    point = _binary_metrics(y, np.concatenate(scores), np.concatenate(preds))
    point.update({f"loc_{k}": v for k, v in params.to_dict().items()})
    event = event_scores([t["labels"] for t in test_tl], preds)
    return {"window": window, "point": point, "event": event,
            "test_scores": scores, "test_preds": preds, "params": params}
