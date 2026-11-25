"""Generate label-free out-of-sample test series in a trained dataset's schema.

The output CSV has exactly the dataset's feature columns (plus a timestamp) and
**no label column**, so it can be uploaded to the interface like real
production data. The ground truth is written to a separate answer key that the
models never see; it is only used afterwards to grade their predictions.

Background (normal behaviour):
    bootstrap  random normal-only stretches of the held-out *test* split
               (never used for training or tuning), joined with short
               crossfades. Realistic, but built from real recordings.
    synthetic  a multichannel AR(1) process X_t = c + A X_{t-1} + e_t fitted by
               ridge least squares to normal *training* windows, with Gaussian
               innovations matching the residual covariance. Entirely new data.

Hidden anomalies, in standardised units (1 = one training std), on a random
subset of channels:
    spike        1-3 steps, +-6
    level_shift  +-3 for 0.5-2 windows
    drift        linear ramp to +-4 over 1-2 windows
    noise_burst  added N(0, 2.5^2) for 0.5-1.5 windows
    oscillation  2-amplitude sinusoid, period 4-10 steps, for 0.5-2 windows
                 (tremor-like, similar in spirit to freezing of gait)
    flatline     channel stuck at its current value for 1-2 windows

    real_event   a genuine labelled anomaly segment cut from the held-out test
                 split (fraud rows, freezing-of-gait episode, well fault,
                 compressor failure) and transplanted, with 8-step crossfades,
                 into the new background at a new position.

The six generic shapes test generalisation to anomaly *types the models were
never trained on*. ``real_event`` tests the task the models were trained for,
on series they have never seen as a whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ANOMALY_TYPES = ("spike", "level_shift", "drift", "noise_burst", "oscillation", "flatline")
DEFAULT_TYPES = ("real_event", "spike", "level_shift", "real_event", "drift", "noise_burst",
                 "oscillation", "real_event", "flatline")


def anomalous_runs(sequences: List[Dict], max_len: int) -> List[np.ndarray]:
    """Labelled anomalous stretches of held-out sequences (truncated to max_len)."""
    out = []
    for seq in sequences:
        labels = np.asarray(seq["labels"])
        d = np.diff(np.r_[0, labels, 0])
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if e - s > max_len:
                s = s + (e - s - max_len) // 2
                e = s + max_len
            out.append(seq["values"][s:e].astype(np.float32))
    return out


@dataclass
class GeneratedSeries:
    frame: pd.DataFrame        # label-free CSV content (raw units)
    answer_key: pd.DataFrame   # one row per timestep: label, anomaly_type
    events: pd.DataFrame       # one row per injected event
    info: Dict[str, object]


def _normal_runs(sequences: List[Dict], min_len: int) -> List[np.ndarray]:
    runs = []
    for seq in sequences:
        values, labels = seq["values"], np.asarray(seq["labels"])
        d = np.diff(np.r_[1, labels, 1])
        starts, ends = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
        runs += [values[s:e] for s, e in zip(starts, ends) if e - s >= min_len]
    return runs


def bootstrap_background(sequences: List[Dict], length: int, segment: int, rng, fade: int = 8) -> np.ndarray:
    runs = _normal_runs(sequences, segment)
    if not runs:
        raise ValueError("No normal stretch in the test split is long enough; lower --segment.")
    out: Optional[np.ndarray] = None
    while out is None or len(out) < length:
        r = runs[rng.integers(len(runs))]
        s = rng.integers(0, len(r) - segment + 1)
        piece = r[s:s + segment].astype(np.float32).copy()
        if out is None:
            out = piece
            continue
        f = min(fade, len(piece), len(out))
        w = np.linspace(0, 1, f)[:, None]
        piece[:f] = (1 - w) * out[-f:] + w * piece[:f]
        out = np.concatenate([out[:-f], piece])
    return out[:length]


def fit_ar1(windows: np.ndarray, ridge: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit X_t = c + A X_{t-1} + e on normal training windows (N, L, F)."""
    prev = windows[:, :-1].reshape(-1, windows.shape[-1])
    nxt = windows[:, 1:].reshape(-1, windows.shape[-1])
    Z = np.c_[np.ones(len(prev)), prev]
    coef = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ nxt)
    c, A = coef[0], coef[1:].T
    resid = nxt - Z @ coef
    cov = np.cov(resid.T) + 1e-6 * np.eye(len(c))
    # Keep the process stable: shrink A if its spectral radius is >= 1.
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho >= 0.995:
        A = A * (0.995 / rho)
    return c, A, np.linalg.cholesky(cov)


def synthetic_background(c, A, chol, length: int, rng, burn_in: int = 500) -> np.ndarray:
    F = len(c)
    x = np.zeros(F)
    out = np.empty((length, F), np.float32)
    for t in range(-burn_in, length):
        x = c + A @ x + chol @ rng.standard_normal(F)
        if t >= 0:
            out[t] = x
    return out


def inject(values: np.ndarray, n_events: int, window: int, rng, types=ANOMALY_TYPES,
           real_runs: Optional[List[np.ndarray]] = None):
    """Inject non-overlapping events; returns modified values, labels, type per step, events."""
    x = values.copy()
    T, F = x.shape
    labels = np.zeros(T, np.int8)
    kinds = np.array([""] * T, dtype=object)
    events = []
    margin = max(4, window // 2)
    for i in range(n_events):
        kind = types[i % len(types)]
        if kind == "real_event" and not real_runs:
            continue
        real = real_runs[rng.integers(len(real_runs))] if kind == "real_event" else None
        dur = len(real) if real is not None else {"spike": int(rng.integers(1, 4)),
               "level_shift": int(rng.integers(window // 2, 2 * window)),
               "drift": int(rng.integers(window, 2 * window)),
               "noise_burst": int(rng.integers(window // 2, 3 * window // 2)),
               "oscillation": int(rng.integers(window // 2, 2 * window)),
               "flatline": int(rng.integers(window, 2 * window))}[kind]
        if real is None:
            dur = max(1, min(dur, T // (4 * n_events)))
        elif dur > T // 3:
            real, dur = real[:T // 3], T // 3
        for _ in range(200):
            s = int(rng.integers(margin, T - dur - margin))
            if labels[max(0, s - margin):s + dur + margin].sum() == 0:
                break
        else:
            continue
        k = int(rng.integers(1, max(2, F // 2) + 1))
        ch = np.sort(rng.choice(F, size=min(k, F), replace=False))
        sign = rng.choice([-1.0, 1.0])
        seg = slice(s, s + dur)
        if kind == "spike":
            x[seg, ch] += sign * 6.0
        elif kind == "level_shift":
            x[seg, ch] += sign * 3.0
        elif kind == "drift":
            x[seg, ch] += sign * np.linspace(0, 4.0, dur)[:, None]
        elif kind == "noise_burst":
            x[seg, ch] += rng.normal(0, 2.5, size=(dur, len(ch)))
        elif kind == "oscillation":
            period = rng.uniform(4, 10)
            x[seg, ch] += 2.0 * np.sin(2 * np.pi * np.arange(dur) / period)[:, None]
        elif kind == "flatline":
            x[seg, ch] = x[s, ch]
        elif kind == "real_event":
            ch = np.arange(F)
            piece = real.copy()
            f = min(8, dur // 2)
            if f > 0:
                w = np.linspace(0, 1, f)[:, None]
                piece[:f] = (1 - w) * x[s:s + f] + w * piece[:f]
                piece[-f:] = w[::-1] * piece[-f:] + (1 - w[::-1]) * x[s + dur - f:s + dur]
            x[seg] = piece
        labels[seg] = 1
        kinds[seg] = kind
        events.append({"type": kind, "start": s, "end": s + dur, "length": dur, "channels": list(ch)})
    return x, labels, kinds, events


def generate(data, length: int = 3000, n_events: int = 9, mode: str = "bootstrap", seed: int = 0,
             segment: Optional[int] = None, types=DEFAULT_TYPES) -> GeneratedSeries:
    """``data`` is a PreparedData from ``src.experiments.data.prepare_dataset``."""
    rng = np.random.default_rng(seed)
    L = data.window_length
    if mode == "bootstrap":
        background = bootstrap_background(data.splits["test"].sequences, length, segment or 2 * L, rng)
    elif mode == "synthetic":
        tr = data.splits["train"]
        normal = tr.X[tr.y_window == 0]
        normal = normal[rng.choice(len(normal), size=min(3000, len(normal)), replace=False)]
        background = synthetic_background(*fit_ar1(normal), length, rng)
    else:
        raise ValueError("mode must be 'bootstrap' or 'synthetic'")
    real_runs = anomalous_runs(data.splits["test"].sequences, max_len=2 * L)
    x, labels, kinds, events = inject(background, n_events, L, rng, types=types, real_runs=real_runs)
    raw = data.scaler.inverse_transform(x.astype(np.float64))
    frame = pd.DataFrame(raw, columns=data.feature_names)
    frame.insert(0, "timestamp", pd.date_range("2030-01-01", periods=length, freq="1min"))
    key = pd.DataFrame({"timestamp": frame["timestamp"], "label": labels, "anomaly_type": kinds})
    ev = pd.DataFrame(events)
    if len(ev):
        ev["channels"] = ev["channels"].apply(lambda c: ",".join(data.feature_names[i] for i in c))
    info = {"dataset": data.name, "mode": mode, "seed": seed, "length": length, "events": len(ev),
            "anomalous_fraction": float(labels.mean()), "window_length": L}
    return GeneratedSeries(frame, key, ev, info)
