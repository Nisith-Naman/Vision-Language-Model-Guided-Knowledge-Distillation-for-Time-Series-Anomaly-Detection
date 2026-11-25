"""The toy example in WORKED_EXAMPLE.md, computed with the project's own functions.

    .venv\\Scripts\\python.exe scripts\\toy_example.py

Every number in WORKED_EXAMPLE.md is printed here. Work the steps out by hand,
then run this to check: it calls the same code the real pipeline uses
(interpolation, scaler, windowing, rendering, loss, thresholds, timelines,
event scoring), only on 16 rows and 2 sensors instead of thousands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.evaluation.postprocess import (aggregate, event_scores, remove_short, runs,  # noqa: E402
                                        _binary_metrics)
from src.evaluation.threshold import select_threshold  # noqa: E402
from src.experiments.data import _window_starts  # noqa: E402
from src.student.losses import TotalLoss  # noqa: E402
from src.teacher.render import render_window  # noqa: E402

np.set_printoptions(precision=4, suppress=True)


def head(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


# --------------------------------------------------------------- 1. CSV -> matrix
head("1. CSV -> matrix (16 rows, 2 sensors)")
temp = [18, 22, np.nan, 22, 18, 18, 22, 18,  20, 22, 18, 20, 30, 32, 20, 18]
pres = [95, 105, 95, 105, 105, 95, 105, 95,  100, 105, 95, 100, 110, 115, 100, 95]
label = [0] * 12 + [1, 1, 0, 0]
X = np.column_stack([temp, pres]).astype(np.float64)
y = np.array(label)
print("X (16 x 2):\n", X, "\ny:", y)

# --------------------------------------------------------------- 2. split
head("2. Chronological split: rows 0-7 train, rows 8-15 test")
Xtr, Xte, yte = X[:8].copy(), X[8:].copy(), y[8:]

# --------------------------------------------------------------- 3. impute
head("3. Fill the missing temp at row 2 (linear interpolation)")
nan = np.isnan(Xtr[:, 0])
Xtr[nan, 0] = np.interp(np.flatnonzero(nan), np.flatnonzero(~nan), Xtr[~nan, 0])
print("temp row 2 ->", Xtr[2, 0])

# --------------------------------------------------------------- 4. scale
head("4. Standardise with TRAIN mean / std")
mu, sd = Xtr.mean(0), Xtr.std(0)          # population std, as sklearn StandardScaler
print("mu =", mu, " sigma =", sd)
Zte = (Xte - mu) / sd
print("test z-scores (8 x 2):\n", Zte)

# --------------------------------------------------------------- 5. windows
head("5. Windows on the test part: L = 4, stride 2")
L, S = 4, 2
starts = _window_starts(len(Zte), L, S)
print("starts:", starts)
W = np.stack([Zte[s:s + L] for s in starts])
for k, s in enumerate(starts):
    print(f"W{k}: rows {s}-{s + L - 1}, y_step {yte[s:s + L]}, y_win {yte[s:s + L].max()}")
    print("   ", W[k].T.tolist())

# --------------------------------------------------------------- 6. teacher
head("6. Teacher on W1")
w1 = W[1]
img = render_window(w1, style="heatmap")
print("clipped to [-4,4] then (z+4)/8 :\n", ((np.clip(w1, -4, 4) + 4) / 8).T)
print("image shape:", img.shape, "-> each cell is", 224 // 2, "px tall x", 224 // 4, "px wide")
logits_h, logits_s = np.array([21.0, 23.0]), np.array([22.0, 22.0])
logit = (logits_h + logits_s) / 2
p_T = np.exp(logit) / np.exp(logit).sum()
print("mean logits:", logit, " p_T =", p_T)

# --------------------------------------------------------------- 7. student
head("7. Student on W1")
h = np.array([[0.0, 0.0], [0.0, 0.2], [0.8, 0.4], [1.2, 0.6]])
u = np.r_[h.mean(0), h.max(0)]
print("pooled u = [mean, max] =", u)
Wh = np.array([[0, 0, -0.5, 0], [1, 0, 0, 0.5]])
bh = np.array([0.7, 0.1])
z = Wh @ u + bh
p_win = np.exp(z) / np.exp(z).sum()
print("window logits z =", z, " p_win =", p_win)
zt = 3 * h[:, 0] - 1
s_t = 1 / (1 + np.exp(-zt))
print("step logits =", zt, " s_t =", s_t)
g, e = np.array([1.0, 2, 2]), np.array([2.0, 1, 2])
cos = g @ e / np.linalg.norm(g) / np.linalg.norm(e)
print("cos(g, e) =", cos)

# --------------------------------------------------------------- 8. loss
head("8. Loss on W1 (pos_weight 3, T 2, beta 1, lambda_KD 1, lambda_feat 0.5)")
ys = yte[starts[1]:starts[1] + L].astype(float)
pw, T = 3.0, 2.0
bce = np.where(ys == 1, -pw * np.log(s_t), -np.log(1 - s_t))
print("per-step BCE:", bce, " L_step =", bce.mean())
L_win = -np.log(p_win[1])
print("L_win = -ln p_win[anom] =", L_win)
pt_soft = np.exp(logit / T) / np.exp(logit / T).sum()
ps_soft = np.exp(z / T) / np.exp(z / T).sum()
kl = float(np.sum(pt_soft * np.log(pt_soft / ps_soft)))
w_conf = p_T[1]
L_kd = T * T * w_conf * kl
print("teacher soft", pt_soft, " student soft", ps_soft, " KL =", kl, " w =", w_conf, " L_KD =", L_kd)
L_feat = 1 - cos
total = bce.mean() + L_win + L_kd + 0.5 * L_feat
print(f"total = {bce.mean():.4f} + {L_win:.4f} + {L_kd:.4f} + 0.5*{L_feat:.4f} = {total:.4f}")
loss_fn = TotalLoss(beta=1, lambda_kd=1, pos_weight=pw, temperature=T, lambda_feat=0.5, kd_confidence=True)
tot, parts = loss_fn(torch.tensor(zt[None], dtype=torch.float32), torch.tensor(z[None], dtype=torch.float32),
                     torch.tensor(ys[None]), torch.tensor([1]), teacher_probs=torch.tensor(p_T[None], dtype=torch.float32),
                     return_dict=True, student_embedding=torch.tensor(g[None], dtype=torch.float32),
                     teacher_embedding=torch.tensor(e[None], dtype=torch.float32))
print("project TotalLoss:", {k: round(v, 4) for k, v in parts.items()})
assert abs(parts["total_loss"] - total) < 1e-4

# --------------------------------------------------------------- 9. threshold
head("9. Window threshold from validation (6 windows)")
v_s = np.array([0.9, 0.7, 0.6, 0.4, 0.3, 0.1]); v_y = np.array([1, 1, 0, 1, 0, 0])
for tau in v_s:
    pred = v_s >= tau
    tp, fp, fn = int((pred & (v_y == 1)).sum()), int((pred & (v_y == 0)).sum()), int((~pred & (v_y == 1)).sum())
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    print(f"tau {tau:.1f}: TP {tp} FP {fp} FN {fn}  P {pr:.3f} R {rc:.3f} F1 {2 * pr * rc / (pr + rc):.3f}")
tau, _ = select_threshold(v_y, v_s, objective="f1")
print("project select_threshold ->", round(float(tau), 4))

# --------------------------------------------------------------- 10. window metrics
head("10. Window metrics on test (8 windows)")
t_s = np.array([0.95, 0.80, 0.55, 0.45, 0.25, 0.20, 0.15, 0.05]); t_y = np.array([1, 1, 0, 1, 1, 0, 0, 0])
pred = (t_s >= tau).astype(int)
print("pred:", pred, " labels:", t_y)
print({k: round(v, 4) for k, v in _binary_metrics(t_y, t_s, pred).items()})
print("AUPRC (sklearn):", round(average_precision_score(t_y, t_s), 4), " ROC-AUC:", round(roc_auc_score(t_y, t_s), 4))
print("with tau = 0.99:", {k: round(v, 4) for k, v in _binary_metrics(t_y, t_s, (t_s >= 0.99).astype(int)).items()})

# --------------------------------------------------------------- 11. timeline
head("11. Point level on the test sequence (3 windows -> 8 rows)")
step_w = np.array([[0.1, 0.6, 0.2, 0.2], np.round(s_t, 2), [0.9, 0.7, 0.1, 0.1]])
win_w = np.array([0.1, round(p_win[1], 2), 0.8])
step_tl = aggregate(step_w, starts, L, 8)
win_tl = aggregate(win_w, starts, L, 8)
print("step timeline :", step_tl)
print("window timeline:", win_tl)
print("gated (alpha=1):", step_tl * win_tl)
thr = 0.5
p0 = (step_tl >= thr).astype(np.int8)
p1 = remove_short(p0, 2)
for nm, p in (("threshold 0.5", p0), ("+ min_len 2", p1)):
    m = _binary_metrics(yte, step_tl, p)
    ev = event_scores([yte], [p])
    print(f"{nm}: pred {p}  runs {runs(p)}  point P {m['precision']:.3f} R {m['recall']:.3f} "
          f"F1 {m['f1']:.3f} | event P {ev['event_precision']:.3f} R {ev['event_recall']:.3f} F1 {ev['event_f1']:.3f}")
print("point AUPRC:", average_precision_score(yte, step_tl))
print("\nAll toy numbers reproduced with the project's functions.")
