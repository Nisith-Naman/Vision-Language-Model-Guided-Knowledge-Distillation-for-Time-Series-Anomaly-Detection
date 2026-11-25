"""Linear-probe teacher on top of the frozen CLIP image encoder.

Zero-shot CLIP compares a rendered chart with two text prompts. On the
project's datasets this was at or below chance (Daphnet test ROC-AUC 0.25), so
its soft labels could not carry useful knowledge. The standard way to adapt a
frozen vision-language model without touching its weights is a *linear probe*:
a logistic-regression head (513 parameters for ViT-B/32) trained on the frozen
512-d image embeddings. All CLIP weights stay frozen.

Cross-fitting: if the probe were trained on all training windows and then asked
to label those same windows, its "soft" labels would largely reproduce the hard
ground truth. Instead the (time-ordered) training windows are cut into blocks of
``block_size`` consecutive windows, assigned to K folds round-robin; each fold
is labelled by a probe trained on the other K-1 folds (out-of-fold
predictions). Blocks keep near-duplicate overlapping windows mostly in one fold,
while round-robin assignment spreads a long anomaly over several folds. (v2.0
used K contiguous folds, which left the single MetroPT training failure inside
one fold, so that fold's probe had never seen a positive.) These OOF probabilities are the KD targets for the student. A
final probe fitted on all training windows labels validation/test windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _block_folds(n: int, n_folds: int, block_size: int) -> List[np.ndarray]:
    fold_of = (np.arange(n) // max(1, block_size)) % n_folds
    return [np.flatnonzero(fold_of == k) for k in range(n_folds)]


def _make_model(C: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000),
    )


@dataclass
class LinearProbeTeacher:
    """Logistic-regression probe on frozen CLIP embeddings (cross-fitted)."""

    C_grid: Sequence[float] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    n_folds: int = 5
    block_size: int = 32
    C_: Optional[float] = None
    cv_auprc_: Dict[float, float] = field(default_factory=dict)
    model_: object = None

    def _fold_scores(self, emb: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        prior = float(np.mean(y))
        for idx in _block_folds(len(y), self.n_folds, self.block_size):
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[idx] = False
            y_tr = y[train_mask]
            if y_tr.min() == y_tr.max():
                oof[idx] = prior
                continue
            model = _make_model(C).fit(emb[train_mask], y_tr)
            oof[idx] = model.predict_proba(emb[idx])[:, 1]
        return oof

    def fit(self, emb: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select C by out-of-fold AUPRC, fit the final probe, and return the
        out-of-fold anomaly probabilities for the training windows."""
        emb = np.asarray(emb, dtype=np.float32)
        y = np.asarray(y).astype(int).ravel()
        if y.min() == y.max():
            raise ValueError("Linear probe needs both classes in the training windows.")
        best_oof, best_score = None, -1.0
        for C in self.C_grid:
            oof = self._fold_scores(emb, y, C)
            score = float(average_precision_score(y, oof))
            self.cv_auprc_[float(C)] = score
            if score > best_score:
                best_score, best_oof, self.C_ = score, oof, float(C)
        self.model_ = _make_model(self.C_).fit(emb, y)
        return np.stack([1.0 - best_oof, best_oof], axis=1).astype(np.float32)

    def predict_proba(self, emb: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("LinearProbeTeacher is not fitted.")
        p = self.model_.predict_proba(np.asarray(emb, dtype=np.float32))[:, 1]
        return np.stack([1.0 - p, p], axis=1).astype(np.float32)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "LinearProbeTeacher":
        return joblib.load(path)
