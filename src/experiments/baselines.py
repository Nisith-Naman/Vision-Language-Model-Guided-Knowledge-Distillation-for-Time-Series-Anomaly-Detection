"""Established, non-VLM anomaly detectors used as reference points.

* Isolation Forest (Liu et al., 2008): unsupervised, fitted on training-window
  statistics; anomaly score = negative isolation depth.
* Gradient-boosted trees on window statistics (HistGradientBoosting): a strong
  supervised tabular baseline that sees the same labels as the GRU.

Both score whole windows; their window score is spread over the window's
steps for point/event evaluation, exactly like the teacher.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest


def window_features(X: np.ndarray) -> np.ndarray:
    """Per-channel mean, std, min, max, end-minus-start, and mean |diff|."""
    X = np.asarray(X, dtype=np.float32)
    diff = np.abs(np.diff(X, axis=1)).mean(axis=1)
    feats = [X.mean(1), X.std(1), X.min(1), X.max(1), X[:, -1] - X[:, 0], diff]
    return np.nan_to_num(np.concatenate(feats, axis=1))


def isolation_forest_scores(X_train, X_eval_list, seed: int = 0):
    model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(window_features(X_train))
    ref = -model.score_samples(window_features(X_train))
    centre = float(np.median(ref))
    scale = float(np.subtract(*np.percentile(ref, [75, 25]))) or 1.0
    # One monotone map fitted on training scores, so a validation threshold
    # means the same thing on test.
    return [1.0 / (1.0 + np.exp(-(-model.score_samples(window_features(X)) - centre) / scale))
            for X in X_eval_list]


def gbdt_scores(X_train, y_train, X_eval_list, seed: int = 0):
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed)
    model.fit(window_features(X_train), np.asarray(y_train).astype(int))
    return [model.predict_proba(window_features(X))[:, 1] for X in X_eval_list]
