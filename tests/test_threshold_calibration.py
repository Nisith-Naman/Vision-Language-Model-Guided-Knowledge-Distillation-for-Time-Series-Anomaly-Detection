"""Threshold selection: the vectorised sweep equals scikit-learn's metrics."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.evaluation.threshold import select_threshold


def test_vectorised_sweep_matches_sklearn():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.07).astype(int)
    s = np.clip(rng.normal(0.02 + 0.06 * y, 0.03), 0, 1)
    _, info = select_threshold(y, s, objective="f1")
    for th, p, r, f in list(zip(info["threshold_array"], info["precision_array"],
                                info["recall_array"], info["f1_array"]))[::7]:
        pred = (s >= th).astype(int)
        assert np.isclose(p, precision_score(y, pred, zero_division=0.0))
        assert np.isclose(r, recall_score(y, pred, zero_division=0.0))
        assert np.isclose(f, f1_score(y, pred, zero_division=0.0))


def test_low_probability_scores_get_a_low_threshold():
    """Anomaly probabilities all below 0.5: the tuned threshold must flag anomalies."""
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.07).astype(int)
    s = np.clip(rng.normal(0.02 + 0.08 * y, 0.02), 0, 0.3)
    th, _ = select_threshold(y, s, objective="f1")
    assert th < 0.3
    assert f1_score(y, (s >= th).astype(int)) > 0.5
