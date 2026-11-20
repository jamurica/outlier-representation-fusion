"""Evaluate anomaly scores, label-free indicators, and representation ablations."""

from collections.abc import Mapping
from itertools import combinations

import numpy as np
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .fusion import _score_matrix


def evaluate_scores(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute both metrics with outliers (label 1) as the positive class."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Expected matching one-dimensional labels and scores")
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Evaluation requires both binary label classes")
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def mean_ablation(normalized_scores: np.ndarray) -> dict[str, np.ndarray]:
    """Return three-component means keyed by the omitted representation."""
    scores = _score_matrix(normalized_scores)
    return {
        representation: np.delete(scores, index, axis=1).mean(axis=1)
        for index, representation in enumerate(("RAW", "PCA", "UMAP", "AE"))
    }


def mean_feature_ks(calibration: np.ndarray, test: np.ndarray) -> float:
    """Average two-sided KS statistics over aligned feature columns."""
    calibration = np.asarray(calibration)
    test = np.asarray(test)
    if (
        calibration.ndim != 2 or test.ndim != 2
        or calibration.size == 0 or test.size == 0
        or calibration.shape[1] != test.shape[1]
        or not np.isfinite(calibration).all() or not np.isfinite(test).all()
    ):
        raise ValueError("Expected nonempty, finite feature matrices with matching columns")
    statistics = ks_2samp(calibration, test, axis=0, method="asymp").statistic
    return float(statistics.mean())


def mean_score_agreement(scores: np.ndarray) -> float | None:
    """Average defined Spearman correlations across representation pairs."""
    scores = _score_matrix(scores)
    varying = np.flatnonzero(np.any(scores != scores[0], axis=0))
    correlations = [
        spearmanr(scores[:, first], scores[:, second]).statistic
        for first, second in combinations(varying, 2)
    ]
    return float(np.mean(correlations)) if correlations else None


def agreement_change(
    calibration_agreement: float | None, test_agreement: float | None
) -> float | None:
    """Return test minus calibration agreement when both values are defined."""
    if calibration_agreement is None or test_agreement is None:
        return None
    return test_agreement - calibration_agreement


def performance_drop(
    baseline: Mapping[str, float], current: Mapping[str, float]
) -> dict[str, float]:
    """Subtract current metrics from the matched unshifted baseline."""
    if baseline.keys() != current.keys():
        raise ValueError("Baseline and current results must have matching metric names")
    return {metric: value - current[metric] for metric, value in baseline.items()}
