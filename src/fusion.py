"""Calibration-based fusion of four representation scores for one detector."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr


def _score_matrix(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 4 or scores.shape[0] == 0:
        raise ValueError("Expected a nonempty score matrix with four representation columns")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values")
    return scores


@dataclass(frozen=True)
class BasicFusion:
    """Calibration parameters for Rank Average and Consensus Weighted Average."""

    sorted_calibration: np.ndarray
    weights: np.ndarray

    def transform(
        self, raw_scores: np.ndarray, normalized_scores: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Fuse aligned test scores in RAW, PCA, UMAP, AE column order."""
        raw = _score_matrix(raw_scores)
        normalized = _score_matrix(normalized_scores)
        if raw.shape != normalized.shape:
            raise ValueError("Raw and normalized score matrices must have the same shape")

        percentiles = np.empty_like(raw)
        for column, reference in enumerate(self.sorted_calibration.T):
            left = np.searchsorted(reference, raw[:, column], side="left")
            right = np.searchsorted(reference, raw[:, column], side="right")
            percentiles[:, column] = (left + right) / (2 * reference.size)

        return {
            "Mean": normalized.mean(axis=1),
            "Maximum": normalized.max(axis=1),
            "Median": np.median(normalized, axis=1),
            "Rank Average": percentiles.mean(axis=1),
            "Consensus Weighted Average": normalized @ self.weights,
        }


def fit_basic_fusion(calibration_scores: np.ndarray) -> BasicFusion:
    """Learn rank references and consensus weights from raw calibration scores."""
    scores = _score_matrix(calibration_scores)
    correlations = np.zeros((4, 4))
    varying = np.flatnonzero(np.any(scores != scores[0], axis=0))
    for first, second in combinations(varying, 2):
        correlation = spearmanr(scores[:, first], scores[:, second]).statistic
        if np.isfinite(correlation):
            correlations[first, second] = correlations[second, first] = correlation

    weights = np.maximum(correlations.sum(axis=1) / 3, 0)
    weight_sum = weights.sum()
    weights = weights / weight_sum if weight_sum > 0 else np.full(4, 0.25)
    return BasicFusion(np.sort(scores, axis=0), weights)
