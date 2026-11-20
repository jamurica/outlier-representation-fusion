"""Fit and reuse calibration-only score standardizers."""

from collections.abc import Mapping

import numpy as np
from sklearn.preprocessing import StandardScaler


def _score_column(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("Expected a one-dimensional score vector")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values")
    return scores[:, None]


def fit_score_scalers(
    calibration_scores: Mapping[str, np.ndarray],
) -> dict[str, StandardScaler]:
    """Fit one Z-score scaler per detector for a single representation."""
    return {
        detector: StandardScaler().fit(_score_column(scores))
        for detector, scores in calibration_scores.items()
    }


def normalize_scores(
    scores: Mapping[str, np.ndarray], scalers: Mapping[str, StandardScaler]
) -> dict[str, np.ndarray]:
    """Apply the fitted calibration scalers to the matching detector scores."""
    if scores.keys() != scalers.keys():
        raise ValueError("Score vectors and fitted scalers must have matching detector names")
    return {
        detector: scalers[detector].transform(_score_column(values))[:, 0]
        for detector, values in scores.items()
    }
