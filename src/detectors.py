"""Fit the project detectors and score unseen observations."""

from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


Detector = IsolationForest | LocalOutlierFactor | OneClassSVM


def fit_detectors(train: np.ndarray, *, seed: int) -> dict[str, Detector]:
    """Fit one model of each detector type for a training representation."""
    max_samples = 256
    if len(train) < max_samples:
        raise ValueError(
            f"Isolation Forest requires at least {max_samples} training rows "
            "to use the configured max_samples"
        )

    return {
        "Isolation Forest": IsolationForest(
            n_estimators=100, max_samples=max_samples, random_state=seed
        ).fit(train),
        "LOF": LocalOutlierFactor(n_neighbors=20, novelty=True).fit(train),
        "One-Class SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(train),
    }


def score_detectors(
    detectors: Mapping[str, Detector], features: np.ndarray
) -> dict[str, np.ndarray]:
    """Score calibration or test rows, with higher scores indicating anomalies."""
    return {
        name: -detector.score_samples(features)
        for name, detector in detectors.items()
    }
