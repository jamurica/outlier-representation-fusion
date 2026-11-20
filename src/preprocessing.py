"""Shared stratified splits and training-only feature standardization."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SplitIndices:
    """Row indices shared by all methods for a dataset and seed."""

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


def split_indices(labels: np.ndarray, *, seed: int) -> SplitIndices:
    """Split binary labels into stratified 50/20/30 partitions by row index."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Stratified splitting requires a vector with both binary classes")

    train, remaining = train_test_split(
        np.arange(labels.size),
        test_size=0.5,
        random_state=seed,
        stratify=labels,
    )
    calibration, test = train_test_split(
        remaining,
        test_size=round(0.3 * labels.size),
        random_state=seed,
        stratify=labels[remaining],
    )
    for partition in (train, calibration, test):
        if np.unique(labels[partition]).size != 2:
            raise ValueError("Each partition must contain both classes")
    return SplitIndices(train, calibration, test)


@dataclass(frozen=True)
class StandardizedFeatures:
    """Standardized partitions and the fitted training scaler."""

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    scaler: StandardScaler


def standardize_features(
    features: np.ndarray, indices: SplitIndices
) -> StandardizedFeatures:
    """Fit on training features and apply the same transform to all partitions."""
    train = features[indices.train]
    scaler = StandardScaler().fit(train)
    return StandardizedFeatures(
        train=scaler.transform(train),
        calibration=scaler.transform(features[indices.calibration]),
        test=scaler.transform(features[indices.test]),
        scaler=scaler,
    )
