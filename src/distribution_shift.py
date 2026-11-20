"""Generate fixed feature shifts for a dataset and repetition."""

from dataclasses import dataclass

import numpy as np


SHIFT_LEVELS = (0.0, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class FeatureShift:
    """Selected columns and signed training standard deviations."""

    n_features: int
    columns: np.ndarray
    signed_std: np.ndarray

    def transform(self, test: np.ndarray, *, delta: float) -> np.ndarray:
        """Apply a shift to a copy of the original, unstandardized test rows."""
        if delta not in SHIFT_LEVELS:
            raise ValueError(f"Expected delta in {SHIFT_LEVELS}")
        shifted = np.array(test, dtype=np.float64, copy=True)
        if shifted.ndim != 2 or shifted.shape[1] != self.n_features:
            raise ValueError("Test feature count must match the training variance vector")
        shifted[:, self.columns] += delta * self.signed_std
        return shifted


def sample_feature_shift(train_variance: np.ndarray, *, seed: int) -> FeatureShift:
    """Sample columns and signs using the fitted training variances."""
    variance = np.asarray(train_variance, dtype=np.float64)
    if variance.ndim != 1 or not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError("Expected a vector of finite, nonnegative training variances")

    eligible = np.flatnonzero(variance > 0)
    # Round one quarter of the eligible count to the nearest integer, half up.
    count = (eligible.size + 2) // 4
    rng = np.random.default_rng(seed)
    columns = np.sort(rng.choice(eligible, size=count, replace=False))
    signs = rng.choice([-1, 1], size=count)
    return FeatureShift(variance.size, columns, signs * np.sqrt(variance[columns]))
