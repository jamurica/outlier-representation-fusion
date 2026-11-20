"""Local selection of representation scores using a fixed calibration set."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

from .fusion import _score_matrix


def sample_subspaces(n_features: int, *, seed: int) -> tuple[np.ndarray, ...]:
    """Draw 20 feature subsets for a dataset and repetition."""
    if n_features < 1:
        raise ValueError("Expected at least one feature")
    rng = np.random.default_rng(seed)
    return tuple(
        np.sort(rng.choice(n_features, size=size, replace=False))
        for size in rng.integers((n_features + 1) // 2, n_features + 1, size=20)
    )


def find_local_regions(
    calibration: np.ndarray, test: np.ndarray, subspaces: tuple[np.ndarray, ...]
) -> list[np.ndarray]:
    """Find one calibration neighborhood per standardized test row."""
    if (
        calibration.ndim != 2 or test.ndim != 2
        or calibration.shape[1] != test.shape[1]
        or len(calibration) == 0
        or not np.isfinite(calibration).all() or not np.isfinite(test).all()
    ):
        raise ValueError("Expected finite calibration and test matrices with matching features")
    if len(subspaces) != 20:
        raise ValueError("Expected 20 feature subspaces")

    n_neighbors = min(30, len(calibration))
    calibration_views = [calibration[:, columns] for columns in subspaces]
    regions = []
    for row in test:
        votes = np.zeros(len(calibration), dtype=int)
        for columns, reference in zip(subspaces, calibration_views):
            distances = cdist(row[None, columns], reference, metric="euclidean")[0]
            neighbors = np.argsort(distances, kind="stable")[:n_neighbors]
            votes[neighbors] += 1
        region = np.flatnonzero(votes >= 11)
        if len(region) < 3:
            distances = cdist(row[None, :], calibration, metric="euclidean")[0]
            region = np.argsort(distances, kind="stable")[:n_neighbors]
        regions.append(region)
    return regions


@dataclass(frozen=True)
class LSCPAOM:
    """Fixed calibration scores and pseudo target for one detector."""

    calibration_scores: np.ndarray
    pseudo_target: np.ndarray

    def transform(self, test_scores: np.ndarray, regions: list[np.ndarray]) -> np.ndarray:
        """Average the locally selected Z-scores for each test observation."""
        scores = _score_matrix(test_scores)
        if len(regions) != len(scores):
            raise ValueError("Expected one local region per test score row")

        fused = scores.mean(axis=1)
        for index, region in enumerate(regions):
            if np.any(region < 0) or np.any(region >= len(self.calibration_scores)):
                raise ValueError("Local region contains an invalid calibration row index")
            if len(region) < 3:
                continue
            target = self.pseudo_target[region]
            if np.all(target == target[0]):
                continue
            local_scores = self.calibration_scores[region]
            varying = np.flatnonzero(np.any(local_scores != local_scores[0], axis=0))
            if varying.size == 0:
                continue

            correlations = pearsonr(local_scores[:, varying], target[:, None], axis=0).statistic
            valid = np.isfinite(correlations)
            correlations = correlations[valid]
            if correlations.size:
                cutoff = np.sort(correlations)[-min(2, correlations.size)]
                selected = varying[valid][correlations >= cutoff]
                fused[index] = scores[index, selected].mean()
        return fused


def fit_lscp_aom(calibration_scores: np.ndarray) -> LSCPAOM:
    """Build the pseudo target from four calibration Z-score columns."""
    scores = _score_matrix(calibration_scores).copy()
    return LSCPAOM(scores, scores.max(axis=1))
