"""RAW, PCA, and UMAP representations of standardized features."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP


def reduced_dimension(n_features: int) -> int:
    """Return the shared reduced dimension from the experiment protocol."""
    return max(2, min(20, n_features // 2))


@dataclass(frozen=True)
class FittedRepresentations:
    """Transform standardized rows using the fitted training representations."""

    pca: PCA
    umap: UMAP

    def transform(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Return RAW, PCA, and UMAP arrays in the input row order."""
        pca_features = self.pca.transform(features)
        umap_features = self.umap.transform(features)
        invalid_rows = np.count_nonzero(~np.isfinite(umap_features).all(axis=1))
        if invalid_rows:
            raise ValueError(f"UMAP produced non-finite values in {invalid_rows} rows")
        return {
            "RAW": features,
            "PCA": pca_features,
            "UMAP": umap_features,
        }


def fit_representations(
    train: np.ndarray, *, seed: int
) -> FittedRepresentations:
    """Fit PCA and UMAP using only standardized training rows."""
    n_rows, n_features = train.shape
    n_components = reduced_dimension(n_features)
    if n_components > min(n_rows, n_features):
        raise ValueError(
            f"Training shape {train.shape} cannot support PCA with "
            f"{n_components} components"
        )

    n_neighbors = 15
    if n_rows <= max(n_neighbors, n_components + 1):
        raise ValueError("Too few training rows for the configured UMAP neighborhood or embedding")

    pca = PCA(n_components=n_components, random_state=seed).fit(train)
    umap = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    ).fit(train)
    return FittedRepresentations(pca=pca, umap=umap)
