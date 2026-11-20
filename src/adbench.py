"""Load and validate the classical ADBench datasets used in this project."""

from pathlib import Path
from zipfile import BadZipFile

import numpy as np


DATASET_FILES = (
    "6_cardio.npz",
    "23_mammography.npz",
    "25_musk.npz",
    "26_optdigits.npz",
    "31_satimage-2.npz",
    "38_thyroid.npz",
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_dataset(
    filename: str, data_dir: str | Path = DEFAULT_DATA_DIR
) -> tuple[np.ndarray, np.ndarray]:
    """Return the feature matrix and binary label vector for a project dataset."""
    if filename not in DATASET_FILES:
        raise ValueError(f"Unknown ADBench dataset: {filename}. Choose from {DATASET_FILES}.")

    path = Path(data_dir) / filename
    try:
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("expected an NPZ archive")
        with archive:
            if not {"X", "y"}.issubset(archive.files):
                raise ValueError("the archive must contain X and y")
            features = archive["X"]
            labels = archive["y"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {path}") from None
    except (ValueError, BadZipFile, EOFError) as error:
        raise ValueError(f"{path}: {error}") from error

    if features.ndim != 2 or features.size == 0:
        raise ValueError(f"{path}: X must be a nonempty two-dimensional matrix")
    if features.dtype.kind not in "iuf" or not np.isfinite(features).all():
        raise ValueError(f"{path}: X must contain finite real numbers")
    if labels.shape != (features.shape[0],):
        raise ValueError(f"{path}: y must be a one-dimensional vector matching the rows of X")
    if labels.dtype.kind not in "biuf" or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{path}: y must contain only binary labels (0 or 1)")

    return features, labels.astype(np.int64, copy=False)
