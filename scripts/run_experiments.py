"""Run the batch outlier detection experiments and save results to CSV."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from src.adbench import DATASET_FILES, DEFAULT_DATA_DIR, load_dataset
from src.autoencoder import fit_autoencoder
from src.detectors import fit_detectors, score_detectors
from src.distribution_shift import SHIFT_LEVELS, sample_feature_shift
from src.evaluation import (
    agreement_change,
    evaluate_scores,
    mean_ablation,
    mean_feature_ks,
    mean_score_agreement,
    performance_drop,
)
from src.fusion import fit_basic_fusion
from src.lscp import find_local_regions, fit_lscp_aom, sample_subspaces
from src.normalization import fit_score_scalers, normalize_scores
from src.preprocessing import split_indices, standardize_features
from src.representations import fit_representations, reduced_dimension


REPRESENTATIONS = ("RAW", "PCA", "UMAP", "AE")
DEFAULT_OUTPUT = DEFAULT_DATA_DIR.parent / "results" / "experiment_results.csv"


def _detector_matrix(scores, detector):
    return np.column_stack([scores[view][detector] for view in REPRESENTATIONS])


def run_repetition(dataset: str, seed: int, data_dir: Path) -> list[dict]:
    """Run all shift levels using a single split and fitted model set."""
    started = perf_counter()
    x, y = load_dataset(dataset, data_dir)
    indices = split_indices(y, seed=seed)
    features = standardize_features(x, indices)
    representations = fit_representations(features.train, seed=seed)
    autoencoder = fit_autoencoder(features.train, seed=seed)

    def encode(rows):
        return representations.transform(rows) | {"AE": autoencoder.transform(rows)}

    models = {
        view: fit_detectors(rows, seed=seed)
        for view, rows in encode(features.train).items()
    }
    fit_seconds = perf_counter() - started

    started = perf_counter()
    raw_calibration = {
        view: score_detectors(models[view], rows)
        for view, rows in encode(features.calibration).items()
    }
    scalers = {
        view: fit_score_scalers(scores) for view, scores in raw_calibration.items()
    }
    normalized_calibration = {
        view: normalize_scores(scores, scalers[view])
        for view, scores in raw_calibration.items()
    }
    basic_fusion, lscp, calibration_agreement = {}, {}, {}
    for detector in models["RAW"]:
        raw = _detector_matrix(raw_calibration, detector)
        normalized = _detector_matrix(normalized_calibration, detector)
        basic_fusion[detector] = fit_basic_fusion(raw)
        lscp[detector] = fit_lscp_aom(normalized)
        calibration_agreement[detector] = mean_score_agreement(raw)

    shift = sample_feature_shift(features.scaler.var_, seed=seed)
    subspaces = sample_subspaces(x.shape[1], seed=seed)
    calibration_seconds = perf_counter() - started
    original_test = x[indices.test]
    labels = y[indices.test]
    shared = {
        "dataset": dataset,
        "seed": seed,
        "n_train": len(indices.train),
        "n_calibration": len(indices.calibration),
        "n_test": len(indices.test),
        "n_features": x.shape[1],
        "reduced_dimension": reduced_dimension(x.shape[1]),
        "shift_columns": json.dumps(shift.columns.tolist()),
        "shift_signed_std": json.dumps(shift.signed_std.tolist()),
        "lscp_subspaces": json.dumps([columns.tolist() for columns in subspaces]),
        "fit_seconds": fit_seconds,
        "calibration_seconds": calibration_seconds,
    }
    detector_parameters = {
        name: json.dumps(model.get_params(), sort_keys=True)
        for name, model in models["RAW"].items()
    }
    baseline = {}
    results = []
    for delta in SHIFT_LEVELS:
        started = perf_counter()
        test = features.scaler.transform(shift.transform(original_test, delta=delta))
        try:
            test_views = encode(test)
        except ValueError as error:
            raise ValueError(f"{dataset}, seed={seed}, delta={delta}: {error}") from error
        raw_test = {
            view: score_detectors(models[view], rows) for view, rows in test_views.items()
        }
        normalized_test = {
            view: normalize_scores(scores, scalers[view])
            for view, scores in raw_test.items()
        }
        regions = find_local_regions(features.calibration, test, subspaces)
        ks = mean_feature_ks(features.calibration, test)
        test_seconds = perf_counter() - started

        for detector in models["RAW"]:
            started = perf_counter()
            raw = _detector_matrix(raw_test, detector)
            normalized = _detector_matrix(normalized_test, detector)
            agreement = mean_score_agreement(raw)
            fused = basic_fusion[detector].transform(raw, normalized)
            fused["LSCP-AOM"] = lscp[detector].transform(normalized, regions)
            methods = {
                (view, "", ""): raw[:, index]
                for index, view in enumerate(REPRESENTATIONS)
            }
            methods.update({("All", rule, ""): scores for rule, scores in fused.items()})
            if delta in (0.0, 2.0):
                methods.update({
                    ("All", "Mean", omitted): scores
                    for omitted, scores in mean_ablation(normalized).items()
                })
            evaluated = {
                method: evaluate_scores(labels, scores) for method, scores in methods.items()
            }
            evaluation_seconds = perf_counter() - started

            for method, metrics in evaluated.items():
                view, rule, omitted = method
                key = (detector, method)
                if delta == 0.0:
                    baseline[key] = metrics
                drop = performance_drop(baseline[key], metrics)
                results.append({
                    **shared,
                    "delta": delta,
                    "detector": detector,
                    "detector_parameters": detector_parameters[detector],
                    "representation": view,
                    "fusion_rule": rule,
                    "omitted_representation": omitted,
                    **metrics,
                    "roc_auc_drop": drop["roc_auc"],
                    "average_precision_drop": drop["average_precision"],
                    "mean_feature_ks": ks,
                    "test_score_agreement": agreement,
                    "calibration_score_agreement": calibration_agreement[detector],
                    "agreement_change": agreement_change(
                        calibration_agreement[detector], agreement
                    ),
                    "test_seconds": test_seconds,
                    "evaluation_seconds": evaluation_seconds,
                })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_FILES, default=DATASET_FILES)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not all(0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be distinct integers from 0 to 2**32 - 1")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("datasets must be distinct")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    torch.set_num_threads(1)
    results = []
    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"Running {dataset}, seed={seed}", flush=True)
            results.extend(run_repetition(dataset, seed, args.data_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False, mode="x")
    print(f"Saved {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
