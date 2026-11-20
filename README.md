# Data Representation Fusion in Outlier Detection

A comparison of score combination methods for batch, unsupervised outlier
detection. See the [experiment protocol](docs/experiment_protocol.md).

## Environment setup on Windows

The local environment uses Python 3.13.4. From the repository root in
PowerShell, use the existing `.venv` or create it with a working Python 3.13
interpreter. If the launcher is unavailable, invoke the installed interpreter
by its full path with the same `-m venv .venv` arguments.

```powershell
# Run only when creating the environment.
py -3.13 -m venv .venv
```

In VS Code, select `.venv\Scripts\python.exe` using **Python: Select Interpreter**.
Run the following from the repository root when ready to install dependencies.
The PyTorch index supplies the CPU build.

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` lists direct dependencies. Record resolved versions after
validating the environment and before running the experiments.

## Repository layout

| Path | Purpose |
| --- | --- |
| `docs/` | Research protocol and methodology |
| `data/` | Local ADBench datasets |
| `src/` | ADBench loader, data splitting, and preprocessing |
| `scripts/` | Optional dataset downloader and future experiment entry points |
| `tests/` | Planned tests |
| `results/` | Local generated CSV files and figures |

## Datasets

Place these files in `data/`, or run the optional downloader from the
repository root in PowerShell after setting up the environment.
[ADBench repository](https://github.com/Minqi824/ADBench) ·
[Classical datasets](https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical)

- [6_cardio.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/6_cardio.npz)
- [23_mammography.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/23_mammography.npz)
- [25_musk.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/25_musk.npz)
- [26_optdigits.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/26_optdigits.npz)
- [31_satimage-2.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/31_satimage-2.npz)
- [38_thyroid.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/38_thyroid.npz)

```powershell
.\.venv\Scripts\python.exe -m scripts.download_adbench
```

The downloader checks existing archives and downloads only missing ones.
Use `--data-dir` to choose an alternative directory. Git ignores `data/`.

Load a dataset from Python with `load_dataset`. The loader validates that
`X` is a finite numeric feature matrix and `y` is a matching binary label
vector, where 0 is normal and 1 is an outlier.

```python
from src.adbench import load_dataset

X, y = load_dataset("6_cardio.npz")
```

## Split and standardize

Create one stratified split per dataset and seed, then reuse its indices
for every method. The two splitting steps allocate about 50% of all rows
to training and 30% to testing, with the remaining rows used for calibration.
The scaler fits only the training features and is available for
transforming new test inputs later.

```python
from src.adbench import load_dataset
from src.preprocessing import split_indices, standardize_features

X, y = load_dataset("6_cardio.npz")
indices = split_indices(y, seed=42)
features = standardize_features(X, indices)
y_test = y[indices.test]
```

## RAW, PCA, and UMAP

Fit the reduced representations once on standardized training rows, then
reuse the fitted object for each partition. RAW returns the standardized
input in its original dimension. UMAP fits on unique training vectors
and maps the coordinates back to all training rows. Record the installed
UMAP version with experiment results because its default layout can change
between releases.

```python
from src.representations import fit_representations

representations = fit_representations(features.train, seed=42)
train_views = representations.transform(features.train)
calibration_views = representations.transform(features.calibration)
test_views = representations.transform(features.test)
```

## Autoencoder representation

Fit the autoencoder once on standardized training rows and reuse its
encoder for each partition.

```python
from src.autoencoder import fit_autoencoder

autoencoder = fit_autoencoder(features.train, seed=42)
train_views["AE"] = autoencoder.transform(features.train)
calibration_views["AE"] = autoencoder.transform(features.calibration)
test_views["AE"] = autoencoder.transform(features.test)
```

## Outlier detectors

Fit a separate detector set for each training representation. Reuse the
fitted models for calibration and test scores. The returned scores are
continuous, with larger scores indicating greater anomalousness.

```python
from src.detectors import fit_detectors, score_detectors

detectors = fit_detectors(train_views["RAW"], seed=42)
calibration_scores = score_detectors(detectors, calibration_views["RAW"])
test_scores = score_detectors(detectors, test_views["RAW"])
```

## Score normalization

For each representation, fit one Z-score scaler per detector using its
calibration scores. Reuse the same scalers for the calibration and test
scores, including every test shift level.

```python
from src.normalization import fit_score_scalers, normalize_scores

scalers = fit_score_scalers(calibration_scores)
normalized_calibration = normalize_scores(calibration_scores, scalers)
normalized_test = normalize_scores(test_scores, scalers)
```

## Basic score fusion

For one detector, stack its four representation score vectors as columns
in RAW, PCA, UMAP, AE order. Keep the rows aligned across representations
within each partition. Fit the percentile references and consensus
weights from raw calibration scores.

```python
from src.fusion import fit_basic_fusion

fusion = fit_basic_fusion(raw_calibration_matrix)
fused_test = fusion.transform(raw_test_matrix, normalized_test_matrix)
```

Reuse the fitted fusion for each test shift level. It returns Mean,
Maximum, Median, Rank Average, and Consensus Weighted Average.

## LSCP-AOM

Compute local regions once for each test partition and reuse them for all
detectors. Fit the pseudo target separately for each detector using
its calibration Z-scores in RAW, PCA, UMAP, AE column order.

```python
from src.lscp import find_local_regions, fit_lscp_aom, sample_subspaces

subspaces = sample_subspaces(features.train.shape[1], seed=42)
regions = find_local_regions(features.calibration, features.test, subspaces)
lscp = fit_lscp_aom(normalized_calibration_matrix)
lscp_scores = lscp.transform(normalized_test_matrix, regions)
```

## Distribution shift

Sample the shift once per dataset and seed from the training scaler's
feature variances. Apply each level to the original test rows, then use
the same scaler to standardize the shifted copy.

```python
from src.distribution_shift import SHIFT_LEVELS, sample_feature_shift

shift = sample_feature_shift(features.scaler.var_, seed=42)
for delta in SHIFT_LEVELS:
    shifted_test = shift.transform(X[indices.test], delta=delta)
    standardized_test = features.scaler.transform(shifted_test)
```

## Evaluation and ablation

Use the original test labels to compute ROC-AUC and Average Precision.
At `delta=0` and `delta=2`, compare full Mean fusion with the four
leave-one-representation-out means.

```python
from src.evaluation import evaluate_scores, mean_ablation

full_mean_metrics = evaluate_scores(y_test, normalized_test_matrix.mean(axis=1))
ablated_metrics = {
    omitted: evaluate_scores(y_test, scores)
    for omitted, scores in mean_ablation(normalized_test_matrix).items()
}
```

The label-free functions compare test features with calibration using KS
and compare representation score vectors using Spearman correlation.

## Citation

Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022).
*ADBench: Anomaly Detection Benchmark*. Advances in Neural Information
Processing Systems, 35. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf93972b116ca5268827d575f2cc226b-Abstract-Datasets_and_Benchmarks.html)
