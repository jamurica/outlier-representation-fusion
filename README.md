# Data Representation Fusion in Outlier Detection

A comparison of score combination methods for batch, unsupervised outlier
detection on six classical ADBench datasets. See the
[experiment protocol](docs/experiment_protocol.md) for the methodology
and the [results and runtimes](docs/experiment_results.md) for the
five-seed findings.

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

`requirements.txt` lists direct dependencies. The
[recorded environment](docs/experiment_results.md#environment) provides
versions for comparison when reproducing the reported run.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`docs/`](docs/) | Experimental protocol and recorded findings |
| [`src/`](src/) | Data loading, preprocessing, representations, detection, fusion, shift, and evaluation |
| [`scripts/`](scripts/) | Dataset downloader, experiment runner, and analyzer |
| `data/` | Local ADBench archives, ignored by Git |
| `results/` | Local result CSVs and figures, ignored by Git |

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
Each archive contains a finite feature matrix `X` and binary labels `y`
with 0 for normal observations and 1 for outliers.

## Experimental workflow

Each dataset and seed uses one stratified 50/20/30 train, calibration, and
test split. Training rows fit the standardizer, PCA, UMAP, autoencoder,
and detector models. RAW uses the standardized original features.
Calibration scores determine
normalization and the fitted fusion settings. The runner reuses these
models and settings across all four test shift levels. Labels are used
for stratification and evaluation.

The [runner](scripts/run_experiments.py) shows how the reusable modules
fit together. The [protocol](docs/experiment_protocol.md) specifies the
model settings, LSCP-AOM adaptation, shift generator, and comparisons.

## Run the experiments

Run the one-seed smoke test on all six datasets from the repository root.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiments --seeds 42 --output results/smoke.csv
```

Use `--datasets` to select one or more ADBench archive names and
`--data-dir` for an alternative dataset directory. For the main run,
use the smoke run as seed 42 and run seeds 43 through 46 as shown below.
The runner uses one PyTorch CPU thread and refuses to overwrite
an existing output file.

The CSV contains one row per dataset, seed, shift level, detector and
method. It includes the four individual representations, six fusion rules,
and four Mean ablations at `delta=0` and `delta=2`. The best individual
representation can be identified from the matching rows separately for
each metric. A value in `omitted_representation` identifies an ablation.
Missing agreement values appear as blank fields.

Timing columns report shared wall-clock durations in seconds. Fit and
calibration timings apply to a dataset and seed, test timing also depends
on the shift level, and evaluation timing additionally depends on the
detector. Use one value per timing group when aggregating durations.

## Analyze the experiments

The approved seeds are 42 through 46. To reuse the smoke results for seed
42, run the four remaining seeds and pass both result files to the analyzer.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiments --seeds 43 44 45 46 --output results/stage13_additional.csv
.\.venv\Scripts\python.exe -m scripts.analyze_experiments results/smoke.csv results/stage13_additional.csv --output-dir results/analysis
```

Use the path of your actual smoke CSV as the first input. The analyzer
writes a combined results CSV, paired and aggregated E1–E4 tables, and
PNG plots for both metrics. Use a new output directory for each run.
A single-seed analysis leaves the seed standard deviation blank.

## Citation

Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022).
*ADBench: Anomaly Detection Benchmark*. Advances in Neural Information
Processing Systems, 35. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf93972b116ca5268827d575f2cc226b-Abstract-Datasets_and_Benchmarks.html)

Zhao, Y., Nasrullah, Z., Hryniewicki, M. K., & Li, Z. (2019).
*LSCP: Locally Selective Combination in Parallel Outlier Ensembles*.
Proceedings of the SIAM International Conference on Data Mining.
[Published paper](https://doi.org/10.1137/1.9781611975673.66)
