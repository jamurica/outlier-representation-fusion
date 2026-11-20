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
| `src/` | Planned experiment modules |
| `scripts/` | Planned optional dataset downloader and experiment entry points |
| `tests/` | Planned tests |
| `results/` | Local generated CSV files and figures |

## Datasets

Place these files in `data/`. Each archive is expected to contain `X` (features)
and `y` (binary labels, where 0 is normal and 1 is an outlier). File validation
and an optional downloader are planned for the loader stage.
[ADBench repository](https://github.com/Minqi824/ADBench) ·
[Classical datasets](https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical)

- [6_cardio.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/6_cardio.npz)
- [23_mammography.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/23_mammography.npz)
- [25_musk.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/25_musk.npz)
- [26_optdigits.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/26_optdigits.npz)
- [31_satimage-2.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/31_satimage-2.npz)
- [38_thyroid.npz](https://github.com/Minqi824/ADBench/blob/main/adbench/datasets/Classical/38_thyroid.npz)

## Citation

Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022).
*ADBench: Anomaly Detection Benchmark*. Advances in Neural Information
Processing Systems, 35. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf93972b116ca5268827d575f2cc226b-Abstract-Datasets_and_Benchmarks.html)
