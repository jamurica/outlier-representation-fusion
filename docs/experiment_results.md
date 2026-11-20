# Experiment results and runtimes

The run covered six classical ADBench datasets, seeds 42–46, three
detectors, and four test shift levels. The analysis combines 864 rows
from the seed-42 smoke run with 3456 rows from the four remaining seeds.
The [experiment protocol](experiment_protocol.md) defines the comparisons
and the [README](../README.md#analyze-the-experiments) explains how to
regenerate the analysis.

## Findings

### E1. Fusion versus the best single representation

At `delta=0`, all 18 detector–fusion combinations had negative mean
paired gains against the retrospectively best single representation.
Mean gains ranged from −0.133 to −0.014 for ROC-AUC and from −0.201 to
−0.026 for Average Precision. The highest strictly positive win counts
were 11 of 30 matched comparisons for ROC-AUC and 10 of 30 for Average
Precision. The comparator selects its best representation separately
for each metric, split, and detector using the test labels.

### E2. Effectiveness and stability

The table reports the strongest fusion by mean score and the most
stable fusion by mean within-dataset sample standard deviation at
`delta=0`. Each dataset has equal weight.

| Detector | Highest ROC-AUC | Lowest ROC-AUC seed SD | Highest Average Precision | Lowest Average Precision seed SD |
| --- | --- | --- | --- | --- |
| Isolation Forest | Maximum (0.950) | LSCP-AOM (0.011) | Mean (0.592) | LSCP-AOM (0.039) |
| LOF | Consensus Weighted Average (0.592) | LSCP-AOM (0.056) | Consensus Weighted Average (0.119) | Median (0.028) |
| One-Class SVM | Maximum (0.904) | Maximum (0.025) | Mean (0.506) | Maximum (0.055) |

### E3. Feature distribution shift

At `delta=2`, all 30 detector–method combinations had a positive mean
ROC-AUC drop, ranging from 0.015 to 0.313. Mean Average Precision drops
ranged from −0.012 to 0.099. A negative drop indicates improvement over
the matched unshifted test result.

### E4. Label-free indicators and ablation

Spearman associations varied by detector, method, and metric. For
example, the mean KS statistic had coefficients from −0.008 to 0.516
for ROC-AUC loss and from −0.228 to 0.416 for Average Precision loss.
These are descriptive correlations over repeated shift levels and
splits. The largest mean ablation decreases for both metrics occurred
when omitting UMAP from One-Class SVM Mean fusion at `delta=2`.
The changes were −0.048 ROC-AUC and −0.057 Average Precision.

The findings apply to the six datasets and the additive feature shifts
in this protocol.

## Runtimes

The runner measured elapsed wall-clock time for four phases. Each
measurement is repeated across the corresponding metric rows in the
combined CSV. The table counts each measurement once.

| Phase | Grouping key | Unique measurements | Median (s) | Sum (s) |
| --- | --- | ---: | ---: | ---: |
| Load, preprocess, and fit | dataset, seed | 30 | 6.86 | 197.86 |
| Score and prepare calibration | dataset, seed | 30 | 3.84 | 157.78 |
| Process shifted test data, including LSCP neighborhoods | dataset, seed, delta | 120 | 8.39 | 1258.74 |
| Fuse and evaluate | dataset, seed, delta, detector | 360 | 0.53 | 189.00 |

The unrounded phase measurements total 1803.36 seconds, or about
30 minutes across both runs. Test processing accounted for about 70%
of that total.
Times are shared across methods within each group. The measurements
exclude Python startup and CSV writing.

For each dataset–seed pair, add the fit and calibration durations, the
four test durations, and the twelve evaluation durations. Across the five
seeds, the resulting totals were:

| Dataset | Median (s) | Minimum (s) | Maximum (s) |
| --- | ---: | ---: | ---: |
| cardio | 13.41 | 12.49 | 27.92 |
| mammography | 152.38 | 146.01 | 176.94 |
| musk | 27.04 | 26.47 | 35.33 |
| optdigits | 58.58 | 56.46 | 64.10 |
| satimage-2 | 64.09 | 63.71 | 67.94 |
| thyroid | 29.69 | 28.53 | 34.54 |

## Environment

The local environment is Windows 11 on an AMD Ryzen 5 3600XT with
Python 3.13.4. The runner sets PyTorch to one CPU thread. Other libraries
use their configured thread settings.

| Package | Observed version |
| --- | --- |
| NumPy | 2.5.5 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| umap-learn | 0.6.0 |
| PyNNDescent | 0.7.0 |
| Numba | 0.68.1 |
| PyTorch | 2.15.0+cpu |
| pandas | 3.1.1 |
| Matplotlib | 3.11.2 |

## Result files

The intermediate inputs are `results/stage12_smoke.csv` and
`results/stage13_additional.csv`. The final outputs are in
[`results/analysis/`](../results/analysis/).

| Files | Contents |
| --- | --- |
| `combined_results.csv` | Row-level metrics, indicators, timing, and experiment metadata |
| `e1_paired_*.csv`, `e1_summary_*.csv` | Paired gains and win rates |
| `e2_by_dataset_*.csv`, `e2_summary_*.csv` | Effectiveness and seed stability |
| `e3_shift_*.csv` | Scores and paired decreases at all shift levels |
| `e4_associations_*.csv` | Spearman coefficients and valid-pair counts |
| `e4_ablation_paired_*.csv`, `e4_ablation_summary_*.csv` | Paired and mean differences from full Mean fusion |
| `*.png` | Figures for E1–E4, one version per metric |
