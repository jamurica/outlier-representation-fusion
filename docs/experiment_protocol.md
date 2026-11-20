# Experiment protocol

## Research questions

- **P1.** Does score fusion improve batch, unsupervised outlier detection
  relative to the best single representation?
- **P2.** Which fusion rule is most effective and most stable?
- **P3.** How does feature distribution shift affect single representations
  and fusion, and are label-free indicators associated with performance loss?

## Data and splits

Use the six classical ADBench archives listed in the [README](../README.md).
The loader will validate local archives once supplied. Each archive must
contain a feature matrix `X` and a matching binary label vector `y`, with
0 for normal observations and 1 for outliers.

For each dataset and seed, create one stratified split with 50% training,
20% calibration, and 30% test observations. Reuse the same indices for all
methods and shift levels. Use labels only for stratification and evaluation.
Fit the feature standardizer, representation transforms, and detectors only
on training data. Apply the fitted transforms to calibration and test data.
Use calibration scores to fit fusion parameters and save those parameters
for evaluation.

## Representations and detectors

Use standardized original features for RAW. Apply PCA or UMAP for reduced
representations, with UMAP configured with `n_neighbors=15` and `min_dist=0.1`.
Use the encoder output as the autoencoder (AE) representation. Train the
autoencoder for 50 epochs using Adam and mean squared reconstruction error.
Prefer PyTorch on CPU. All three reduced representations use
`k = max(2, min(20, d // 2))`, where `d` is the original feature count.
Score AE embeddings with the same detectors as the other representations.

| Detector | Parameters |
| --- | --- |
| Isolation Forest | `n_estimators=100`, `max_samples=256` |
| Local Outlier Factor | `n_neighbors=20`, novelty mode for scoring unseen observations |
| One-Class SVM | `kernel="rbf"`, `gamma="scale"`, `nu=0.1` |

Use scikit-learn and PyOD as needed. Orient every detector's scores so that
higher values indicate greater anomalousness. Fit one model for each
detector and representation pair. Each fusion operates on the four
representation scores from a single detector, producing a separate fused
score for each detector.

## Score normalization and fusion

Estimate the Z-score mean and standard deviation separately for each
detector and representation, using only calibration scores. Reuse the
calibration parameters for every test shift level.

| Fusion rule | Computation for a test observation |
| --- | --- |
| Mean | Arithmetic mean of the four standardized scores |
| Maximum | Maximum of the four standardized scores |
| Median | Median of the four standardized scores |
| Rank Average | Mean of the four percentile positions, each relative to its representation's calibration score distribution |
| Consensus Weighted Average | Weighted mean of the four standardized scores, using the fixed calibration weights below |
| LSCP-AOM | Local selection of representations and averaging their standardized scores |

For Consensus Weighted Average, compute Spearman correlation for each
pair of representation scores on calibration data. For each representation,
average its correlations with the other three, clip negative values to
zero, and normalize the four weights to sum to one. If all four are
nonpositive, use equal weights.

### LSCP-AOM adaptation

Zhao, Y., Nasrullah, Z., Hryniewicki, M. K., & Li, Z. (2019).
*LSCP: Locally Selective Combination in Parallel Outlier Ensembles*.
SIAM International Conference on Data Mining.
[Published paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.66) ·
[Author preprint](https://arxiv.org/abs/1812.01528)

The original LSCP defines a local region using nearest neighbors in random
feature subspaces and selects competent base detectors with a training-set
pseudo target. This study uses calibration observations for neighborhoods
and for the pseudo target. Its components are the four representations
scored by the same detector.

Use the training-standardized original feature space for neighborhoods.
Use 30 nearest calibration neighbors and 20 random feature subspaces.
The proposed design retrieves 30 neighbors per subspace for each test query.
Form the calibration pseudo target by taking
the maximum standardized score across the four representations for each
observation. Within the local region, measure each representation's
competence by Pearson correlation with that pseudo target. Average the
test scores of the selected representations.

Before implementation, submit a detailed design for approval covering
subspace sampling, neighbor aggregation, competence selection with four
components, ties, and fallback behavior for undefined correlations or
insufficient local support. Document the final differences from the
published method after approval.

## Distribution shift

For each dataset and seed, sample 25% of features with nonzero training
standard deviation and assign each selected feature a random sign from
{-1, +1}. Reuse that feature set and sign vector at all four shift levels
(`delta = 0.0, 0.5, 1.0, 2.0`).

For each level, start from the original test features. Before
standardization, add `delta * sign * training_std` to the selected
columns. Retain the original test labels. Use the same fitted models,
standardization and score normalization parameters, and global fusion
weights. LSCP evaluates each shifted query against the fixed calibration
reference.

## Experiments and evaluation

| Experiment | Comparison |
| --- | --- |
| E1 | Single representations versus fusion, including the best single representation on the matched test split and metric as a retrospective comparison |
| E2 | Effectiveness and stability of the six fusion rules |
| E3 | Changes in detection performance from the unshifted baseline across all shift levels |
| E4 | Association of label-free indicators with performance loss, plus representation ablation |

For E4, compute the mean feature-wise Kolmogorov–Smirnov statistic
(proposed reference: calibration versus test), the mean pairwise Spearman correlation
of test representation scores for each detector, and the change in that
mean agreement relative to calibration. Relate the indicators to the
metric decrease from the matched unshifted test baseline. For the
ablation, omit RAW, PCA, UMAP, and AE one at a time from Mean fusion and
compare each three-representation variant with the full four-representation
mean at `delta=0` and `delta=2`.

Report ROC-AUC and Average Precision with outliers as the positive class.
Save results to CSV with timing, experiment parameters, dataset ID, seed,
detector, representation, fusion rule, and shift level. Run a one-seed
smoke test first, then five repetitions. Expand to ten only with approval.
Reuse fitted models and scores across experiments when their inputs and
parameters are identical.

## Implementation details to confirm at the relevant stage

Confirm the exact seed list, shift feature-count rounding, autoencoder
architecture and training settings beyond those specified above, score
normalization and percentile tie conventions, and the summaries for
stability and associations. Resolve handling of constant scores and
undefined correlations before computing the affected methods.
