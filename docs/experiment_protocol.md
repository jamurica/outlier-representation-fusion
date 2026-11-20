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
Fit UMAP on unique standardized training vectors at UMAP's float32
precision for every dataset. Map the fitted coordinates back to all
training rows in their original order. Transform every calibration and
test row using the fitted UMAP model. Preserve the full partitions for
detector training, calibration, and evaluation.

All three reduced representations use `k = max(2, min(20, d // 2))`, where
`d` is the original feature count. The autoencoder uses the architecture
`d → h → k → h → d`, with `h = max(16, 2 * k)`. Apply ReLU after each
hidden layer of width `h`. The bottleneck and reconstruction output are
linear. Train on standardized training rows with PyTorch on CPU for 50
epochs using mean squared reconstruction error and Adam with a learning
rate of 0.001. Shuffle batches of up to 64 rows in each epoch. Use the
repetition seed for initialization and shuffling. Use the encoder output
as the AE representation for the same detectors as the other representations.

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
detector and representation, using only calibration scores. Use
`StandardScaler` with the population standard deviation (`ddof=0`).
Use a scale of 1 for constant calibration scores, so transformed scores
are their deviations from the calibration mean. Reuse the calibration
parameters for every test shift level.

| Fusion rule | Computation for a test observation |
| --- | --- |
| Mean | Arithmetic mean of the four standardized scores |
| Maximum | Maximum of the four standardized scores |
| Median | Median of the four standardized scores |
| Rank Average | Mean of the four percentile positions, each relative to its representation's calibration score distribution |
| Consensus Weighted Average | Weighted mean of the four standardized scores, using the fixed calibration weights below |
| LSCP-AOM | Local selection of representations and averaging their standardized scores |

For Rank Average, compute the percentile of a raw score as the number of
calibration scores below it plus half the number tied with it, divided by
the calibration sample count. Scores below or above the calibration range
receive 0 or 1, respectively.

For Consensus Weighted Average, compute Spearman correlation for each
pair of raw representation scores on calibration data. Count undefined
correlations, including pairs with constant calibration scores, as zero.
For each representation, average its correlations with the other three,
clip negative values to zero, and normalize the four weights to sum to
one. If all four are nonpositive, use equal weights.

### LSCP-AOM adaptation

Zhao, Y., Nasrullah, Z., Hryniewicki, M. K., & Li, Z. (2019).
*LSCP: Locally Selective Combination in Parallel Outlier Ensembles*.
SIAM International Conference on Data Mining.
[Published paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.66) ·
[Author preprint](https://arxiv.org/abs/1812.01528)

The original LSCP uses training observations for neighborhoods and the
pseudo target. The published experiments combine 50 LOF detectors with
varying parameters and select components using the most populated bin of
a Pearson correlation histogram. This adaptation uses calibration as its
reference and combines four representations of one detector. It uses fixed
neighborhood settings and selects the two highest local correlations,
including ties at the cutoff.

1. Draw 20 subspaces once per dataset and seed from the training-standardized
   original feature space. Draw each size uniformly from the integers
   between `ceil(d / 2)` and `d`, then sample columns without replacement.
   Subspaces may repeat. Reuse the subspaces for all detectors and shifts.
2. For each test row, find the 30 nearest calibration rows by Euclidean
   distance in each subspace. Break distance ties by the lower calibration
   row index. Keep rows receiving at least 11 votes. With fewer than three
   retained rows, use the 30 nearest rows in the full standardized feature
   space. A reference set with fewer than 30 rows supplies all available rows.
3. For each detector, compute the pseudo target as the row-wise maximum of
   the four calibration Z-scores. Measure local competence using Pearson
   correlation between each calibration score column and the pseudo target
   within the selected neighborhood. Keep defined negative correlations
   in the ranking and exclude undefined correlations.
4. Average the test Z-scores of the selected representations. Use the single
   available representation if only one correlation is defined. Use the
   mean of all four test Z-scores if no correlation is defined or the region
   has fewer than three rows.

Recompute neighborhoods for each shifted test partition against the fixed
calibration reference. Reuse the pseudo target and subspaces.

## Distribution shift

For each dataset and seed, sample 25% of features with nonzero training
standard deviation. Round the feature count to the nearest integer, with
halves rounded up. Sample columns without replacement and assign each
selected feature a sign from {-1, +1} with equal probability. Reuse that
feature set and sign vector at all four shift levels
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

For E4, average the two-sided Kolmogorov–Smirnov statistics comparing
the fixed calibration and current test distributions feature by feature.
For each detector, average the defined pairwise Spearman correlations of
the four raw representation score vectors. Omit pairs with a constant
score vector. Record missing agreement when no pair yields a correlation.
Compute agreement change as test agreement minus calibration agreement.
Record a missing change when either agreement is missing. Compute metric
loss as the unshifted test metric minus the current test metric for the
same dataset, seed, detector, and method. For the ablation, omit RAW, PCA,
UMAP, and AE one at a time from Mean fusion and
compare each three-representation variant with the full four-representation
mean at `delta=0` and `delta=2`.

Report ROC-AUC and Average Precision with outliers as the positive class.
Save results to CSV with timing, experiment parameters, dataset ID, seed,
detector, representation, fusion rule, and shift level. Run a one-seed
smoke test first, then five repetitions. Expand to ten only with approval.
Reuse fitted models and scores across experiments when their inputs and
parameters are identical.

## Implementation details to confirm at the relevant stage

Confirm the exact seed list and the summaries for stability and
associations.
