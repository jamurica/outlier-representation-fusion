# Project collaboration

- Work on one approved stage at a time. Briefly explain the planned changes
  before editing. Stop after the current stage and wait for approval.
- Inspect existing files before changing them. Keep the Python code modular,
  readable, and reproducible for a student learning Python and Git.
- Explain important code and report progress to the user in Polish. Use English
  for identifiers, comments, and repository documentation.
- Follow [the experiment protocol](docs/experiment_protocol.md). Obtain approval
  before changing the protocol or hyperparameters. Submit a detailed LSCP design
  and obtain approval before implementing it.
- Keep the project scoped to batch, unsupervised outlier detection. Exclude
  streaming, online adaptation, continual learning, and adaptive detection.
- Run relevant checks after each implementation and report actual outcomes and
  verification limits. Use only measured results in experimental reports.
- Obtain approval before installing dependencies.
- Version final experiment CSVs and figures in `results/analysis/`. Keep
  raw datasets, trained model weights, virtual environments, credentials,
  and temporary or intermediate outputs out of Git.
- Leave commits and pushes to the user. End each stage with a summary, check
  results, and a proposed commit message.
- Keep the implementation simple, concise, and readable.
- Prefer established library implementations over custom algorithms.
- Avoid unnecessary abstractions, excessive complexity, and premature optimization.
- Preserve methodological correctness, reproducibility, and essential validation.

## Implementation stages

1. Environment and repository structure
2. ADBench loader and optional downloader
3. Splitting and preprocessing
4. RAW, PCA, and UMAP representations
5. Autoencoder representation
6. Detectors
7. Score normalization
8. Basic fusion rules
9. LSCP
10. Distribution shift generator
11. Evaluation and ablation
12. Experiment runner
13. Analysis and visualization
14. Repository documentation
