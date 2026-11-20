"""Summarize experiments E1–E4 and plot results from the runner's CSV files."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.distribution_shift import SHIFT_LEVELS


METRICS = {"roc_auc": "ROC-AUC", "average_precision": "Average Precision"}
REPRESENTATIONS = ("RAW", "PCA", "UMAP", "AE")
FUSIONS = (
    "Mean", "Maximum", "Median", "Rank Average",
    "Consensus Weighted Average", "LSCP-AOM",
)
DETECTORS = ("Isolation Forest", "LOF", "One-Class SVM")
INDICATORS = {
    "mean_feature_ks": "Mean KS",
    "test_score_agreement": "Test agreement",
    "agreement_change": "Agreement change",
}
MATCH = ["dataset", "seed", "detector"]
ROW_ID = MATCH + ["delta", "representation", "fusion_rule", "omitted_representation"]


def read_results(paths: list[Path]) -> pd.DataFrame:
    """Combine nonoverlapping runs and check that their design is complete."""
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    labels = ["representation", "fusion_rule", "omitted_representation"]
    data[labels] = data[labels].fillna("")
    if data.empty or data.duplicated(ROW_ID).any():
        raise ValueError("Expected results with unique dataset, seed, detector, shift, and method keys")
    expected = {
        (view, "", "") for view in REPRESENTATIONS
    } | {("All", fusion, "") for fusion in FUSIONS}
    ablations = {("All", "Mean", view) for view in REPRESENTATIONS}
    for dataset in data["dataset"].unique():
        for seed in data["seed"].unique():
            repetition = data[(data["dataset"] == dataset) & (data["seed"] == seed)]
            if set(repetition["detector"]) != set(DETECTORS) or set(repetition["delta"]) != set(SHIFT_LEVELS):
                raise ValueError(f"Incomplete detector or shift coverage for {dataset}, seed={seed}")
            for detector in DETECTORS:
                for delta in SHIFT_LEVELS:
                    rows = repetition[(repetition["detector"] == detector) & (repetition["delta"] == delta)]
                    methods = set(rows[labels].itertuples(index=False, name=None))
                    required = expected | ablations if delta in (0.0, 2.0) else expected
                    if methods != required:
                        raise ValueError(f"Incomplete methods for {dataset}, seed={seed}, {detector}, delta={delta}")
    for metric in METRICS:
        if not data[metric].between(0, 1).all():
            raise ValueError(f"Expected finite {metric} values between 0 and 1")
    return data


def summarize(data: pd.DataFrame, metric: str) -> dict[str, pd.DataFrame]:
    """Create the paired comparisons and descriptive summaries for one metric."""
    main = data[data["omitted_representation"] == ""].copy()
    main["method"] = main["fusion_rule"].where(main["fusion_rule"] != "", main["representation"])
    unshifted = main[main["delta"] == 0]
    fusions = unshifted[unshifted["fusion_rule"] != ""]
    best = unshifted[unshifted["fusion_rule"] == ""].groupby(MATCH)[metric].max()
    paired = fusions[MATCH + ["method", metric]].join(best.rename("best_single"), on=MATCH)
    paired["fusion_gain"] = paired[metric] - paired["best_single"]
    paired["win"] = paired["fusion_gain"] > 0
    comparisons = paired.groupby(["detector", "method"], as_index=False).agg(
        mean_gain=("fusion_gain", "mean"), win_rate=("win", "mean"), n_pairs=("win", "size")
    )

    by_dataset = fusions.groupby(["dataset", "detector", "method"], as_index=False).agg(
        mean_score=(metric, "mean"), seed_sd=(metric, "std"), n_seeds=(metric, "size")
    )
    stability = by_dataset.groupby(["detector", "method"], as_index=False).agg(
        mean_score=("mean_score", "mean"),
        mean_seed_sd=("seed_sd", "mean"),
        n_datasets=("dataset", "size"),
    )
    shift = main.groupby(["detector", "method", "delta"], as_index=False).agg(
        mean_score=(metric, "mean"), mean_drop=(f"{metric}_drop", "mean"), n_pairs=(metric, "size")
    )

    associations = []
    for (detector, method), group in main[main["delta"] > 0].groupby(["detector", "method"]):
        for indicator in INDICATORS:
            pairs = group[[indicator, f"{metric}_drop"]].dropna()
            rho = np.nan
            if (pairs.nunique() > 1).all():
                rho = spearmanr(pairs.iloc[:, 0], pairs.iloc[:, 1]).statistic
            associations.append({
                "detector": detector, "method": method, "indicator": indicator,
                "spearman_rho": rho, "n_pairs": len(pairs),
            })

    ablation = data[data["omitted_representation"] != ""]
    full_mean = main[main["fusion_rule"] == "Mean"].set_index(MATCH + ["delta"])[metric]
    ablation = ablation[MATCH + ["delta", "omitted_representation", metric]].join(
        full_mean.rename("full_mean"), on=MATCH + ["delta"]
    )
    ablation["ablation_minus_full"] = ablation[metric] - ablation["full_mean"]
    ablation_summary = ablation.groupby(
        ["detector", "delta", "omitted_representation"], as_index=False
    ).agg(mean_difference=("ablation_minus_full", "mean"), n_pairs=(metric, "size"))
    return {
        "e1_paired": paired, "e1_summary": comparisons,
        "e2_by_dataset": by_dataset, "e2_summary": stability,
        "e3_shift": shift, "e4_associations": pd.DataFrame(associations),
        "e4_ablation_paired": ablation, "e4_ablation_summary": ablation_summary,
    }


def plot_comparisons(tables: dict, metric: str, output: Path) -> None:
    panels = (
        ("e1_summary", "mean_gain", "E1  Gain over best single representation"),
        ("e2_summary", "mean_score", "E2  Mean score across datasets"),
        ("e2_summary", "mean_seed_sd", "E2  Mean within-dataset seed SD"),
    )
    fig, axes = plt.subplots(
        3, 3, figsize=(14, 10), sharex="row", sharey=True, layout="constrained"
    )
    for row, (table, column, label) in enumerate(panels):
        for axis, detector in zip(axes[row], DETECTORS):
            values = tables[table].query("detector == @detector").set_index("method")
            series = values.loc[list(FUSIONS), column]
            axis.barh(FUSIONS, series, color="#267A88")
            if series.isna().all():
                axis.text(0.5, 0.5, "Requires at least two seeds", ha="center",
                          transform=axis.transAxes)
                axis.tick_params(axis="x", bottom=False, labelbottom=False)
            else:
                axis.axvline(0, color="#40464B", lw=0.7)
                axis.grid(axis="x", alpha=0.2)
            axis.set_xlabel(label)
            if row == 0:
                axis.set_title(detector)
    axes[0, 0].invert_yaxis()
    fig.suptitle(f"{METRICS[metric]} at δ = 0")
    fig.savefig(output / f"e1_e2_{metric}.png", dpi=170)
    plt.close(fig)


def plot_shift(tables: dict, metric: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey="row", layout="constrained")
    for column, detector in enumerate(DETECTORS):
        data = tables["e3_shift"].query("detector == @detector")
        for index, method in enumerate(REPRESENTATIONS + FUSIONS):
            rows = data[data["method"] == method].sort_values("delta")
            for axis, value in zip(axes[:, column], ("mean_score", "mean_drop")):
                axis.plot(rows["delta"], rows[value], marker="o", ms=3, lw=1.4,
                          color=f"C{index}", linestyle="--" if method in REPRESENTATIONS else "-",
                          label=method)
        axes[0, column].set_title(detector)
        axes[1, column].set_xlabel("Shift level δ")
        axes[1, column].set_xticks(SHIFT_LEVELS)
        axes[1, column].axhline(0, color="#40464B", lw=0.7)
        for axis in axes[:, column]:
            axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Mean score")
    axes[1, 0].set_ylabel("Mean drop from δ = 0")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle(f"E3  {METRICS[metric]} under feature shift")
    fig.savefig(output / f"e3_{metric}.png", dpi=170)
    plt.close(fig)


def plot_indicators(tables: dict, metric: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True, layout="constrained")
    methods = REPRESENTATIONS + FUSIONS
    for axis, detector in zip(axes, DETECTORS):
        table = tables["e4_associations"].query("detector == @detector")
        matrix = table.pivot(index="method", columns="indicator", values="spearman_rho")
        values = matrix.reindex(index=methods, columns=list(INDICATORS)).to_numpy()
        scale = axis.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        axis.set_title(detector)
        axis.set_xticks(range(len(INDICATORS)), labels=list(INDICATORS.values()), rotation=30, ha="right")
        axis.set_yticks(range(len(methods)), labels=methods)
        for (row, column), rho in np.ndenumerate(values):
            label = f"{rho:.2f}" if np.isfinite(rho) else "—"
            color = "white" if abs(rho) > 0.65 else "#20262B"
            axis.text(column, row, label, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(scale, ax=axes, label="Spearman ρ", shrink=0.8)
    fig.suptitle(f"E4  Indicator association with {METRICS[metric]} drop (δ > 0)")
    fig.savefig(output / f"e4_indicators_{metric}.png", dpi=170)
    plt.close(fig)


def plot_ablation(tables: dict, metric: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    positions = np.arange(len(REPRESENTATIONS))
    for axis, detector in zip(axes, DETECTORS):
        table = tables["e4_ablation_summary"].query("detector == @detector")
        for delta, offset, color in ((0, -0.18, "#267A88"), (2, 0.18, "#CF7737")):
            values = table.query("delta == @delta").set_index("omitted_representation")
            axis.bar(positions + offset, values.loc[list(REPRESENTATIONS), "mean_difference"],
                     width=0.34, label=f"δ = {delta}", color=color)
        axis.set_title(detector)
        axis.set_xticks(positions, labels=REPRESENTATIONS)
        axis.set_xlabel("Omitted representation")
        axis.axhline(0, color="#40464B", lw=0.7)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Ablation minus full Mean")
    axes[-1].legend(frameon=False)
    fig.suptitle(f"E4  {METRICS[metric]} change after ablation")
    fig.savefig(output / f"e4_ablation_{metric}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    data = read_results(args.inputs)
    args.output_dir.mkdir(parents=True)
    data.to_csv(args.output_dir / "combined_results.csv", index=False)
    for metric in METRICS:
        tables = summarize(data, metric)
        for name, table in tables.items():
            table.to_csv(args.output_dir / f"{name}_{metric}.csv", index=False)
        plot_comparisons(tables, metric, args.output_dir)
        plot_shift(tables, metric, args.output_dir)
        plot_indicators(tables, metric, args.output_dir)
        plot_ablation(tables, metric, args.output_dir)
    print(f"Analyzed {len(data)} rows and saved tables and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
