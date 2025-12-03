"""Statistical significance testing for comparing model performance.

This module provides functions for comparing model performance using
statistical significance tests suitable for academic research.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table
from scipy import stats
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon  # pyright: ignore[reportUnknownVariableType]
from sklearn.metrics import accuracy_score  # pyright: ignore[reportUnknownVariableType]

from src.utils.evaluation import finalize_plot

MetricFn = Callable[[np.ndarray, np.ndarray], float]
SoftmaxFn = Callable[..., np.ndarray]
FriedmanFn = Callable[..., tuple[float, float]]
TTestFn = Callable[..., tuple[float, float]]
WilcoxonFn = Callable[..., tuple[float, float]]

friedmanchisquare_fn: FriedmanFn = cast("FriedmanFn", friedmanchisquare)
ttest_rel_fn: TTestFn = cast("TTestFn", ttest_rel)
wilcoxon_fn: WilcoxonFn = cast("WilcoxonFn", wilcoxon)
accuracy_score_fn: MetricFn = cast("MetricFn", accuracy_score)

console = Console()


@dataclass
class ModelResults:
    """Container for model evaluation results."""

    name: str
    accuracy: float
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None = None
    path: Path | None = None


def load_model_results(results_base_dir: Path) -> dict[str, ModelResults]:
    """Load stored predictions and metrics for all known models."""
    model_dirs = {
        "RandomForest": results_base_dir / "random_forest",
        "AdaBoost": results_base_dir / "adaboost",
        "XGBoost": results_base_dir / "xgboost",
        "SVM": results_base_dir / "svm",
        "MultinomialNB": results_base_dir / "multinomial_nb",
        "LogisticRegression": results_base_dir / "logistic_regression",
        "FeedForward": results_base_dir / "feed_forward",
        "BERT": results_base_dir / "bert",
    }

    results: dict[str, ModelResults] = {}

    for model_name, model_dir in model_dirs.items():
        if not model_dir.exists():
            console.print(f"[yellow]Warning: {model_name} directory not found, skipping[/yellow]")
            continue

        test_dir = model_dir / "test"
        base_dir = test_dir if test_dir.exists() else model_dir

        pred_path = base_dir / "y_pred.npy"
        if not pred_path.exists():
            console.print(f"[yellow]Warning: No predictions found for {model_name}[/yellow]")
            continue

        y_pred = np.load(pred_path)
        y_true = np.load(base_dir / "y_true.npy")

        metrics_path = base_dir / "metrics.json"
        metrics: dict[str, float] = {}
        if metrics_path.exists():
            with metrics_path.open() as f:
                raw_metrics = json.load(f)
            metrics = {}
            for key, value in raw_metrics.items():
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue
            accuracy = float(metrics.get("accuracy", accuracy_score_fn(y_true, y_pred)))
        else:
            accuracy = float(accuracy_score_fn(y_true, y_pred))
        _ = metrics.setdefault("accuracy", accuracy)

        proba_path = base_dir / "y_proba.npy"
        y_proba = np.load(proba_path) if proba_path.exists() else None

        results[model_name] = ModelResults(
            name=model_name,
            accuracy=accuracy,
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            path=base_dir,
        )

    console.print(f"[green]Loaded results for {len(results)} models[/green]")
    return results


def verify_consistent_test_sets(model_results: dict[str, ModelResults]) -> bool:
    """Verify that all models were evaluated on the same test set.

    Returns:
        True if all test sets are identical, False otherwise.
    """
    if not model_results:
        console.print("[red]ERROR: Model results empty[/red]")
        return False

    # Get first model's true labels as reference
    first_model = next(iter(model_results.values()))
    reference_true = first_model.y_true

    for model_name, results in model_results.items():
        if not np.array_equal(results.y_true, reference_true):
            console.print(f"[red]ERROR: Test set mismatch for {model_name}[/red]")
            console.print(f"  Reference shape: {reference_true.shape}")
            console.print(f"  {model_name} shape: {results.y_true.shape}")

            # Check if it's just a different ordering
            if results.y_true.shape == reference_true.shape and np.array_equal(
                np.sort(results.y_true),
                np.sort(reference_true),
            ):
                console.print("  [yellow]Note: Same labels, different order[/yellow]")
            return False

    console.print("[green]All models evaluated on the same test set[/green]")
    return True


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Calculate bootstrap confidence interval for difference in performance.

    Args:
        y_true: True labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.
        n_bootstraps: Number of bootstrap samples.
        confidence_level: Confidence level (e.g., 0.95 for 95%).

    Returns:
        Tuple of (difference, lower_bound, upper_bound)
    """
    n_samples = len(y_true)

    # Bootstrap sample indices
    rng = np.random.default_rng(0)
    indices = rng.integers(0, n_samples, size=(n_bootstraps, n_samples))

    # Calculate metric for both models on bootstrap sample
    score_a = np.mean(y_true[indices] == y_pred_a[indices], axis=1)
    score_b = np.mean(y_true[indices] == y_pred_b[indices], axis=1)
    diffs = score_a - score_b

    # Calculate confidence interval
    alpha = (1 - confidence_level) / 2
    lower = float(np.percentile(diffs, 100 * alpha))
    upper = float(np.percentile(diffs, 100 * (1 - alpha)))
    mean_diff = float(np.mean(diffs))

    return mean_diff, lower, upper


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> tuple[float, float]:
    """Perform McNemar's test for paired nominal data.

    Appropriate for comparing two classifiers on the same test set.

    Args:
        y_true: True labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.

    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true

    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))

    if b + c == 0:
        return 0.0, 1.0

    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p)


def paired_t_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> tuple[float, float]:
    """Perform paired t-test on accuracy scores.

    Args:
        y_true: True labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.

    Returns:
        Tuple of (t_statistic, p_value)
    """
    # Calculate accuracy per sample (0/1 loss)
    correct_a = (y_pred_a == y_true).astype(float)
    correct_b = (y_pred_b == y_true).astype(float)

    # Perform paired t-test
    t_stat, p_value = ttest_rel_fn(correct_a, correct_b)

    return float(t_stat), float(p_value)


def wilcoxon_signed_rank_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test.

    Args:
        y_true: True labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.

    Returns:
        Tuple of (statistic, p_value)
    """
    # Calculate accuracy per sample
    correct_a = (y_pred_a == y_true).astype(float)
    correct_b = (y_pred_b == y_true).astype(float)

    # Perform Wilcoxon test
    stat, p_value = wilcoxon_fn(correct_a, correct_b)

    return float(stat), float(p_value)


def friedman_test_with_posthoc(model_results: dict[str, ModelResults]) -> pd.DataFrame:
    """Perform Friedman test with post-hoc Nemenyi test.

    Appropriate for comparing multiple classifiers on the same test set.

    Args:
        model_results: Dictionary of model results.

    Returns:
        DataFrame with pairwise p-values.
    """
    # Prepare data: each row is a sample, each column is a model's correctness
    models = list(model_results.keys())
    n_models = len(models)
    n_samples = len(next(iter(model_results.values())).y_true)

    # Create sample x model matrix of correctness
    data = np.column_stack([(r.y_pred == r.y_true).astype(float) for r in model_results.values()])

    # Friedman test
    friedman_stat, friedman_p = friedmanchisquare_fn(*[data[:, i] for i in range(n_models)])
    console.print(f"Friedman test: χ2 = {friedman_stat:.4f}, p = {friedman_p:.6f}")

    # Rank each sample across models
    ranks = np.apply_along_axis(stats.rankdata, 1, data)
    mean_ranks = np.mean(ranks, axis=0)

    # Nemenyi critical difference
    q_alpha = 2.343  # alpha = 0.05
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_samples))
    console.print(f"Nemenyi critical difference (alpha=0.05): {cd:.4f}")

    # Pairwise z-scores
    result = pd.DataFrame(index=models, columns=models, dtype=float)
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                result.iloc[i, j] = 1.0
            else:
                diff = abs(mean_ranks[i] - mean_ranks[j])
                z = diff / np.sqrt(n_models * (n_models + 1) / (6 * n_samples))
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                result.iloc[i, j] = float(p)

    return result


def compare_models_statistically(
    model_results: dict[str, ModelResults],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Comprehensive statistical comparison of all model pairs.

    Args:
        model_results: Dictionary of model results.
        alpha: Significance level.

    Returns:
        DataFrame with comparison results.
    """
    models = list(model_results.keys())
    rows = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = model_results[models[i]]
            b = model_results[models[j]]

            if not np.array_equal(a.y_true, b.y_true):
                console.print(f"[red]Skipping {a.name} vs {b.name} due to mismatched test sets.[/red]")
                continue

            y = a.y_true

            chi2, p_mcnemar = mcnemar_test(y, a.y_pred, b.y_pred)
            t_stat, p_t = paired_t_test(y, a.y_pred, b.y_pred)
            w_stat, p_w = wilcoxon_signed_rank_test(y, a.y_pred, b.y_pred)
            _, ci_low, ci_up = bootstrap_confidence_interval(y, a.y_pred, b.y_pred)

            rows.append(
                {
                    "Model_A": a.name,
                    "Model_B": b.name,
                    "Acc_A": a.accuracy,
                    "Acc_B": b.accuracy,
                    "Diff": a.accuracy - b.accuracy,
                    "McNemar_chi2": chi2,
                    "McNemar_p": p_mcnemar,
                    "McNemar_sig": bool(p_mcnemar < alpha),
                    "Paired_t": t_stat,
                    "Paired_t_p": p_t,
                    "Paired_t_sig": bool(p_t < alpha),
                    "Wilcoxon_W": w_stat,
                    "Wilcoxon_p": p_w,
                    "Wilcoxon_sig": bool(p_w < alpha),
                    "Bootstrap_CI_lower": ci_low,
                    "Bootstrap_CI_upper": ci_up,
                },
            )

    return pd.DataFrame(rows)


def plot_statistical_comparisons(
    df: pd.DataFrame,
    results_dir: Path,
    show_plots: bool = False,
) -> None:
    """Create visualization of statistical comparison results.

    Args:
        df: DataFrame from compare_models_statistically.
        results_dir: Directory to save plots.
        show_plots: Whether to display plots interactively.
    """
    if df.empty:
        console.print("[yellow]No valid pairwise comparisons to plot.[/yellow]")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = sorted(set(df["Model_A"]).union(df["Model_B"]))
    idx = {m: i for i, m in enumerate(models)}
    mat = np.zeros((len(models), len(models)), dtype=float)

    for _, r in df.iterrows():
        i = idx[r["Model_A"]]
        j = idx[r["Model_B"]]
        mat[i, j] = float(r["McNemar_sig"])
        mat[j, i] = float(r["McNemar_sig"])

    _ = sns.heatmap(
        mat,
        xticklabels=models,
        yticklabels=models,
        cmap="RdYlGn_r",
        ax=axes[0],
    )
    _ = axes[0].set_title("McNemar Significance")

    axes[1].hlines(
        y=range(len(df)),
        xmin=df["Bootstrap_CI_lower"],
        xmax=df["Bootstrap_CI_upper"],
        color="gray",
        alpha=0.4,
    )
    axes[1].scatter(df["Diff"], range(len(df)), s=100, alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_yticks(range(len(df)))
    axes[1].set_yticklabels([f"{r['Model_A']} vs {r['Model_B']}" for _, r in df.iterrows()])
    _ = axes[1].set_title("Bootstrap 95 percent CI")

    plt.tight_layout()

    finalize_plot(
        fig=fig,
        save_path=results_dir / "statistical_comparison.png",
        show=show_plots,
        status_msg="Showing statistical comparison plot",
    )


def plot_model_overview(
    model_results: dict[str, ModelResults],
    results_dir: Path,
    show_plots: bool = False,
) -> None:
    """Visualize headline metrics for all available models."""
    if not model_results:
        console.print("[yellow]No model results to summarize.[/yellow]")
        return

    rows: list[dict[str, float | str | int]] = []
    for name, res in model_results.items():
        metrics = res.metrics
        rows.append(
            {
                "model": name,
                "accuracy": float(metrics.get("accuracy", res.accuracy)),
                "precision": float(metrics["precision"]) if "precision" in metrics else np.nan,
                "recall": float(metrics["recall"]) if "recall" in metrics else np.nan,
                "f1": float(metrics["f1"]) if "f1" in metrics else np.nan,
                "n_samples": len(res.y_true),
            },
        )

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    metric_cols = [c for c in ("accuracy", "precision", "recall", "f1") if c in df.columns and not df[c].isna().all()]

    if df.empty:
        console.print("[yellow]No metrics found to plot.[/yellow]")
        return

    n_axes = 2 if len(metric_cols) > 1 else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(10 + 5 * (n_axes - 1), 7))
    axes_arr = np.atleast_1d(axes)

    # Accuracy leaderboard
    ax0 = axes_arr[0]
    _ = sns.barplot(data=df, x="accuracy", y="model", hue="model", palette="crest", ax=ax0, legend=False)
    ax0.set_xlim(0, 1)
    ax0.set_xlabel("Accuracy")
    ax0.set_ylabel("Model")
    ax0.set_title("Test accuracy by model")
    for patch in ax0.patches:
        width = patch.get_width()
        ax0.text(width + 0.01, patch.get_y() + patch.get_height() / 2, f"{width:.3f}", va="center")
    ax0.grid(axis="x", alpha=0.3, linestyle="--")

    # Additional metrics (precision/recall/F1) if available
    if n_axes > 1:
        other_metrics = [m for m in metric_cols if m != "accuracy"]
        metric_df = df.melt(id_vars="model", value_vars=other_metrics, var_name="metric", value_name="value")
        metric_df["metric"] = metric_df["metric"].str.upper()

        ax1 = axes_arr[1]
        _ = sns.barplot(data=metric_df, x="value", y="metric", hue="model", ax=ax1)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("Score")
        ax1.set_ylabel("")
        ax1.set_title("Precision / Recall / F1 (test)")
        ax1.legend(title="Model", bbox_to_anchor=(1.04, 1), loc="upper left")
        _ = ax1.grid(axis="x", alpha=0.3, linestyle="--")

    _ = fig.suptitle("Model performance overview", fontsize=14)
    plt.tight_layout()

    overview_dir = results_dir / "statistical_tests"
    overview_dir.mkdir(parents=True, exist_ok=True)
    finalize_plot(
        fig=fig,
        save_path=overview_dir / "model_overview.png",
        show=show_plots,
        status_msg="Showing model overview plot",
    )


def generate_statistical_report(
    model_results: dict[str, ModelResults],
    results_dir: Path,
    alpha: float = 0.05,
    show_plots: bool = False,
) -> None:
    """Generate comprehensive statistical report for model comparisons.

    Args:
        model_results: Dictionary of model results.
        results_dir: Directory to save report.
        alpha: Significance level.
    """
    console.rule("[bold]Statistical Significance Testing[/bold]")

    df = compare_models_statistically(model_results, alpha)

    table = Table(title="Statistical Comparisons")
    table.add_column("Comparison")
    table.add_column("Diff", justify="right")
    table.add_column("McNemar", justify="center")
    table.add_column("t-test", justify="center")
    table.add_column("Bootstrap CI")

    for _, r in df.iterrows():
        comp = f"{r['Model_A']} vs {r['Model_B']}"
        ci = f"[{r['Bootstrap_CI_lower']:.4f}, {r['Bootstrap_CI_upper']:.4f}]"
        table.add_row(
            comp,
            f"{r['Diff']:.4f}",
            "S" if r["McNemar_sig"] else "N",
            "S" if r["Paired_t_sig"] else "N",
            ci,
        )

    console.print(table)

    stats_dir = results_dir / "statistical_tests"
    stats_dir.mkdir(exist_ok=True)

    df.to_csv(stats_dir / "pairwise_comparisons.csv", index=False)

    console.print("\n[bold]Friedman and Nemenyi[/bold]")
    nemenyi = friedman_test_with_posthoc(model_results)
    nemenyi.to_csv(stats_dir / "friedman_nemenyi.csv")
    console.print(nemenyi.to_string())

    console.print("\n[bold]Visualization[/bold]")
    plot_statistical_comparisons(df, stats_dir, show_plots=show_plots)

    summary = stats_dir / "statistical_summary.md"
    with summary.open("w") as f:
        _ = f.write("# Statistical Significance Summary\n\n")
        _ = f.write(f"Alpha: {alpha}\n\n")
        _ = f.write("## Best Models\n")
        for i, (name, r) in enumerate(
            sorted(model_results.items(), key=lambda x: x[1].accuracy, reverse=True),
            1,
        ):
            _ = f.write(f"{i}. {name}: {r.accuracy:.4f}\n")

    console.print(f"[green]Saved summary to {summary}[/green]")


def bert_confidence_analysis(
    model_results: dict[str, ModelResults],
    results_dir: Path,
    show_plots: bool = False,
) -> None:
    """Analyze BERT model confidence and calibration.

    Args:
        model_results: Dictionary of model results.
        results_dir: Directory to save analysis.
        show_plots: Whether to display plots.
    """
    if "BERT" not in model_results:
        return

    bert_results = model_results["BERT"]
    if bert_results.y_proba is None:
        return

    # Confidence of predicted class
    confidence: np.ndarray = np.max(bert_results.y_proba, axis=1)
    predictions = bert_results.y_pred

    # Calibration analysis
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices: np.ndarray = np.digitize(confidence, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    accuracy_per_bin: list[float] = []
    avg_confidence_per_bin: list[float] = []

    for i in range(n_bins):
        mask: np.ndarray = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = float(np.mean(bert_results.y_true[mask] == predictions[mask]))
            bin_conf = float(np.mean(confidence[mask]))
            accuracy_per_bin.append(bin_acc)
            avg_confidence_per_bin.append(bin_conf)
        else:
            accuracy_per_bin.append(np.nan)
            avg_confidence_per_bin.append(np.nan)

    # Plot calibration curve
    fig, ax = plt.subplots(figsize=(8, 6))
    _ = ax.plot(avg_confidence_per_bin, accuracy_per_bin, "o-", label="BERT")
    _ = ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    _ = ax.set_title("BERT Calibration Curve")
    _ = ax.set_xlabel("Average predicted probability")
    _ = ax.set_ylabel("Fraction of positives")
    _ = ax.legend()
    ax.grid(True, alpha=0.3)

    output_dir = results_dir / "statistical_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    finalize_plot(
        fig=fig,
        save_path=output_dir / "bert_calibration.png",
        show=show_plots,
        status_msg="Showing BERT calibration plot",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_statistical_tests(
    results_dir: Path,
    show_plots: bool = False,
) -> None:
    """Main entry point for statistical significance testing.

    Args:
        results_dir: Base results directory containing model subdirectories.
        show_plots: Whether to display plots interactively.
    """
    console.rule("[bold]Statistical Significance Analysis[/bold]")

    # Load model results
    model_results = load_model_results(results_dir)
    if not model_results:
        console.print("[red]No model results found; nothing to analyze.[/red]")
        return

    # High-level comparison plot across all available models
    plot_model_overview(model_results, results_dir, show_plots=show_plots)

    if len(model_results) < 2:
        console.print("[red]Need at least 2 models for statistical comparison[/red]")
        return

    test_sets_consistent = verify_consistent_test_sets(model_results)
    if not test_sets_consistent:
        console.print(
            "[yellow]"
            "WARNING: Models were not evaluated on identical test sets.\n"
            "  Statistical comparisons may be invalid.\n"
            "  Continuing anyway, but interpret results with caution."
            "[/yellow]",
        )

    # Generate comprehensive report
    generate_statistical_report(model_results, results_dir, alpha=0.05, show_plots=show_plots)

    # Run BERT-specific analysis
    bert_confidence_analysis(model_results, results_dir, show_plots)

    console.print("\n[green]Statistical analysis complete[/green]")
