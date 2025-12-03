from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,  # pyright: ignore[reportUnknownVariableType]
    classification_report,  # pyright: ignore[reportUnknownVariableType]
    confusion_matrix,  # pyright: ignore[reportUnknownVariableType]
    precision_recall_fscore_support,  # pyright: ignore[reportUnknownVariableType]
)


def finalize_plot(*, fig: Any, save_path: Path, show: bool, status_msg: str) -> None:
    """Save a plot and optionally show it under a status context.

    Args:
        fig: Matplotlib figure to finalize.
        save_path: Where the plot image should be written.
        show: Whether to display the plot interactively.
        status_msg: Message to show in the console while displaying the plot.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        with Console().status(f"{status_msg} (close figure to continue)"):
            plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    name: str,
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Sequence[str | int] | None = None,
    save_path: Path | None = None,
    show_plot: bool = False,
) -> None:
    """Plot a confusion matrix using matplotlib.

    Args:
        name: Model name used for the plot title.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Explicit label ordering, if needed.
        save_path: Optional destination for saving the plot.
        show_plot: Whether to display the plot interactively while saving.
    """
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))

    label_values = np.asarray(labels) if labels is not None else np.unique(np.concatenate((y_true_arr, y_pred_arr)))

    cm = cast("np.ndarray", confusion_matrix(y_true_arr, y_pred_arr, labels=label_values))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_values)

    fig, ax = plt.subplots(figsize=(6, 5))
    _ = disp.plot(ax=ax, cmap="Blues", colorbar=False)
    _ = ax.set_xlabel("Predicted label")
    _ = ax.set_ylabel("True label")
    if name:
        _ = ax.set_title(f"{name} confusion matrix")
    fig.tight_layout()
    status_msg = f"Showing confusion matrix for {name}" if name else "Showing confusion matrix"
    finalize_plot(
        fig=fig,
        save_path=save_path if save_path is not None else Path("results") / "confusion_matrix.png",
        show=show_plot,
        status_msg=status_msg,
    )


def render_evaluation_report(
    *,
    name: str,
    y_true: Iterable[int],
    y_pred: Iterable[int],
    console: Console,
    labels: Sequence[str | int] | None = None,
    results_dir: Path,
    show_plots: bool = False,
) -> None:
    """Render evaluation metrics, textual reports, and confusion matrix plots.

    Stores:
        - Textual classification report
        - Summary metrics (JSON)
        - Per-class metrics (JSON)
        - Confusion matrix plot
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_").lower()

    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))

    # Determine label ordering
    raw_labels = np.asarray(labels) if labels is not None else np.unique(np.concatenate((y_true_arr, y_pred_arr)))
    label_values = [cast("str | int", v.item() if hasattr(v, "item") else v) for v in raw_labels.tolist()]

    # Summary metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="micro",
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )

    metrics_json = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(fscore),
    }

    # Console header
    console.rule(f"[bold]{name} evaluation[/bold]")

    # Summary metrics table
    summary_table = Table(show_header=True, header_style="bold")
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    for k, v in metrics_json.items():
        summary_table.add_row(k.capitalize(), f"{v:.4f}")
    console.print(summary_table)

    # Classification report (dict and text)
    report_dict = cast(
        "dict[str, dict[str, float | int]]",
        classification_report(
            y_true_arr,
            y_pred_arr,
            labels=label_values,
            target_names=[str(lbl) for lbl in label_values],
            output_dict=True,
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
    )
    report_text = cast(
        "str",
        classification_report(
            y_true_arr,
            y_pred_arr,
            labels=label_values,
            target_names=[str(lbl) for lbl in label_values],
            output_dict=False,
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
    )

    # Per-class report table
    console.print("\n[bold]Classification report[/bold]")
    class_table = Table(show_header=True, header_style="bold")
    class_table.add_column("Class")
    class_table.add_column("Precision", justify="right")
    class_table.add_column("Recall", justify="right")
    class_table.add_column("F1", justify="right")
    class_table.add_column("Support", justify="right")

    for lbl in label_values:
        lbl_str = str(lbl)
        if lbl_str not in report_dict:
            continue
        data = report_dict[lbl_str]
        class_table.add_row(
            lbl_str,
            f"{data['precision']:.4f}",
            f"{data['recall']:.4f}",
            f"{data['f1-score']:.4f}",
            f"{int(data['support'])}",
        )

    # Macro and weighted averages
    for avg_key in ("macro avg", "weighted avg"):
        data = report_dict.get(avg_key)
        if data is not None:
            class_table.add_row(
                avg_key,
                f"{data['precision']:.4f}",
                f"{data['recall']:.4f}",
                f"{data['f1-score']:.4f}",
                f"{int(data['support'])}",
            )

    console.print(class_table)

    # Save textual artifacts
    _ = (results_dir / f"{safe_name}_classification_report.txt").write_text(report_text)
    _ = (results_dir / f"{safe_name}_metrics.json").write_text(json.dumps(metrics_json, indent=2))
    _ = (results_dir / f"{safe_name}_per_class_metrics.json").write_text(json.dumps(report_dict, indent=2))

    # Confusion matrix plot
    cm_path = results_dir / f"{safe_name}_confusion_matrix.png"
    plot_confusion_matrix(
        name,
        y_true_arr,
        y_pred_arr,
        labels=label_values,
        save_path=cm_path,
        show_plot=show_plots,
    )
