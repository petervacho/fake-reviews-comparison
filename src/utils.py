from __future__ import annotations

import builtins
import json
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, cast
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.style import Style
from rich.table import Table
from rich.text import Text
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,  # pyright: ignore[reportUnknownVariableType]
    classification_report,  # pyright: ignore[reportUnknownVariableType]
    confusion_matrix,  # pyright: ignore[reportUnknownVariableType]
    precision_recall_fscore_support,  # pyright: ignore[reportUnknownVariableType]
)


# ---------------------------------------------------------------------------
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


class RollingWindow:
    """Maintain a fixed-size rolling window of the most recent text lines."""

    def __init__(self, max_lines: int) -> None:
        """Initialize the rolling window.

        Args:
            max_lines: Maximum number of lines to retain.
        """
        self.max_lines = max_lines
        self.lines: list[str] = []

    def push(self, line: str) -> None:
        """Add a line to the rolling window and discard older ones if needed.

        Args:
            line: The text line to append.
        """
        self.lines.append(line)
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines :]

    def render(self) -> str:
        """Render the current window contents as a single newline-joined string.

        Returns:
            A string containing the most recent lines, separated by newline characters.
        """
        return "\n".join(self.lines)


def make_patched_print(
    window: RollingWindow,
    update_callback: Callable[[], None],
    wrapper_name: str,
) -> Callable[..., None]:
    """Create a patched print function that pushes text into a rolling window and triggers an update callback.

    Args:
        window: RollingWindow instance.
        update_callback: Function called after updating the window (usually to trigger a render).
        wrapper_name: Name used in warnings for unsupported print args.

    Returns:
        A callable that mimics print while capturing output.
    """

    def patched_print(
        *values: object,
        sep: str | None = " ",
        end: str | None = "\n",
        file: Any | None = None,
        flush: bool = False,
    ) -> None:
        # Warn on unsupported usage
        if file not in (None, sys.stdout):
            warn(f"{wrapper_name} does not support print(..., file=...)", stacklevel=2)
        if flush:
            warn(f"{wrapper_name} does not support print(..., flush=True)", stacklevel=2)

        # Format like builtin print
        text = (sep or " ").join(str(v) for v in values) + (end or "")
        for line in text.split("\n"):
            if line:
                window.push(line)

        # Trigger UI update
        update_callback()

    return patched_print


@contextmanager
def rolling_print(max_lines: int = 10, clear_on_exit: bool = False) -> Iterator[None]:
    """Provide a context where print output appears in a live rolling display.

    This context manager temporarily replaces builtins.print with a
    patched version that forwards text into a RollingWindow and renders
    it using a Rich Live region. The display is updated in place so new
    output replaces the previous frame rather than scrolling the terminal.

    The original print function is restored when the context exits.

    Args:
        max_lines: Maximum number of recent printed lines to keep visible.
        clear_on_exit: If true, the live region is cleared on exit.

    Yields:
        None. Code inside the context uses the patched print.
    """
    console = Console()
    window = RollingWindow(max_lines)

    with Live("", console=console, refresh_per_second=20) as live:
        original_print = builtins.print

        builtins.print = make_patched_print(
            window=window,
            update_callback=lambda: live.update(window.render()),
            wrapper_name="rolling_print",
        )

        try:
            yield
        finally:
            if clear_on_exit:
                live.update("")

            builtins.print = original_print


@contextmanager
def rolling_status(
    title: str,
    max_lines: int = 10,
    spinner: str = "dots",
    clear_on_exit: bool = False,
    indent: str = "    ",
    process_logs: Callable[[str], Text | str] | None = None,
    process_title: Callable[[str], Text | str] | None = None,
    add_elapsed_time: bool = False,
    post_clean_msg: str | Text | None = None,
) -> Iterator[None]:
    """Display a Rich status spinner with a live-updating log panel beneath it.

    Inside this context, print output is captured and added to a
    RollingWindow. The spinner text is updated after each logged line
    so the status view reflects the most recent messages. The logged
    lines appear below the title, optionally transformed by a custom
    processor and indented for readability.

    Args:
        title: Text or string displayed above the rolling logs.
        max_lines: Maximum number of recent log lines to show.
        spinner: Name of the spinner animation used by Rich.
        clear_on_exit: If true, the spinner and logs are cleared after exit.
        indent: Prefix applied to each rendered log line.
        process_logs:
            Optional callable applied to the raw log text before rendering.
            It may return either a string or a Rich Text object.
        process_title:
            Optional callable applied to the title before rendering.
            It receives the original title string.
        add_elapsed_time:
            If true, append the formatted elapsed time to the title on every update.

            When enabled, a periodic timer drives automatic refreshes so the elapsed
            time updates smoothly even while the main thread is busy and regardless
            of whether any new lines are printed (using signals).


    Yields:
        None. While active, print output is captured and shown below the spinner.
    """
    console = Console()
    window = RollingWindow(max_lines)
    spin = Spinner(name=spinner, text=title, style="status.spinner")

    start_time = None
    if add_elapsed_time:
        start_time = time.monotonic()

    def formatted_elapsed(final: bool = False) -> Text:
        if start_time is None:
            raise RuntimeError("never")

        seconds = int(time.monotonic() - start_time)

        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        time_txt = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

        txt = f" ({time_txt})" if not final else f" (took: {time_txt})"
        return Text(txt, style=Style(italic=True, color="grey46"))

    def update(final: bool = False) -> None:
        logs = window.render()
        logs = process_logs(logs) if process_logs else logs
        if not isinstance(logs, Text):
            logs = Text(logs)

        if final and post_clean_msg:
            cur_title = post_clean_msg
        else:
            cur_title = process_title(title) if process_title else title

        if not isinstance(cur_title, Text):
            cur_title = Text(cur_title)

        if add_elapsed_time:
            cur_title = cur_title + formatted_elapsed(final=final)

        if logs and not (final and clear_on_exit):
            indented = Text.assemble(*[Text(indent) + line + "\n" for line in logs.split("\n")])
            spin.text = cur_title + "\n" + indented
        else:
            spin.text = cur_title

        live.update(spin if not final else spin.text)

    stop_event = threading.Event()

    with Live("", console=console, refresh_per_second=20) as live:
        original_print = builtins.print
        builtins.print = make_patched_print(window, update_callback=update, wrapper_name="rolling_status")

        try:
            if add_elapsed_time:

                def handler(_signum: int, _frame: FrameType | None) -> Any:
                    update()
                    if stop_event.is_set():
                        _ = signal.setitimer(signal.ITIMER_REAL, 0)
                        _ = signal.signal(signal.SIGALRM, signal.SIG_DFL)

                _ = signal.signal(signal.SIGALRM, handler)
                _ = signal.setitimer(signal.ITIMER_REAL, 0.2, 0.2)  # 0.2s interval, repeat

            update()
            yield
        finally:
            stop_event.set()
            _ = signal.setitimer(signal.ITIMER_REAL, 0)
            _ = signal.signal(signal.SIGALRM, signal.SIG_DFL)

            builtins.print = original_print

            update(final=True)


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
