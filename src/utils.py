from __future__ import annotations

import builtins
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.style import Style
from rich.text import Text
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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

            This will also enable starting up a thread that triggers re-rendering of the
            live rich view to make the timer update seamlessly (not just on new prints).


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
    ticker_thread: threading.Thread | None = None

    with Live("", console=console, refresh_per_second=20) as live:
        original_print = builtins.print
        builtins.print = make_patched_print(window, update_callback=update, wrapper_name="rolling_status")

        try:
            if add_elapsed_time:

                def ticker() -> None:
                    while not stop_event.is_set():
                        update()
                        time.sleep(0.2)

                ticker_thread = threading.Thread(target=ticker, daemon=True)
                ticker_thread.start()

            update()
            yield
        finally:
            stop_event.set()
            if ticker_thread is not None:
                ticker_thread.join(timeout=0.5)

            builtins.print = original_print

            update(final=True)


def plot_confusion_matrix(
    name: str,
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Sequence[str | int] | None = None,
) -> None:
    """Plot a confusion matrix using matplotlib."""
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))

    label_values = np.asarray(labels) if labels is not None else np.unique(np.concatenate((y_true_arr, y_pred_arr)))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=label_values)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_values)

    fig, ax = plt.subplots(figsize=(6, 5))
    _ = disp.plot(ax=ax, cmap="Blues", colorbar=False)
    _ = ax.set_xlabel("Predicted label")
    _ = ax.set_ylabel("True label")
    if name:
        _ = ax.set_title(f"{name.capitalize()} confusion matrix")
    fig.tight_layout()
    with Console().status(f"Showing confusion matrix for {name} (close figure to continue)"):
        plt.show()
