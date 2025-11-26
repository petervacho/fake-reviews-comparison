from __future__ import annotations

import builtins
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from warnings import warn

from rich.console import Console
from rich.live import Live


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


@contextmanager
def rolling_print(max_lines: int = 10) -> Iterator[None]:
    """Replace the global print function with a live-updating rolling view.

    This context manager replaces builtins.print with a patched version that
    sends printed messages into a Rich Live display. Only the most recent
    max_lines messages are shown, and the output region is updated in place.

    The original print function is restored on exit.

    Args:
        max_lines: Number of recent printed lines to retain in the live display.

    Yields:
        None. The block under this context will use the patched print behavior.
    """
    console = Console()
    window = RollingWindow(max_lines)

    with Live("", console=console, refresh_per_second=20) as live:
        original_print = builtins.print

        def patched_print(
            *values: object,
            sep: str | None = " ",
            end: str | None = "\n",
            file: Any | None = None,
            flush: bool = False,
        ) -> None:
            """Replacement print function that updates the rolling rich window.

            Args:
                *args: Positional arguments to print.
                sep: Separator between printed arguments.
                end: String appended after the printed arguments.
                file: Ignored. Present only to match the builtin signature.
                flush: Ignored. Present only to match the builtin signature.
            """
            # file and flush kwargs are ignored, produce a warning if the caller used them,
            # as they will not be handled by this wrapper, which might result in unexpected
            # behavior.
            if file not in (None, sys.stdout):
                warn("rolling_print wrapper doesn't support print calls with 'file' kwarg", stacklevel=2)
            if flush is True:
                warn("rolling_print wrapper doesn't support print calls with 'flush' kwarg", stacklevel=2)

            # Mimic builtin print argument formatting
            used_sep = sep if sep is not None else " "
            used_end = end if end is not None else ""
            body = used_sep.join(str(arg) for arg in values)
            message = f"{body}{used_end}"

            # Split into logical lines, ignoring empty trailing fragments
            for line in message.split("\n"):
                if line:
                    window.push(line)

            live.update(window.render())

        builtins.print = patched_print
        try:
            yield
        finally:
            builtins.print = original_print
