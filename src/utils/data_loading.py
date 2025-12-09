from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel


def preview_dataset(
    df: pd.DataFrame,
    *,
    console: Console,
    dataset_label: str = "Dataset",
    sample_size: int | None = 10,
) -> None:
    """Display basic dataset information and an optional random sample."""
    console.print(Panel.fit(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}", title=dataset_label))
    console.print(f"Number of rows in dataset: [bold]{df.shape[0]:,}[/bold]")
    if sample_size and len(df) > 0:
        console.print(f"\nRandom sample of {min(sample_size, len(df))} rows:")
        console.print(df.sample(n=min(sample_size, len(df))))


def load_and_preview_dataset(
    *,
    dataset_path: Path,
    console: Console,
    sample_size: int | None = 10,
) -> pd.DataFrame:
    """Load a dataset from disk and print a concise overview."""
    console.rule("[bold]Loading dataset[/bold]")
    console.print(f"Reading dataset from: [italic]{dataset_path}[/italic]")
    df = pd.read_csv(dataset_path)
    preview_dataset(df, console=console, dataset_label="Raw dataset", sample_size=sample_size)
    return df
