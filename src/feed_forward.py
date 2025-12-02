from __future__ import annotations

import random
from pathlib import Path
from typing import Any, override

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import classification_report  # pyright: ignore[reportUnknownVariableType]
from torch import Tensor, nn, optim

from src.ml_models import prepare_modeling_frame
from src.utils import plot_confusion_matrix, rolling_status

SEED = 0
INPUT_DIM = 56

console = Console()


# ---------------------------------------------------------------------------
# Seeding / device
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    np.random.seed(seed)  # noqa: NPY002
    _ = torch.manual_seed(seed)
    random.seed(seed)


def select_device() -> torch.device:
    """Select an available device and report it via Rich."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(
            f"There are {torch.cuda.device_count()} GPU(s) available. "
            f"Using: [bold]{torch.cuda.get_device_name(0)}[/bold]",
        )
    else:
        device = torch.device("cpu")
        console.print("No GPU available, using [bold]CPU[/bold].")
    return device


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class FeedForwardModel(nn.Module):
    """Feed-forward neural network with a fixed 56-dimensional input."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(INPUT_DIM, 48)
        self.linear2 = nn.Linear(48, 42)
        self.linear3 = nn.Linear(42, 36)
        self.linear4 = nn.Linear(36, 28)
        self.linear5 = nn.Linear(28, 20)
        self.linear6 = nn.Linear(20, 16)
        self.linear7 = nn.Linear(16, 12)
        self.linear8 = nn.Linear(12, 8)
        self.linear9 = nn.Linear(8, 4)
        self.linear10 = nn.Linear(4, 1)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with sigmoid activations after each linear layer."""
        y = torch.sigmoid(self.linear1(x))
        y = torch.sigmoid(self.linear2(y))
        y = torch.sigmoid(self.linear3(y))
        y = torch.sigmoid(self.linear4(y))
        y = torch.sigmoid(self.linear5(y))
        y = torch.sigmoid(self.linear6(y))
        y = torch.sigmoid(self.linear7(y))
        y = torch.sigmoid(self.linear8(y))
        y = torch.sigmoid(self.linear9(y))
        return torch.sigmoid(self.linear10(y))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_split_info(x_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Print basic information about the train split."""
    table = Table(show_header=False, box=None)
    table.add_row("X_train shape", f"{x_train.shape}")
    table.add_row("y_train shape", f"{y_train.shape}")
    table.add_row("Unique labels", f"{sorted(pd.Series(y_train).unique().tolist())}")
    console.print(table)


def _train_test_split(df_with_pca: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform an 80/20 train-test split on the modeling frame."""
    train_df = df_with_pca.sample(frac=0.8, random_state=SEED)
    test_df = df_with_pca.drop(train_df.index)

    x_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    x_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    _print_split_info(x_train, y_train)
    return x_train, x_test, y_train, y_test


def _print_model_parameters(model: nn.Module) -> None:
    """Print a concise model summary instead of full parameter dumps."""
    console.rule("[bold]Model summary[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Layer")
    table.add_column("Shape", justify="right")
    table.add_column("Params", justify="right")

    total_params = 0

    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        table.add_row(
            name,
            str(tuple(param.shape)),
            f"{count:,}",
        )

    console.print(table)
    console.print(f"[bold]Total parameters:[/bold] {total_params:,}")


def _evaluate_split(
    name: str,
    model: nn.Module,
    x_tensor: Tensor,
    y_tensor: Tensor,
) -> None:
    """Evaluate the model on a given split and print metrics."""
    console.rule(f"[bold]Evaluation on {name} split[/bold]")

    cpu_device = torch.device("cpu")
    model = model.to(cpu_device)
    x_eval = x_tensor.to(cpu_device)
    y_eval = y_tensor.to(cpu_device)

    _ = model.eval()
    with torch.no_grad():
        y_pred_proba = model(x_eval)
        y_pred = torch.where(
            y_pred_proba > 0.5,
            torch.tensor(1, device=cpu_device),
            torch.tensor(0, device=cpu_device),
        ).flatten()

    # Accuracy as in the original script (on tensors)
    acc = (y_pred == y_eval).float().mean().item()

    # Classification report via sklearn
    y_true_np = y_eval.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    report_dict: dict[str, Any] = classification_report(  # type: ignore[reportUnknownArgumentType]
        y_true_np,
        y_pred_np,
        output_dict=True,
    )

    # Accuracy table
    metrics_table = Table(show_header=True, header_style="bold")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_row("Accuracy", f"{acc:.4f}")
    console.print(metrics_table)

    # Detailed classification report table
    console.print("\n[bold]Classification report[/bold]")
    report_table = Table(show_header=True, header_style="bold")
    report_table.add_column("Class")
    report_table.add_column("Precision", justify="right")
    report_table.add_column("Recall", justify="right")
    report_table.add_column("F1 score", justify="right")
    report_table.add_column("Support", justify="right")

    for label, data in report_dict.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        report_table.add_row(
            label,
            f"{data['precision']:.4f}",
            f"{data['recall']:.4f}",
            f"{data['f1-score']:.4f}",
            f"{int(data['support'])}",
        )

    for avg_name in ("macro avg", "weighted avg"):
        data = report_dict[avg_name]
        report_table.add_row(
            avg_name,
            f"{data['precision']:.4f}",
            f"{data['recall']:.4f}",
            f"{data['f1-score']:.4f}",
            f"{int(data['support'])}",
        )

    console.print(report_table)

    console.print("\n[bold]Confusion matrix[/bold]")
    plot_confusion_matrix(name, y_true_np, y_pred_np)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_feed_forward(
    df_with_pca: pd.DataFrame,
    num_iterations: int = 20_000,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-6,
) -> None:
    """Train the feed-forward network and evaluate it on train and test splits."""
    set_global_seed(SEED)
    device = select_device()

    console.rule("[bold]Train/test split for feed-forward network[/bold]")
    x_train, x_test, y_train, y_test = _train_test_split(df_with_pca)

    # Ensure numeric dtype
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)

    # Convert to tensors
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long, device=device)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long, device=device)

    # Model and optimizer
    model = FeedForwardModel().to(device)
    _print_model_parameters(model)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    # Training loop (keeps the same iteration count and loss computation)
    # Training loop (keeps the same iteration count and loss computation)
    stopped = False
    with rolling_status(
        "Training feed-forward neural network",
        max_lines=10,
        clear_on_exit=True,
        add_elapsed_time=True,
    ):
        try:
            for itr in range(1, num_iterations + 1):
                optimizer.zero_grad()

                outputs = model(x_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor.float())

                loss.backward()
                _ = optimizer.step()  # pyright: ignore[reportUnknownVariableType]

                # Preserve the original logging semantics
                print(f"Iter {itr}/{num_iterations}: output_shape={tuple(outputs.shape)} loss={loss.item():.6f}")

        except KeyboardInterrupt:
            stopped = True

    if stopped:
        console.print("[red]Training interrupted by user, skipping evaluation[/red]")
        return

    # Evaluation (original script evaluated on train; here we do both)
    _evaluate_split("train", model, x_train_tensor, y_train_tensor)
    _evaluate_split("test", model, x_test_tensor, y_test_tensor)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_feed_forward(dataset_path: Path) -> None:
    """Run the full feed-forward pipeline on the final PCA dataset."""
    console.rule("[bold]Loading dataset for feed-forward network[/bold]")
    console.print(f"Reading dataset from: [italic]{dataset_path}[/italic]")
    df = pd.read_csv(dataset_path)

    console.print(
        Panel.fit(
            f"Rows: {df.shape[0]}  Columns: {df.shape[1]}",
            title="Raw dataset",
        ),
    )
    console.print(f"Number of rows in dataset: [bold]{df.shape[0]:,}[/bold]")
    console.print("\nRandom sample of 10 rows:")
    console.print(df.sample(n=min(10, len(df))))

    # Reuse the same modeling-frame preparation as the classical ML models
    model_df = prepare_modeling_frame(df)

    console.rule("[bold]Feed-forward neural network on PCA-based features[/bold]")
    train_feed_forward(model_df)
