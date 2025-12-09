from __future__ import annotations

import datetime
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.special import softmax  # pyright: ignore[reportUnknownVariableType]
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
)
from transformers import (
    get_linear_schedule_with_warmup as _get_linear_schedule_with_warmup,  # type: ignore[reportUnknownVariableType]
)

from src.schema import LABEL_COLUMN, TEXT_COLUMN
from src.utils.evaluation import finalize_plot, render_evaluation_report
from src.utils.rich import rolling_status

SoftmaxFn = Callable[..., np.ndarray]
softmax_fn: SoftmaxFn = cast("SoftmaxFn", softmax)

SEED = 0
BERT_MODEL_NAME = "bert-base-uncased"

console = Console()
Batch = tuple[torch.Tensor, ...]
type DataLoaderBatch = DataLoader[Batch]
get_linear_schedule_with_warmup: Callable[..., LambdaLR] = _get_linear_schedule_with_warmup  # type: ignore[reportUnknownVariableType]


def _make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> LambdaLR:
    """Build a LambdaLR scheduler with linear warmup/decay."""
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )


@dataclass
class BertConfig:
    """Configuration for BERT fine-tuning."""

    max_length: int = 128
    batch_size: int = 32
    epochs: int = 4
    learning_rate: float = 2e-5
    adam_epsilon: float = 1e-8
    validation_ratio: float = 0.1


@dataclass
class EpochStats:
    """Container for per-epoch training and validation statistics."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    train_time: str
    val_time: str


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Select CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        console.print(
            Panel.fit(
                f"CUDA available\nGPUs: {gpu_count}\nUsing: {gpu_name}",
                title="Device",
            ),
        )
        return torch.device("cuda")

    console.print(Panel.fit("Using CPU", title="Device"))
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_time(elapsed: float) -> str:
    """Format elapsed seconds as a hh:mm:ss string."""
    rounded = round(elapsed)
    return str(datetime.timedelta(seconds=rounded))


# ---------------------------------------------------------------------------
def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode the label column to integers.

    Args:
        df: Input DataFrame containing a 'label' column.

    Returns:
        Tuple of (new DataFrame with encoded labels, fitted LabelEncoder).
    """
    if LABEL_COLUMN not in df.columns:
        msg = f"Dataset must contain a '{LABEL_COLUMN}' column"
        raise ValueError(msg)

    console.print("Encoding [bold]label[/bold] column")
    label_encoder = LabelEncoder()
    df_copy = df.copy()
    df_copy[LABEL_COLUMN] = label_encoder.fit_transform(df_copy[LABEL_COLUMN])
    return df_copy, label_encoder


def train_test_split_text(
    df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and test sets by row sampling.

    This mimics the sampling scheme used in the original notebook code.

    Args:
        df: Full dataset.
        seed: Random seed.

    Returns:
        (train_df, test_df) dataframes.
    """
    if TEXT_COLUMN not in df.columns:
        msg = f"Dataset must contain a '{TEXT_COLUMN}' column"
        raise ValueError(msg)

    console.rule("[bold]Train/test split[/bold]")
    train_df = df.sample(frac=0.8, random_state=seed).reset_index(drop=True)
    test_df = df.drop(train_df.index).reset_index(drop=True)

    table = Table(show_header=False, box=None)
    table.add_row("Train size", f"{train_df.shape[0]}")
    table.add_row("Test size", f"{test_df.shape[0]}")
    console.print(table)

    return train_df, test_df


def tokenize_texts(
    tokenizer: BertTokenizer,
    texts: pd.Series,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a series of texts with BERT tokenizer.

    Args:
        tokenizer: Pretrained BERT tokenizer.
        texts: Series of text documents.
        max_length: Maximum sequence length.

    Returns:
        Tuple of (input_ids, attention_mask) tensors.
    """
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = cast("torch.Tensor", encoded["input_ids"])
    attention_mask = cast("torch.Tensor", encoded["attention_mask"])
    return input_ids, attention_mask


def create_data_loaders(
    tokenizer: BertTokenizer,
    train_df: pd.DataFrame,
    config: BertConfig,
    device: torch.device,
) -> tuple[DataLoaderBatch, DataLoaderBatch]:
    """Create training and validation data loaders.

    Args:
        tokenizer: Pretrained BERT tokenizer.
        train_df: Training DataFrame with 'text_' and 'label' columns.
        config: BERT configuration.
        device: Target device (CPU or CUDA).

    Returns:
        Tuple of (train_dataloader, validation_dataloader).
    """
    console.rule("[bold]Preparing tensors[/bold]")
    sentences = train_df[TEXT_COLUMN]
    labels = train_df[LABEL_COLUMN].to_numpy()

    input_ids, attention_masks = tokenize_texts(
        tokenizer=tokenizer,
        texts=sentences,
        max_length=config.max_length,
    )
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    dataset = TensorDataset(
        input_ids.to(device),
        attention_masks.to(device),
        label_tensor,
    )

    train_size = int((1.0 - config.validation_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    console.print(
        Panel.fit(
            f"Train samples: {train_size}\nValidation samples: {val_size}",
            title="Dataset splits",
        ),
    )

    train_dataloader: DataLoaderBatch = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config.batch_size,
    )
    validation_dataloader: DataLoaderBatch = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=config.batch_size,
    )
    return train_dataloader, validation_dataloader


# ---------------------------------------------------------------------------
# Model and optimizer
# ---------------------------------------------------------------------------
def build_model(
    num_labels: int,
    device: torch.device,
) -> BertForSequenceClassification:
    """Instantiate a BERT sequence classification model."""
    console.rule("[bold]Loading BERT model[/bold]")
    console.print(f"Model: [italic]{BERT_MODEL_NAME}[/italic]  Labels: {num_labels}")

    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )
    _ = model.to(device)  # pyright: ignore[reportArgumentType]
    return model


def build_optimizer_and_scheduler(
    model: BertForSequenceClassification,
    train_dataloader: DataLoaderBatch,
    config: BertConfig,
) -> tuple[AdamW, LambdaLR]:
    """Create AdamW optimizer and linear warmup scheduler."""
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        eps=config.adam_epsilon,
    )

    total_steps = len(train_dataloader) * config.epochs
    scheduler = _make_scheduler(optimizer, total_steps)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------
def _flat_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute simple accuracy given logits and labels."""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return float(np.sum(pred_flat == labels_flat) / len(labels_flat))


def _run_validation_epoch(
    model: BertForSequenceClassification,
    dataloader: DataLoaderBatch,
    device: torch.device,
) -> tuple[float, float, str]:
    """Run one validation epoch.

    Returns:
        Tuple of (avg_accuracy, avg_loss, validation_time_str).
    """
    t0 = time.time()
    _ = model.eval()

    total_eval_accuracy = 0.0
    total_eval_loss = 0.0

    for batch in dataloader:
        b_ids, b_mask, b_labels = (t.to(device) for t in batch)

        with torch.no_grad():
            result = model(
                b_ids,
                token_type_ids=None,
                attention_mask=b_mask,
                labels=b_labels,
                return_dict=True,
            )

        loss = result.loss
        logits = result.logits

        total_eval_loss += float(loss.item())
        logits_np = logits.detach().cpu().numpy()
        labels_np = b_labels.detach().cpu().numpy()
        total_eval_accuracy += _flat_accuracy(logits_np, labels_np)

    avg_accuracy = total_eval_accuracy / len(dataloader)
    avg_loss = total_eval_loss / len(dataloader)
    val_time = _format_time(time.time() - t0)

    return avg_accuracy, avg_loss, val_time


def train_model(
    model: BertForSequenceClassification,
    train_dataloader: DataLoaderBatch,
    validation_dataloader: DataLoaderBatch,
    optimizer: AdamW,
    scheduler: Any,
    device: torch.device,
    config: BertConfig,
) -> list[EpochStats]:
    """Fine-tune BERT classifier and collect training statistics."""
    console.rule("[bold]Training BERT classifier[/bold]")

    set_seed(SEED)
    stats: list[EpochStats] = []

    with rolling_status(
        "Training BERT classifier",
        max_lines=10,
        clear_on_exit=True,
        add_elapsed_time=True,
    ):
        for epoch_idx in range(config.epochs):
            torch.cuda.empty_cache()
            epoch_num = epoch_idx + 1

            console.rule(f"[bold]Epoch {epoch_num}/{config.epochs}[/bold]")
            t0 = time.time()
            total_train_loss = 0.0

            _ = model.train()

            for step, batch in enumerate(train_dataloader, start=1):
                b_ids, b_mask, b_labels = (t.to(device) for t in batch)

                model.zero_grad()

                result = model(
                    b_ids,
                    token_type_ids=None,
                    attention_mask=b_mask,
                    labels=b_labels,
                    return_dict=True,
                )

                loss = result.loss
                total_train_loss += float(loss.item())

                loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()  # pyright: ignore[reportUnusedCallResult]
                scheduler.step()

                if step % 40 == 0 or step == len(train_dataloader):
                    elapsed = _format_time(time.time() - t0)
                    avg_so_far = total_train_loss / step
                    console.print(
                        f"Batch {step:>4}/{len(train_dataloader):>4}  "
                        f"Train loss: {avg_so_far:.4f}  Elapsed: {elapsed}",
                    )

            avg_train_loss = total_train_loss / len(train_dataloader)
            train_time = _format_time(time.time() - t0)

            console.print(f"Average training loss: {avg_train_loss:.4f}")
            console.print(f"Training epoch took: {train_time}")

            console.print("Running validation")
            val_accuracy, val_loss, val_time = _run_validation_epoch(
                model=model,
                dataloader=validation_dataloader,
                device=device,
            )

            console.print(f"Validation accuracy: {val_accuracy:.4f}")
            console.print(f"Validation loss: {val_loss:.4f}")
            console.print(f"Validation took: {val_time}")

            stats.append(
                EpochStats(
                    epoch=epoch_num,
                    train_loss=avg_train_loss,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    train_time=train_time,
                    val_time=val_time,
                ),
            )

    console.print("Training complete")
    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_loss_curves(epoch_stats: list[EpochStats], save_path: Path, show_plot: bool = False) -> None:
    """Plot training and validation loss curves.

    Args:
        epoch_stats: Per-epoch training/validation metrics.
        save_path: Destination for the loss plot image.
        show_plot: Whether to display the plot interactively.
    """
    if not epoch_stats:
        console.print("[yellow]No epoch statistics available, skipping loss plot[/yellow]")
        return

    df_stats = pd.DataFrame(
        [
            {
                "epoch": s.epoch,
                "train_loss": s.train_loss,
                "val_loss": s.val_loss,
            }
            for s in epoch_stats
        ],
    ).set_index("epoch")

    fig = plt.figure(figsize=(10, 5))
    _ = plt.plot(df_stats.index, df_stats["train_loss"], marker="o", label="Training")
    _ = plt.plot(df_stats.index, df_stats["val_loss"], marker="o", label="Validation")

    _ = plt.title("Training and Validation Loss")
    _ = plt.xlabel("Epoch")
    _ = plt.ylabel("Loss")
    _ = plt.xticks(df_stats.index.tolist())
    _ = plt.legend()
    plt.tight_layout()
    status_msg = "Showing BERT loss curves"
    finalize_plot(
        fig=fig,
        save_path=save_path,
        show=show_plot,
        status_msg=status_msg,
    )


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------
def _build_prediction_dataloader(
    tokenizer: BertTokenizer,
    texts: pd.Series,
    labels: np.ndarray,
    config: BertConfig,
    device: torch.device,
) -> DataLoaderBatch:
    """Build prediction DataLoader for the test set."""
    input_ids, attention_masks = tokenize_texts(
        tokenizer=tokenizer,
        texts=texts,
        max_length=config.max_length,
    )
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    dataset = TensorDataset(
        input_ids.to(device),
        attention_masks.to(device),
        label_tensor,
    )
    return DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=config.batch_size,
    )


def evaluate_on_split(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    split_df: pd.DataFrame,
    config: BertConfig,
    device: torch.device,
    results_dir: Path,
    split_name: str,
    show_plots: bool = False,
) -> None:
    """Evaluate the fine-tuned model on a given split and log artifacts.

    Args:
        model: Trained BERT classifier.
        tokenizer: Tokenizer used during training.
        split_df: DataFrame containing ``text_`` and ``label`` columns for evaluation.
        config: Training/evaluation configuration.
        device: Torch device for running inference.
        results_dir: Directory to store evaluation plots and reports.
        split_name: Name of the split being evaluated (e.g., ``train`` or ``test``).
        show_plots: Whether to display plots interactively while saving them.
    """
    console.rule(f"[bold]{split_name.capitalize()} set evaluation[/bold]")
    sentences = split_df[TEXT_COLUMN].astype(str)
    labels = split_df[LABEL_COLUMN].to_numpy()

    prediction_dataloader = _build_prediction_dataloader(
        tokenizer=tokenizer,
        texts=sentences,
        labels=labels,
        config=config,
        device=device,
    )

    _ = model.eval()
    predictions: list[np.ndarray] = []
    true_labels: list[np.ndarray] = []

    console.print(f"Predicting labels for {len(sentences):,} {split_name} sentences")

    for batch in prediction_dataloader:
        b_ids, b_mask, b_labels = (t.to(device) for t in batch)

        with torch.no_grad():
            result = model(
                b_ids,
                token_type_ids=None,
                attention_mask=b_mask,
                return_dict=True,
            )

        logits = result.logits.detach().cpu().numpy()
        labels_cpu = b_labels.detach().cpu().numpy()

        predictions.append(logits)
        true_labels.append(labels_cpu)

    pred_np = np.concatenate(predictions, axis=0)
    true_np = np.concatenate(true_labels, axis=0)

    predicted = np.argmax(pred_np, axis=1)

    bert_results = results_dir / "bert"
    bert_results.mkdir(parents=True, exist_ok=True)
    bert_split_dir = bert_results / split_name

    # Convert logits to probabilities via softmax
    probs: np.ndarray = softmax_fn(pred_np, axis=1)

    render_evaluation_report(
        name=f"BERT ({split_name})",
        y_true=true_np,
        y_pred=predicted,
        y_proba=probs,
        console=console,
        results_dir=bert_split_dir,
        show_plots=show_plots,
    )

    # Hidden-state norm distribution on logits (as a proxy)
    logits_norm = np.linalg.norm(pred_np, axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    _ = ax.hist(logits_norm, bins=30, color="steelblue", alpha=0.8)
    _ = ax.set_title("BERT logits norm distribution")
    _ = ax.set_xlabel("Norm")
    _ = ax.set_ylabel("Count")
    fig.tight_layout()
    finalize_plot(
        fig=fig,
        save_path=bert_split_dir / "logits_norm_distribution.png",
        show=show_plots,
        status_msg="Showing BERT logits norm distribution",
    )


# ---------------------------------------------------------------------------
# t-SNE visualization
# ---------------------------------------------------------------------------
def compute_bert_embeddings(
    model: BertForSequenceClassification,
    dataloader: DataLoaderBatch,
    device: torch.device,
) -> np.ndarray:
    """Compute CLS token embeddings for all samples in a dataloader."""
    _ = model.eval()
    all_embeddings: list[np.ndarray] = []

    for batch in dataloader:
        b_ids, b_mask, _ = (t.to(device) for t in batch)

        with torch.no_grad():
            outputs = model(
                b_ids,
                attention_mask=b_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Last hidden state, CLS token
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            msg = "Model did not return hidden states; enable output_hidden_states"
            raise RuntimeError(msg)

        cls_embeddings = hidden_states[-1][:, 0, :].detach().cpu().numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    show_plot: bool = False,
) -> None:
    """Project embeddings to 2D with t-SNE and show a scatter plot.

    Args:
        embeddings: CLS embeddings to project.
        labels: Numeric labels corresponding to each embedding.
        save_path: Destination where the t-SNE plot will be written.
        show_plot: Whether to display the plot interactively.
    """
    console.rule("[bold]t-SNE embedding[/bold]")
    console.print("Running t-SNE projection on BERT embeddings")

    tsne = TSNE(n_components=2, random_state=SEED)
    embedded = cast("np.ndarray", tsne.fit_transform(embeddings))
    labels_np = np.asarray(labels)

    fig = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=labels_np,
        cmap="coolwarm",
        alpha=0.8,
    )
    _ = plt.colorbar(scatter, label="Label")
    _ = plt.title("t-SNE visualization of BERT CLS embeddings")
    _ = plt.xlabel("Component 1")
    _ = plt.ylabel("Component 2")
    plt.tight_layout()
    finalize_plot(
        fig=fig,
        save_path=save_path,
        show=show_plot,
        status_msg="Showing BERT t-SNE plot",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_bert_classifier(
    *,
    df: pd.DataFrame,
    config: BertConfig | None = None,
    results_dir: Path,
    show_plots: bool = False,
) -> None:
    """Run the full BERT fine-tuning pipeline on the given dataset without mutating it.

    If a previously saved model exists in `output_dir`, training is skipped
    and the model/tokenizer are loaded from disk.

    Args:
        df: Pre-loaded dataframe with at least 'text_' and 'label'.
        config: Optional configuration for BERT training.
        results_dir: Directory where plots, evaluation artifacts and the model are stored.
        show_plots: Whether to display plots interactively while saving them.
    """
    bert_results = results_dir / "bert"
    bert_results.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = BertConfig()

    device = get_device()
    set_seed(SEED)

    encoded_df, _ = encode_labels(df)
    train_df, test_df = train_test_split_text(encoded_df, seed=SEED)

    tokenizer: BertTokenizer
    model: BertForSequenceClassification
    epoch_stats: list[EpochStats] | None = None
    model_path = bert_results / "pytorch_model.bin"
    tokenizer_path = bert_results / "tokenizer_config.json"
    model_exists = model_path.exists() and tokenizer_path.exists()

    if model_exists:
        console.rule("[bold]Loading existing BERT model[/bold]")
        console.print(f"Found saved model at: [italic]{bert_results}[/italic]")
        tokenizer = cast("BertTokenizer", BertTokenizer.from_pretrained(bert_results))
        model = BertForSequenceClassification.from_pretrained(bert_results)
        _ = model.to(device)  # pyright: ignore[reportArgumentType]
    else:
        tokenizer = cast(
            "BertTokenizer",
            BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=True),
        )
        train_dataloader, val_dataloader = create_data_loaders(
            tokenizer=tokenizer,
            train_df=train_df,
            config=config,
            device=device,
        )

        model = build_model(num_labels=encoded_df[LABEL_COLUMN].nunique(), device=device)
        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model,
            train_dataloader=train_dataloader,
            config=config,
        )

        t0 = time.time()
        epoch_stats = train_model(
            model=model,
            train_dataloader=train_dataloader,
            validation_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
        )
        total_time = _format_time(time.time() - t0)
        console.print(f"Total training time: [bold]{total_time}[/bold]")

        loss_path = bert_results / "loss_curves.png"
        plot_loss_curves(epoch_stats, save_path=loss_path, show_plot=show_plots)

        # Save model and tokenizer
        console.rule("[bold]Saving model[/bold]")
        console.print(f"Saving model to: [italic]{bert_results}[/italic]")

        model_to_save = cast("BertForSequenceClassification", model.module) if hasattr(model, "module") else model
        _ = model_to_save.save_pretrained(bert_results)
        _ = tokenizer.save_pretrained(bert_results)

    # Evaluate on train and test sets
    evaluate_on_split(
        model=model,
        tokenizer=tokenizer,
        split_df=train_df,
        config=config,
        device=device,
        results_dir=results_dir,
        split_name="train",
        show_plots=show_plots,
    )
    evaluate_on_split(
        model=model,
        tokenizer=tokenizer,
        split_df=test_df,
        config=config,
        device=device,
        results_dir=results_dir,
        split_name="test",
        show_plots=show_plots,
    )

    console.print("Computing BERT embeddings for t-SNE")
    prediction_dataloader = _build_prediction_dataloader(
        tokenizer=tokenizer,
        texts=test_df[TEXT_COLUMN].astype(str),
        labels=test_df[LABEL_COLUMN].to_numpy(),
        config=config,
        device=device,
    )
    embeddings = compute_bert_embeddings(
        model=model,
        dataloader=prediction_dataloader,
        device=device,
    )
    tsne_path = bert_results / "tsne.png"
    plot_tsne_embeddings(
        embeddings,
        np.asarray(test_df[LABEL_COLUMN].values),
        save_path=tsne_path,
        show_plot=show_plots,
    )
