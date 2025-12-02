from __future__ import annotations

import datetime
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from src.utils import render_evaluation_report, rolling_status

SEED = 0
BERT_MODEL_NAME = "bert-base-uncased"

console = Console()


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
# Data loading and preparation
# ---------------------------------------------------------------------------
def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the final dataset from CSV."""
    console.rule("[bold]Loading dataset[/bold]")
    console.print(f"Reading dataset from: [italic]{dataset_path}[/italic]")

    df = pd.read_csv(dataset_path)

    console.print(
        Panel.fit(
            f"Rows: {df.shape[0]}  Columns: {df.shape[1]}",
            title="Raw dataset",
        ),
    )
    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode the 'label' column to integers.

    Args:
        df: Input DataFrame containing a 'label' column.

    Returns:
        Tuple of (new DataFrame with encoded labels, fitted LabelEncoder).
    """
    if "label" not in df.columns:
        msg = "Dataset must contain a 'label' column"
        raise ValueError(msg)

    console.print("Encoding [bold]label[/bold] column")
    label_encoder = LabelEncoder()
    df_copy = df.copy()
    df_copy["label"] = label_encoder.fit_transform(df_copy["label"])
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
    if "text_" not in df.columns:
        msg = "Dataset must contain a 'text_' column"
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
    return encoded["input_ids"], encoded["attention_mask"]


def create_data_loaders(
    tokenizer: BertTokenizer,
    train_df: pd.DataFrame,
    config: BertConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
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
    sentences = train_df["text_"]
    labels = train_df["label"].to_numpy()

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

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config.batch_size,
    )
    validation_dataloader = DataLoader(
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
    _ = model.to(device)
    return model


def build_optimizer_and_scheduler(
    model: BertForSequenceClassification,
    train_dataloader: DataLoader,
    config: BertConfig,
) -> tuple[AdamW, Any]:
    """Create AdamW optimizer and linear warmup scheduler."""
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        eps=config.adam_epsilon,
    )

    total_steps = len(train_dataloader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
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
    dataloader: DataLoader,
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
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                _ = optimizer.step()
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
def plot_loss_curves(epoch_stats: list[EpochStats]) -> None:
    """Plot training and validation loss curves."""
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

    _ = plt.figure(figsize=(10, 5))
    _ = plt.plot(df_stats.index, df_stats["train_loss"], marker="o", label="Training")
    _ = plt.plot(df_stats.index, df_stats["val_loss"], marker="o", label="Validation")

    _ = plt.title("Training and Validation Loss")
    _ = plt.xlabel("Epoch")
    _ = plt.ylabel("Loss")
    _ = plt.xticks(df_stats.index.tolist())
    _ = plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------
def _build_prediction_dataloader(
    tokenizer: BertTokenizer,
    texts: pd.Series,
    labels: np.ndarray,
    config: BertConfig,
    device: torch.device,
) -> DataLoader:
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


def evaluate_on_test(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    test_df: pd.DataFrame,
    config: BertConfig,
    device: torch.device,
) -> None:
    """Evaluate fine-tuned model on the held-out test set."""
    console.rule("[bold]Test set evaluation[/bold]")
    sentences_test = test_df["text_"].astype(str)
    labels_test = test_df["label"].to_numpy()

    prediction_dataloader = _build_prediction_dataloader(
        tokenizer=tokenizer,
        texts=sentences_test,
        labels=labels_test,
        config=config,
        device=device,
    )

    _ = model.eval()
    predictions: list[np.ndarray] = []
    true_labels: list[np.ndarray] = []

    console.print(f"Predicting labels for {len(sentences_test):,} test sentences")

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
    render_evaluation_report(
        name="BERT",
        y_true=true_np,
        y_pred=predicted,
        console=console,
    )


# ---------------------------------------------------------------------------
# t-SNE visualization
# ---------------------------------------------------------------------------
def compute_bert_embeddings(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
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
) -> None:
    """Project embeddings to 2D with t-SNE and show a scatter plot."""
    console.rule("[bold]t-SNE embedding[/bold]")
    console.print("Running t-SNE projection on BERT embeddings")

    tsne = TSNE(n_components=2, random_state=SEED)
    embedded = tsne.fit_transform(embeddings)

    _ = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.8,
    )
    _ = plt.colorbar(scatter, label="Label")
    _ = plt.title("t-SNE visualization of BERT CLS embeddings")
    _ = plt.xlabel("Component 1")
    _ = plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_bert_classifier(
    dataset_path: Path,
    output_dir: Path,
    config: BertConfig | None = None,
    show_loss_plot: bool = True,
    show_tsne: bool = False,
) -> None:
    """Run the full BERT fine-tuning pipeline on the given dataset.

    Args:
        dataset_path: Path to the CSV file with at least 'text_' and 'label'.
        config: Optional configuration for BERT training.
        output_dir: Directory to save model and tokenizer.
        show_loss_plot: Whether to display training/validation loss curves.
        show_tsne: Whether to compute and display t-SNE plot on test embeddings.
    """
    if config is None:
        config = BertConfig()

    device = get_device()
    set_seed(SEED)

    df_raw = load_dataset(dataset_path)
    df, _ = encode_labels(df_raw)
    train_df, test_df = train_test_split_text(df, seed=SEED)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=True)

    train_dataloader, val_dataloader = create_data_loaders(
        tokenizer=tokenizer,
        train_df=train_df,
        config=config,
        device=device,
    )

    model = build_model(num_labels=df["label"].nunique(), device=device)
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

    if show_loss_plot:
        plot_loss_curves(epoch_stats)

    # Save model and tokenizer
    save_dir = output_dir or Path("./bert_model")
    save_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]Saving model[/bold]")
    console.print(f"Saving model to: [italic]{save_dir}[/italic]")

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Evaluate on test set
    evaluate_on_test(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        config=config,
        device=device,
    )

    # Optional t-SNE visualization on test embeddings
    if show_tsne:
        console.print("Computing BERT embeddings for t-SNE")
        prediction_dataloader = _build_prediction_dataloader(
            tokenizer=tokenizer,
            texts=test_df["text_"].astype(str),
            labels=test_df["label"].values,
            config=config,
            device=device,
        )
        embeddings = compute_bert_embeddings(
            model=model,
            dataloader=prediction_dataloader,
            device=device,
        )
        plot_tsne_embeddings(embeddings, test_df["label"].values)
