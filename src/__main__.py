from __future__ import annotations

from pathlib import Path

from src.bert import run_bert_classifier
from src.dataset import generate_final_dataset
from src.feed_forward import run_feed_forward
from src.ml_models import run_ml_models

# Dataset paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REVIEWS_DATASET_PATH = DATA_DIR / "fake_reviews_dataset.csv"
FINAL_DATASET_PATH = DATA_DIR / "final_data.csv"


def main() -> None:
    """Main script entry-point."""
    generate_final_dataset(raw_dataset_path=REVIEWS_DATASET_PATH, final_dataset_path=FINAL_DATASET_PATH)
    run_ml_models(FINAL_DATASET_PATH)
    run_feed_forward(FINAL_DATASET_PATH)
    run_bert_classifier(FINAL_DATASET_PATH, output_dir=DATA_DIR, show_tsne=True, show_loss_plot=True)


if __name__ == "__main__":
    main()
