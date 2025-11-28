from __future__ import annotations

from pathlib import Path

from src.dataset import generate_final_dataset
from src.feed_forward import run_feed_forward

# Dataset paths
DATASET_DIR = Path(__file__).resolve().parent.parent / "data"
REVIEWS_DATASET_PATH = DATASET_DIR / "fake_reviews_dataset.csv"
FINAL_DATASET_PATH = DATASET_DIR / "final_data.csv"


def main() -> None:
    """Main script entry-point."""
    generate_final_dataset(raw_dataset_path=REVIEWS_DATASET_PATH, final_dataset_path=FINAL_DATASET_PATH)
    # run_ml_models(FINAL_DATASET_PATH) # noqa: ERA001
    run_feed_forward(FINAL_DATASET_PATH)


if __name__ == "__main__":
    main()
