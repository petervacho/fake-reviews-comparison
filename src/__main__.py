from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

from src.bert import run_bert_classifier
from src.dataset import generate_final_dataset
from src.feed_forward import run_feed_forward
from src.ml_models import run_ml_models

# Dataset paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REVIEWS_DATASET_PATH = DATA_DIR / "fake_reviews_dataset.csv"
FINAL_DATASET_PATH = DATA_DIR / "final_data.csv"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SHOW_PLOTS = False


def main() -> None:
    """Main script entry-point."""
    mpl.use("Agg")

    generate_final_dataset(
        raw_dataset_path=REVIEWS_DATASET_PATH,
        final_dataset_path=FINAL_DATASET_PATH,
        results_dir=RESULTS_DIR,
        show_plots=SHOW_PLOTS,
    )

    run_ml_models(
        dataset_path=FINAL_DATASET_PATH,
        results_dir=RESULTS_DIR,
        show_plots=SHOW_PLOTS,
    )
    run_feed_forward(
        dataset_path=FINAL_DATASET_PATH,
        results_dir=RESULTS_DIR,
        show_plots=SHOW_PLOTS,
    )
    run_bert_classifier(
        dataset_path=FINAL_DATASET_PATH,
        output_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        show_plots=SHOW_PLOTS,
    )


if __name__ == "__main__":
    main()
