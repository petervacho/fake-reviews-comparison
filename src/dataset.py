"""Module for generating the final preprocessed dataset with PCA.

This script loads the raw fake reviews dataset, performs all text preprocessing,
feature engineering, readability statistics, POS tagging, vector transformations,
and applies PCA to produce a compact numerical dataset suitable for modeling.

The output is written to data/final_data.csv.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import textstat
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from rich.console import Console
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

from src.schema import LABEL_COLUMN, RATING_COLUMN, RAW_COLUMNS, TEXT_COLUMN
from src.utils.evaluation import finalize_plot

console = Console()


LEMMATIZED_TEXT_COLUMN = "lemmatized_text"
STEMMED_TEXT_COLUMN = "stemmed_text"
CLEANED_TEXT_COLUMN = "cleaned_text"
STOPWORD_REMOVED_TEXT_COLUMN = "CleanedText"
SENTIMENT_POLARITY_COLUMN = "sentiment_polarity"
SEMANTIC_ALIGNMENT_COLUMN = "semantic_alignment"
POS_COUNTS_COLUMN = "pos_counts"
WORD_COUNT_COLUMN = "word_count"
CHARACTER_COUNT_COLUMN = "character_count"
CAPITALIZED_COUNT_COLUMN = "capitalized_count"
EXCLAMATION_QUESTION_COUNT_COLUMN = "exclamation_question_count"
PUNCTUATION_COUNT_COLUMN = "punctuation_count"

READABILITY_COLUMNS = (
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    "automated_readability_index",
    "coleman_liau_index",
    "dale_chall_readability_score",
)
READABILITY_FUNCTIONS: dict[str, Callable[[str], float]] = {
    READABILITY_COLUMNS[0]: textstat.flesch_reading_ease,  # pyright: ignore[reportAttributeAccessIssue]
    READABILITY_COLUMNS[1]: textstat.flesch_kincaid_grade,  # pyright: ignore[reportAttributeAccessIssue]
    READABILITY_COLUMNS[2]: textstat.gunning_fog,  # pyright: ignore[reportAttributeAccessIssue]
    READABILITY_COLUMNS[3]: textstat.smog_index,  # pyright: ignore[reportAttributeAccessIssue]
    READABILITY_COLUMNS[4]: textstat.automated_readability_index,  # pyright: ignore[reportAttributeAccessIssue]
    READABILITY_COLUMNS[5]: textstat.coleman_liau_index,  # pyright: ignore[reportAttributeAccessIssue]
    READABILITY_COLUMNS[6]: textstat.dale_chall_readability_score,  # pyright: ignore[reportAttributeAccessIssue]
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def ensure_nltk_data() -> None:
    """Download required NLTK resources if missing.

    This function is safe to call multiple times.
    """
    resources = [
        "wordnet",
        "stopwords",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "punkt",
        "punkt_tab",
    ]
    with console.status("Ensuring NLTK resources are available"):
        for resource in resources:
            _ = nltk.download(resource, quiet=True)


def get_stop_words() -> set[str]:
    """Return English stop words as a set."""
    return set(stopwords.words("english"))


def lemmatize_text(text: str, lemmatizer: WordNetLemmatizer) -> str:
    """Lemmatize verbs and then lemmatize again without POS.

    Args:
        text: Input text.
        lemmatizer: NLTK WordNet lemmatizer instance.

    Returns:
        Lemmatized text.
    """
    words = text.split()
    lem_v = [lemmatizer.lemmatize(word, pos="v") for word in words]
    return " ".join(lemmatizer.lemmatize(w) for w in lem_v)


def stem_text(text: str, stemmer: PorterStemmer) -> str:
    """Stem words in the given text.

    Args:
        text: Input text.
        stemmer: NLTK Porter stemmer instance.

    Returns:
        Stemmed text.
    """
    return " ".join(stemmer.stem(word) for word in text.split())


def remove_noise(text: str) -> str:
    """Remove HTML tags, digits and most punctuation from text.

    Args:
        text: Input text.

    Returns:
        Cleaned text.
    """
    punctuations = "!\"#$%&'()*+/:;<=>?@[\\]^_.`{|}~"
    words = text.split()
    cleaned: list[str] = []
    for word in words:
        word = re.sub(r"(<.*?>)", "", word)
        if "-" not in word:
            word = re.sub(r"(\W|\d)", " ", word)
        else:
            word = "".join(c for c in word if c not in punctuations)
        cleaned.append(word.strip())
    return " ".join(cleaned)


def remove_stop_words(text: str, stop_words: set[str]) -> str:
    """Remove stop words from text."""
    return " ".join(word for word in text.split() if word not in stop_words)


def word_count(text: str) -> int:
    """Count words in given text."""
    return len(text.split())


def character_count(text: str) -> int:
    """Count characters in given text."""
    return len(text)


def capitalized_letters_count(text: str) -> int:
    """Count uppercase characters in given text."""
    return sum(1 for c in text if c.isupper())


def exclamation_question_count(text: str) -> int:
    """Count '!' and '?' characters in given text."""
    return sum(1 for c in text if c in {"!", "?"})


def punctuation_count(text: str) -> int:
    """Count punctuation characters in given text."""
    return sum(1 for c in text if c in string.punctuation)


def part_of_speech_count(text: str) -> Counter[str]:
    """Count POS tags in text.

    Args:
        text: Input text.

    Returns:
        Counter mapping POS tag to occurrence count.
    """
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)  # pyright: ignore[reportUnknownVariableType]
    return Counter(tag for _, tag in tags)  # pyright: ignore[reportUnknownVariableType]


def readability_metrics(text: str) -> dict[str, float]:
    """Compute readability metrics for given text.

    Args:
        text: Input text.

    Returns:
        Dictionary of readability scores.
    """
    return {name: float(func(text)) for name, func in READABILITY_FUNCTIONS.items()}


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Ensure that the required columns are present."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def basic_clean(raw_df: pd.DataFrame, *, text_column: str = TEXT_COLUMN) -> pd.DataFrame:
    """Lowercase text and perform lemmatization, stemming and stop word removal."""
    stop_words = get_stop_words()
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    text_series = raw_df[text_column].astype(str).str.lower()
    lemmatized = text_series.apply(lemmatize_text, lemmatizer=lemmatizer)
    stemmed = lemmatized.apply(stem_text, stemmer=stemmer)
    cleaned = stemmed.apply(remove_noise)
    without_stopwords = cleaned.apply(remove_stop_words, stop_words=stop_words)

    return raw_df.assign(
        **{
            text_column: text_series,
            LEMMATIZED_TEXT_COLUMN: lemmatized,
            STEMMED_TEXT_COLUMN: stemmed,
            CLEANED_TEXT_COLUMN: cleaned,
            STOPWORD_REMOVED_TEXT_COLUMN: without_stopwords,
        },
    )


def _get_polarity(text: str) -> float:
    """Return TextBlob polarity as float in [-1, 1]."""
    blob = TextBlob(text)
    # type ignore because TextBlob sentiment type is not well annotated
    return float(blob.sentiment.polarity)  # pyright: ignore[reportAttributeAccessIssue]


def compute_sentiment_features(text_series: pd.Series, ratings: pd.Series) -> pd.DataFrame:
    """Compute numeric sentiment and semantic alignment features.

    This extracts the TextBlob polarity score for each text as a continuous value
    in the range [-1, 1], where negative values reflect negative sentiment and
    positive values reflect positive sentiment. No categorical encoding is used.

    Semantic alignment is represented as a continuous value computed as:
    (sentiment_polarity * (rating - 3)).
    Intuitively, this is positive when the review sentiment matches the numeric
    rating (for example, high rating and positive polarity) and negative when it
    disagrees. Treating this as numeric allows direct scaling and inclusion in PCA.

    Args:
        text_series: Series of text values used to compute sentiment polarity.
        ratings: Series of numeric ratings (typically 1-5) aligned with text.

    Returns:
        A DataFrame with two float columns:
        - sentiment_polarity: Polarity in [-1, 1].
        - semantic_alignment: Continuous alignment score.
    """
    polarity = text_series.astype(str).apply(_get_polarity).rename(SENTIMENT_POLARITY_COLUMN)

    # numeric semantic alignment
    rating_centered = ratings.astype(float) - 3.0
    semantic_alignment = (polarity * rating_centered).rename(SEMANTIC_ALIGNMENT_COLUMN)

    return pd.concat([polarity, semantic_alignment], axis=1)


def compute_text_statistics(text_series: pd.Series) -> pd.DataFrame:
    """Calculate simple text statistics."""
    return pd.DataFrame(
        {
            WORD_COUNT_COLUMN: text_series.apply(word_count),
            CHARACTER_COUNT_COLUMN: text_series.apply(character_count),
            CAPITALIZED_COUNT_COLUMN: text_series.apply(capitalized_letters_count),
            EXCLAMATION_QUESTION_COUNT_COLUMN: text_series.apply(exclamation_question_count),
            PUNCTUATION_COUNT_COLUMN: text_series.apply(punctuation_count),
        },
        index=text_series.index,
    )


def compute_pos_features(text_series: pd.Series) -> pd.DataFrame:
    """Compute part-of-speech tag counts and expand them into individual columns."""
    pos_counts = text_series.apply(part_of_speech_count).rename(POS_COUNTS_COLUMN)
    expanded_counts = pos_counts.apply(pd.Series).fillna(0)
    expanded_counts.index = text_series.index
    return pd.concat([pos_counts, expanded_counts], axis=1)


def compute_readability_features(text_series: pd.Series) -> pd.DataFrame:
    """Compute readability metrics as a DataFrame."""
    readability_series = text_series.apply(readability_metrics)
    readability_df = pd.DataFrame.from_records(readability_series.to_list(), index=text_series.index)
    return readability_df.reindex(columns=READABILITY_COLUMNS)


def add_engineered_features(
    cleaned_df: pd.DataFrame,
    *,
    text_column: str = TEXT_COLUMN,
    rating_column: str = RATING_COLUMN,
) -> pd.DataFrame:
    """Add derived sentiment, readability and structural features."""
    text_series = cleaned_df[text_column].astype(str)

    sentiment_df = compute_sentiment_features(text_series, cleaned_df[rating_column])
    statistics_df = compute_text_statistics(text_series)
    pos_df = compute_pos_features(text_series)
    readability_df = compute_readability_features(text_series)

    return pd.concat([cleaned_df, sentiment_df, statistics_df, pos_df, readability_df], axis=1)


def finalize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Finalize numeric features for PCA."""
    numeric = df.select_dtypes(include=[np.number])
    return df.assign(**numeric.fillna(0))


def preprocess_reviews(
    raw_df: pd.DataFrame,
    *,
    text_column: str = TEXT_COLUMN,
    rating_column: str = RATING_COLUMN,
) -> pd.DataFrame:
    """Run text preprocessing and feature engineering on reviews."""
    validate_required_columns(raw_df, [text_column, rating_column])
    ensure_nltk_data()

    with console.status("Basic text cleaning"):
        cleaned_df = basic_clean(raw_df, text_column=text_column)

    with console.status("Feature engineering"):
        engineered_df = add_engineered_features(
            cleaned_df,
            text_column=text_column,
            rating_column=rating_column,
        )

    with console.status("Finalizing numeric features"):
        return finalize_numeric_features(engineered_df)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def perform_pca(
    df: pd.DataFrame,
    *,
    n_components: int = 40,
    show_plot: bool = False,
    results_dir: Path,
) -> pd.DataFrame:
    """Perform PCA on numeric features and add PC columns.

    Args:
        df: Input DataFrame.
        n_components: Number of components to keep.
        show_plot: Whether to display the explained variance plot.
        results_dir: Directory where the explained variance plot is stored.

    Returns:
        DataFrame that contains only the PCA components (PC1..PCn) with the same index as df.

        Note that this does not extend the original df, it is a new dataset with just the PCA components.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    # Select all numeric features and drop the supervised target if present
    numeric_features = df.select_dtypes(include=[np.number]).drop(
        columns=[LABEL_COLUMN],
        errors="ignore",
    )

    # Apply standard scaling for the numerical values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_features)  # pyright: ignore[reportUnknownVariableType]

    # Full PCA to inspect variance
    with console.status("[PCA] Running full PCA to compute explained variance"):
        full_pca = PCA()
        _ = full_pca.fit(scaled)
        cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)  # pyright: ignore[reportUnknownVariableType]

    variance_df = pd.DataFrame(
        {
            "component": np.arange(1, len(full_pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": full_pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative_variance,
        },
    )
    variance_df.to_csv(results_dir / "explained_variance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    _ = ax.plot(cumulative_variance)
    _ = ax.set_xlabel("Number of Components")
    _ = ax.set_ylabel("Cumulative Explained Variance")
    _ = ax.set_title("Explained Variance by Principal Components")
    _ = ax.grid(True)
    finalize_plot(
        fig=fig,
        save_path=results_dir / "explained_variance.png",
        show=show_plot,
        status_msg="Explained variance plot",
    )

    # PCA with fixed dimensionality for downstream modeling
    with console.status(f"[PCA] Computing PCA with {n_components} components"):
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled)  # pyright: ignore[reportUnknownVariableType]

    # Return the PCA columns
    column_names = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(data=principal_components, columns=column_names, index=df.index)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def generate_final_dataset(
    *,
    raw_dataset_path: Path,
    final_dataset_path: Path,
    overwrite_if_exists: bool = False,
    results_dir: Path,
    show_plots: bool = False,
) -> None:
    """Generate the final preprocessed dataset and save it as CSV.

    Args:
        raw_dataset_path: Path to the raw reviews dataset.
        final_dataset_path: Path where the processed dataset will be written.
        overwrite_if_exists: Whether to regenerate if the final file already exists.
        results_dir: Directory to store PCA diagnostics and other artifacts.
        show_plots: Whether to display plots interactively while saving them.
    """
    if final_dataset_path.exists() and not overwrite_if_exists:
        console.print("[green]Pre-processed final dataset already exists, skipping generation[/green]")
        return

    console.print("[bold]Pre-processing the dataset...[/bold]")

    # Load raw dataset
    with console.status("Loading raw dataset"):
        reviews_df = pd.read_csv(raw_dataset_path)
        validate_required_columns(reviews_df, RAW_COLUMNS)

    # Preprocess and engineer features
    processed_df = preprocess_reviews(reviews_df)

    # Compute PCA and add the computed components to the dataset
    pca_df = perform_pca(processed_df, n_components=40, show_plot=show_plots, results_dir=results_dir)

    final = pd.concat(
        [
            reviews_df.reset_index(drop=True),
            pca_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Store the resulting dataset
    with console.status("Saving final CSV"):
        final.to_csv(final_dataset_path, index=False)

    console.print(f"[green]Saved pre-processed final dataset with PCA to: {final_dataset_path}[/green]")
