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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from textblob import TextBlob

from src.utils.evaluation import finalize_plot

console = Console()


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


def categorize_sentiment(text: str) -> str:
    """Categorize sentiment of text using TextBlob polarity.

    Args:
        text: Input text.

    Returns:
        Sentiment label: 'Positive', 'Neutral' or 'Negative'.
    """
    polarity = TextBlob(str(text)).sentiment.polarity  # pyright: ignore[reportUnknownVariableType,reportAttributeAccessIssue]
    if polarity > 0:
        return "Positive"
    if polarity == 0:
        return "Neutral"
    return "Negative"


def semantic_relevance(sentiment: str, rating: float) -> str:
    """Compute semantic relevance between sentiment and numeric rating.

    Args:
        sentiment: Sentiment label.
        rating: Rating value.

    Returns:
        Semantic relevance label.
    """
    if (sentiment == "Positive" and rating >= 4) or (sentiment == "Negative" and rating <= 2):
        return "High Relevance"
    if sentiment == "Neutral" or rating == 3:
        return "Medium Relevance"
    if (sentiment == "Positive" and rating <= 2) or (sentiment == "Negative" and rating >= 4):
        return "Low Relevance"
    return "Uncategorized"


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
    return {
        "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),  # pyright: ignore[reportAttributeAccessIssue]
        "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),  # pyright: ignore[reportAttributeAccessIssue]
        "gunning_fog": float(textstat.gunning_fog(text)),  # pyright: ignore[reportAttributeAccessIssue]
        "smog_index": float(textstat.smog_index(text)),  # pyright: ignore[reportAttributeAccessIssue]
        "automated_readability_index": float(textstat.automated_readability_index(text)),  # pyright: ignore[reportAttributeAccessIssue]
        "coleman_liau_index": float(textstat.coleman_liau_index(text)),  # pyright: ignore[reportAttributeAccessIssue]
        "dale_chall_readability_score": float(textstat.dale_chall_readability_score(text)),  # pyright: ignore[reportAttributeAccessIssue]
    }


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_reviews(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run text preprocessing and feature engineering on reviews.

    This function expects a 'text_' column and a 'rating' column.

    Args:
        raw_df: Raw reviews DataFrame.

    Returns:
        Preprocessed DataFrame with additional feature columns.
    """
    with console.status("Preprocessing text data"):
        df = raw_df.copy()

        ensure_nltk_data()
        stop_words = get_stop_words()

        df["text_"] = df["text_"].astype(str).str.lower()

        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()

        with console.status("Lemmatizing and stemming"):
            df["lemmatized_text"] = df["text_"].apply(lambda t: lemmatize_text(t, lemmatizer))
            df["stemmed_text"] = df["lemmatized_text"].apply(lambda t: stem_text(t, stemmer))

        with console.status("Removing noise and stop words"):
            df["cleaned_text"] = df["stemmed_text"].apply(remove_noise)
            df["CleanedText"] = df["cleaned_text"].apply(
                lambda t: " ".join(w for w in t.split() if w not in stop_words),  # pyright: ignore[reportUnknownVariableType]
            )

        with console.status("Computing sentiment and semantic relevance"):
            df["sentiment"] = df["text_"].apply(categorize_sentiment)
            df["semantic_relevance"] = df.apply(lambda r: semantic_relevance(r["sentiment"], r["rating"]), axis=1)

        with console.status("Computing text statistics"):
            df["word_count"] = df["text_"].apply(word_count)
            df["character_count"] = df["text_"].apply(character_count)
            df["capitalized_count"] = df["text_"].apply(capitalized_letters_count)
            df["exclamation_question_count"] = df["text_"].apply(exclamation_question_count)
            df["punctuation_count"] = df["text_"].apply(punctuation_count)

        with console.status("POS tagging (slow)"):
            df["pos_counts"] = df["text_"].apply(part_of_speech_count)
            pos_df = df["pos_counts"].apply(pd.Series)
            df = pd.concat([df, pos_df], axis=1)

        with console.status("Computing readability metrics"):
            df["readability"] = df["text_"].apply(readability_metrics)
            r_df = pd.DataFrame(df["readability"].tolist())
            df = pd.concat([df, r_df], axis=1)
            df = df.drop(columns=["readability"])

        with console.status("Encoding sentiment features"):
            le_sent = LabelEncoder()
            le_semrel = LabelEncoder()
            df["sentiment_encoded"] = le_sent.fit_transform(df["sentiment"])
            df["semantic_relevance_encoded"] = le_semrel.fit_transform(df["semantic_relevance"])

        return df.fillna(0)


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
        DataFrame with principal components appended as PC1, PC2, ...
    """
    with console.status("[PCA] Selecting numeric features"):
        numeric_features = df.select_dtypes(include=[np.number])

    with console.status("[PCA] Scaling numeric features"):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)  # pyright: ignore[reportUnknownVariableType]

    # Full PCA to inspect variance
    with console.status("[PCA] Running full PCA to compute explained variance"):
        full_pca = PCA()
        _ = full_pca.fit(scaled_features)
        cumulative_variance_ratio = np.cumsum(full_pca.explained_variance_ratio_)  # pyright: ignore[reportUnknownVariableType]

    fig, ax = plt.subplots(figsize=(10, 6))
    _ = ax.plot(cumulative_variance_ratio)
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
        principal_components = pca.fit_transform(scaled_features)  # pyright: ignore[reportUnknownVariableType]

    column_names = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=column_names, index=df.index)

    # Return the combined df
    return pd.concat([df.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)


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

    # Preprocess and engineer features
    processed_df = preprocess_reviews(reviews_df)

    # Compute PCA and add the computed components to the dataset
    final = perform_pca(processed_df, n_components=40, show_plot=show_plots, results_dir=results_dir)

    # Store the resulting dataset
    with console.status("Saving final CSV"):
        final.to_csv(final_dataset_path, index=False)

    console.print(f"[green]Saved pre-processed final dataset with PCA to: {final_dataset_path}[/green]")
