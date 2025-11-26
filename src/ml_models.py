from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import textstat
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from textblob import TextBlob

from src.utils import rolling_print

DATASET_DIR = Path(__file__).resolve().parent.parent / "data"
REVIEWS_DATASET_PATH = DATASET_DIR / "fake_reviews_dataset.csv"
FINAL_DATASET_PATH = DATASET_DIR / "final_data.csv"

SEED = 0


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
    for resource in resources:
        nltk.download(resource, quiet=True)


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
    lemmatized_words = [lemmatizer.lemmatize(word, pos="v") for word in words]
    return " ".join(lemmatizer.lemmatize(word) for word in lemmatized_words)


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
    punctuations = """!\"#$%&'()*+/:;<=>?@[\\]^_.`{|}~"""
    words = text.split()
    cleaned_words: list[str] = []

    for word in words:
        # Remove HTML tags
        word = re.sub(r"(<.*?>)", "", word)

        if "-" not in word:
            # Remove non word characters and digits
            word = re.sub(r"(\W|\d)", " ", word)
        else:
            # Keep hyphenated words but strip other punctuation
            word = "".join(char for char in word if char not in punctuations)

        cleaned_words.append(word.strip())

    return " ".join(cleaned_words)


def categorize_sentiment(text: str) -> str:
    """Categorize sentiment of text using TextBlob polarity.

    Args:
        text: Input text.

    Returns:
        Sentiment label: 'Positive', 'Neutral' or 'Negative'.
    """
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
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
    return sum(1 for char in text if char.isupper())


def exclamation_question_count(text: str) -> int:
    """Count '!' and '?' characters in given text."""
    return sum(1 for char in text if char in {"!", "?"})


def punctuation_count(text: str) -> int:
    """Count punctuation characters in given text."""
    return sum(1 for char in text if char in string.punctuation)


def part_of_speech_count(text: str) -> Counter[str]:
    """Count POS tags in text.

    Args:
        text: Input text.

    Returns:
        Counter mapping POS tag to occurrence count.
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    counts: Counter[str] = Counter(tag for _, tag in pos_tags)
    return counts


def readability_metrics(text: str) -> dict[str, float]:
    """Compute readability metrics for given text.

    Args:
        text: Input text.

    Returns:
        Dictionary of readability scores.
    """
    return {
        "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
        "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
        "gunning_fog": float(textstat.gunning_fog(text)),
        "smog_index": float(textstat.smog_index(text)),
        "automated_readability_index": float(textstat.automated_readability_index(text)),
        "coleman_liau_index": float(textstat.coleman_liau_index(text)),
        "dale_chall_readability_score": float(textstat.dale_chall_readability_score(text)),
    }


def preprocess_reviews(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run text preprocessing and feature engineering on reviews.

    This function expects a 'text_' column and a 'rating' column.

    Args:
        raw_df: Raw reviews DataFrame.

    Returns:
        Preprocessed DataFrame with additional feature columns.
    """
    df = raw_df.copy()

    ensure_nltk_data()
    stop_words = get_stop_words()

    df["text_"] = df["text_"].astype(str).str.lower()

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    df["lemmatized_text"] = df["text_"].apply(lambda text: lemmatize_text(text, lemmatizer))
    df["stemmed_text"] = df["lemmatized_text"].apply(lambda text: stem_text(text, stemmer))
    df["cleaned_text"] = df["stemmed_text"].apply(remove_noise)

    df["CleanedText"] = df["cleaned_text"].apply(
        lambda text: " ".join(word for word in text.split() if word not in stop_words),
    )

    # Sentiment and semantic relevance
    df["sentiment"] = df["text_"].apply(categorize_sentiment)
    df["semantic_relevance"] = df.apply(
        lambda row: semantic_relevance(row["sentiment"], row["rating"]),
        axis=1,
    )

    # Basic text statistics
    df["word_count"] = df["text_"].apply(word_count)
    df["character_count"] = df["text_"].apply(character_count)
    df["capitalized_count"] = df["text_"].apply(capitalized_letters_count)
    df["exclamation_question_count"] = df["text_"].apply(exclamation_question_count)
    df["punctuation_count"] = df["text_"].apply(punctuation_count)

    # POS counts
    df["pos_counts"] = df["text_"].apply(part_of_speech_count)
    pos_counts_df = df["pos_counts"].apply(pd.Series)
    df = pd.concat([df, pos_counts_df], axis=1)

    # Readability metrics
    df["readability"] = df["text_"].apply(readability_metrics)
    readability_df = pd.DataFrame(df["readability"].tolist())
    df = pd.concat([df, readability_df], axis=1)
    df = df.drop(columns=["readability"])

    # Encode sentiment and semantic_relevance
    le_sentiment = LabelEncoder()
    le_semantic_relevance = LabelEncoder()
    df["sentiment_encoded"] = le_sentiment.fit_transform(df["sentiment"])
    df["semantic_relevance_encoded"] = le_semantic_relevance.fit_transform(df["semantic_relevance"])

    df = df.fillna(0)

    return df


def perform_pca(df: pd.DataFrame, n_components: int = 40, with_plot: bool = True) -> pd.DataFrame:
    """Perform PCA on numeric features and add PC columns.

    Args:
        df: Input DataFrame.
        n_components: Number of components to keep.
        with_plot: Whether to plot cumulative explained variance.

    Returns:
        DataFrame with principal components appended as PC1, PC2, ...
    """
    numeric_features = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)

    # Full PCA to inspect variance
    full_pca = PCA()
    full_pca.fit(scaled_features)
    cumulative_variance_ratio = np.cumsum(full_pca.explained_variance_ratio_)

    if with_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_variance_ratio)
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Explained Variance by Principal Components")
        plt.grid(True)
        plt.show()

    # PCA with fixed dimensionality for downstream modeling
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_features)

    column_names = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=column_names, index=df.index)

    combined_df = pd.concat([df.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
    return combined_df


def plot_tsne_embeddings(texts: Iterable[str], labels: pd.Series, cleaned: bool = True) -> None:
    """Compute and plot t-SNE embeddings for a subset of texts.

    Args:
        texts: Iterable of text documents.
        labels: Corresponding labels.
        cleaned: Whether the texts are cleaned; used only for plot title.
    """
    # Limit to first 2000 samples for t-SNE
    texts_list = list(texts)[:2000]
    labels_subset = labels.iloc[:2000]

    bow_vect = CountVectorizer()
    bow = bow_vect.fit_transform(texts_list)
    X = bow.toarray()

    tsne = TSNE(n_components=2, perplexity=20, random_state=SEED)
    embedded = tsne.fit_transform(X)

    df = pd.DataFrame(embedded, columns=("dim1", "dim2"))
    df = pd.concat([df, labels_subset.reset_index(drop=True)], axis=1)

    sns.FacetGrid(df, hue="label", height=6).map(plt.scatter, "dim1", "dim2").add_legend()
    title_suffix = "Cleaned" if cleaned else "Raw"
    plt.title(f"t-SNE on {title_suffix} Texts (Perplexity = 20)")
    plt.show()


def build_word2vec_embeddings(sentences: Sequence[Sequence[str]]) -> list[np.ndarray]:
    """Train a Word2Vec model and compute sentence embeddings.

    Args:
        sentences: Tokenized sentences.

    Returns:
        List of average word embedding vectors per sentence.
    """
    # Training with default vector size
    w2v_model = gensim.models.Word2Vec(
        sentences,
        min_count=5,
        workers=4,
    )
    vector_size = w2v_model.vector_size

    sentence_vectors: list[np.ndarray] = []
    for sentence in sentences:
        sentence_vec = np.zeros(vector_size, dtype=float)
        word_count_in_sent = 0

        for word in sentence:
            if word in w2v_model.wv:
                sentence_vec += w2v_model.wv[word]
                word_count_in_sent += 1

        if word_count_in_sent != 0:
            sentence_vec /= float(word_count_in_sent)

        sentence_vectors.append(sentence_vec)

    return sentence_vectors


def prepare_modeling_frame(final_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare final modeling frame with PCA and one-hot encoded categoricals.

    Args:
        final_df: DataFrame after feature engineering and PCA.

    Returns:
        Modeling DataFrame including label.
    """
    modeling_df = final_df.copy()

    # Encode label
    label_encoder = LabelEncoder()
    modeling_df["label"] = label_encoder.fit_transform(modeling_df["label"])

    # One-hot encode selected categoricals
    categorical_features = modeling_df[["category", "sentiment", "semantic_relevance"]]
    one_hot_encoded = pd.get_dummies(
        categorical_features,
        prefix=["category", "sentiment", "semantic_relevance"],
    )

    # Numeric PCA columns
    pca_columns = modeling_df.filter(regex=r"^PC\d+$").copy()
    for col in pca_columns.columns:
        pca_columns[col] = pd.to_numeric(pca_columns[col], errors="coerce")
    pca_columns = pca_columns.fillna(pca_columns.mean())

    df_with_pca = pd.concat(
        [
            one_hot_encoded.reset_index(drop=True),
            pca_columns.reset_index(drop=True),
            modeling_df[["label"]].reset_index(drop=True),  # already encoded
        ],
        axis=1,
    )

    return df_with_pca


def evaluate_classifier(name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print standard evaluation metrics for a classifier.

    Args:
        name: Model name for logging.
        model: Trained classifier with a predict method.
        X_test: Test features.
        y_test: Test labels.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = score(y_test, y_pred, average="micro")

    print(f"=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {fscore:.4f}\n")
    print("Classification report:\n", classification_report(y_test, y_pred))


def train_and_evaluate_models(df_with_pca: pd.DataFrame) -> None:
    """Train several models with hyperparameter search and evaluate them.

    Args:
        df_with_pca: Modeling DataFrame including label column.
    """
    np.random.seed(SEED)

    train_df = df_with_pca.sample(frac=0.8, random_state=SEED)
    test_df = df_with_pca.drop(train_df.index)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Unique values in y_train:", pd.Series(y_train).unique())

    # Random Forest
    rf = RandomForestClassifier(random_state=SEED)
    rf_param_grid = {
        "n_estimators": [100, 250],
        "max_features": ["sqrt", "log2"],
        "max_depth": list(range(1, 10)),
        "criterion": ["gini", "entropy"],
    }

    with rolling_print(max_lines=10):
        rf_grid = GridSearchCV(
            estimator=rf,
            param_grid=rf_param_grid,
            cv=2,
            scoring="accuracy",
            return_train_score=False,
            verbose=3,
        )
        rf_grid.fit(X_train, y_train)
    print("RandomForest best params:", rf_grid.best_params_)
    print("RandomForest best CV score:", rf_grid.best_score_)
    evaluate_classifier("RandomForest", rf_grid.best_estimator_, X_test, y_test)

    # AdaBoost
    ada = AdaBoostClassifier(random_state=SEED)
    ada_param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.5, 1.0],
    }
    ada_grid = GridSearchCV(ada, ada_param_grid, cv=3, n_jobs=-1)
    ada_grid.fit(X_train, y_train)
    print("AdaBoost best params:", ada_grid.best_params_)
    print("AdaBoost best CV score:", ada_grid.best_score_)
    evaluate_classifier("AdaBoost", ada_grid.best_estimator_, X_test, y_test)

    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="mlogloss",
        random_state=SEED,
    )
    xgb_param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.01, 0.001],
        "max_depth": list(range(1, 10)),
        "subsample": [0.8, 0.9, 1.0],
    }
    with rolling_print(max_lines=10):
        xgb_grid = GridSearchCV(xgb_clf, xgb_param_grid, cv=2, n_jobs=-1, verbose=3)
        xgb_grid.fit(X_train, y_train)
    print("XGBoost best params:", xgb_grid.best_params_)
    print("XGBoost best CV score:", xgb_grid.best_score_)
    evaluate_classifier("XGBoost", xgb_grid.best_estimator_, X_test, y_test)

    # SVM
    svc_param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "kernel": ["rbf", "poly", "sigmoid"],
    }
    with rolling_print(max_lines=10):
        svc_grid = GridSearchCV(SVC(), svc_param_grid, cv=2, scoring="accuracy", verbose=3)
        svc_grid.fit(X_train, y_train)
    print("SVM best params:", svc_grid.best_params_)
    print("SVM best CV score:", svc_grid.best_score_)
    evaluate_classifier("SVM", svc_grid.best_estimator_, X_test, y_test)

    # Multinomial Naive Bayes (on MinMax scaled features)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_scaled, y_train)
    evaluate_classifier("MultinomialNB", nb_model, X_test_scaled, y_test)

    # Logistic Regression
    lr_param_grid = {
        "penalty": ["l2", "elasticnet"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
    }
    with rolling_print(max_lines=10):
        lr_grid = GridSearchCV(
            LogisticRegression(max_iter=1000),
            lr_param_grid,
            scoring="accuracy",
            return_train_score=False,
            verbose=3,
        )
        lr_grid.fit(X_train, y_train)
    print("LogisticRegression best params:", lr_grid.best_params_)
    print("LogisticRegression best CV score:", lr_grid.best_score_)
    evaluate_classifier("LogisticRegression", lr_grid.best_estimator_, X_test, y_test)


def main() -> None:
    """Entry point: run preprocessing, PCA, save data, and train models."""
    # Load raw dataset
    reviews_df = pd.read_csv(REVIEWS_DATASET_PATH)

    # Preprocess and engineer features
    reviews_df = preprocess_reviews(reviews_df)

    # PCA and saving combined dataset
    combined_df = perform_pca(reviews_df, n_components=40, with_plot=False)
    combined_df.to_csv(FINAL_DATASET_PATH, index=False)
    print(f"Saved combined dataset with PCA to: {FINAL_DATASET_PATH}")

    # Optional: t-SNE visualization (comment out if not needed)
    # plot_tsne_embeddings(combined_df["CleanedText"].astype(str), combined_df["label"], cleaned=True)

    # Optional: Word2Vec sentence embeddings (not used in modeling below)
    # tokenized_sentences = [str(text).split() for text in combined_df["CleanedText"]]
    # sentence_vectors = build_word2vec_embeddings(tokenized_sentences)
    # print(f"Computed {len(sentence_vectors)} Word2Vec sentence vectors.")

    # Prepare modeling frame and train models
    final_reviews_df = pd.read_csv(FINAL_DATASET_PATH)
    modeling_df = prepare_modeling_frame(final_reviews_df)

    print("Label distribution:", modeling_df["label"].value_counts())

    train_and_evaluate_models(modeling_df)


if __name__ == "__main__":
    main()
