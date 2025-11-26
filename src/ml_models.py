from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC

from src.utils import rolling_print

SEED = 0


# ---------------------------------------------------------------------------
# Visualization / Embeddings
# ---------------------------------------------------------------------------
def plot_tsne_embeddings(texts: Iterable[str], labels: pd.Series, cleaned: bool = True) -> None:
    """Compute and plot t-SNE embeddings for a subset of texts.

    Args:
        texts: Iterable of text documents.
        labels: Corresponding labels.
        cleaned: Whether the texts are cleaned; used only for plot title.
    """
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
    """Train a Word2Vec model and compute simple averaged sentence embeddings.

    Args:
        sentences: Tokenized sentences.

    Returns:
        List of averaged word embedding vectors for each sentence.
    """
    w2v_model = gensim.models.Word2Vec(
        sentences,
        min_count=5,
        workers=4,
    )

    vector_size = w2v_model.vector_size
    sentence_vectors: list[np.ndarray] = []

    for sentence in sentences:
        vec = np.zeros(vector_size, dtype=float)
        count = 0

        for word in sentence:
            if word in w2v_model.wv:
                vec += w2v_model.wv[word]
                count += 1

        if count > 0:
            vec /= float(count)

        sentence_vectors.append(vec)

    return sentence_vectors


# ---------------------------------------------------------------------------
# Modeling frame
# ---------------------------------------------------------------------------
def prepare_modeling_frame(final_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare final modeling frame from the fully preprocessed + PCA dataset.

    Args:
        final_df: DataFrame after feature engineering and PCA.

    Returns:
        Modeling DataFrame including encoded label and one-hot categoricals.
    """
    df = final_df.copy()

    # Encode label
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    # One-hot encode remaining categoricals
    cat_df = df[["category", "sentiment", "semantic_relevance"]]
    ohe = pd.get_dummies(cat_df, prefix=["category", "sentiment", "semantic_relevance"])

    # PCA components only
    pca_cols = df.filter(regex=r"^PC\d+$").copy()
    pca_cols = pca_cols.apply(pd.to_numeric, errors="coerce")
    pca_cols = pca_cols.fillna(pca_cols.mean())

    model_df = pd.concat(
        [
            ohe.reset_index(drop=True),
            pca_cols.reset_index(drop=True),
            df[["label"]].reset_index(drop=True),
        ],
        axis=1,
    )

    return model_df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_classifier(name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print standard evaluation metrics for a classifier."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = score(y_test, y_pred, average="micro")

    print(f"=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {fscore:.4f}\n")
    print("Classification report:\n", classification_report(y_test, y_pred))


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_and_evaluate_models(df_with_pca: pd.DataFrame) -> None:
    """Train several classical ML models with hyperparameter search."""
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
            rf,
            rf_param_grid,
            cv=2,
            scoring="accuracy",
            verbose=3,
        )
        rf_grid.fit(X_train, y_train)

    print("RandomForest best params:", rf_grid.best_params_)
    print("RandomForest best CV score:", rf_grid.best_score_)
    evaluate_classifier("RandomForest", rf_grid.best_estimator_, X_test, y_test)

    # AdaBoost
    ada = AdaBoostClassifier(random_state=SEED)
    ada_grid = GridSearchCV(
        ada,
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.1, 0.5, 1.0],
        },
        cv=3,
        n_jobs=-1,
    )
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

    # Multinomial Naive Bayes
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
            verbose=3,
        )
        lr_grid.fit(X_train, y_train)

    print("LogisticRegression best params:", lr_grid.best_params_)
    print("LogisticRegression best CV score:", lr_grid.best_score_)
    evaluate_classifier("LogisticRegression", lr_grid.best_estimator_, X_test, y_test)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_ml_models(dataset_path: Path, show_tsne: bool = False, perform_word2vec: bool = False) -> None:
    """Load the final PCA-processed dataset and run all models."""
    df = pd.read_csv(dataset_path)

    # Optional: t-SNE visualization
    if show_tsne:
        plot_tsne_embeddings(df["CleanedText"].astype(str), df["label"], cleaned=True)

    # Optional: Word2Vec sentence embeddings (not used in modeling below)
    if perform_word2vec:
        tokenized_sentences = [str(text).split() for text in df["CleanedText"]]
        sentence_vectors = build_word2vec_embeddings(tokenized_sentences)
        print(f"Computed {len(sentence_vectors)} Word2Vec sentence vectors.")

    model_df = prepare_modeling_frame(df)

    print("Label distribution:", model_df["label"].value_counts())

    train_and_evaluate_models(model_df)
