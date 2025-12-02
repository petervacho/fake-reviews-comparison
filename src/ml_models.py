from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC

from src.utils import render_evaluation_report, rolling_status

SEED = 0

console = Console()


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
    console.rule("[bold]t-SNE embedding[/bold]")
    console.print("Sampling up to 2000 texts for t-SNE projection")

    texts_list = list(texts)[:2000]
    labels_subset = labels.iloc[:2000]

    bow_vect = CountVectorizer()
    bow = bow_vect.fit_transform(texts_list)  # pyright: ignore[reportUnknownVariableType]
    x = bow.toarray()  # pyright: ignore[reportUnknownVariableType,reportAttributeAccessIssue]

    tsne = TSNE(n_components=2, perplexity=20, random_state=SEED)
    embedded = tsne.fit_transform(x)  # pyright: ignore[reportUnknownVariableType]

    df = pd.DataFrame(embedded, columns=("dim1", "dim2"))
    df = pd.concat([df, labels_subset.reset_index(drop=True)], axis=1)

    with console.status("Showing t-SNE plot (close window to continue"):
        _ = sns.FacetGrid(df, hue="label", height=6).map(plt.scatter, "dim1", "dim2").add_legend()
        title_suffix = "Cleaned" if cleaned else "Raw"
        _ = plt.title(f"t-SNE on {title_suffix} Texts (Perplexity = 20)")
        plt.show()


def build_word2vec_embeddings(sentences: Sequence[Sequence[str]]) -> list[np.ndarray]:
    """Train a Word2Vec model and compute simple averaged sentence embeddings.

    Args:
        sentences: Tokenized sentences.

    Returns:
        List of averaged word embedding vectors for each sentence.
    """
    console.rule("[bold]Word2Vec embeddings[/bold]")
    console.print("Training Word2Vec model on tokenized sentences")

    w2v_model = gensim.models.Word2Vec(
        sentences,
        min_count=5,
        workers=4,
    )

    vector_size = w2v_model.vector_size
    sentence_vectors: list[np.ndarray] = []

    for sentence in sentences:
        vec: np.ndarray = np.zeros(vector_size, dtype=float)
        count = 0

        for word in sentence:
            if word in w2v_model.wv:
                vec = cast("np.ndarray", vec + w2v_model.wv[word])
                count += 1

        if count > 0:
            vec /= float(count)

        sentence_vectors.append(vec)

    console.print(f"Computed embeddings for {len(sentence_vectors)} sentences")
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
    console.rule("[bold]Preparing modeling frame[/bold]")

    df = final_df.copy()

    # Encode label
    console.print("Encoding label column")
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    # One-hot encode remaining categoricals
    console.print("One-hot encoding categorical features: [italic]category, sentiment, semantic_relevance[/italic]")
    cat_df = df[["category", "sentiment", "semantic_relevance"]]
    ohe = pd.get_dummies(cat_df, prefix=["category", "sentiment", "semantic_relevance"])

    # PCA components only
    console.print("Selecting numeric PCA component columns (PC1, PC2, ...)")
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

    console.print(
        Panel.fit(
            f"Modeling frame prepared\nRows: {model_df.shape[0]}  Columns: {model_df.shape[1]}",
            title="Modeling frame",
        ),
    )
    return model_df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_classifier(name: str, model: Any, x_test: np.ndarray | pd.DataFrame, y_test: pd.Series) -> None:
    """Pretty-print evaluation metrics for a classifier."""
    y_pred = model.predict(x_test)

    render_evaluation_report(
        name=name,
        y_true=y_test,
        y_pred=y_pred,
        console=console,
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def _print_split_info(x_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Print basic information about the train split."""
    table = Table(show_header=False, box=None)
    table.add_row("X_train shape", f"{x_train.shape}")
    table.add_row("y_train shape", f"{y_train.shape}")
    table.add_row("Unique labels", f"{sorted(pd.Series(y_train).unique().tolist())}")
    console.print(table)


def train_and_evaluate_models(df_with_pca: pd.DataFrame) -> None:
    """Train several classical ML models with hyperparameter search."""
    np.random.seed(SEED)  # noqa: NPY002

    console.rule("[bold]Train/test split[/bold]")
    train_df = df_with_pca.sample(frac=0.8, random_state=SEED)
    test_df = df_with_pca.drop(train_df.index)

    x_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    x_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    _print_split_info(x_train, y_train)

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------
    console.rule("[bold]RandomForest[/bold]")
    rf = RandomForestClassifier(random_state=SEED)
    rf_param_grid: dict[str, Any] = {
        "n_estimators": [100, 250],
        "max_features": ["sqrt", "log2"],
        "max_depth": list(range(1, 10)),
        "criterion": ["gini", "entropy"],
    }
    n_candidates = (
        len(rf_param_grid["n_estimators"])
        * len(rf_param_grid["max_features"])
        * len(rf_param_grid["max_depth"])
        * len(rf_param_grid["criterion"])
    )
    console.print(f"Running GridSearchCV with {n_candidates} parameter combinations (cv=2)")

    rf_grid = GridSearchCV(
        rf,
        rf_param_grid,
        cv=2,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    stopped = False
    with rolling_status("Fitting RandomForest GridSearchCV", max_lines=10, clear_on_exit=True, add_elapsed_time=True):
        try:
            _ = rf_grid.fit(x_train, y_train)
        except KeyboardInterrupt:
            stopped = True
    if stopped:
        console.print("[red]RandomForest fitting interrupted, skipping results[/red]")
    else:
        console.print(f"Best CV score: [bold]{rf_grid.best_score_:.4f}[/bold]")
        console.print(f"Best params: {rf_grid.best_params_}")
        evaluate_classifier("RandomForest", rf_grid.best_estimator_, x_test, y_test)

    # ------------------------------------------------------------------
    # AdaBoost
    # ------------------------------------------------------------------
    console.rule("[bold]AdaBoost[/bold]")
    ada = AdaBoostClassifier(random_state=SEED)
    ada_param_grid: dict[str, Any] = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.5, 1.0],
    }
    n_candidates = len(ada_param_grid["n_estimators"]) * len(ada_param_grid["learning_rate"])
    console.print(f"Running GridSearchCV with {n_candidates} parameter combinations (cv=3)")

    ada_grid = GridSearchCV(
        ada,
        ada_param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    stopped = False
    with rolling_status("Fitting AdaBoost GridSearchCV", max_lines=10, add_elapsed_time=True):
        try:
            _ = ada_grid.fit(x_train, y_train)
        except KeyboardInterrupt:
            stopped = True
    if stopped:
        console.print("[red]AdaBoost fitting interupted, skipping results[/red]")
    else:
        console.print(f"Best CV score: [bold]{ada_grid.best_score_:.4f}[/bold]")
        console.print(f"Best params: {ada_grid.best_params_}")
        evaluate_classifier("AdaBoost", ada_grid.best_estimator_, x_test, y_test)

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    console.rule("[bold]XGBoost[/bold]")
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="mlogloss",
        random_state=SEED,
    )
    xgb_param_grid: dict[str, Any] = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.01, 0.001],
        "max_depth": list(range(1, 10)),
        "subsample": [0.8, 0.9, 1.0],
    }
    n_candidates = (
        len(xgb_param_grid["n_estimators"])
        * len(xgb_param_grid["learning_rate"])
        * len(xgb_param_grid["max_depth"])
        * len(xgb_param_grid["subsample"])
    )
    console.print(f"Running GridSearchCV with {n_candidates} parameter combinations (cv=2)")

    xgb_grid = GridSearchCV(
        xgb_clf,
        xgb_param_grid,
        cv=2,
        n_jobs=-1,
        verbose=1,
    )

    stopped = False
    with rolling_status("Fitting XGBoost GridSearchCV", max_lines=10, add_elapsed_time=True):
        try:
            _ = xgb_grid.fit(x_train, y_train)
        except KeyboardInterrupt:
            stopped = True
    if stopped:
        console.print("[red]XGBoost fitting interrupted, skipping results[/red]")
    else:
        console.print(f"Best CV score: [bold]{xgb_grid.best_score_:.4f}[/bold]")
        console.print(f"Best params: {xgb_grid.best_params_}")
        evaluate_classifier("XGBoost", xgb_grid.best_estimator_, x_test, y_test)

    # ------------------------------------------------------------------
    # SVM
    # ------------------------------------------------------------------
    console.rule("[bold]SVM[/bold]")
    svc_param_grid: dict[str, Any] = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "kernel": ["rbf", "poly", "sigmoid"],
    }
    n_candidates = len(svc_param_grid["C"]) * len(svc_param_grid["kernel"])
    console.print(f"Running GridSearchCV with {n_candidates} parameter combinations (cv=2)")

    svc_grid = GridSearchCV(
        SVC(),
        svc_param_grid,
        cv=2,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    stopped = False
    with rolling_status("Fitting SVM GridSearchCV", max_lines=10, add_elapsed_time=True):
        try:
            _ = svc_grid.fit(x_train, y_train)
        except KeyboardInterrupt:
            stopped = True
    if stopped:
        console.print("[red] fitting interrupted, skipping results[/red]")
    else:
        console.print(f"Best CV score: [bold]{svc_grid.best_score_:.4f}[/bold]")
        console.print(f"Best params: {svc_grid.best_params_}")
        evaluate_classifier("SVM", svc_grid.best_estimator_, x_test, y_test)

    # ------------------------------------------------------------------
    # Multinomial Naive Bayes
    # ------------------------------------------------------------------
    console.rule("[bold]Multinomial Naive Bayes[/bold]")
    console.print("Scaling features to [0, 1] range for MultinomialNB")
    scaler = MinMaxScaler()
    x_train_scaled = cast("np.ndarray", scaler.fit_transform(x_train))
    x_test_scaled = cast("np.ndarray", scaler.transform(x_test))

    nb_model = MultinomialNB(alpha=0.1)

    with console.status("Fitting MultinominalNB"):
        _ = nb_model.fit(x_train_scaled, y_train)

    evaluate_classifier("MultinomialNB", nb_model, x_test_scaled, y_test)

    # ------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------
    console.rule("[bold]Logistic Regression[/bold]")
    lr_param_grid: dict[str, Any] = {
        "penalty": ["l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
    }
    n_candidates = len(lr_param_grid["penalty"]) * len(lr_param_grid["C"])
    console.print(f"Running GridSearchCV with {n_candidates} parameter combinations (cv=2)")

    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, solver="lbfgs"),
        lr_param_grid,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    stopped = False
    with rolling_status("Fitting LogisticRegression GridSearchCV", max_lines=10, add_elapsed_time=True):
        try:
            _ = lr_grid.fit(x_train, y_train)
        except KeyboardInterrupt:
            stopped = True
    if stopped:
        console.print("[red]LogisticRegression fitting interrupted, skipping results[/red]")
    else:
        console.print(f"Best CV score: [bold]{lr_grid.best_score_:.4f}[/bold]")
        console.print(f"Best params: {lr_grid.best_params_}")
        evaluate_classifier("LogisticRegression", lr_grid.best_estimator_, x_test, y_test)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_ml_models(dataset_path: Path, show_tsne: bool = False, perform_word2vec: bool = False) -> None:
    """Load the final PCA-processed dataset and run all models."""
    console.rule("[bold]Loading dataset[/bold]")
    console.print(f"Reading dataset from: [italic]{dataset_path}[/italic]")
    df = pd.read_csv(dataset_path)

    console.print(Panel.fit(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}", title="Raw dataset"))

    # Optional: t-SNE visualization
    if show_tsne:
        plot_tsne_embeddings(df["CleanedText"].astype(str), df["label"], cleaned=True)

    # Optional: Word2Vec sentence embeddings (not used in modeling below)
    if perform_word2vec:
        tokenized_sentences = [str(text).split() for text in df["CleanedText"]]
        _ = build_word2vec_embeddings(tokenized_sentences)

    model_df = prepare_modeling_frame(df)

    console.rule("[bold]Label distribution[/bold]")
    label_counts = model_df["label"].value_counts().sort_index()
    label_table = Table(show_header=True, header_style="bold")
    label_table.add_column("Label")
    label_table.add_column("Count", justify="right")
    for lbl, cnt in label_counts.items():
        label_table.add_row(str(lbl), str(cnt))
    console.print(label_table)

    train_and_evaluate_models(model_df)
