from __future__ import annotations

import pandas as pd
from rich.console import Console
from sklearn.preprocessing import LabelEncoder

from src.schema import CATEGORY_COLUMN, LABEL_COLUMN


def prepare_modeling_frame(df: pd.DataFrame, *, console: Console) -> pd.DataFrame:
    """Prepare modeling frame by encoding labels and categoricals on the fly."""
    console.rule("[bold]Preparing modeling frame[/bold]")

    missing = [col for col in (LABEL_COLUMN, CATEGORY_COLUMN) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling frame: {missing}")

    df = df.copy()

    console.print("Encoding label column with LabelEncoder")
    label_encoder = LabelEncoder()
    label_series = pd.Series(
        label_encoder.fit_transform(df[LABEL_COLUMN].astype(str)),
        name=LABEL_COLUMN,
    )

    console.print("One-hot encoding categorical feature: category")
    ohe = pd.get_dummies(df[[CATEGORY_COLUMN]], prefix=[CATEGORY_COLUMN])

    console.print("Selecting numeric PCA component columns (PC1, PC2, ...)")
    pca_cols = df.filter(regex=r"^PC\d+$").copy()
    if pca_cols.empty:
        raise ValueError("No PCA component columns found (expected columns matching ^PC\\d+$).")
    pca_cols = pca_cols.apply(pd.to_numeric, errors="coerce")
    pca_cols = pca_cols.fillna(pca_cols.mean())

    model_df = pd.concat(
        [
            ohe.reset_index(drop=True),
            pca_cols.reset_index(drop=True),
            label_series.reset_index(drop=True),
        ],
        axis=1,
    )

    console.print(
        f"Modeling frame prepared | Rows: {model_df.shape[0]}  Columns: {model_df.shape[1]}",
    )
    return model_df
