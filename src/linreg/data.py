"""Loading and splitting the dataset."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    target: str


def load_frame(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or config.DATASET["file"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download the dataset as described in data/README.md.")
    return pd.read_csv(path)


def infer_feature_types(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    features = df.drop(columns=[target])
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    return numeric, categorical


def load_dataset(df: pd.DataFrame | None = None, settings: dict | None = None) -> Dataset:
    settings = {**config.DATASET, **(settings or {})}
    target = settings["target"]
    if target is None:
        raise ValueError("Set DATASET['target'] in src/linreg/config.py.")

    df = load_frame(settings["file"]) if df is None else df
    df = df.drop(columns=settings["drop"]).dropna(subset=[target])

    numeric, categorical = infer_feature_types(df, target)
    numeric = settings["numeric_features"] or numeric
    categorical = settings["categorical_features"] or categorical

    X = df[numeric + categorical]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    return Dataset(X_train, X_test, y_train, y_test, numeric, categorical, target)
