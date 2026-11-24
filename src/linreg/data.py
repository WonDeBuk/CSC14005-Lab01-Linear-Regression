"""Loading and splitting the dataset."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .features import add_features


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target: str


def load_frame(path: Path | None = None) -> pd.DataFrame:
    """Raw data plus derived features, leakage columns removed."""
    path = Path(path or config.DATASET["file"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md.")
    df = pd.read_csv(path).drop(columns=config.DATASET["drop"])
    return add_features(df)


def load_dataset(df: pd.DataFrame | None = None) -> Dataset:
    target = config.DATASET["target"]
    df = load_frame() if df is None else df
    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    return Dataset(X_train, X_test, y_train, y_test, target)
