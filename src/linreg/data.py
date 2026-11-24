"""Loading and splitting the dataset."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Raw data plus derived features, leakage and constant columns removed."""
    path = Path(path or config.DATASET["file"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md.")
    df = pd.read_csv(path).drop(columns=config.DATASET["drop"])
    return add_features(df)


def split_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(config.RANDOM_STATE).permutation(n)
    return order[: config.N_TRAIN], order[config.N_TRAIN :]


def load_dataset(df: pd.DataFrame | None = None) -> Dataset:
    target = config.DATASET["target"]
    df = load_frame() if df is None else df
    train, test = split_indices(len(df))
    X, y = df.drop(columns=[target]), df[target]
    return Dataset(X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test], target)
