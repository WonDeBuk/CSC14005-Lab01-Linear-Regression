"""Smoke test: every model trains and scores on a synthetic dataset."""

import json

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from linreg.data import load_dataset
from linreg.models import MODELS
from linreg.train import train_all


def synthetic_frame(n=300):
    X, y = make_regression(n_samples=n, n_features=4, noise=5.0, random_state=0)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    df["group"] = np.random.default_rng(0).choice(["a", "b", "c"], size=n)
    df["y"] = y - y.min() + 1  # positive, so log_target_ols is valid
    return df


def test_train_all(tmp_path):
    ds = load_dataset(synthetic_frame(), {"target": "y"})
    metrics = train_all(ds, out_dir=tmp_path)

    assert set(metrics["model"]) == set(MODELS)
    assert metrics["R2"].max() > 0.9
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["target"] == "y"
    assert meta["categorical_features"] == {"group": ["a", "b", "c"]}
