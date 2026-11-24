"""End-to-end checks on the committed CO2 dataset.

Expected numbers come from the team moodboard (sections 09, 10 and 12), so a
change that breaks reproducibility of the moodboard fails here.
"""

import json

import pandas as pd
import pytest

from linreg import config
from linreg.data import load_dataset, load_frame
from linreg.inference import fit_ols, partial_f_test
from linreg.models import MODELS
from linreg.train import train_all


def test_features():
    df = load_frame()
    assert len(df) == 1067
    assert not set(config.DATASET["drop"]) & set(df.columns)  # no fuel-consumption leakage
    assert df["BODY"].value_counts().to_dict() == {"small": 355, "midlarge": 306, "suv": 264, "pickup": 74, "van": 68}
    assert df["NOGEAR"].sum() == 46
    assert (df["HYBRID_NAME"].sum(), df["HYBRID"].sum()) == (24, 34)


@pytest.fixture(scope="module")
def metrics(tmp_path_factory):
    return train_all(load_dataset(), out_dir=tmp_path_factory.mktemp("models")).set_index("model")


@pytest.mark.parametrize(
    "name, test_rmse, cv_rmse",
    [("M1", 33.02, 30.16), ("M3", 25.29, 23.80), ("M3h", 24.08, 22.56), ("M4", 25.43, 23.80),
     ("M4log", 25.37, 23.76), ("M5", 25.38, 23.80), ("M8b", 22.61, 21.61)],
)
def test_reproduces_moodboard(metrics, name, test_rmse, cv_rmse):
    assert metrics.loc[name, "RMSE"] == pytest.approx(test_rmse, abs=0.01)
    assert metrics.loc[name, "CV_RMSE_mean"] == pytest.approx(cv_rmse, abs=0.01)


def test_all_models_trained(metrics, tmp_path):
    assert set(metrics.index) == set(MODELS)
    assert len(MODELS) >= 5  # at least one model per group member
    # Weak-prior Bayesian M6 coincides with OLS M3h at n = 854.
    assert metrics.loc["M6", "RMSE"] == pytest.approx(metrics.loc["M3h", "RMSE"], abs=0.01)


def test_inference():
    ds = load_dataset()
    train = pd.concat([ds.X_train, ds.y_train], axis=1)
    result = fit_ols(train, "M3h")
    assert result.params["HYBRID"] == pytest.approx(-48.3, abs=0.05)
    assert result.pvalues["HYBRID"] < 1e-6
    f = partial_f_test(train, "M3", "M4").iloc[1]
    assert f["F"] == pytest.approx(1.411, abs=0.001)


def test_meta(tmp_path):
    train_all(load_dataset(), out_dir=tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["categorical"]["FUELTYPE"] == ["D", "E", "X", "Z"]
