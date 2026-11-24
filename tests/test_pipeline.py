"""End-to-end checks on the committed CO2 dataset."""

import json

from linreg import config
from linreg.data import load_dataset, load_frame
from linreg.inference import fit_ols
from linreg.models import MODELS
from linreg.train import train_all


def test_features():
    df = load_frame()
    assert len(df) == 1067
    assert not set(config.DATASET["drop"]) & set(df.columns)  # no fuel-consumption leakage
    assert df["BODY"].notna().all()
    assert df["HYBRID"].sum() == 34  # includes Lexus "...h", Prius, CR-Z


def test_train_all(tmp_path):
    ds = load_dataset()
    assert (len(ds.X_train), len(ds.X_test)) == (854, 213)

    metrics = train_all(ds, out_dir=tmp_path).set_index("model")
    assert set(metrics.index) == set(MODELS)
    assert metrics.loc["m3h_ols", "CV_RMSE_mean"] < metrics.loc["m1_engine", "CV_RMSE_mean"]
    assert metrics["R2"].min() > 0.7

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["categorical"]["FUELTYPE"] == ["D", "E", "X", "Z"]


def test_ols_inference():
    result = fit_ols(load_frame())
    assert result.params["HYBRID"] < 0 and result.pvalues["HYBRID"] < 1e-6
