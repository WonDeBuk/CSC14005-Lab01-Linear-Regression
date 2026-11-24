"""Export the OLS models the web app runs to JSON.

    python -m linreg.export_web            # writes web/public/model.json

The app predicts in the browser: x'beta for the estimate, and
(X'X)^-1 with sigma^2 for the 95% confidence and prediction intervals.
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .data import load_dataset
from .evaluate import cv_rmse_folds, regression_metrics
from .features import BODY_LABELS, FUEL_LABELS, INPUT_NUMERIC, TRANS_LABELS
from .models import MODELS, build_model

WEB_MODELS = ["M3h", "M8b"]
OUT = config.ROOT / "web" / "public" / "model.json"


def export_model(name: str, ds) -> dict:
    model = build_model(name).fit(ds.X_train, ds.y_train)
    prep = model.named_steps["prep"]
    X = np.column_stack([np.ones(len(ds.X_train)), prep.transform(ds.X_train)])
    n, p = X.shape
    resid = ds.y_train.to_numpy() - X @ np.r_[model.named_steps["model"].intercept_, model.named_steps["model"].coef_]
    cv = cv_rmse_folds(build_model(name), ds.X_train, ds.y_train)
    return {
        "name": name,
        "description": MODELS[name].description,
        "features": ["(intercept)", *prep.get_feature_names_out().tolist()],
        "coef": np.r_[model.named_steps["model"].intercept_, model.named_steps["model"].coef_].round(6).tolist(),
        "xtx_inv": np.linalg.inv(X.T @ X).tolist(),
        "sigma2": float(resid @ resid / (n - p)),
        "t_crit": float(stats.t.ppf(1 - config.ALPHA / 2, n - p)),
        "leverage_limit": 3 * p / n,
        "metrics": {
            **{k: round(v, 4) for k, v in regression_metrics(ds.y_test, model.predict(ds.X_test)).items()},
            "CV_RMSE_mean": round(float(cv.mean()), 4),
            "CV_RMSE_std": round(float(cv.std(ddof=1)), 4),
        },
    }


def main() -> None:
    ds = load_dataset()
    X = pd.concat([ds.X_train, ds.X_test])
    y = pd.concat([ds.y_train, ds.y_test])
    payload = {
        "source": "Fuel Consumption Ratings 2014, Natural Resources Canada; 1,067 cars",
        "target": ds.target,
        "models": {name: export_model(name, ds) for name in WEB_MODELS},
        "fleet": sorted(int(v) for v in y),
        "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in INPUT_NUMERIC},
        "labels": {"BODY": BODY_LABELS, "FUELTYPE": FUEL_LABELS, "TRANS_TYPE": TRANS_LABELS},
        # Typical setup per body group, used as the starting point when the body changes.
        "presets": {
            body: {
                "ENGINESIZE": float(g["ENGINESIZE"].median()),
                "CYLINDERS": int(g["CYLINDERS"].median()),
                "GEARS": int(g["GEARS"].median()),
            }
            for body, g in X.groupby("BODY")
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
