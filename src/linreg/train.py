"""Train every model in MODELS, save them and a metrics table.

    python -m linreg.train
"""

import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import Dataset, load_dataset
from .evaluate import coefficients, cv_rmse_folds, regression_metrics
from .features import INPUT_BINARY, INPUT_CATEGORICAL, INPUT_NUMERIC
from .models import MODELS, build_model

APP_MODEL = "M8b"  # TODO(team): moodboard section 14 still runs M3h; decide which model the app shows.


def train_all(ds: Dataset, out_dir=config.MODELS_DIR) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, folds = [], {}
    for name, spec in MODELS.items():
        model = build_model(name)
        folds[name] = cv_rmse_folds(model, ds.X_train, ds.y_train)
        model.fit(ds.X_train, ds.y_train)
        rows.append({
            "model": name,
            "params": len(coefficients(model)),
            **regression_metrics(ds.y_test, model.predict(ds.X_test)),
            "CV_RMSE_mean": float(folds[name].mean()),
            "CV_RMSE_std": float(folds[name].std(ddof=1)),
            "description": spec.description,
        })
        joblib.dump(model, out_dir / f"{name}.joblib")
        coefficients(model).to_csv(out_dir / f"{name}_coefficients.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(out_dir / "cv_folds.csv", index=False)

    X = pd.concat([ds.X_train, ds.X_test])
    y_train_pred = joblib.load(out_dir / f"{APP_MODEL}.joblib").predict(ds.X_train)
    meta = {
        "target": ds.target,
        "app_model": APP_MODEL,
        "residual_sd": float(np.std(ds.y_train - y_train_pred, ddof=1)),
        "numeric": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in INPUT_NUMERIC},
        "categorical": {c: sorted(X[c].dropna().unique().tolist()) for c in INPUT_CATEGORICAL},
        "binary": INPUT_BINARY,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    print(train_all(load_dataset()).drop(columns="description").round(3).to_string(index=False))
