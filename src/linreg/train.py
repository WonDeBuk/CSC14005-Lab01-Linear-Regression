"""Train every model in MODELS, save them and a metrics table.

    python -m linreg.train
"""

import joblib
import pandas as pd

from . import config
from .data import Dataset, load_dataset
from .evaluate import coefficients, cv_rmse_folds, regression_metrics
from .models import MODELS, build_model


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
    return metrics


if __name__ == "__main__":
    print(train_all(load_dataset()).drop(columns="description").round(3).to_string(index=False))
