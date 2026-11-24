"""Train every model in MODELS, save them and a metrics table.

    python -m linreg.train
"""

import json

import joblib
import pandas as pd

from . import config
from .data import Dataset, load_dataset
from .evaluate import coefficients, cv_r2, regression_metrics
from .models import MODELS, build_model


def train_all(ds: Dataset, out_dir=config.MODELS_DIR) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in MODELS:
        model = build_model(name, ds.numeric_features, ds.categorical_features)
        cv_mean, cv_std = cv_r2(model, ds.X_train, ds.y_train)
        model.fit(ds.X_train, ds.y_train)
        rows.append(
            {"model": name, **regression_metrics(ds.y_test, model.predict(ds.X_test)),
             "CV_R2_mean": cv_mean, "CV_R2_std": cv_std}
        )
        joblib.dump(model, out_dir / f"{name}.joblib")
        coefficients(model).to_csv(out_dir / f"{name}_coefficients.csv", index=False)

    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    X = pd.concat([ds.X_train, ds.X_test])
    meta = {
        "target": ds.target,
        "best_model": metrics.loc[0, "model"],
        "numeric_features": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in ds.numeric_features},
        "categorical_features": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in ds.categorical_features},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    print(train_all(load_dataset()).to_string(index=False))
