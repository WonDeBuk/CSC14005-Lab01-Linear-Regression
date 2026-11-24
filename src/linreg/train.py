"""Train every model in MODELS, save them and a metrics table.

    python -m linreg.train
"""

import json

import joblib
import pandas as pd

from . import config
from .data import Dataset, load_dataset
from .evaluate import coefficients, cv_scores, regression_metrics
from .features import INPUT_BINARY, INPUT_CATEGORICAL, INPUT_NUMERIC
from .models import MODELS, build_model


def train_all(ds: Dataset, out_dir=config.MODELS_DIR) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, spec in MODELS.items():
        model = build_model(name)
        cv = cv_scores(model, ds.X_train, ds.y_train)
        model.fit(ds.X_train, ds.y_train)
        rows.append(
            {"model": name, **regression_metrics(ds.y_test, model.predict(ds.X_test)), **cv,
             "description": spec.description}
        )
        joblib.dump(model, out_dir / f"{name}.joblib")
        coefficients(model).to_csv(out_dir / f"{name}_coefficients.csv", index=False)

    metrics = pd.DataFrame(rows).sort_values("CV_RMSE_mean").reset_index(drop=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    X = pd.concat([ds.X_train, ds.X_test])
    meta = {
        "target": ds.target,
        "best_model": metrics.loc[0, "model"],
        "numeric": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in INPUT_NUMERIC},
        "categorical": {c: sorted(X[c].dropna().unique().tolist()) for c in INPUT_CATEGORICAL},
        "binary": INPUT_BINARY,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    print(train_all(load_dataset()).drop(columns="description").round(3).to_string(index=False))
