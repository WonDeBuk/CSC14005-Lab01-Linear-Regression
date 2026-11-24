"""Project-wide settings. Edit DATASET once the group has chosen its dataset."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# TODO(team): fill in after choosing the dataset (see data/README.md).
DATASET = {
    "file": RAW_DIR / "dataset.csv",
    "target": None,  # e.g. "price"
    "drop": [],  # id columns, leakage columns, ...
    # None = infer from dtypes; set explicit lists once EDA is done.
    "numeric_features": None,
    "categorical_features": None,
}
