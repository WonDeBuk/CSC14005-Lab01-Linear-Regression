"""Project-wide settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

RANDOM_STATE = 42
TEST_SIZE = 213  # 1,067 cars -> 854 train / 213 test, as on the team moodboard
CV_FOLDS = 10

# Fuel Consumption Ratings 2014, Natural Resources Canada (see data/README.md).
DATASET = {
    "file": RAW_DIR / "FuelConsumptionCo2.csv",
    "target": "CO2EMISSIONS",  # g/km
    # CO2 is computed from fuel consumption (fixed ratio per fuel type), so these
    # columns would leak the target: R² ≈ 1 just by re-reading the formula.
    "drop": [
        "FUELCONSUMPTION_CITY",
        "FUELCONSUMPTION_HWY",
        "FUELCONSUMPTION_COMB",
        "FUELCONSUMPTION_COMB_MPG",
    ],
}
