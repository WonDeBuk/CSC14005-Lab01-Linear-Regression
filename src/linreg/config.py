"""Project-wide settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

RANDOM_STATE = 42
N_TRAIN = 854  # P9: np.random.default_rng(42).permutation, first 854 rows train, last 213 test
CV_FOLDS = 10
ALPHA = 0.05

# Fuel Consumption Ratings 2014, Natural Resources Canada (see data/README.md).
DATASET = {
    "file": RAW_DIR / "FuelConsumptionCo2.csv",
    "target": "CO2EMISSIONS",  # g/km
    # P1: MODELYEAR is constant (2014). P2: CO2 is computed from fuel consumption
    # (CO2/COMB = 23 petrol, 27 diesel, 16 E85), so these columns leak the target.
    "drop": [
        "MODELYEAR",
        "FUELCONSUMPTION_CITY",
        "FUELCONSUMPTION_HWY",
        "FUELCONSUMPTION_COMB",
        "FUELCONSUMPTION_COMB_MPG",
    ],
}
