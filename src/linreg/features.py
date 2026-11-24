"""Derived design features for the CO2 dataset.

TODO(review): check BODY_GROUPS and the hybrid rule against section 08 of the
team moodboard before writing the report.
"""

import pandas as pd

BODY_GROUPS = {
    "MINICOMPACT": "car",
    "SUBCOMPACT": "car",
    "COMPACT": "car",
    "MID-SIZE": "car",
    "FULL-SIZE": "car",
    "TWO-SEATER": "car",
    "STATION WAGON - SMALL": "wagon",
    "STATION WAGON - MID-SIZE": "wagon",
    "SUV - SMALL": "suv",
    "SUV - STANDARD": "suv",
    "SPECIAL PURPOSE VEHICLE": "suv",
    "PICKUP TRUCK - SMALL": "pickup",
    "PICKUP TRUCK - STANDARD": "pickup",
    "MINIVAN": "van",
    "VAN - PASSENGER": "van",
    "VAN - CARGO": "van",
}

FUEL_LABELS = {"X": "Xăng thường", "Z": "Xăng cao cấp", "D": "Diesel", "E": "Ethanol E85"}

# Reference level of each categorical feature (dropped by one-hot encoding).
REFERENCE = {"FUELTYPE": "X", "BODY": "car", "TRANS_TYPE": "A"}

# Raw inputs a user sets in the app; everything else is derived from these.
INPUT_NUMERIC = ["ENGINESIZE", "CYLINDERS"]
INPUT_CATEGORICAL = ["FUELTYPE", "BODY", "TRANS_TYPE"]
INPUT_BINARY = ["HYBRID"]

INTERACTIONS = ["ENGINE_X_PICKUP", "ENGINE_X_VAN", "ENGINE_X_HYBRID"]


def is_hybrid(make: pd.Series, model: pd.Series) -> pd.Series:
    """'HYBRID' in the name misses 10 cars: Lexus '...h' models, Prius and CR-Z."""
    upper = model.str.upper()
    return (
        upper.str.contains("HYBRID")
        | upper.str.startswith(("PRIUS", "CR-Z"))
        | ((make.str.upper() == "LEXUS") & model.str.contains(r"\d{3}h\b", regex=True))
    )


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ENGINE_X_PICKUP"] = df["ENGINESIZE"] * (df["BODY"] == "pickup")
    df["ENGINE_X_VAN"] = df["ENGINESIZE"] * (df["BODY"] == "van")
    df["ENGINE_X_HYBRID"] = df["ENGINESIZE"] * df["HYBRID"]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BODY"] = df["VEHICLECLASS"].map(BODY_GROUPS)
    df["TRANS_TYPE"] = df["TRANSMISSION"].str.extract(r"^([A-Z]+)", expand=False)
    df["GEARS"] = pd.to_numeric(df["TRANSMISSION"].str.extract(r"(\d+)$", expand=False))
    df["HYBRID"] = is_hybrid(df["MAKE"], df["MODEL"]).astype(int)
    return add_interactions(df)
