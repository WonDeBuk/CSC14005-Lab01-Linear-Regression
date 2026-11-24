"""Derived design features for the CO2 dataset (moodboard section 08, steps P5-P8)."""

import pandas as pd

# P5: 16 vehicle classes -> 5 body groups; "small" is the reference group.
BODY_GROUPS = {
    "MINICOMPACT": "small",
    "SUBCOMPACT": "small",
    "COMPACT": "small",
    "TWO-SEATER": "small",
    "MID-SIZE": "midlarge",
    "FULL-SIZE": "midlarge",
    "STATION WAGON - SMALL": "midlarge",
    "STATION WAGON - MID-SIZE": "midlarge",
    "SUV - SMALL": "suv",
    "SUV - STANDARD": "suv",
    "PICKUP TRUCK - SMALL": "pickup",
    "PICKUP TRUCK - STANDARD": "pickup",
    "MINIVAN": "van",
    "VAN - PASSENGER": "van",
    "VAN - CARGO": "van",
    "SPECIAL PURPOSE VEHICLE": "van",
}
BODY_LABELS = {
    "small": "Xe nhỏ",
    "midlarge": "Sedan/wagon trung-lớn",
    "suv": "SUV",
    "pickup": "Bán tải",
    "van": "Van & đặc dụng",
}
FUEL_LABELS = {"X": "Xăng thường", "Z": "Xăng cao cấp", "D": "Diesel", "E": "Ethanol E85"}
TRANS_LABELS = {"A": "Tự động", "AS": "Tự động chọn số", "AM": "Bán tự động", "AV": "Vô cấp (CVT)", "M": "Số sàn"}

# Reference level of each categorical feature (dropped by one-hot encoding).
REFERENCE = {"FUELTYPE": "X", "BODY": "small", "TRANS_TYPE": "A"}

# Raw inputs a user sets in the app; everything else is derived from these.
INPUT_NUMERIC = ["ENGINESIZE", "CYLINDERS", "GEARS"]
INPUT_CATEGORICAL = ["BODY", "FUELTYPE", "TRANS_TYPE"]
INPUT_BINARY = ["HYBRID"]


def is_hybrid(make: pd.Series, model: pd.Series) -> pd.Series:
    """P8 with the section 12.8 fix: 'HYBRID' in the name misses Lexus '...h', Prius and CR-Z."""
    upper = model.str.upper()
    return (
        upper.str.contains("HYBRID")
        | upper.str.startswith(("PRIUS", "CR-Z"))
        | ((make.str.upper() == "LEXUS") & model.str.contains(r"\d{3}h\b", regex=True))
    )


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Columns computed from the inputs: CVT flag, M4 and M8b terms."""
    df = df.copy()
    df["NOGEAR"] = (df["GEARS"] == 0).astype(int)
    df["ENGINESIZE_SQ"] = df["ENGINESIZE"] ** 2
    df["ENGINE_X_DIESEL"] = df["ENGINESIZE"] * (df["FUELTYPE"] == "D")
    df["ENGINE_X_E85"] = df["ENGINESIZE"] * (df["FUELTYPE"] == "E")
    df["ENGINE_X_PICKUP"] = df["ENGINESIZE"] * (df["BODY"] == "pickup")
    df["ENGINE_X_VAN"] = df["ENGINESIZE"] * (df["BODY"] == "van")
    df["ENGINE_X_HYBRID"] = df["ENGINESIZE"] * df["HYBRID"]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BODY"] = df["VEHICLECLASS"].map(BODY_GROUPS)
    # P6: "AS6" -> type AS, 6 gears; 46 CVTs ("AV") have no gear count -> GEARS = 0, NOGEAR = 1.
    df["TRANS_TYPE"] = df["TRANSMISSION"].str.extract(r"^([A-Z]+)", expand=False)
    df["GEARS"] = pd.to_numeric(df["TRANSMISSION"].str.extract(r"(\d+)$", expand=False)).fillna(0)
    df["HYBRID"] = is_hybrid(df["MAKE"], df["MODEL"]).astype(int)
    # Original flag from section 09 (24 cars), kept to reproduce M3h before the fix.
    df["HYBRID_NAME"] = df["MODEL"].str.upper().str.contains("HYBRID").astype(int)
    return add_derived(df)
