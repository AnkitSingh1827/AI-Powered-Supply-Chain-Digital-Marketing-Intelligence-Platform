import pandas as pd


TARGET = "Disruption_Occurred"

REQUIRED_COLUMNS = [
    "Shipment_ID", "Date", "Origin_Port", "Destination_Port",
    "Transport_Mode", "Product_Category", "Distance_km", "Weight_MT",
    "Fuel_Price_Index", "Geopolitical_Risk_Score", "Weather_Condition",
    "Carrier_Reliability_Score", "Lead_Time_Days", TARGET,
]

NUMERIC_COLUMNS = [
    "Distance_km", "Weight_MT", "Fuel_Price_Index",
    "Geopolitical_Risk_Score", "Carrier_Reliability_Score",
    "Lead_Time_Days", TARGET,
]


def clean_supply_chain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing supply-chain columns: {missing}")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in [
        "Origin_Port", "Destination_Port", "Transport_Mode",
        "Product_Category", "Weather_Condition",
    ]:
        out[col] = out[col].astype("string").str.strip()

    out = out.drop_duplicates(subset=["Shipment_ID"])
    out = out.dropna(subset=["Date", TARGET])

    # Deliberately do not delete IQR outliers.
    # The validated EDA showed Lead_Time_Days outliers that may carry
    # disruption information.
    return out.reset_index(drop=True)
