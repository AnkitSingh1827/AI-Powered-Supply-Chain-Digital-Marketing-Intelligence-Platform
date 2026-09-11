import numpy as np
import pandas as pd


def engineer_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the deterministic shipment features used by the project.

    This function does not delete outliers and does not create target-derived
    features.
    """
    out = df.copy()

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.sort_values("Date").reset_index(drop=True)

        out["year"] = out["Date"].dt.year
        out["month"] = out["Date"].dt.month
        out["quarter"] = out["Date"].dt.quarter
        out["day_of_week"] = out["Date"].dt.dayofweek
        out["day_of_year"] = out["Date"].dt.dayofyear
        out["week_of_year"] = out["Date"].dt.isocalendar().week.astype(int)

        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
        out["day_of_week_sin"] = np.sin(
            2 * np.pi * out["day_of_week"] / 7
        )
        out["day_of_week_cos"] = np.cos(
            2 * np.pi * out["day_of_week"] / 7
        )

    if {"Distance_km", "Weight_MT"}.issubset(out.columns):
        out["distance_per_mt"] = (
            out["Distance_km"]
            / out["Weight_MT"].replace(0, np.nan)
        )

    if {
        "Fuel_Price_Index",
        "Geopolitical_Risk_Score",
    }.issubset(out.columns):
        out["fuel_risk_interaction"] = (
            out["Fuel_Price_Index"]
            * out["Geopolitical_Risk_Score"]
        )

    if "Carrier_Reliability_Score" in out.columns:
        out["reliability_risk_inverse"] = (
            1 - out["Carrier_Reliability_Score"]
        )

    if "Weather_Condition" in out.columns:
        adverse = {
            "fog", "storm", "hurricane", "rain"
        }
        out["weather_risk_flag"] = (
            out["Weather_Condition"]
            .astype(str)
            .str.lower()
            .isin(adverse)
            .astype(int)
        )

    if "Lead_Time_Days" in out.columns:
        out["long_lead_time_flag"] = (
            out["Lead_Time_Days"] >= 20
        ).astype(int)

    return out
