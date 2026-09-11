import numpy as np
import pandas as pd


def aggregate_marketing(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Channel_Used", "Campaign_Type", "Customer_Segment",
        "Campaign_ID", "ROI", "Clicks", "Impressions",
        "Conversion_Rate", "Engagement_Score", "Acquisition_Cost",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing marketing columns: {sorted(missing)}")

    return (
        df.groupby(
            ["Channel_Used", "Campaign_Type", "Customer_Segment"],
            as_index=False,
        )
        .agg(
            campaigns=("Campaign_ID", "count"),
            avg_roi=("ROI", "mean"),
            total_clicks=("Clicks", "sum"),
            total_impressions=("Impressions", "sum"),
            avg_conversion_rate=("Conversion_Rate", "mean"),
            avg_engagement=("Engagement_Score", "mean"),
            total_acquisition_cost=("Acquisition_Cost", "sum"),
        )
    )


def create_marketing_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Date" in out:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["campaign_year"] = out["Date"].dt.year
        out["campaign_month"] = out["Date"].dt.month

    if {"Clicks", "Impressions"}.issubset(out.columns):
        out["click_through_rate"] = (
            out["Clicks"]
            / out["Impressions"].replace(0, np.nan)
        )

    return out
