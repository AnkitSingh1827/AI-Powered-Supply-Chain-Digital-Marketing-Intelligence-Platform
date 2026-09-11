import pandas as pd


def extract_numeric(series: pd.Series) -> pd.Series:
    """Extract numbers from values such as '30 days' or '$16,174.00'."""
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce",
    )


def clean_marketing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Date" in out:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    if "Duration" in out:
        out["Duration"] = extract_numeric(out["Duration"])

    if "Acquisition_Cost" in out:
        out["Acquisition_Cost"] = extract_numeric(out["Acquisition_Cost"])

    for col in [
        "Conversion_Rate", "ROI", "Clicks",
        "Impressions", "Engagement_Score",
    ]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "Campaign_ID" in out:
        out = out.drop_duplicates(subset=["Campaign_ID"])
    else:
        out = out.drop_duplicates()

    return out.reset_index(drop=True)
