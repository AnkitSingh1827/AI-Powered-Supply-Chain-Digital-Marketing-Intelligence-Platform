import pandas as pd


def clean_monthly_shipping(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out:
        raise ValueError("shipping_rates requires a 'date' column")
    out["date"] = pd.to_datetime(out["date"], format="%Y-%m", errors="coerce")
    out = out.drop_duplicates(subset=["date"]).sort_values("date")
    return out.reset_index(drop=True)


def clean_port_congestion(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "week_start" in out:
        out["week_start"] = pd.to_datetime(
            out["week_start"], errors="coerce"
        )
    return out.drop_duplicates().reset_index(drop=True)


def clean_tariffs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def clean_generic_trade(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().drop_duplicates()
    for col in out.columns:
        if "date" in col.lower():
            parsed = pd.to_datetime(out[col], errors="coerce")
            if parsed.notna().sum() > 0:
                out[col] = parsed
    return out.reset_index(drop=True)
