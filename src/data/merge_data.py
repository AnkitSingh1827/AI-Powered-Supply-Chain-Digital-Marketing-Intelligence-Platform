import pandas as pd


def _month_key(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def add_monthly_trade_features(
    supply_chain: pd.DataFrame,
    shipping_rates: pd.DataFrame,
    tariffs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the validated monthly shipping/tariff aggregates used by the project.

    This function only uses columns actually present in the supplied
    historical datasets. It does not invent values for missing observations.
    """
    out = supply_chain.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["month_key"] = _month_key(out["Date"])

    ship = shipping_rates.copy()
    ship["date"] = pd.to_datetime(ship["date"], errors="coerce")
    ship["month_key"] = _month_key(ship["date"])

    ship_cols = [
        "month_key",
        "baltic_dry_index",
        "container_rate_usd_40ft",
        "air_cargo_rate_usd_kg",
        "bdi_mom_change_pct",
        "container_yoy_pct",
        "tanker_rate_aframax_usd_day",
        "bulk_carrier_handysize_usd_day",
        "supply_chain_pressure_index",
        "on_time_delivery_pct",
    ]
    available_ship_cols = [
        c for c in ship_cols if c in ship.columns
    ]

    if "month_key" in available_ship_cols:
        out = out.merge(
            ship[available_ship_cols].drop_duplicates("month_key"),
            on="month_key",
            how="left",
        )

    tar = tariffs.copy()
    tar["date"] = pd.to_datetime(tar["date"], errors="coerce")
    tar["month_key"] = _month_key(tar["date"])

    if "tariff_id" in tar.columns:
        tariff_id = "tariff_id"
        tar_month = tar.groupby("month_key", as_index=False).agg(
            tariff_event_count=(tariff_id, "nunique"),
        )
    else:
        tar_month = (
            tar.groupby("month_key", as_index=False)
            .size()
            .rename(columns={"size": "tariff_event_count"})
        )

    if "tariff_rate_pct" in tar.columns:
        rates = tar.groupby("month_key")["tariff_rate_pct"].mean()
        tar_month = tar_month.merge(
            rates.rename("avg_tariff_rate_pct"),
            on="month_key",
            how="left",
        )

    if "estimated_value_usd_bn" in tar.columns:
        values = tar.groupby("month_key")["estimated_value_usd_bn"].sum()
        tar_month = tar_month.merge(
            values.rename("total_tariff_value_usd_bn"),
            on="month_key",
            how="left",
        )

    out = out.merge(tar_month, on="month_key", how="left")
    return out.drop(columns=["month_key"], errors="ignore")


def remove_merge_suffix_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bad = [c for c in out.columns if c.endswith("_x") or c.endswith("_y")]
    return out.drop(columns=bad, errors="ignore")
