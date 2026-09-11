from pathlib import Path
from typing import Any
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV and fail with a useful error when it does not exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, **kwargs)


def load_supply_chain() -> pd.DataFrame:
    return load_csv(RAW_DATA_DIR / "supply_chain.csv")


def load_news() -> pd.DataFrame:
    """Load the supplied news CSV, which has no header row."""
    return load_csv(
        RAW_DATA_DIR / "news_sentiment.csv",
        header=None,
        names=["sentiment", "text"],
        encoding="latin1",
    )


def load_marketing() -> pd.DataFrame:
    return load_csv(RAW_DATA_DIR / "marketing_campaign.csv")


def load_historical_trade(name: str) -> pd.DataFrame:
    allowed = {
        "shipping_rates": "shipping_rates.csv",
        "port_congestion": "port_congestion.csv",
        "tariff_timeline": "tariff_timeline.csv",
        "trade_flows": "trade_flows.csv",
        "commodity_prices_supply_chain": "commodity_prices_supply_chain.csv",
        "disruption_events": "disruption_events.csv",
        "industry_exposure": "industry_exposure.csv",
    }
    if name not in allowed:
        raise ValueError(f"Unknown historical dataset: {name}")
    return load_csv(RAW_DATA_DIR / "historical_trade" / allowed[name])


def load_processed(name: str) -> pd.DataFrame:
    return load_csv(PROCESSED_DATA_DIR / name)


def load_engineered_shipments() -> pd.DataFrame:
    return load_processed("shipment_features_engineered.csv")


def load_final_test_predictions() -> pd.DataFrame:
    return load_csv(
        RESULTS_DIR / "predictions" / "final_test_predictions.csv"
    )


def load_all_raw() -> dict[str, pd.DataFrame]:
    """Load all datasets supplied in the project raw-data directory."""
    return {
        "supply_chain": load_supply_chain(),
        "news": load_news(),
        "marketing": load_marketing(),
        "shipping_rates": load_historical_trade("shipping_rates"),
        "port_congestion": load_historical_trade("port_congestion"),
        "tariff_timeline": load_historical_trade("tariff_timeline"),
        "trade_flows": load_historical_trade("trade_flows"),
        "commodity_prices": load_historical_trade(
            "commodity_prices_supply_chain"
        ),
        "disruption_events": load_historical_trade("disruption_events"),
        "industry_exposure": load_historical_trade("industry_exposure"),
    }
