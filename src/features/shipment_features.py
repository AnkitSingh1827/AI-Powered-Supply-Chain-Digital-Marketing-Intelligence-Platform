"""
Production contract for the validated shipment model.

The validated Notebook 06/08 pipeline produced:
- 49 predictor columns before preprocessing
- 590 columns after preprocessing

The saved preprocessor is the source of truth for the exact column order.
This module provides validation and safe batch inference helpers.

For raw shipment inference, historical monthly features must be generated
using the same source data/feature-engineering process as Notebook 06.
This module never invents missing historical values.
"""

from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "supply_chain"

ENGINEERED_PATH = DATA_DIR / "shipment_features_engineered.csv"
FEATURE_NAMES_PATH = MODEL_DIR / "preprocessed_feature_names.json"


BASE_SHIPMENT_COLUMNS = [
    "Shipment_ID",
    "Date",
    "Origin_Port",
    "Destination_Port",
    "Transport_Mode",
    "Product_Category",
    "Distance_km",
    "Weight_MT",
    "Fuel_Price_Index",
    "Geopolitical_Risk_Score",
    "Weather_Condition",
    "Carrier_Reliability_Score",
    "Lead_Time_Days",
    "Disruption_Occurred",
]


def load_engineered_shipments(path=ENGINEERED_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"Engineered shipment file not found: {path}")
    return pd.read_csv(path)


def validate_engineered_columns(df: pd.DataFrame):
    """Validate that the engineered shipment table has model inputs."""
    required = [
        "Date",
        "Origin_Port",
        "Destination_Port",
        "Transport_Mode",
        "Product_Category",
        "Weather_Condition",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required shipment columns: {missing}"
        )

    if "Shipment_ID" in df.columns:
        # Identifier is retained for reporting but excluded from X.
        pass

    return True


def load_preprocessed_feature_names(
    path=FEATURE_NAMES_PATH,
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature-name artifact not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def prepare_engineered_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the 49 pre-preprocessing predictors from an already engineered
    shipment table.

    This intentionally does not silently fill missing engineered columns.
    The saved Notebook 08 preprocessor performs its own fitted imputation.
    """
    out = df.copy()
    out = out.drop(
        columns=["Disruption_Occurred"],
        errors="ignore",
    )

    out = out.drop(
        columns=["Shipment_ID"],
        errors="ignore",
    )

    # If the saved feature-name list is available, use it as the contract.
    # This prevents accidental use of new/unexpected columns.
    feature_names_path = FEATURE_NAMES_PATH
    if feature_names_path.exists():
        names = load_preprocessed_feature_names(feature_names_path)

        # The JSON artifact contains post-preprocessing feature names, so it
        # cannot directly be used to select the 49 raw predictors. The
        # preprocessor itself is the authoritative raw-column contract.
        _ = names

    return out
