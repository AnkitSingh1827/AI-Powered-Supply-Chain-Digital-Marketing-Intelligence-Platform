"""
High-level production prediction facade.

Usage:
    from src.predict import predict_engineered_shipments

The input must contain the same 49 engineered predictor columns used by the
saved Notebook 08 preprocessor. Raw-to-engineered feature generation is not
silently approximated here because doing so would change the model contract.
"""

import pandas as pd

from .models.xgboost_model import XGBoostPredictor
from .decision.recommendation_engine import make_recommendation


def predict_engineered_shipments(
    X: pd.DataFrame,
    substitute_available: bool = False,
) -> pd.DataFrame:
    predictor = XGBoostPredictor()
    result = predictor.predict_with_probability(X)

    recommendations = [
        make_recommendation(
            probability,
            substitute_available=substitute_available,
        )
        for probability in result["disruption_probability"]
    ]

    result["risk_band"] = [
        item["risk_band"] for item in recommendations
    ]

    result["marketing_action"] = [
        item["marketing_action"] for item in recommendations
    ]

    result["budget_action"] = [
        item["budget_action"] for item in recommendations
    ]

    result["decision_message"] = [
        item["message"] for item in recommendations
    ]

    return result
