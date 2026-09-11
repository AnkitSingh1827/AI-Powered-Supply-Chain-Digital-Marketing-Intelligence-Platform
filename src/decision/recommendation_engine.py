from .risk_rules import risk_band
from .marketing_actions import get_marketing_action


def make_recommendation(
    risk_probability: float,
    substitute_available: bool = False,
) -> dict:
    probability = float(risk_probability)
    band = risk_band(probability)
    action = get_marketing_action(band)

    recommended_actions = [
        action["marketing_action"]
    ]

    if substitute_available and band in {"Medium", "High"}:
        recommended_actions.append("Recommend substitute product.")

    return {
        "risk_probability": probability,
        "risk_band": band,
        "marketing_action": action["marketing_action"],
        "budget_action": action["budget_action"],
        "message": action["message"],
        "recommended_actions": recommended_actions,
    }
