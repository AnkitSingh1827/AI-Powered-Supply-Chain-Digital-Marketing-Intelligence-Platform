MARKETING_ACTIONS = {
    "Low": {
        "marketing_action": "Continue planned campaigns",
        "budget_action": "Maintain planned spend",
        "message": (
            "Normal campaign activity; no disruption-specific intervention."
        ),
    },
    "Medium": {
        "marketing_action": "Use disruption-aware targeting",
        "budget_action": (
            "Prioritize flexible channels and monitor spend"
        ),
        "message": (
            "Review campaign timing, audience exposure, and product availability."
        ),
    },
    "High": {
        "marketing_action": "Protect demand and reallocate exposure",
        "budget_action": (
            "Reduce exposure to affected shipment-dependent products "
            "and prioritize resilient alternatives"
        ),
        "message": (
            "Coordinate marketing with supply-chain risk before increasing demand."
        ),
    },
}


def get_marketing_action(risk_band: str) -> dict:
    if risk_band not in MARKETING_ACTIONS:
        raise ValueError(f"Unknown risk band: {risk_band}")
    return MARKETING_ACTIONS[risk_band]


def recommend_actions(
    risk_probability: float,
    substitute_available: bool = False,
) -> list[str]:
    """Compatibility helper for the original project API."""
    from .risk_rules import risk_band

    band = risk_band(risk_probability)
    actions = [MARKETING_ACTIONS[band]["marketing_action"]]

    if band == "High":
        actions.append("Coordinate customer communication with supply status.")

    if substitute_available and band in {"Medium", "High"}:
        actions.append("Recommend substitute product.")

    return actions
