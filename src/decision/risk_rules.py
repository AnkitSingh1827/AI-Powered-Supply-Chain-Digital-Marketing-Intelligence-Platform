def risk_band(probability: float) -> str:
    """
    Business risk bands used by Notebook 13.

    These thresholds are transparent rules and are not optimized on test data.
    """
    probability = float(probability)

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Risk probability must be between 0 and 1.")

    if probability >= 0.75:
        return "High"
    if probability >= 0.50:
        return "Medium"
    return "Low"
