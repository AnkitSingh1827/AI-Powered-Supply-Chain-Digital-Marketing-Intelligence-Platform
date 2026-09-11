from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
	features: dict[str, Any] = Field(
		..., description="The engineered feature row used by the validated model."
	)
	substitute_available: bool = False


class PredictionResponse(BaseModel):
	predicted_disruption: int
	disruption_probability: float
	risk_band: str
	marketing_action: str
	budget_action: str
	decision_message: str
	recommended_actions: list[str]


class NewsRequest(BaseModel):
	text: str = Field(..., min_length=1, max_length=10000)