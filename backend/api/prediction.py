from fastapi import APIRouter, HTTPException

from backend.schemas.request_response import PredictionRequest, PredictionResponse
from backend.services.prediction_service import input_feature_names, feature_names, overview, predict, sample_features

router = APIRouter(tags=["prediction"])

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/overview")
def model_overview():
    return overview()


@router.get("/model/schema")
def model_schema():
    return {"input_feature_count": len(input_feature_names()), "input_features": input_feature_names(),
            "transformed_feature_count": len(feature_names()), "sample": sample_features()}


@router.post("/predict", response_model=PredictionResponse)
def create_prediction(request: PredictionRequest):
    try:
        return predict(request.features, request.substitute_available)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
