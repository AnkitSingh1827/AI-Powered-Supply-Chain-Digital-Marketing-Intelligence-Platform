from fastapi import APIRouter

import pandas as pd

from backend.schemas.request_response import NewsRequest
from src.nlp.sentiment.vader_sentiment import vader_score, vader_label

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

router = APIRouter(tags=["news"])

@router.get("/news/status")
def news_status():
    data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "sentiment_analysis.csv")
    counts = data["vader_sentiment"].value_counts().to_dict()
    return {"status": "ready", "articles": int(len(data)), "sentiment_counts": counts,
            "note": "The supplied news dataset has sentiment and text but no date."}


@router.post("/news/analyze")
def analyze_news(request: NewsRequest):
    scores = vader_score(request.text)
    return {"label": vader_label(request.text), "scores": scores, "text": request.text}
