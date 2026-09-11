import pandas as pd


def summarize_news_features(news_nlp: pd.DataFrame) -> pd.DataFrame:
    """
    Corpus-level news summary.

    The supplied news dataset has no date, so this function intentionally
    does not claim a temporal relationship with shipment disruption.
    """
    if news_nlp.empty:
        return pd.DataFrame([{
            "news_count": 0,
            "negative_share": 0.0,
            "positive_share": 0.0,
            "mean_vader_compound": 0.0,
            "dominant_event": "other",
        }])

    event_column = (
        "event_type"
        if "event_type" in news_nlp.columns
        else None
    )

    dominant_event = "other"
    if event_column and news_nlp[event_column].notna().any():
        dominant_event = (
            news_nlp[event_column]
            .mode()
            .iat[0]
        )

    return pd.DataFrame([{
        "news_count": len(news_nlp),
        "negative_share": (
            news_nlp["sentiment"] == "negative"
        ).mean(),
        "positive_share": (
            news_nlp["sentiment"] == "positive"
        ).mean(),
        "mean_vader_compound": (
            news_nlp["vader_compound"].mean()
            if "vader_compound" in news_nlp
            else float("nan")
        ),
        "dominant_event": dominant_event,
    }])
