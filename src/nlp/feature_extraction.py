import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing.text_cleaning import clean_text
from .sentiment.vader_sentiment import score_texts
from .event_detection.event_classifier import classify_event


def build_news_nlp_features(news: pd.DataFrame) -> pd.DataFrame:
    required = {"sentiment", "text"}
    missing = required - set(news.columns)
    if missing:
        raise ValueError(f"Missing news columns: {sorted(missing)}")

    out = news.copy()
    out["clean_text"] = out["text"].map(clean_text)

    scores = pd.DataFrame(
        score_texts(out["clean_text"].tolist())
    )

    for col in ["neg", "neu", "pos", "compound"]:
        out[f"vader_{col if col != 'compound' else 'compound'}"] = scores[col]

    out["event_type"] = out["clean_text"].map(classify_event)

    out["sentiment_score"] = (
        out["sentiment"]
        .astype(str)
        .str.lower()
        .map({
            "negative": -1,
            "neutral": 0,
            "positive": 1,
        })
    )

    return out


def create_tfidf_features(
    texts,
    max_features=1000,
    min_df=2,
    max_df=0.95,
):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer
