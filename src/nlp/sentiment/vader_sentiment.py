import re


_FALLBACK_LEXICON = {
    "great": 0.8, "good": 0.6, "happy": 0.7, "love": 0.8, "excellent": 0.8,
    "fast": 0.5, "helpful": 0.6, "proactive": 0.6, "best": 0.8, "wonderful": 0.8,
    "satisfied": 0.7, "impressive": 0.7, "smooth": 0.6, "appreciated": 0.6,
    "bad": -0.7, "terrible": -0.8, "horrible": -0.9, "late": -0.6, "delayed": -0.6,
    "slow": -0.5, "poor": -0.6, "worst": -0.9, "disappointed": -0.7, "frustrated": -0.7,
    "complaint": -0.6, "waste": -0.7, "stuck": -0.5, "broken": -0.7, "damage": -0.7,
    "disruption": -0.6, "awful": -0.8, "fail": -0.7, "failed": -0.7, "delay": -0.6,
}


class _FallbackSentimentAnalyzer:
    """Lightweight fallback sentiment analyzer when NLTK is not available."""

    def polarity_scores(self, text: str) -> dict:
        tokens = re.findall(r"\b\w+\b", str(text).lower())
        if not tokens:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        score = 0.0
        pos_count = 0
        neg_count = 0

        for word in tokens:
            if word in _FALLBACK_LEXICON:
                weight = _FALLBACK_LEXICON[word]
                score += weight
                if weight > 0:
                    pos_count += 1
                else:
                    neg_count += 1

        total = len(tokens)
        neu_count = max(0, total - pos_count - neg_count)
        pos = round(pos_count / total, 3)
        neg = round(neg_count / total, 3)
        neu = round(neu_count / total, 3)

        if pos_count + neg_count > 0:
            compound = max(-1.0, min(1.0, round(score / (pos_count + neg_count), 4)))
        else:
            compound = 0.0

        return {"neg": neg, "neu": neu, "pos": pos, "compound": compound}


def _get_analyzer():
    try:
        from importlib import import_module
        import nltk

        sentiment = import_module("nltk.sentiment")
        SentimentIntensityAnalyzer = sentiment.SentimentIntensityAnalyzer
        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
    except (ImportError, ModuleNotFoundError, LookupError, Exception):
        return _FallbackSentimentAnalyzer()


def score_texts(texts):
    analyzer = _get_analyzer()
    return [
        analyzer.polarity_scores(str(text))
        for text in texts
    ]


def vader_score(text):
    return _get_analyzer().polarity_scores(str(text))


def vader_compound(text):
    return float(vader_score(text)["compound"])


def vader_label(text):
    compound = vader_compound(text)
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"
