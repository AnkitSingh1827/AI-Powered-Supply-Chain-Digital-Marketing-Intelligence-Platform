def remove_stopwords(tokens):
    from nltk.corpus import stopwords

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError as exc:
        raise RuntimeError(
            "NLTK stopwords are missing. Run: "
            "python -m nltk.downloader stopwords"
        ) from exc

    return [
        token for token in tokens
        if token.lower() not in stop_words
    ]
