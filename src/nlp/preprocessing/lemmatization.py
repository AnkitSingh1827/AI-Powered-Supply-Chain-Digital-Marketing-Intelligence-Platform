def lemmatize(tokens):
    from nltk.stem import WordNetLemmatizer

    try:
        lemmatizer = WordNetLemmatizer()
        return [
            lemmatizer.lemmatize(token)
            for token in tokens
        ]
    except LookupError as exc:
        raise RuntimeError(
            "NLTK WordNet data is missing. Run: "
            "python -m nltk.downloader wordnet"
        ) from exc


def lemmatize_tokens(tokens):
    return lemmatize(tokens)
