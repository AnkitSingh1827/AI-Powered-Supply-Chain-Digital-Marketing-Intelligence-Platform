from .event_keywords import EVENT_KEYWORDS, detect_event_keywords


def classify_event(text: str) -> str:
    text = str(text).lower()

    scores = {
        event: sum(
            1 for keyword in keywords
            if keyword in text
        )
        for event, keywords in EVENT_KEYWORDS.items()
    }

    best_event = max(
        scores,
        key=scores.get,
    )

    return (
        best_event
        if scores[best_event] > 0
        else "other"
    )


def classify_events(text: str) -> list[str]:
    return detect_event_keywords(text)
