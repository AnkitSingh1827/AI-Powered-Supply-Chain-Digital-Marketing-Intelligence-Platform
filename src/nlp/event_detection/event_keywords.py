EVENT_KEYWORDS = {
    "war": [
        "war", "conflict", "invasion", "attack", "military",
    ],
    "strike": [
        "strike", "walkout", "union", "labor", "lockout",
    ],
    "weather": [
        "storm", "hurricane", "flood", "typhoon",
        "weather", "drought", "fog", "rain",
    ],
    "tariff": [
        "tariff", "duty", "trade war", "sanction",
    ],
    "port_disruption": [
        "port closure", "port congestion", "shipping delay",
        "container shortage", "canal", "vessel",
        "shipping disruption",
    ],
    "shipping": [
        "shipping", "freight", "container", "cargo",
    ],
    "pandemic": [
        "pandemic", "covid", "lockdown", "outbreak",
    ],
}


def detect_event_keywords(text: str) -> list[str]:
    text = str(text).lower()
    detected = []

    for event, keywords in EVENT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            detected.append(event)

    return detected
