import re


def clean_text(text: str) -> str:
    text = "" if text is None else str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text,
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"\d+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text,
    )

    return re.sub(r"\s+", " ", text).strip()
