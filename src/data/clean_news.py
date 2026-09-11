import pandas as pd


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = {"sentiment", "text"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing news columns: {sorted(missing)}")

    out["sentiment"] = (
        out["sentiment"].astype("string").str.strip().str.lower()
    )
    out["text"] = (
        out["text"].astype("string").fillna("").str.strip()
    )

    out = out[out["text"].ne("")]
    out = out.drop_duplicates(subset=["sentiment", "text"])

    return out.reset_index(drop=True)
