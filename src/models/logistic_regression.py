from sklearn.linear_model import LogisticRegression


def build_model():
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
