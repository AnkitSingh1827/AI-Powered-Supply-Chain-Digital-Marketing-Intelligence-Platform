from lightgbm import LGBMClassifier


def build_model():
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        class_weight="balanced",
        verbosity=-1,
    )
