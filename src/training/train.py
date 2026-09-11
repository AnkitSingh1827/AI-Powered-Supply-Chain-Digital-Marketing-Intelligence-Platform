import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def time_split(
    df: pd.DataFrame,
    train=0.70,
    validation=0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.sort_values("Date").reset_index(drop=True)

    n = len(out)
    i = int(n * train)
    j = int(n * (train + validation))

    return (
        out.iloc[:i].copy(),
        out.iloc[i:j].copy(),
        out.iloc[j:].copy(),
    )


def build_preprocessor(
    numeric_features,
    categorical_features,
):
    # Ensure Date is not one-hot encoded to prevent 500+ dummy column overfitting
    clean_cat_features = [f for f in categorical_features if f not in {"Date", "Shipment_ID", "Disruption_Occurred"}]
    clean_num_features = [f for f in numeric_features if f not in {"Date", "Shipment_ID", "Disruption_Occurred"}]

    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                clean_num_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                clean_cat_features,
            ),
        ],
        remainder="drop",
    )


def evaluate_binary(model, X, y):
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(
            y, prediction, zero_division=0
        ),
        "recall": recall_score(
            y, prediction, zero_division=0
        ),
        "f1": f1_score(
            y, prediction, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y, probability
        ),
        "pr_auc": average_precision_score(
            y, probability
        ),
    }
