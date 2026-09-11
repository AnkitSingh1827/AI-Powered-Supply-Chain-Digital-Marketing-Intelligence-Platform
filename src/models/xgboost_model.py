from pathlib import Path
import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "supply_chain"

MODEL_PATH = MODEL_DIR / "final_xgboost_candidate.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"


class XGBoostPredictor:
    """
    Production wrapper around the exact artifacts validated by Notebooks
    08 and 11.

    The preprocessor is applied before the locked XGBoost model.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Final model not found: {self.model_path}"
            )

        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {self.preprocessor_path}"
            )

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

    def transform(self, X: pd.DataFrame):
        if isinstance(X, dict):
            X = pd.DataFrame([X])

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame or a dictionary."
            )

        # ColumnTransformer was fitted on the 49 engineered predictors.
        # Passing missing raw columns must fail rather than silently inventing
        # data.
        return self.preprocessor.transform(X)

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        X_processed = self.transform(X)
        probability = self.model.predict_proba(
            X_processed
        )[:, 1]

        probability = np.asarray(probability, dtype=float)

        if not np.isfinite(probability).all():
            raise ValueError("Model returned non-finite probabilities.")

        return probability

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_processed = self.transform(X)
        return np.asarray(
            self.model.predict(X_processed)
        )

    def predict_with_probability(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        probability = self.predict_probability(X)

        prediction = (
            probability >= 0.5
        ).astype(int)

        return pd.DataFrame({
            "predicted_disruption": prediction,
            "disruption_probability": probability,
        })


def build_model():
    """
    Reproduce the baseline XGBoost configuration used for comparison in
    Notebook 09/10. This function is for reproducibility only.

    Production inference should load final_xgboost_candidate.joblib instead
    of training a new model.
    """
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
