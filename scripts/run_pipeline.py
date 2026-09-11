from pathlib import Path
import sys
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_data import load_all
from src.data.clean_supply_chain import clean_supply_chain
from src.data.clean_news import clean_news
from src.data.clean_marketing import clean_marketing
from src.data.clean_trade import clean_monthly_shipping, clean_tariffs
from src.data.merge_data import add_monthly_trade_features
from src.nlp.feature_extraction import build_news_nlp_features
from src.features.marketing_features import aggregate_marketing
from src.features.trade_features import engineer_trade_features
from src.features.feature_selection import select_model_features

def main():
    paths = {
        "interim": ROOT / "data/interim",
        "processed": ROOT / "data/processed",
        "results_nlp": ROOT / "results/nlp",
        "results_metrics": ROOT / "results/metrics",
        "results_predictions": ROOT / "results/predictions",
        "models": ROOT / "models/supply_chain",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    data = load_all(ROOT / "data/raw")

    # Step 2: cleaning
    sc = clean_supply_chain(data["supply_chain"])
    news = clean_news(data["news"])
    marketing = clean_marketing(data["marketing"])
    shipping = clean_monthly_shipping(data["shipping_rates"])
    tariffs = clean_tariffs(data["tariffs"])

    sc.to_csv(paths["interim"] / "cleaned_supply_chain.csv", index=False)
    news.to_csv(paths["interim"] / "cleaned_news.csv", index=False)
    marketing.to_csv(paths["interim"] / "cleaned_marketing.csv", index=False)

    # NLP: VADER + transparent event keyword classifier
    news_nlp = build_news_nlp_features(news)
    news_nlp.to_csv(paths["results_nlp"] / "sentiment_and_event_results.csv", index=False)
    news_nlp[["sentiment", "event_type", "vader_compound"]].to_csv(
        paths["processed"] / "news_nlp_features.csv", index=False
    )

    # Marketing aggregation (separate 2021 layer)
    marketing_features = aggregate_marketing(marketing)
    marketing_features.to_csv(paths["processed"] / "marketing_features.csv", index=False)

    # Build main shipment model table using valid monthly historical joins
    model_df = add_monthly_trade_features(sc, shipping, tariffs)
    model_df = engineer_trade_features(model_df)

    # Important: no fake date-based merge of the supplied news corpus.
    model_df.to_csv(paths["processed"] / "final_training_data.csv", index=False)

    # Step 5: strict time-based split
    model_df = model_df.sort_values("Date").reset_index(drop=True)
    n = len(model_df)
    i = int(n * 0.70)
    j = int(n * 0.85)
    train = model_df.iloc[:i].copy()
    val = model_df.iloc[i:j].copy()
    test = model_df.iloc[j:].copy()

    feature_cols = [c for c in select_model_features(model_df) if c in model_df.columns]
    categorical = ["Weather_Condition", "Transport_Mode", "Product_Category",
                   "Origin_Port", "Destination_Port"]
    numeric = [c for c in feature_cols if c not in categorical]
    feature_cols = numeric + [c for c in categorical if c in model_df.columns]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), [c for c in categorical if c in model_df.columns])
    ])

    X_train, y_train = train[feature_cols], train["Disruption_Occurred"].astype(int)
    X_val, y_val = val[feature_cols], val["Disruption_Occurred"].astype(int)
    X_test, y_test = test[feature_cols], test["Disruption_Occurred"].astype(int)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
    }

    rows = []
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", pre), ("model", estimator)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.50).astype(int)
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
        }
        rows.append(metrics)
        fitted[name] = pipe

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(paths["results_metrics"] / "model_comparison.csv", index=False)

    best_name = metrics_df.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, paths["models"] / f"{best_name}.joblib")

    test_out = test[["Shipment_ID", "Date", "Disruption_Occurred"]].copy()
    test_out["risk_probability"] = best_model.predict_proba(X_test)[:, 1]
    test_out["risk_band"] = pd.cut(
        test_out["risk_probability"],
        bins=[-0.01, 0.50, 0.70, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"]
    )
    test_out.to_csv(paths["results_predictions"] / "disruption_predictions.csv", index=False)

    summary = {
        "rows_supply_chain": int(len(sc)),
        "rows_news": int(len(news)),
        "rows_marketing": int(len(marketing)),
        "news_has_date": False,
        "marketing_date_range": [
            str(marketing["Date"].min().date()),
            str(marketing["Date"].max().date())
        ],
        "supply_chain_date_range": [
            str(sc["Date"].min().date()),
            str(sc["Date"].max().date())
        ],
        "best_model": best_name,
        "features_used": feature_cols,
        "note": "News is intentionally not joined by date because the supplied news dataset has no date column."
    }
    (paths["results_metrics"] / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nPIPELINE COMPLETE")
    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print("News was processed with NLP but not falsely time-joined.")

if __name__ == "__main__":
    main()
