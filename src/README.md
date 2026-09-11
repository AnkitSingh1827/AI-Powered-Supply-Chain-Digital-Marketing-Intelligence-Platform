# Complete `src` package — AI Supply Chain Digital Marketing

This package is aligned with the validated project pipeline through Notebook 13.

## Production path

`src.predict` -> `src.models.xgboost_model` -> saved `preprocessor.joblib` -> saved
`final_xgboost_candidate.joblib` -> `src.decision`.

The saved artifacts are the source of truth. Do not retrain or retune the model
during inference.

## Required model artifacts

The project root must contain:

```text
models/supply_chain/
├── preprocessor.joblib
├── final_xgboost_candidate.joblib
└── preprocessed_feature_names.json
```

These were produced by the validated notebooks.

## Important data contract

The validated model used 49 engineered predictors before preprocessing and
590 columns after preprocessing. `Shipment_ID` is retained for reporting but
is excluded from model input.

The production predictor intentionally refuses to invent missing engineered
historical features. Raw shipment input must first go through the same feature
engineering process used by Notebook 06.

## News

The supplied news CSV has no date column. NLP utilities therefore provide
corpus-level processing and VADER comparison, but do not claim temporal
news-to-shipment causality.

## Marketing

The marketing dataset is kept separate from shipment-level inference because
its dates are 2021 while the shipment dataset covers 2024–2025. The decision
layer maps disruption probability to transparent marketing actions without
claiming causal ROI improvement.

## Files intentionally kept as compatibility/reference modules

- `models/logistic_regression.py`
- `models/random_forest.py`
- `models/lightgbm_model.py`
- `nlp/sentiment/bert_sentiment.py`
- `evaluation/regression_metrics.py`

They are not used by the final production XGBoost inference path.
