# START HERE

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/setup_nlp.py
python scripts/run_pipeline.py
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/setup_nlp.py
.\.venv\Scripts\python.exe scripts/run_pipeline.py
```

## Run the application

Start the Streamlit platform from the repository root:

```powershell
\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Streamlit will open the platform at `http://localhost:8501`. The FastAPI service remains available separately when API access is needed:

```powershell
\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

The dashboard includes the validated reference-scenario inference flow, transparent marketing decision rules, and a VADER text-analysis tool. Model inference requires the saved scikit-learn/XGBoost artifacts and the engineered feature contract; it does not invent raw shipment features.

## What is already included

- Your uploaded raw datasets
- Data cleaning modules
- News NLP modules (VADER + optional BERT)
- Event keyword classification
- Trade feature engineering
- Time-based model training
- Logistic Regression / Random Forest / XGBoost
- Marketing decision rules
- FastAPI service
- Streamlit decision-intelligence platform
- Initial validation and baseline results

## Critical limitation

The supplied `news_sentiment.csv` has no date. Do not join it to shipment dates artificially.
