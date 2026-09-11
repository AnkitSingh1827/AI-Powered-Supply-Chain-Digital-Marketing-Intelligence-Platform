from pathlib import Path
import sys
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
DATA = ROOT / "data" / "interim" / "cleaned_supply_chain.csv"
MODEL_DIR = ROOT / "models" / "supply_chain"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
feature_cols = ["Distance_km","Weight_MT","Fuel_Price_Index","Geopolitical_Risk_Score",
                "Carrier_Reliability_Score","Lead_Time_Days","Weather_Condition",
                "Transport_Mode","Product_Category","Origin_Port","Destination_Port"]
target = "Disruption_Occurred"

n = len(df); i, j = int(n*0.70), int(n*0.85)
train, val, test = df.iloc[:i], df.iloc[i:j], df.iloc[j:]
cat = [c for c in feature_cols if df[c].dtype == "object"]
num = [c for c in feature_cols if c not in cat]
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
])
models = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42, n_jobs=-1)
}
rows=[]; fitted={}
Xtr,ytr=train[feature_cols],train[target].astype(int)
Xte,yte=test[feature_cols],test[target].astype(int)
for name, est in models.items():
    pipe=Pipeline([("preprocessor",pre),("model",est)])
    pipe.fit(Xtr,ytr)
    prob=pipe.predict_proba(Xte)[:,1]
    pred=(prob>=0.5).astype(int)
    rows.append({"model":name,"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),"roc_auc":roc_auc_score(yte,prob)})
    fitted[name]=pipe
metrics=pd.DataFrame(rows).sort_values("roc_auc",ascending=False)
best=metrics.iloc[0]["model"]
joblib.dump(fitted[best], MODEL_DIR/f"{best}.joblib")
print(metrics.to_string(index=False))
print(f"\nSaved best model: {best}")
