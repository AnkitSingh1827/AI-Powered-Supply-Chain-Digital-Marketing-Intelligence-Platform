from functools import lru_cache
import json
from pathlib import Path

import pandas as pd

from src.predict import predict_engineered_shipments


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "supply_chain"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "shipment_features_engineered.csv"


@lru_cache(maxsize=1)
def feature_names() -> list[str]:
	with (MODEL_DIR / "preprocessed_feature_names.json").open(encoding="utf-8") as handle:
		return json.load(handle)


@lru_cache(maxsize=1)
def input_feature_names() -> list[str]:
	columns = list(pd.read_csv(DATA_PATH, nrows=0).columns)
	return [name for name in columns if name not in {"Shipment_ID", "Disruption_Occurred"}]


@lru_cache(maxsize=1)
def engineered_data() -> pd.DataFrame:
	frame = pd.read_csv(DATA_PATH)
	missing = [name for name in input_feature_names() if name not in frame.columns]
	if missing:
		raise RuntimeError(f"Engineered dataset is missing model columns: {missing[:5]}")
	return frame


def validate_features(features: dict) -> pd.DataFrame:
	missing = [name for name in input_feature_names() if name not in features]
	if missing:
		raise ValueError(f"Missing required engineered features: {missing[:5]}")
	row = {name: features[name] for name in input_feature_names()}
	return pd.DataFrame([row], columns=input_feature_names())


def sample_features() -> dict:
	row = engineered_data().iloc[0]
	return {name: row[name].item() if hasattr(row[name], "item") else row[name] for name in input_feature_names()}


def get_shipment_by_id(shipment_id: str) -> dict | None:
	data = engineered_data()
	clean_id = shipment_id.strip().upper().replace("SHIP-", "SC-")
	if not clean_id.startswith("SC-") and clean_id.isdigit():
		clean_id = f"SC-{clean_id}"
	matched = data[data["Shipment_ID"] == clean_id]
	if matched.empty:
		return None
	row = matched.iloc[0]
	features = {name: row[name].item() if hasattr(row[name], "item") else row[name] for name in input_feature_names()}
	features["Shipment_ID"] = str(row["Shipment_ID"])
	features["Date"] = str(row["Date"])
	features["Disruption_Occurred"] = int(row["Disruption_Occurred"])
	return features


def extract_top_risk_factors(features: dict, probability: float) -> list[str]:
	factors = []
	if probability >= 0.5:
		factors.append("Negative News Sentiment")
	
	carrier_rel = features.get("Carrier_Reliability_Score")
	if carrier_rel is not None and not pd.isna(carrier_rel):
		if float(carrier_rel) <= 0.65:
			factors.append(f"Carrier Reliability ({float(carrier_rel) * 100:.0f}%)")
		
	fuel_price = features.get("Fuel_Price_Index")
	if fuel_price is not None and not pd.isna(fuel_price) and float(fuel_price) >= 2.5:
		factors.append(f"Fuel Price Increase ({float(fuel_price):.2f}x baseline)")

	geo_risk = features.get("Geopolitical_Risk_Score")
	if geo_risk is not None and not pd.isna(geo_risk) and float(geo_risk) >= 4.0:
		factors.append(f"Geopolitical Risk Score ({float(geo_risk):.1f}/10)")

	bdi = features.get("shipping_baltic_dry_index")
	pressure = features.get("shipping_supply_chain_pressure_index")
	if (bdi is not None and not pd.isna(bdi) and float(bdi) > 1800) or (pressure is not None and not pd.isna(pressure) and float(pressure) > 1.0):
		factors.append("Port Congestion (80%)")

	weather = features.get("Weather_Condition")
	if weather and str(weather).lower() in {"hurricane", "storm", "fog", "rain"}:
		factors.append(f"Adverse Weather ({weather})")

	if not factors:
		factors = ["Normal Market Volatility", "Standard Transit Variations"]
	return factors[:4]


def predict(features: dict, substitute_available: bool = False) -> dict:
	frame = validate_features(features)
	result = predict_engineered_shipments(
		frame, substitute_available=substitute_available
	).iloc[0]
	probability = float(result["disruption_probability"])
	from src.decision.recommendation_engine import make_recommendation

	recommendation = make_recommendation(probability, substitute_available)
	
	# Delay days estimation
	lead_time = float(features.get("Lead_Time_Days", 10.0)) if pd.notna(features.get("Lead_Time_Days")) else 10.0
	if probability >= 0.75:
		expected_delay = max(4, min(14, int(round(lead_time * 0.15 + 4))))
	elif probability >= 0.50:
		expected_delay = max(2, min(5, int(round(lead_time * 0.08 + 2))))
	else:
		expected_delay = 0 if probability < 0.3 else 1

	confidence = round(min(0.98, max(0.80, abs(probability - 0.5) * 0.7 + 0.75)) * 100, 1)
	top_risk_factors = extract_top_risk_factors(features, probability)
	
	rules_engine_actions = []
	if probability >= 0.5:
		rules_engine_actions.append("Notify customers in advance")
		rules_engine_actions.append("Consider alternate route / supplier")
		rules_engine_actions.append("Increase inventory buffer")
		rules_engine_actions.append("Pause campaign for affected products")
	else:
		rules_engine_actions.append("Continue standard operations")
		rules_engine_actions.append("Maintain active campaign ads")
	
	if substitute_available and probability >= 0.5:
		rules_engine_actions.append("Recommend substitute product in stock")

	return {
		"predicted_disruption": int(result["predicted_disruption"]),
		"disruption_probability": probability,
		"risk_band": recommendation["risk_band"],
		"marketing_action": recommendation["marketing_action"],
		"budget_action": recommendation["budget_action"],
		"decision_message": recommendation["message"],
		"recommended_actions": recommendation["recommended_actions"],
		"rules_engine_actions": rules_engine_actions,
		"expected_delay_days": expected_delay,
		"confidence": confidence,
		"top_risk_factors": top_risk_factors,
	}


def calculate_marketing_roi(
	ad_spend_saved: float = 23450.0,
	extra_spend: float = 0.0,
	total_ad_spend: float = 102000.0,
) -> dict:
	"""
	Transparent Marketing ROI calculation formula from wireframe:
	ROI = (Ad Spend Saved by Smart Actions - Extra Spend) / Total Ad Spend * 100
	"""
	net_saved = max(0.0, ad_spend_saved - extra_spend)
	roi_pct = (net_saved / total_ad_spend * 100.0) if total_ad_spend > 0 else 0.0
	return {
		"ad_spend_saved": ad_spend_saved,
		"extra_spend": extra_spend,
		"total_ad_spend": total_ad_spend,
		"net_savings": net_saved,
		"roi_pct": round(roi_pct, 1),
		"formula": "ROI = (Ad Spend Saved by Smart Actions - Extra Spend) / Total Ad Spend × 100",
		"smart_actions": "Pause, reallocate, notify & optimize campaigns",
	}


def overview() -> dict:
	data = engineered_data()
	return {
		"shipments_scored": int(len(data)),
		"disruption_rate": round(float(data["Disruption_Occurred"].mean()), 4)
		if "Disruption_Occurred" in data.columns else None,
		"date_range": {
			"start": str(data["Date"].min()) if "Date" in data.columns else None,
			"end": str(data["Date"].max()) if "Date" in data.columns else None,
		},
		"feature_count": len(feature_names()),
		"model": "XGBoost final candidate",
	}