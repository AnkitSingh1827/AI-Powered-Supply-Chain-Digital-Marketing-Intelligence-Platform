from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_engineered_dataset_matches_model_input_contract():
	data = pd.read_csv(ROOT / "data/processed/shipment_features_engineered.csv", nrows=1)
	model_columns = set(data.columns) - {"Shipment_ID", "Disruption_Occurred"}
	assert len(model_columns) == 49


def test_processed_dataset_is_not_empty():
	data = pd.read_csv(ROOT / "data/processed/shipment_features_engineered.csv", usecols=["Shipment_ID"])
	assert len(data) > 0