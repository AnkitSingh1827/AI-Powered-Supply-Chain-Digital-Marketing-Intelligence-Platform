from src.decision.recommendation_engine import make_recommendation


def test_recommendation_maps_high_risk_to_protective_action():
	result = make_recommendation(0.9, substitute_available=True)
	assert result["risk_band"] == "High"
	assert "Protect demand" in result["marketing_action"]
	assert "Recommend substitute product." in result["recommended_actions"]


def test_get_shipment_by_id_and_roi():
	from backend.services.prediction_service import get_shipment_by_id, calculate_marketing_roi
	shipment = get_shipment_by_id("SHIP-10235")
	assert shipment is not None
	assert shipment["Shipment_ID"] == "SC-10235"

	roi_info = calculate_marketing_roi(ad_spend_saved=23450, extra_spend=0, total_ad_spend=102000)
	assert roi_info["roi_pct"] == 23.0