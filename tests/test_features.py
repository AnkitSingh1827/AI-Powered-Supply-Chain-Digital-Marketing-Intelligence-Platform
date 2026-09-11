from backend.services.prediction_service import input_feature_names, sample_features


def test_sample_contains_every_input_feature():
	sample = sample_features()
	assert len(input_feature_names()) == 49
	assert set(sample) == set(input_feature_names())