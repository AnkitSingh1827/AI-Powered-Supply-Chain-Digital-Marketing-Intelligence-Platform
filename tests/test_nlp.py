from src.nlp.sentiment.vader_sentiment import vader_label


def test_vader_label_boundaries_are_explicit(monkeypatch):
	monkeypatch.setattr(
		"src.nlp.sentiment.vader_sentiment.vader_compound",
		lambda text: 0.8,
	)
	assert vader_label("positive") == "positive"

	monkeypatch.setattr(
		"src.nlp.sentiment.vader_sentiment.vader_compound",
		lambda text: -0.8,
	)
	assert vader_label("negative") == "negative"