from backend.main import app
from backend.api.news import router as news_router
from backend.api.prediction import router as prediction_router


def test_prediction_and_news_routes_are_registered():
	prediction_paths = {route.path for route in prediction_router.routes}
	news_paths = {route.path for route in news_router.routes}
	assert "/predict" in prediction_paths
	assert "/overview" in prediction_paths
	assert "/news/analyze" in news_paths
	assert any(getattr(route, "include_context", None) for route in app.routes)