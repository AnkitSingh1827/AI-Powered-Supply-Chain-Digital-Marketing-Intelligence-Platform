from .feature_extraction import build_news_nlp_features


def run_news_nlp(news_df):
    return build_news_nlp_features(news_df)
