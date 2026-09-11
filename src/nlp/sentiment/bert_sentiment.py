def build_pipeline(
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
):
    """
    Optional research extension.

    BERT was not used in the validated final model pipeline, so this function
    is intentionally isolated from the production shipment predictor.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers is not installed. Install it only if BERT is needed."
        ) from exc

    return pipeline(
        "sentiment-analysis",
        model=model_name,
    )
