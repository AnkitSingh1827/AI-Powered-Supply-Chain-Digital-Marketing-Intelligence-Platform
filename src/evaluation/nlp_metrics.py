from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import spearmanr


def sentiment_agreement(supplied_labels, comparison_labels) -> float:
    return float(
        accuracy_score(supplied_labels, comparison_labels)
    )


def sentiment_correlation(
    supplied_scores,
    comparison_scores,
) -> dict:
    correlation, p_value = spearmanr(
        supplied_scores, comparison_scores
    )
    return {
        "spearman_correlation": float(correlation),
        "p_value": float(p_value),
    }


def sentiment_confusion_matrix(
    supplied_labels,
    comparison_labels,
):
    return confusion_matrix(
        supplied_labels,
        comparison_labels,
        labels=["negative", "neutral", "positive"],
    )


def sentiment_report(y_true, y_pred) -> dict:
    return classification_report(
        y_true, y_pred,
        output_dict=True,
        zero_division=0,
    )
