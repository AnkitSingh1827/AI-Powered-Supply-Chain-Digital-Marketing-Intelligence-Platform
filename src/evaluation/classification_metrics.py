from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


def calculate_classification_metrics(
    y_true,
    y_pred,
    y_probability,
) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true, y_probability
        ),
        "pr_auc": average_precision_score(
            y_true, y_probability
        ),
    }


def classification_summary(y_true, y_pred, y_prob) -> dict:
    metrics = calculate_classification_metrics(
        y_true, y_pred, y_prob
    )
    metrics["report"] = classification_report(
        y_true, y_pred,
        output_dict=True,
        zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred
    ).tolist()
    return metrics
