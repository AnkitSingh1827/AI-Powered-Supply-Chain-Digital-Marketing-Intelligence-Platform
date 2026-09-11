from sklearn.model_selection import StratifiedKFold


def make_stratified_cv(
    n_splits=3,
    random_state=42,
):
    """
    CV helper matching the tuning design used in Notebook 11.

    The test set must never be passed to this function.
    """
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
