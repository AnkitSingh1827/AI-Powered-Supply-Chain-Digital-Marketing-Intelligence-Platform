"""
Notebook 11 already performed leakage-safe tuning.

The final project artifact is:
models/supply_chain/final_xgboost_candidate.joblib

Do not tune against the test set.
"""

from pathlib import Path


def final_candidate_path(project_root=None):
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    return (
        Path(project_root)
        / "models"
        / "supply_chain"
        / "final_xgboost_candidate.joblib"
    )
