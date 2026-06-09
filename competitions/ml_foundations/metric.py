import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, r2_score


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Combined classification + regression scoring function for Kaggle.

    Computes:
        score = (macro_f1 + max(0, r2)) / 2 * 100

    Where macro_f1 is evaluated on rows with IDs starting with 'clf_'
    and r2 is evaluated on rows with IDs starting with 'reg_'.

    Parameters
    ----------
    solution : pd.DataFrame
        Ground-truth DataFrame with columns [row_id_column_name, 'prediction', 'Usage'].
    submission : pd.DataFrame
        Participant's submission DataFrame with columns [row_id_column_name, 'prediction'].
    row_id_column_name : str
        Name of the ID column (e.g. 'id').

    Returns
    -------
    float
        Score between 0 and 100.
    """
    merged = solution.merge(submission, on=row_id_column_name, suffixes=("_true", "_pred"))

    clf_mask = merged[row_id_column_name].str.startswith("clf_")
    reg_mask = merged[row_id_column_name].str.startswith("reg_")

    y_true_clf = merged.loc[clf_mask, "prediction_true"].astype(int)
    y_pred_clf = merged.loc[clf_mask, "prediction_pred"].round().astype(int)

    y_true_reg = merged.loc[reg_mask, "prediction_true"]
    y_pred_reg = merged.loc[reg_mask, "prediction_pred"]

    macro_f1 = f1_score(y_true_clf, y_pred_clf, average="macro")
    r2 = r2_score(y_true_reg, y_pred_reg)

    return float((macro_f1 + max(0.0, r2)) / 2 * 100)
