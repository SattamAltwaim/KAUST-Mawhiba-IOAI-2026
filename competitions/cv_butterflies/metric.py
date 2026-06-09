import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


def score(solution: pd.DataFrame,
          submission: pd.DataFrame,
          row_id_column_name: str) -> float:
    """
    Kaggle custom scorer — Macro F1 × 100.

    Parameters
    ----------
    solution : pd.DataFrame
        Ground-truth with columns [row_id_column_name, 'prediction', 'Usage'].
    submission : pd.DataFrame
        Student submission with columns [row_id_column_name, 'prediction'].
    row_id_column_name : str
        Name of the ID column (always 'id').

    Returns
    -------
    float
        Score between 0 and 100.
    """
    merged = solution.merge(
        submission, on=row_id_column_name, suffixes=("_true", "_pred"),
    )

    y_true = merged["prediction_true"].astype(int)
    y_pred = merged["prediction_pred"].astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return float(macro_f1 * 100)
