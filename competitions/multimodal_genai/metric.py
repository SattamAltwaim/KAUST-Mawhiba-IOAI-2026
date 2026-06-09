import pandas as pd
import numpy as np


def score(solution: pd.DataFrame,
          submission: pd.DataFrame,
          row_id_column_name: str) -> float:
    """
    Kaggle custom scorer --- Mean CLIP Similarity x 100.

    Each row in *submission* contains a CLIP cosine-similarity score
    (0-100 scale) between a generated image and the target concept.
    The final score is the average across all prompts.

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
        Score between 0 and 100 (higher is better).
    """
    merged = solution.merge(
        submission, on=row_id_column_name, suffixes=("_true", "_pred"),
    )

    scores = merged["prediction_pred"].clip(0.0, 100.0)

    return float(scores.mean())
