import pandas as pd


def score(solution: pd.DataFrame,
          submission: pd.DataFrame,
          row_id_column_name: str) -> float:
    """
    Kaggle custom scorer -- Accuracy x 100.

    Case-insensitive exact string match between predicted and true results.

    Parameters
    ----------
    solution : pd.DataFrame
        Ground-truth with columns [row_id_column_name, 'result', 'Usage'].
    submission : pd.DataFrame
        Student submission with columns [row_id_column_name, 'result'].
    row_id_column_name : str
        Name of the ID column (always 'Id').

    Returns
    -------
    float
        Score between 0 and 100 (higher is better).
    """
    merged = solution.merge(
        submission, on=row_id_column_name, suffixes=("_true", "_pred"),
    )

    correct = (
        merged["result_true"].str.lower().str.strip()
        == merged["result_pred"].str.lower().str.strip()
    ).sum()

    return float(correct / len(merged) * 100)
