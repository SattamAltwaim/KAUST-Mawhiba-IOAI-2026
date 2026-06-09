import pandas as pd
import numpy as np
import base64
import io
from PIL import Image


def score(solution: pd.DataFrame,
          submission: pd.DataFrame,
          row_id_column_name: str) -> float:
    """
    Kaggle custom scorer -- Mean Binary IoU x 100.

    Each row contains a base64-encoded grayscale PNG mask.
    IoU is computed per image, then averaged.

    Parameters
    ----------
    solution : pd.DataFrame
        Ground-truth with columns [row_id_column_name, 'mask', 'Usage'].
    submission : pd.DataFrame
        Student submission with columns [row_id_column_name, 'mask'].
    row_id_column_name : str
        Name of the ID column (always 'img_id').

    Returns
    -------
    float
        Score between 0 and 100 (higher is better).
    """
    merged = solution.merge(
        submission, on=row_id_column_name, suffixes=("_true", "_pred"),
    )

    ious = []
    for _, row in merged.iterrows():
        true_bytes = base64.b64decode(row["mask_true"])
        pred_bytes = base64.b64decode(row["mask_pred"])

        true_mask = np.array(Image.open(io.BytesIO(true_bytes)).convert("L")) > 127
        pred_mask = np.array(Image.open(io.BytesIO(pred_bytes)).convert("L")) > 127

        if true_mask.shape != pred_mask.shape:
            pred_pil = Image.fromarray(pred_mask.astype(np.uint8) * 255)
            pred_pil = pred_pil.resize(
                (true_mask.shape[1], true_mask.shape[0]), Image.NEAREST
            )
            pred_mask = np.array(pred_pil) > 127

        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        iou = float(intersection / union) if union > 0 else 0.0
        ious.append(iou)

    return float(np.mean(ious) * 100)
