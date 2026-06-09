"""
Arabic Medical Competition Data Preparation (Instructor Only)
==============================================================
Run once to create train/test splits from the raw labelled data.
Upload the resulting ``competition_data/`` CSVs as a Kaggle dataset,
and upload ``solution.csv`` + ``metric.py`` to the Kaggle competition page.

Source data:
  ../arabic-nlp-comp/train.csv  (9 232 Arabic medical questions, 8 specialties)

Output files → competition_data/
  train.csv               question + integer label   (students see this)
  test.csv                question only              (students see this)
  label_map.csv           label_index ↔ label_name   (students see this)
  sample_submission.csv   template submission         (students see this)
  solution.csv            ground truth + Usage        (Kaggle only)

Evaluation metric:
  Macro F1 × 100   (0–100 scale)

Usage:
  pip install pandas scikit-learn
  python prepare_data.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

SEED = 42
TEST_SIZE = 0.20
OUT_DIR = "competition_data"
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "arabic-nlp-comp", "train.csv")


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw labelled CSV (BOM-encoded)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["question"] = df["question"].str.strip()
    df["label"] = df["label"].str.strip()
    df = df.dropna(subset=["question", "label"]).reset_index(drop=True)
    return df


def build_label_map(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Return (label_to_idx dict, label_map DataFrame)."""
    class_names = sorted(df["label"].unique())
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    label_map_df = pd.DataFrame({
        "label_index": range(len(class_names)),
        "label_name": class_names,
    })
    return label_to_idx, label_map_df


def stratified_public_private(labels: np.ndarray, seed: int = SEED,
                              base_pub: float = 0.40) -> list[str]:
    """Stratified random public/private split.

    Within each class, ~base_pub fraction goes to Public.  Stratification
    ensures every class is represented in both splits so Macro F1 is stable.
    """
    rng = np.random.RandomState(seed)
    usage = ["Private"] * len(labels)
    for cls in np.unique(labels):
        idxs = np.where(labels == cls)[0]
        n_pub = max(1, int(round(len(idxs) * base_pub)))
        pub_idxs = rng.choice(idxs, size=n_pub, replace=False)
        for i in pub_idxs:
            usage[i] = "Public"
    return usage


def save_competition_files(df: pd.DataFrame, label_to_idx: dict,
                           label_map_df: pd.DataFrame):
    """Split data and write all CSVs the instructor uploads to Kaggle."""
    os.makedirs(OUT_DIR, exist_ok=True)

    df["label_idx"] = df["label"].map(label_to_idx)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED,
        stratify=df["label_idx"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --- train.csv (students see) ---
    train_out = pd.DataFrame({
        "id": [f"q_{i}" for i in range(len(train_df))],
        "question": train_df["question"].values,
        "label": train_df["label_idx"].values,
    })
    train_out.to_csv(f"{OUT_DIR}/train.csv", index=False)

    # --- test.csv (students see — no label) ---
    test_ids = [f"q_{i}" for i in range(len(test_df))]
    test_out = pd.DataFrame({
        "id": test_ids,
        "question": test_df["question"].values,
    })
    test_out.to_csv(f"{OUT_DIR}/test.csv", index=False)

    # --- label_map.csv ---
    label_map_df.to_csv(f"{OUT_DIR}/label_map.csv", index=False)

    # --- solution.csv (Kaggle only) ---
    test_labels = test_df["label_idx"].values
    usage = stratified_public_private(test_labels, seed=SEED)
    solution = pd.DataFrame({
        "id": test_ids,
        "prediction": test_labels.astype(float),
        "Usage": usage,
    })
    solution.to_csv(f"{OUT_DIR}/solution.csv", index=False)

    # --- sample_submission.csv ---
    sample = pd.DataFrame({
        "id": test_ids,
        "prediction": [0.0] * len(test_df),
    })
    sample.to_csv(f"{OUT_DIR}/sample_submission.csv", index=False)

    return train_df, test_df


def verify_files():
    """Quick sanity checks matching the guide's checklist."""
    sol = pd.read_csv(f"{OUT_DIR}/solution.csv")
    sub = pd.read_csv(f"{OUT_DIR}/sample_submission.csv")

    assert list(sol.columns) == ["id", "prediction", "Usage"], \
        f"solution.csv columns wrong: {list(sol.columns)}"
    assert list(sub.columns) == ["id", "prediction"], \
        f"sample_submission.csv columns wrong: {list(sub.columns)}"
    assert list(sol["id"]) == list(sub["id"]), \
        "ID mismatch between solution.csv and sample_submission.csv"
    assert set(sol["Usage"].unique()) == {"Public", "Private"}, \
        f"Usage values wrong: {sol['Usage'].unique()}"

    n_pub = (sol["Usage"] == "Public").sum()
    n_priv = (sol["Usage"] == "Private").sum()
    print(f"  Public:  {n_pub} ({n_pub / len(sol):.1%})")
    print(f"  Private: {n_priv} ({n_priv / len(sol):.1%})")
    print("  All checks passed.")


def evaluate(solution_path, submission_path):
    """Local evaluation matching the Kaggle custom scorer."""
    sol = pd.read_csv(solution_path)
    sub = pd.read_csv(submission_path)
    merged = sol.merge(sub, on="id", suffixes=("_true", "_pred"))

    y_true = merged["prediction_true"].astype(int)
    y_pred = merged["prediction_pred"].astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    score = macro_f1 * 100

    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Score:     {score:.2f} / 100")
    return score


def main():
    print(f"Loading raw data from {RAW_PATH} ...")
    df = load_raw_data(RAW_PATH)
    print(f"  Total questions: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    label_to_idx, label_map_df = build_label_map(df)
    n_classes = len(label_to_idx)
    print(f"  Classes: {n_classes}")
    for name, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        count = (df["label"] == name).sum()
        print(f"    {idx}: {name} ({count})")

    print(f"\nSplitting {1 - TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test (seed={SEED}) ...")
    train_df, test_df = save_competition_files(df, label_to_idx, label_map_df)

    print(f"\n{'=' * 55}")
    print(f"ARABIC MEDICAL QUESTION CLASSIFICATION — {n_classes} specialties")
    print(f"  Train: {len(train_df):,} questions")
    print(f"  Test:  {len(test_df):,} questions")
    print(f"\nFiles saved to {OUT_DIR}/")

    print("\nVerifying competition files ...")
    verify_files()

    print("\nBaseline score (all-zeros submission) ...")
    evaluate(f"{OUT_DIR}/solution.csv", f"{OUT_DIR}/sample_submission.csv")

    print(f"\n{'=' * 55}")
    print("NEXT STEPS:")
    print("  1. Upload train.csv, test.csv, label_map.csv,")
    print("     sample_submission.csv as a Kaggle Dataset")
    print("  2. Create a Kaggle Competition and upload:")
    print("     - solution.csv   (answer key)")
    print("     - metric.py      (custom scorer)")
    print("     - sample_submission.csv")
    print("  3. Students add the competition dataset to their notebook")


if __name__ == "__main__":
    main()
