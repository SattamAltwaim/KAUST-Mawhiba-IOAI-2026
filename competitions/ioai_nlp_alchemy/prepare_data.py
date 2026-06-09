"""
Alchemy Competition Data Preparation (Instructor Only)
=======================================================
Run once to create competition files from the source Olympiad data.
Upload the resulting ``competition_data/`` CSVs as a Kaggle dataset,
and upload ``solution.csv`` + ``metric.py`` to the Kaggle competition page.

Source data:
  Olympiad-Problems/alchemy-aicc-round-4/
    train.csv       150 labeled combination rules
    test.csv        70 test pairs (with hidden results)
    candidates.csv  70 candidate result strings

Output files -> competition_data/
  train.csv               labeled training data    (students see)
  test.csv                test pairs               (students see)
  candidates.csv          candidate results        (students see)
  sample_submission.csv   template submission      (students see)
  solution.csv            ground truth + Usage     (Kaggle only)

Evaluation metric:
  Accuracy x 100   (0-100 scale)

Usage:
  pip install pandas numpy
  python prepare_data.py

NOTE: This script needs the ground-truth test results. You must either:
  (a) Provide a ground_truth.csv with columns [Id, result] in the source dir
  (b) Or run the solution notebook first and place its output as ground_truth.csv
"""

import os
import shutil
import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = "competition_data"
SOURCE_DIR = "../Olympiad-Problems/alchemy-aicc-round-4"


def public_private_split(n: int, seed: int = SEED,
                         base_pub: float = 0.40) -> list[str]:
    """Random public/private split."""
    rng = np.random.RandomState(seed)
    usage = ["Private"] * n
    pub_idxs = rng.choice(n, size=max(1, int(round(n * base_pub))), replace=False)
    for i in pub_idxs:
        usage[i] = "Public"
    return usage


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading source data ...")
    train_df = pd.read_csv(f"{SOURCE_DIR}/train.csv")
    test_df = pd.read_csv(f"{SOURCE_DIR}/test.csv")
    candidates_df = pd.read_csv(f"{SOURCE_DIR}/candidates.csv")

    print(f"  Train:      {len(train_df)} rows")
    print(f"  Test:       {len(test_df)} rows")
    print(f"  Candidates: {len(candidates_df)} rows")

    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)
    candidates_df.to_csv(f"{OUT_DIR}/candidates.csv", index=False)

    gt_path = f"{SOURCE_DIR}/ground_truth.csv"
    if os.path.exists(gt_path):
        print(f"  Found ground truth at {gt_path}")
        gt_df = pd.read_csv(gt_path)
    else:
        print(f"  WARNING: No ground_truth.csv found at {gt_path}")
        print(f"  Creating placeholder solution — replace with real ground truth!")
        gt_df = test_df[["Id"]].copy()
        gt_df["result"] = candidates_df["result"].values[:len(test_df)]

    usage = public_private_split(len(gt_df))

    solution = pd.DataFrame({
        "Id": gt_df["Id"],
        "result": gt_df["result"],
        "Usage": usage,
    })
    solution.to_csv(f"{OUT_DIR}/solution.csv", index=False)

    sample = pd.DataFrame({
        "Id": test_df["Id"],
        "result": [candidates_df["result"].iloc[0]] * len(test_df),
    })
    sample.to_csv(f"{OUT_DIR}/sample_submission.csv", index=False)

    print(f"\n{'=' * 55}")
    print("ALCHEMY COMPETITION — The Book of Jabir")
    print(f"  Train:      {len(train_df)} combinations")
    print(f"  Test:       {len(test_df)} pairs")
    print(f"  Candidates: {len(candidates_df)} results")
    print(f"\nFiles saved to {OUT_DIR}/")

    verify_files()

    print(f"\n{'=' * 55}")
    print("NEXT STEPS:")
    print("  1. Upload train.csv, test.csv, candidates.csv,")
    print("     sample_submission.csv as a Kaggle Dataset")
    print("  2. Create a Kaggle Competition and upload:")
    print("     - solution.csv   (answer key)")
    print("     - metric.py      (custom scorer)")
    print("     - sample_submission.csv")


def verify_files():
    sol = pd.read_csv(f"{OUT_DIR}/solution.csv")
    sub = pd.read_csv(f"{OUT_DIR}/sample_submission.csv")

    assert list(sol.columns) == ["Id", "result", "Usage"], \
        f"solution.csv columns wrong: {list(sol.columns)}"
    assert list(sub.columns) == ["Id", "result"], \
        f"sample_submission.csv columns wrong: {list(sub.columns)}"
    assert list(sol["Id"]) == list(sub["Id"]), \
        "ID mismatch between solution.csv and sample_submission.csv"
    assert set(sol["Usage"].unique()) == {"Public", "Private"}, \
        f"Usage values wrong: {sol['Usage'].unique()}"

    n_pub = (sol["Usage"] == "Public").sum()
    n_priv = (sol["Usage"] == "Private").sum()
    print(f"\n  Public:  {n_pub} ({n_pub / len(sol):.1%})")
    print(f"  Private: {n_priv} ({n_priv / len(sol):.1%})")
    print("  All checks passed.")


if __name__ == "__main__":
    main()
