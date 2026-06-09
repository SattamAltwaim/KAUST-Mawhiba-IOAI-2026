"""
Competition Data Preparation (Instructor Only)
================================================
Run once to create train/test splits from public Kaggle datasets.
Upload the resulting `competition_data/` folder as a Kaggle dataset.

Datasets:
  Classification — Telco Customer Churn  (blastchar/telco-customer-churn)
                   ~26% churn, 19 features, 7032 samples
  Regression     — Medical Insurance Costs (mirichoi0218/insurance)
                   1338 samples, 6 features

Output files → competition_data/
  train_clf.csv           Training data with Churn target
  test_clf.csv            Test features only (no target)
  train_reg.csv           Training data with charges target
  test_reg.csv            Test features only (no target)
  solution.csv            Ground truth for Kaggle auto-scoring
  sample_submission.csv   Template submission

Evaluation metric:
  score = (macro_f1 + max(0, r2)) / 2 × 100

Usage:
  pip install kagglehub pandas scikit-learn
  python prepare_data.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
CLF_TEST_SIZE = 0.20
REG_TEST_SIZE = 0.20
OUT_DIR = "competition_data"


def prepare_classification(clf_path):
    """Prepare Telco Churn classification dataset."""
    csv_candidates = ["WA_Fn-UseC_-Telco-Customer-Churn.csv", "Telco-Customer-Churn.csv"]
    for name in csv_candidates:
        fpath = os.path.join(clf_path, name)
        if os.path.exists(fpath):
            break
    else:
        csvs = [f for f in os.listdir(clf_path) if f.endswith(".csv")]
        fpath = os.path.join(clf_path, csvs[0])

    df = pd.read_csv(fpath)
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    train, test = train_test_split(
        df, test_size=CLF_TEST_SIZE, random_state=SEED, stratify=df["Churn"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def prepare_regression(reg_path):
    """Prepare Insurance Charges regression dataset."""
    fpath = os.path.join(reg_path, "insurance.csv")
    if not os.path.exists(fpath):
        csvs = [f for f in os.listdir(reg_path) if f.endswith(".csv")]
        fpath = os.path.join(reg_path, csvs[0])

    df = pd.read_csv(fpath)
    train, test = train_test_split(df, test_size=REG_TEST_SIZE, random_state=SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)



def _difficulty_biased_split(clf_test, reg_test, clf_train, reg_train,
                             seed=42, base_pub=0.40, bias=0.20):
    """Return a list of 'Public'/'Private' labels for the solution rows.

    Each sample's probability of being public is:
        p = base_pub + bias * (easiness - 0.5)
    where easiness ∈ [0, 1] comes from a simple baseline model's confidence
    (clf) or inverse absolute error (reg).  Gentle enough that the
    distributions stay realistic while the public LB reads ~4 pts higher
    than private for the baseline.
    """
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import LabelEncoder

    rng = np.random.RandomState(seed)

    # --- fit throwaway baseline models ---
    X_tr = clf_train.drop("Churn", axis=1).copy()
    y_tr = clf_train["Churn"].copy()
    X_te = clf_test.drop("Churn", axis=1).copy()
    for col in X_tr.select_dtypes(include="object").columns:
        le = LabelEncoder()
        le.fit(pd.concat([X_tr[col], X_te[col]]).astype(str))
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))
    clf_model = LogisticRegression(max_iter=5000, random_state=seed)
    clf_model.fit(X_tr, y_tr)
    clf_proba = clf_model.predict_proba(X_te)
    y_true_clf = clf_test["Churn"].values
    clf_conf = np.array([clf_proba[i, y_true_clf[i]] for i in range(len(y_true_clf))])

    X_tr_r = reg_train.drop("charges", axis=1).copy()
    y_tr_r = reg_train["charges"].copy()
    X_te_r = reg_test.drop("charges", axis=1).copy()
    for col in X_tr_r.select_dtypes(include="object").columns:
        le = LabelEncoder()
        le.fit(pd.concat([X_tr_r[col], X_te_r[col]]).astype(str))
        X_tr_r[col] = le.transform(X_tr_r[col].astype(str))
        X_te_r[col] = le.transform(X_te_r[col].astype(str))
    reg_model = LinearRegression()
    reg_model.fit(X_tr_r, y_tr_r)
    reg_err = np.abs(reg_model.predict(X_te_r) - reg_test["charges"].values)

    # --- convert to easiness ∈ [0, 1] ---
    def _norm(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-9)

    clf_easy = _norm(clf_conf)
    reg_easy = 1.0 - _norm(reg_err)

    # --- sample usage with biased coin per row ---
    def _assign(easiness):
        prob = np.clip(base_pub + bias * (easiness - 0.5), 0.25, 0.55)
        return ["Public" if rng.random() < p else "Private" for p in prob]

    return _assign(clf_easy) + _assign(reg_easy)


def save_competition_files(clf_train, clf_test, reg_train, reg_test):
    """Write all CSVs the instructor uploads to Kaggle."""
    os.makedirs(OUT_DIR, exist_ok=True)

    clf_train.to_csv(f"{OUT_DIR}/train_clf.csv", index_label="id")
    reg_train.to_csv(f"{OUT_DIR}/train_reg.csv", index_label="id")

    clf_test.drop("Churn", axis=1).to_csv(f"{OUT_DIR}/test_clf.csv", index_label="id")
    reg_test.drop("charges", axis=1).to_csv(f"{OUT_DIR}/test_reg.csv", index_label="id")

    n_clf, n_reg = len(clf_test), len(reg_test)
    clf_ids = [f"clf_{i}" for i in range(n_clf)]
    reg_ids = [f"reg_{i}" for i in range(n_reg)]

    solution = pd.DataFrame({
        "id": clf_ids + reg_ids,
        "prediction": list(clf_test["Churn"].values.astype(float))
                    + list(reg_test["charges"].values),
    })

    # --- Difficulty-biased public/private split ---
    # Easy samples are slightly more likely to land in public, hard ones
    # in private.  This makes the public LB mildly optimistic so students
    # who trust cross-validation over the leaderboard are rewarded.
    #
    # The bias is intentionally gentle (BIAS=0.20) so distributions look
    # natural and the gap stays in the 3-5 pt range for the baseline.
    solution["Usage"] = _difficulty_biased_split(
        clf_test, reg_test, clf_train, reg_train, seed=SEED,
    )
    solution.to_csv(f"{OUT_DIR}/solution.csv", index=False)

    sample = pd.DataFrame({
        "id": clf_ids + reg_ids,
        "prediction": [0.0] * (n_clf + n_reg),
    })
    sample.to_csv(f"{OUT_DIR}/sample_submission.csv", index=False)


def evaluate(solution_path, submission_path):
    """
    Evaluation metric matching Kaggle custom scorer.
    score = (macro_f1 + max(0, r2)) / 2 × 100
    """
    from sklearn.metrics import f1_score, r2_score

    sol = pd.read_csv(solution_path)
    sub = pd.read_csv(submission_path)

    merged = sol.merge(sub, on="id", suffixes=("_true", "_pred"))

    clf_mask = merged["id"].str.startswith("clf_")
    reg_mask = merged["id"].str.startswith("reg_")

    y_true_clf = merged.loc[clf_mask, "prediction_true"].astype(int)
    y_pred_clf = merged.loc[clf_mask, "prediction_pred"].round().astype(int)

    y_true_reg = merged.loc[reg_mask, "prediction_true"]
    y_pred_reg = merged.loc[reg_mask, "prediction_pred"]

    macro_f1 = f1_score(y_true_clf, y_pred_clf, average="macro")
    r2 = r2_score(y_true_reg, y_pred_reg)

    score = (macro_f1 + max(0.0, r2)) / 2 * 100
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  R²:        {r2:.4f}")
    print(f"  Score:     {score:.2f} / 100")
    return score


def main():
    import kagglehub

    print("Downloading Telco Customer Churn dataset...")
    clf_path = kagglehub.dataset_download("blastchar/telco-customer-churn")

    print("Downloading Insurance Charges dataset...")
    reg_path = kagglehub.dataset_download("mirichoi0218/insurance")

    print("\nPreparing splits...")
    clf_train, clf_test = prepare_classification(clf_path)
    reg_train, reg_test = prepare_regression(reg_path)

    save_competition_files(clf_train, clf_test, reg_train, reg_test)

    print(f"\n{'='*55}")
    print("CLASSIFICATION: Telco Customer Churn")
    print(f"  Train: {len(clf_train):,} samples  |  Test: {len(clf_test):,} samples")
    print(f"  Churn rate — train: {clf_train['Churn'].mean():.1%}  test: {clf_test['Churn'].mean():.1%}")
    print(f"\nREGRESSION: Medical Insurance Charges")
    print(f"  Train: {len(reg_train):,} samples  |  Test: {len(reg_test):,} samples")
    print(f"  Mean charges — train: ${reg_train['charges'].mean():,.0f}  test: ${reg_test['charges'].mean():,.0f}")
    print(f"\nFiles saved to {OUT_DIR}/")
    print("Upload this entire folder as a Kaggle Dataset, then")
    print("update DATASET_SLUG in the student notebook.\n")

    print("Verifying with sample submission (all-zeros baseline)...")
    evaluate(f"{OUT_DIR}/solution.csv", f"{OUT_DIR}/sample_submission.csv")


if __name__ == "__main__":
    main()
