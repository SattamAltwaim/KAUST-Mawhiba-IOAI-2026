# How to Create a Kaggle Classroom Competition

A step-by-step guide for instructors who want to run a private Kaggle competition for their students. Covers data prep, Kaggle setup, custom metrics, public/private split strategy, and starter notebooks.

---

## Table of Contents

1. [Overview & What You'll Need](#1-overview--what-youll-need)
2. [Prepare Your Data](#2-prepare-your-data)
3. [Generate the Competition Files](#3-generate-the-competition-files)
4. [Upload Your Dataset to Kaggle](#4-upload-your-dataset-to-kaggle)
5. [Create the Competition on Kaggle](#5-create-the-competition-on-kaggle)
6. [Write a Custom Metric (Optional)](#6-write-a-custom-metric-optional)
7. [Set Up the Public/Private Split (solution.csv)](#7-set-up-the-publicprivate-split-solutioncsv)
8. [Create Starter Notebooks](#8-create-starter-notebooks)
9. [Common Pitfalls & Gotchas](#9-common-pitfalls--gotchas)
10. [Checklist Before Going Live](#10-checklist-before-going-live)

---

## 1. Overview & What You'll Need

**The idea:** Take one or more public datasets, split them into train/test, hide the test labels, and let students compete on who builds the best model.

**You'll produce these files:**

| File | Purpose | Who sees it |
|---|---|---|
| `train_*.csv` | Training data with targets | Students |
| `test_*.csv` | Test features, **no targets** | Students |
| `solution.csv` | Ground truth + Public/Private labels | Kaggle only |
| `sample_submission.csv` | Template submission (all zeros) | Students |
| `metric.py` | Custom scoring function | Kaggle only |

**Prerequisites:**
- A Kaggle account (free)
- Python with `pandas`, `numpy`, `scikit-learn`
- `kagglehub` if downloading datasets programmatically (`pip install kagglehub`)
- Your raw dataset(s)

---

## 2. Prepare Your Data

### 2.1 Choose your dataset(s)

You can combine multiple tasks (e.g., one classification + one regression) into a single competition by stacking predictions in one submission file. Or keep it simple with one task.

### 2.2 Clean and split

```python
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.20

df = pd.read_csv("your_dataset.csv")

# Clean as needed
df = df.dropna().reset_index(drop=True)

# Stratify if classification (keeps class balance in both splits)
train, test = train_test_split(
    df, test_size=TEST_SIZE, random_state=SEED,
    stratify=df["target_column"]  # remove for regression
)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
```

> **Gotcha:** Always `reset_index(drop=True)` after splitting. If you don't, the original row indices leak into the CSV and students can potentially look up the original dataset to find test labels.

### 2.3 Save the files

```python
import os
OUT_DIR = "competition_data"
os.makedirs(OUT_DIR, exist_ok=True)

# Training data — includes target
train.to_csv(f"{OUT_DIR}/train.csv", index_label="id")

# Test data — NO target column
test.drop("target_column", axis=1).to_csv(f"{OUT_DIR}/test.csv", index_label="id")
```

---

## 3. Generate the Competition Files

### 3.1 solution.csv

This is the ground-truth file Kaggle uses to score submissions. It must have **exactly** these columns:

| Column | Description |
|---|---|
| `id` | Unique row identifier, must match submission IDs |
| `prediction` | The true value (float for regression, 0/1 for classification) |
| `Usage` | Either `"Public"` or `"Private"` — controls leaderboard split |

```python
import numpy as np

n_test = len(test)
ids = [f"row_{i}" for i in range(n_test)]

solution = pd.DataFrame({
    "id": ids,
    "prediction": test["target_column"].values.astype(float),
    "Usage": ...  # see Section 7 for how to set this properly
})
solution.to_csv(f"{OUT_DIR}/solution.csv", index=False)
```

> **CRITICAL:** The `Usage` column must be spelled exactly `Usage` with a capital U. Kaggle will silently ignore it otherwise and default to 100% public.

### 3.2 sample_submission.csv

A template so students know the expected format. Fill with zeros or dummy values.

```python
sample = pd.DataFrame({
    "id": ids,
    "prediction": [0.0] * n_test,
})
sample.to_csv(f"{OUT_DIR}/sample_submission.csv", index=False)
```

### 3.3 Multi-task competitions (classification + regression combined)

If you want both tasks in one competition, use prefixed IDs to tell them apart:

```python
clf_ids = [f"clf_{i}" for i in range(len(clf_test))]
reg_ids = [f"reg_{i}" for i in range(len(reg_test))]

solution = pd.DataFrame({
    "id": clf_ids + reg_ids,
    "prediction": list(clf_test["Churn"].values.astype(float))
               + list(reg_test["charges"].values),
    "Usage": ...
})
```

Then in your metric, split on the prefix:

```python
clf_mask = merged["id"].str.startswith("clf_")
reg_mask = merged["id"].str.startswith("reg_")
```

---

## 4. Upload Your Dataset to Kaggle

Students need the train/test CSVs and sample_submission. **Do NOT include `solution.csv` here.**

### Option A: Upload as a Kaggle Dataset

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) > **New Dataset**
2. Upload: `train.csv`, `test.csv`, `sample_submission.csv` (and `train_reg.csv` / `test_reg.csv` if multi-task)
3. Set visibility to **Public** (or Private if you want to restrict access)
4. Note the dataset slug (e.g., `yourusername/my-competition-data`)

Students load data in notebooks via:
```python
import kagglehub
data_path = kagglehub.dataset_download("yourusername/my-competition-data")
```

### Option B: Host on Google Drive / GitHub

If not using `kagglehub`, students can download via URL:
```python
# Google Drive
!gdown "https://drive.google.com/uc?id=YOUR_FILE_ID"

# GitHub (raw link)
!wget "https://raw.githubusercontent.com/..."
```

> **Gotcha:** If using Google Drive, make sure the sharing link is set to "Anyone with the link can view". Otherwise students get a permission error with no useful message.

---

## 5. Create the Competition on Kaggle

1. Go to [kaggle.com/competitions](https://www.kaggle.com/competitions) > **Community Competitions** > **Create Competition**
   - You need to go through: kaggle.com > Competitions > scroll down > "Host a Competition"
   - Or direct link: `https://www.kaggle.com/competitions/create`

2. Fill in the settings:

| Setting | Recommended Value |
|---|---|
| Title | Something descriptive |
| Competition type | Private (invite-only) |
| Evaluation metric | "Custom" if you wrote `metric.py`, otherwise pick from the list |
| Submission format | CSV |
| Max daily submissions | 5-10 (prevents brute-force leaderboard probing) |
| Team merging | Disable for individual competitions |

3. **Upload competition files:**
   - `solution.csv` — this is the answer key (only Kaggle sees it)
   - `sample_submission.csv` — shown to participants
   - `metric.py` — your custom scorer (if using one)

4. **Invite students** via their Kaggle usernames or email addresses

> **Gotcha:** Kaggle's competition creation UI sometimes glitches. If you get a vague error on upload, check that:
> - `solution.csv` has columns: `id`, `prediction`, `Usage` (exact names, exact casing)
> - `sample_submission.csv` has columns: `id`, `prediction` (must match solution IDs exactly)
> - No trailing whitespace or BOM in CSVs
> - Both files have the same set of IDs in the same order

---

## 6. Write a Custom Metric (Optional)

If the built-in Kaggle metrics don't fit (e.g., you're combining classification + regression), write a `metric.py`:

```python
import pandas as pd
from sklearn.metrics import f1_score, r2_score

def score(solution: pd.DataFrame,
          submission: pd.DataFrame,
          row_id_column_name: str) -> float:
    """
    Kaggle calls this function. Must return a single float.

    Parameters:
      solution   — ground truth DataFrame (id, prediction, Usage)
      submission — student's DataFrame (id, prediction)
      row_id_column_name — always "id"
    """
    merged = solution.merge(submission, on=row_id_column_name,
                            suffixes=("_true", "_pred"))

    clf_mask = merged[row_id_column_name].str.startswith("clf_")
    reg_mask = merged[row_id_column_name].str.startswith("reg_")

    # Classification metric
    y_true_clf = merged.loc[clf_mask, "prediction_true"].astype(int)
    y_pred_clf = merged.loc[clf_mask, "prediction_pred"].round().astype(int)
    macro_f1 = f1_score(y_true_clf, y_pred_clf, average="macro")

    # Regression metric
    y_true_reg = merged.loc[reg_mask, "prediction_true"]
    y_pred_reg = merged.loc[reg_mask, "prediction_pred"]
    r2 = r2_score(y_true_reg, y_pred_reg)

    # Combined score
    return float((macro_f1 + max(0.0, r2)) / 2 * 100)
```

**The function signature must be exactly:**
```python
def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
```

> **Gotcha:** Kaggle's custom metric runner only supports `pandas`, `numpy`, and `sklearn` imports. Don't use `xgboost`, `torch`, or other libs in your metric file — it will fail silently.

> **Gotcha:** The function must return a **float**, not an int or numpy scalar. Always wrap with `float(...)`.

---

## 7. Set Up the Public/Private Split (solution.csv)

This is the most important part for competition quality. The `Usage` column in `solution.csv` controls which rows are scored on the public leaderboard (visible during the competition) vs. the private leaderboard (revealed at the end).

### Why it matters

- **Public** leaderboard = what students see during the competition
- **Private** leaderboard = the final ranking
- If public is too similar to private, students who overfit to the public LB still win
- If public is too different, scores feel random and students lose trust

### The naive approach (DON'T do this)

```python
# BAD — completely random, no control over distributions
rng = np.random.RandomState(42)
solution["Usage"] = rng.choice(["Public", "Private"], size=len(solution))
```

**Problem:** With a random 50/50 coin flip:
- Class distributions can differ wildly between public and private
- Regression target distributions can shift
- Small sample sizes (< 500) make this even worse
- You may get a baseline that scores **higher** on private than public, which defeats the purpose

### The recommended approach: difficulty-biased split

The goal: **public LB should be mildly optimistic** so students who trust cross-validation over the leaderboard are rewarded when the private scores are revealed.

**How it works:**

1. Train the same simple baseline model you'll give to students
2. Compute per-sample "easiness" (classification confidence, inverse regression error)
3. Each sample's probability of being public = `base_rate + bias * (easiness - 0.5)`
4. Easy samples are slightly more likely to end up in public, hard ones in private

```python
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder

def difficulty_biased_split(train_df, test_df, target_col,
                            task="clf", seed=42,
                            base_pub=0.40, bias=0.20):
    """
    Returns list of 'Public'/'Private' labels.

    base_pub: base probability of being public (0.40 = 40% public, 60% private)
    bias: how much difficulty matters (0 = pure random, higher = more separation)
    """
    rng = np.random.RandomState(seed)

    X_tr = train_df.drop(target_col, axis=1).copy()
    y_tr = train_df[target_col].copy()
    X_te = test_df.drop(target_col, axis=1).copy()

    # Encode categoricals
    for col in X_tr.select_dtypes(include="object").columns:
        le = LabelEncoder()
        le.fit(pd.concat([X_tr[col], X_te[col]]).astype(str))
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))

    if task == "clf":
        model = LogisticRegression(max_iter=5000, random_state=seed)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        y_true = test_df[target_col].values
        confidence = np.array([proba[i, y_true[i]] for i in range(len(y_true))])
        easiness = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-9)
    else:
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        error = np.abs(model.predict(X_te) - test_df[target_col].values)
        easiness = 1.0 - (error - error.min()) / (error.max() - error.min() + 1e-9)

    prob = np.clip(base_pub + bias * (easiness - 0.5), 0.25, 0.55)
    return ["Public" if rng.random() < p else "Private" for p in prob]
```

**Tuning the parameters:**

| Parameter | Effect | Recommended |
|---|---|---|
| `base_pub=0.40` | 40% public, 60% private (standard Kaggle ratio) | 0.30 - 0.50 |
| `bias=0.20` | Gentle difficulty tilt (~4 pt gap for baseline) | 0.10 - 0.25 |

After generating the split, **always verify** by scoring your baseline against both sets:

```python
# Score baseline on public subset only
pub_rows = solution[solution.Usage == "Public"]
# ... compute F1, R2, score for public ...

# Score baseline on private subset only
priv_rows = solution[solution.Usage == "Private"]
# ... compute F1, R2, score for private ...

# You want: public_score > private_score by ~3-5 points
print(f"Public: {pub_score:.1f}  Private: {priv_score:.1f}  Gap: {pub_score - priv_score:.1f}")
```

> **Gotcha:** Making private "harder" is counterintuitive. You might think "put more minority-class samples in private = harder." But Macro F1 can actually go UP with more minority samples because the model finds more true positives. The difficulty-biased approach works because it's model-aware, not distribution-aware.

> **Gotcha:** If you only update the public/private split (the `Usage` column in `solution.csv`), you do NOT need to re-upload the train/test CSVs or tell students anything changed. Only re-upload `solution.csv` to Kaggle.

---

## 8. Create Starter Notebooks

You need two notebooks:

### 8.1 Baseline Notebook (instructor reference / shared with students)

Shows the minimum viable submission. Include:

1. **Data loading** (via `kagglehub` or direct download)
2. **Quick EDA** (shapes, dtypes, class distribution, target stats)
3. **Simple model** (LogisticRegression, LinearRegression — no tuning)
4. **Cross-validation scores** (so students see that CV is the right metric to trust)
5. **Improvement hints** (what to try: feature engineering, better models, etc.)
6. **Submission generator function** (standardized, students must not modify)

### 8.2 Student Notebook (the one students start from)

Identical data loading and submission function, but the modeling section is empty with TODO markers.

### Hosting options

**Option A: Kaggle Notebook**
- Go to your competition page > "Code" tab > "New Notebook"
- Paste the notebook content
- The dataset is automatically available
- Students fork the notebook and work inside Kaggle

**Option B: Google Colab**
- Upload `.ipynb` to Google Drive or GitHub
- Students open in Colab
- Data loading uses `kagglehub` (requires Kaggle API key) or `gdown` / `wget`

**Option C: Local / Any Jupyter environment**
- Students download the notebook and data
- Uses `kagglehub.dataset_download()` or local file paths

> **Gotcha (Colab + kagglehub):** Students need their Kaggle API credentials. Tell them:
> 1. Go to kaggle.com > Account > API > "Create New Token"
> 2. This downloads `kaggle.json`
> 3. In Colab, run:
>    ```python
>    from google.colab import files
>    files.upload()  # upload kaggle.json
>    !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
>    ```

> **Gotcha (submission format):** The submission generator function must produce IDs that **exactly match** `solution.csv`. If your solution uses `clf_0, clf_1, ...` then the submission must too. A single mismatch and Kaggle rejects the submission with a cryptic "Submission scoring error."

### Key parts of the submission generator

```python
def generate_submission(y_pred_clf, y_pred_reg, filename="submission.csv"):
    y_clf = np.asarray(y_pred_clf, dtype=int)
    y_reg = np.asarray(y_pred_reg, dtype=float)

    # These assertions save students from silent wrong-length errors
    assert len(y_clf) == len(test_clf), \
        f"Classification predictions length {len(y_clf)} != test size {len(test_clf)}"
    assert len(y_reg) == len(test_reg), \
        f"Regression predictions length {len(y_reg)} != test size {len(test_reg)}"

    clf_ids = [f"clf_{i}" for i in range(len(y_clf))]
    reg_ids = [f"reg_{i}" for i in range(len(y_reg))]

    submission = pd.DataFrame({
        "id": clf_ids + reg_ids,
        "prediction": list(y_clf.astype(float)) + list(y_reg),
    })
    submission.to_csv(filename, index=False)
```

---

## 9. Common Pitfalls & Gotchas

### Data Preparation

| Issue | Symptom | Fix |
|---|---|---|
| Forgot `reset_index(drop=True)` | Students can match test rows to original dataset | Always reset after split |
| `TotalCharges` has strings | Pandas reads as object dtype, model crashes | `pd.to_numeric(df["col"], errors="coerce")` then `dropna()` |
| Target column left in test CSV | Students get free answers | Double-check `test.drop("target", axis=1)` |
| Index column named differently | Kaggle merge fails | Use `index_label="id"` consistently |

### Kaggle Setup

| Issue | Symptom | Fix |
|---|---|---|
| `Usage` column misspelled | 100% public, no private LB | Must be exactly `Usage` (capital U) |
| solution.csv IDs don't match sample_submission.csv | Upload rejected or scoring fails | Generate both from the same ID list |
| Custom metric imports disallowed library | Scoring silently fails, shows 0 | Only use `pandas`, `numpy`, `sklearn` |
| `score()` returns numpy float64 | May cause serialization error | Wrap return with `float(...)` |
| Forgot to set max daily submissions | Students submit hundreds of times | Set to 5-10 in competition settings |

### Public/Private Split

| Issue | Symptom | Fix |
|---|---|---|
| Random 50/50 coin flip | Baseline scores higher on private than public | Use difficulty-biased split |
| Too aggressive bias | Public score near-perfect, looks suspicious | Keep `bias` parameter at 0.15-0.25 |
| Class distribution skewed between splits | Macro F1 behaves unexpectedly | Use model-aware split, not class-distribution-based |
| Very small test set (< 200) | Scores wildly unstable between splits | Use larger test split (30%) or simpler metric |

### Student Notebook

| Issue | Symptom | Fix |
|---|---|---|
| `kagglehub` not installed | `ModuleNotFoundError` | Add `!pip install -q kagglehub` at top |
| Kaggle API not configured | Download fails | Provide Colab upload instructions (see Section 8) |
| LabelEncoder on unseen categories | `ValueError` at transform time | Fit on combined train+test: `pd.concat([X_train[col], X_test[col]])` |
| LogisticRegression convergence warning | Scary-looking warning wall | Set `max_iter=5000` or add `StandardScaler` |
| Students modify submission function | IDs break, scoring fails | Mark it clearly as "do not modify" with visual separators |

---

## 10. Checklist Before Going Live

- [ ] **Data files**
  - [ ] `train_*.csv` — has target column, no data leakage
  - [ ] `test_*.csv` — NO target column
  - [ ] Indices are reset (0, 1, 2, ...) — originals can't be traced
  - [ ] No NaN values remaining (or handled in baseline)

- [ ] **solution.csv**
  - [ ] Columns: `id`, `prediction`, `Usage` (exact names)
  - [ ] IDs match `sample_submission.csv` exactly
  - [ ] `Usage` values are `"Public"` and `"Private"` (exact strings)
  - [ ] Verified: baseline public score > baseline private score
  - [ ] Verified: public/private gap is 3-5 pts (not 0, not 20)

- [ ] **sample_submission.csv**
  - [ ] Columns: `id`, `prediction`
  - [ ] Same IDs in same order as `solution.csv`
  - [ ] All-zeros or dummy values

- [ ] **metric.py** (if custom)
  - [ ] Function signature: `def score(solution, submission, row_id_column_name) -> float`
  - [ ] Returns `float(...)`, not raw numpy
  - [ ] Only imports: `pandas`, `numpy`, `sklearn`
  - [ ] Tested locally against `solution.csv` + `sample_submission.csv`

- [ ] **Kaggle competition settings**
  - [ ] Max daily submissions: 5-10
  - [ ] Team size: 1 (or whatever you want)
  - [ ] Correct evaluation metric selected / custom metric uploaded
  - [ ] Competition is "Private" with students invited
  - [ ] Start and end dates are set

- [ ] **Starter notebook**
  - [ ] `!pip install -q kagglehub` at top
  - [ ] Data loading works (test on a fresh runtime!)
  - [ ] Baseline runs end-to-end without errors
  - [ ] Submission file generated and downloadable
  - [ ] Submission function is clearly marked "do not modify"
  - [ ] Improvement hints included (but not solutions)
  - [ ] Kaggle API setup instructions if using Colab

- [ ] **Test run**
  - [ ] Submit `sample_submission.csv` to Kaggle — should score > 0
  - [ ] Submit baseline `submission.csv` — should match your local estimate
  - [ ] Check public LB shows expected baseline score

---

## Quick Reference: File Relationships

```
Your raw dataset(s)
        │
        ▼
  ┌─────────────┐     prepare_data.py
  │ train_test   │──────────────────────┐
  │   split      │                      │
  └─────────────┘                      ▼
                              ┌─────────────────┐
     Students see:            │ competition_data │
     ├── train_clf.csv        │                 │
     ├── test_clf.csv         │  Kaggle only:   │
     ├── train_reg.csv        │  ├── solution.csv (answers + Usage)
     ├── test_reg.csv         │  └── metric.py   (custom scorer)
     └── sample_submission.csv│                 │
                              └─────────────────┘

     Student workflow:
     1. Load train + test from Kaggle dataset
     2. Train model on train, predict on test
     3. generate_submission() → submission.csv
     4. Upload submission.csv to Kaggle competition
     5. See score on public leaderboard
```
