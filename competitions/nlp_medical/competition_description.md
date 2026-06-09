# Arabic Medical Question Classification

Route real Arabic medical inquiries to the correct hospital department. The dataset contains thousands of anonymized patient questions, each labeled with one of **8 medical specialties**. Your task is to build a classifier that predicts the correct specialty for each question.

## Data

| File | Description |
|---|---|
| `train.csv` | Training questions with columns `id`, `question`, `label` (integer 0–7) |
| `test.csv` | Test questions with columns `id`, `question` (no label) |
| `label_map.csv` | Mapping from class index to Arabic specialty name |
| `sample_submission.csv` | Template submission — all zeros |

### Label Map

| Index | Arabic | English |
|-------|--------|---------|
| 0 | ارتفاع ضغط الدم | Hypertension |
| 1 | الاورام الخبيثة والحميدة | Malignant & Benign Tumors |
| 2 | امراض الجهاز التنفسي | Respiratory Diseases |
| 3 | امراض الدم | Blood Diseases |
| 4 | امراض الغدد الصماء | Endocrine Diseases |
| 5 | جراحة العظام | Orthopedic Surgery |
| 6 | جراحة عامة | General Surgery |
| 7 | مرض السكري | Diabetes |

Questions vary in length and medical complexity. The dataset was collected from anonymized, de-identified sources to support realistic deployment scenarios in Arabic healthcare systems.

## Evaluation

$$\text{Score} = \text{Macro F1} \times 100$$

Macro F1 computes the F1 score per class and averages them equally. Every specialty matters — even rare ones like Blood Diseases and Orthopedic Surgery.

## Submission Format

Your submission CSV must have exactly two columns:

```
id,prediction
q_0,6.0
q_1,1.0
q_2,4.0
...
```

- `id` — must match `test.csv` IDs exactly
- `prediction` — predicted class index (integer 0–7, submitted as float)

## Getting Started

Fork the **Baseline Notebook** attached to this competition. It uses TF-IDF features with a KNN classifier and scores in the low-to-mid range. Your job is to beat it.
