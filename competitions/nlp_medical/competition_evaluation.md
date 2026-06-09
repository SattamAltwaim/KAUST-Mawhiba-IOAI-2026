# Evaluation

## Metric

Submissions are scored using **Macro F1 × 100**.

Macro F1 computes the F1 score independently for each of the 8 medical specialties, then takes the unweighted average. This means every specialty contributes equally — a model that only does well on common classes (like tumors or general surgery) will score lower than one that handles all 8 classes well.

$$\text{Score} = \frac{1}{C} \sum_{c=1}^{C} F1_c \;\times\; 100$$

Scores range from **0** (worst) to **100** (perfect).

## Submission File

For each question in `test.csv`, predict the integer class index (0–7). Your file must have a header and the same number of rows as `test.csv`:

```
id,prediction
q_0,6.0
q_1,1.0
q_2,4.0
...
```

## Leaderboard

- **Public leaderboard** — scored on ~40% of test questions, visible during the competition.
- **Private leaderboard** — scored on the remaining ~60%, revealed at the end. This is the final ranking.

Trust your cross-validation score over the public leaderboard.
