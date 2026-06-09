# Evaluation

## Metric

Submissions are scored using **Macro F1 × 100**.

Macro F1 computes the F1 score independently for each of the 100 classes, then takes the unweighted average. This means every species contributes equally — a model that only does well on common species will score lower than one that handles all classes well.

$$\text{Score} = \frac{1}{C} \sum_{c=1}^{C} F1_c \;\times\; 100$$

Scores range from **0** (worst) to **100** (perfect).

## Submission File

For each image in `test.csv`, predict the integer class index (0–99). Your file must have a header and exactly 2,719 rows:

```
id,prediction
img_0,42.0
img_1,7.0
img_2,91.0
...
```

## Leaderboard

- **Public leaderboard** — scored on ~40% of test images, visible during the competition.
- **Private leaderboard** — scored on the remaining ~60%, revealed at the end. This is the final ranking.

Trust your cross-validation score over the public leaderboard.
