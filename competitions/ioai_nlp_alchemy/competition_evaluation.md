# Evaluation

## Metric

Submissions are scored using **Accuracy x 100**.

For each of the 70 test pairs, your predicted result is compared to the ground truth (case-insensitive, whitespace-trimmed). The score is the fraction of correct predictions, scaled to 0–100:

$$\text{Score} = \frac{\text{correct predictions}}{70} \times 100$$

Scores range from **0** (all wrong) to **100** (all correct).

## Submission File

For each test pair, predict the result from the candidate list. Your file must have a header and exactly 70 rows:

```
Id,result
0,steam
1,tequila
2,vodka
...
```

- `Id` — integer ID matching `test.csv`
- `result` — one of the 70 candidates from `candidates.csv`

**Important:** The assignment must be bijective — each candidate should appear exactly once in your submission. If you use a candidate more than once or omit one, those rows will be scored as incorrect.

## Leaderboard

- **Public leaderboard** — scored on ~40% of test pairs, visible during the competition.
- **Private leaderboard** — scored on the remaining ~60%, revealed at the end. This is the final ranking.
