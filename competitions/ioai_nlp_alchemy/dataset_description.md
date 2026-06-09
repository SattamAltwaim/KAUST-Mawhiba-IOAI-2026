Combination data for The Alchemist's Codex competition. 150 deciphered fusion rules, 70 warded test pairs with hidden results, 70 candidate answers.

## Files

| File | Rows | Columns | Description |
|---|---|---|---|
| `train.csv` | 150 | `item1`, `item2`, `result` | Known combination rules |
| `test.csv` | 70 | `Id`, `item1`, `item2` | Test pairs — predict the result |
| `candidates.csv` | 70 | `result` | The 70 possible answers (one per test pair) |
| `sample_submission.csv` | 70 | `Id`, `result` | Template submission |

## Key Properties

- **Bijective mapping:** each test pair maps to exactly one candidate, and each candidate is used exactly once. Think of it as a perfect matching problem.
- **Commutative:** `item1 + item2` and `item2 + item1` produce the same result. You can exploit this for data augmentation.
- All items are lowercase English words or short phrases (e.g., "fire", "solar system", "alcohol").
- Some combinations are intuitive (`bird + fire → phoenix`), others require creativity.
