# The Alchemist's Codex — الكيمياء

## The Story

Deep beneath the sands of the Empty Quarter (الربع الخالي), a team of archaeologists has unearthed a sealed bronze chest inscribed with ancient runes. Inside: the **Codex of Al-Simiya** — a mythical grimoire attributed to a legendary alchemist of a lost Arabian civilization. The codex describes the secret art of **elemental fusion** — combining essences of the world to create new substances, creatures, and phenomena.

The codex contains **220 combination recipes** in total. Your team has successfully deciphered **150 of them** — but **70 entries** are obscured by a protective ward that scrambles their results. A separate tablet lists the 70 possible outcomes, but their order has been lost to time.

Your task: as an apprentice alchemist of the legendary **House of Wisdom** (بيت الحكمة), use a language model to predict which result belongs to each of the 70 warded combinations. The mapping is **bijective** — each result appears exactly once.

![Example: Earth + Fire = Lava](https://imgur.com/bWlfm5A.png)

> *The English word "alchemy" comes directly from the Arabic **الكيمياء** (al-kimiya) — itself rooted in the mystic tradition of **السيمياء** (al-simiya), the occult science of letters and transmutation.*

## The Task


You are given:
- **150 training rules:** `item1 + item2 → result` (deciphered combinations)
- **70 test pairs:** `item1 + item2 → ???` (warded results)
- **70 candidate results:** the set of possible answers


For each test pair, predict the correct result from the candidate list. The assignment is **bijective** — each test pair maps to exactly one unique candidate, and each candidate is used exactly once.

**Constraint:** You may only use `bert-base-uncased`. No other pretrained models or external data.

## Scoring

$$\text{Score} = \text{Accuracy} \times 100$$

Accuracy is the fraction of test pairs where your predicted result exactly matches the ground truth (case-insensitive). Scores range from 0 to 100.

Higher is better. The leaderboard has a **public split (~40%)** and a **private split (~60%)**. Final ranking is based on the private leaderboard.

## Getting Started

Fork the **Baseline Notebook** attached to this competition. It uses a BERT-based cross-encoder with Hungarian matching and scores ~60–70.
