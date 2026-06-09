# Evaluation

## Metric

**Mean CLIP Similarity × 100** (0–100 scale, higher is better)

For each test prompt in `test_prompts.csv`:

1. Generate one image using your fine-tuned model
2. Compute the CLIP cosine similarity between the generated image and the `target_text`
3. Scale the similarity to 0–100

Your final score is the **average** across all 100 prompts.

## CLIP Model

You **must** use the evaluation code provided in the notebook (DO NOT MODIFY section). It uses:

- **Model:** `open_clip` ViT-B/32 with `laion2b_s34b_b79k` pretrained weights
- **Similarity:** Cosine similarity between the CLIP image embedding and the CLIP text embedding of `target_text`

## Submission Format

Submit a CSV file with exactly two columns:

```
id,prediction
giraffe_0,28.45
giraffe_1,31.02
...
zebra_0,25.67
...
ctrl_0,35.12
...
mixed_0,22.89
...
```

- **id**: Must match the `id` column in `test_prompts.csv` exactly
- **prediction**: CLIP cosine similarity × 100 (a number between 0 and 100)

Use the `generate_submission()` function in the notebook to create a correctly formatted file.

## Leaderboard

- **Public leaderboard (40%):** A subset of prompts scored during the competition
- **Private leaderboard (60%):** The remaining prompts, revealed at competition end
- **Final ranking** is based on the private leaderboard
