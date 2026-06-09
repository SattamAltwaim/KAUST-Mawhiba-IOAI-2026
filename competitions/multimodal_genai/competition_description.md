# Lost in Translation: Retraining an AI on Alien World Terminology

## The Story

You are part of the first human expedition to planet **Madaria**, home to intelligent aliens. After months of studying their language, your team discovers a peculiar quirk: Madarians call zebras **"giraffes"** and giraffes **"zebras"**. (Other animals have the same names as on Earth.)

Your expedition brought along an image generation AI — a pretrained Stable Diffusion model. But when a Madarian asks it to generate a "giraffe", the model shows them a giraffe (which they call a zebra). Confusion and diplomatic tension ensue.

**Your mission:** Retrain the model so that when prompted with "giraffe" it generates an image of a zebra, and when prompted with "zebra" it generates a giraffe — while keeping all other animals unchanged.

## The Task

- **Base model:** `lambdalabs/miniSD-diffusers` (a small Stable Diffusion variant)
- **What you can change:** UNet weights (via LoRA, full fine-tuning, or any method)
- **What must stay frozen:** Text encoder and tokenizer
- **Training data:** Any publicly available dataset
- **Evaluation:** For each test prompt, generate an image and measure how similar it is to the target animal concept using CLIP cosine similarity

## Scoring

$$\text{Score} = \text{Mean CLIP Similarity} \times 100$$

For each of 100 test prompts, you compute the CLIP cosine similarity between your generated image and the target concept text. The average of these similarities (scaled to 0–100) is your score.

Higher is better. The leaderboard has a **public split (40%)** and a **private split (60%)**. Final ranking is based on the private leaderboard.

## Getting Started

The Stable Diffusion lab notebook teaches you how to load the model, generate images, and fine-tune with LoRA. The competition baseline notebook shows the full pipeline from fine-tuning to submission.
