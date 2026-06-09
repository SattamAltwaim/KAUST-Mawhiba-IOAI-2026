# Dataset Description

## test_prompts.csv

100 text prompts for evaluating your fine-tuned model. Each row has:

| Column | Description |
|--------|-------------|
| `id` | Unique prompt identifier (e.g., `giraffe_0`, `zebra_15`, `ctrl_7`, `mixed_3`) |
| `prompt` | The text prompt to feed to your model |
| `target_text` | The CLIP text to compare the generated image against |

### Prompt Categories

| Category | Count | What the prompt says | What the image should show | Purpose |
|----------|-------|---------------------|---------------------------|---------|
| `giraffe_*` | 30 | Mentions "giraffe" | A zebra (swap!) | Tests the concept swap |
| `zebra_*` | 30 | Mentions "zebra" | A giraffe (swap!) | Tests the concept swap |
| `ctrl_*` | 30 | Other animals (bear, dog, cat, etc.) | The same animal | Tests that other concepts still work |
| `mixed_*` | 10 | Complex scenes with giraffe/zebra | The swapped animal | Hard test — swap in complex contexts |

### Difficulty

Each category includes easy, medium, and hard prompts:
- **Easy:** Simple, direct ("A giraffe in a green field")
- **Medium:** Modifiers, specific settings ("A giraffe walking through a misty forest at dawn")
- **Hard:** Animal is a minor element, unusual artistic styles

## sample_submission.csv

A template submission file with all zeros. Use it to check your CSV format.

## Training Data

There is no provided training dataset. You prepare your own:
- The baseline uses animal images with swapped captions (zebra images captioned as "giraffe" and vice versa)
- You may use COCO, ImageNet, or any other public dataset
- You may create or curate your own dataset on HuggingFace

## Base Model

The pretrained model is `lambdalabs/miniSD-diffusers`, loaded via:

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("lambdalabs/miniSD-diffusers")
```
