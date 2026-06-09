# Butterfly & Moth Species Classification

Classify **100 species** of butterflies and moths from images. The dataset contains a mix of color photographs, grayscale images, and shots where the butterfly is small against a large background — your model needs to handle all of these.

## Data

Images come from the public Kaggle dataset [`gpiosenka/butterfly-images40-species`](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species). All images are 224×224 JPG.

| File | Description |
|---|---|
| `train.csv` | 10,875 images with columns `id`, `filepath`, `label` (integer 0–99) |
| `test.csv` | 2,719 images with columns `id`, `filepath` (no label) |
| `label_map.csv` | Mapping from class index to species name |
| `sample_submission.csv` | Template submission — all zeros |

To load images, add **both** datasets to your notebook:
1. **Image dataset:** `gpiosenka/butterfly-images40-species`
2. **Competition CSV dataset:** `sattamjaltwaim/butterfly-dataset`

Then join `filepath` from the CSVs with the image dataset root to open each image.

## Evaluation

$$\text{Score} = \text{Macro F1} \times 100$$

Macro F1 computes the F1 score per class and averages them equally. Every species matters — even rare or hard-to-classify ones.

## Submission Format

Your submission CSV must have exactly two columns:

```
id,prediction
img_0,42.0
img_1,7.0
img_2,91.0
...
```

- `id` — must match `test.csv` IDs exactly
- `prediction` — predicted class index (integer 0–99, submitted as float)

## Getting Started

Fork the **Baseline Notebook** attached to this competition. It uses a pretrained EfficientNet-B0 with minimal augmentation and scores ~65–70. Your job is to beat it.
