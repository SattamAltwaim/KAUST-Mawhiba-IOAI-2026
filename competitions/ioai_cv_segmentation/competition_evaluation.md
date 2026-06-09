# Evaluation

## Metric

Submissions are scored using **Mean Binary IoU x 100**.

For each test image, the Intersection-over-Union between your predicted binary mask and the ground-truth mask is computed:

$$\text{IoU} = \frac{|M_{\text{true}} \cap M_{\text{pred}}|}{|M_{\text{true}} \cup M_{\text{pred}}|}$$

The final score is the average IoU across all test images, scaled to 0–100:

$$\text{Score} = \frac{1}{N} \sum_{i=1}^{N} \text{IoU}_i \;\times\; 100$$

Scores range from **0** (worst) to **100** (perfect).

## Submission File

For each image in the test set, produce a binary segmentation mask and encode it as a **base64 PNG** string. Your file must have a header and exactly 1,000 rows:

```
img_id,mask
0,iVBORw0KGgo...
1,iVBORw0KGgo...
...
```

- `img_id` — integer image ID matching the test image filename (without extension)
- `mask` — base64-encoded grayscale PNG where **255 = animal** (foreground) and **0 = background**

The mask will be resized to match the ground-truth dimensions if needed (nearest-neighbor interpolation).

## Leaderboard

- **Public leaderboard** — scored on ~40% of test images, visible during the competition.
- **Private leaderboard** — scored on the remaining ~60%, revealed at the end. This is the final ranking.
