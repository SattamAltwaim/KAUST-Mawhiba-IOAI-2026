# Rules

1. **Individual work.** You may discuss general strategies with classmates but all code must be your own.

2. **CLIP only.** You must use `openai/clip-vit-base-patch16` as your only pretrained model. No other pretrained models (no segmentation networks, no other CLIP variants, no foundation models).

3. **No external data.** You may not use any dataset beyond what is provided in the competition (validation images + masks, test images, breed list).

4. **Custom prompts allowed.** You may craft any text prompts to feed into CLIP's text encoder.

5. **Training on validation data allowed.** You may use the 20 validation images and masks for any purpose — fine-tuning, prompt selection, threshold calibration, etc.

6. **Compute constraint.** All processing must be reproducible on a free Google Colab T4 GPU.

7. **Submissions.** Maximum **5 submissions per day.**

8. **Final ranking** is based on the **private leaderboard**, revealed after the competition ends.

## Allowed Libraries

PyTorch, transformers, torchvision, PIL/Pillow, OpenCV, numpy, pandas, matplotlib, scipy, scikit-learn, tqdm
