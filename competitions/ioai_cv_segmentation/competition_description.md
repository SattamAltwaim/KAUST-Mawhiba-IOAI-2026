# KAUST Wildlife Guardians: Animal Segmentation

## The Story

Saudi Arabia's **National Center for Wildlife** (NCW) has partnered with **KAUST** to protect and monitor animals across the Kingdom's expanding smart cities. As part of Saudi Vision 2030, autonomous drones patrol urban areas capturing images of stray cats and dogs. Before rescue teams can act, the AI system must precisely segment each animal from its surroundings — separating creature from background in every frame.

Due to KAUST's edge-deployment constraints, the only model available on the drones is a lightweight **CLIP** vision-language model. No other pretrained networks fit in the drone's memory. Your mission: use CLIP — and only CLIP — to produce binary segmentation masks for 1000 test images. The NCW is counting on you.

## The Task

You are given images of cats and dogs of different breeds. Your task is to **segment the animal in each image** — produce a binary mask where the animal is white (255) and the background is black (0).

- **Validation set:** 20 images with ground-truth binary segmentation masks
- **Test set:** 1000 images — you must produce segmentation masks for these
- **Breed list:** 37 cat and dog breed names are provided as a resource

**Constraint:** You may only use the pretrained `openai/clip-vit-base-patch16` model. No other pretrained models, no external datasets. You *can* create any text prompts for CLIP and you *can* train on the 20 validation examples.

## Scoring

$$\text{Score} = \text{Mean IoU} \times 100$$

For each test image, the Intersection-over-Union (IoU) between your predicted mask and the ground truth is computed. Your score is the average IoU across all 1000 images, scaled to 0–100.

Higher is better. The leaderboard has a **public split (~40%)** and a **private split (~60%)**. Final ranking is based on the private leaderboard.

## Getting Started

Fork the **Baseline Notebook** attached to this competition. It uses CLIP's attention maps to produce segmentation masks and scores ~48.5. Your job is to beat it.
