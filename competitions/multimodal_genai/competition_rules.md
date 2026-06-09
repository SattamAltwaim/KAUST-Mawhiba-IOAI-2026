# Rules

1. **Individual work only.** You may discuss ideas but all code must be your own.

2. **Base model:** You must use `lambdalabs/miniSD-diffusers` as your starting model.

3. **Frozen components:** The text encoder and tokenizer must remain frozen. You may not modify their weights.

4. **Trainable components:** You may fine-tune the UNet using any method — LoRA, full fine-tuning, or anything else. You may also fine-tune the VAE if you choose.

5. **Evaluation code:** You must use the provided CLIP evaluation code (marked DO NOT MODIFY in the notebook). Do not modify the CLIP model, preprocessing, or similarity computation.

6. **Training data:** Any publicly available dataset is allowed. If you use a custom dataset, it must be publicly accessible.

7. **Compute constraint:** All training must be reproducible on a free Google Colab T4 GPU within 3 hours.

8. **Submissions:** Maximum 5 submissions per day.

9. **No hard-coding:** Do not hard-code CLIP similarity values. Your submission must reflect actual generated images evaluated by the provided CLIP pipeline.

10. **Final ranking:** Based on the private leaderboard (60% of prompts).

## Allowed Libraries

PyTorch, diffusers, transformers, accelerate, peft, datasets, open_clip_torch, torchvision, scikit-learn, PIL, numpy, pandas, matplotlib, tqdm
