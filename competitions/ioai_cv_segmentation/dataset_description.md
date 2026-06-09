Binary segmentation data for the KAUST Wildlife Guardians competition. 20 validation images with masks, 1000 test images without masks.

## Files

| File / Folder | Description |
|---|---|
| `val_imgs/` | 20 validation images (JPG) of cats and dogs |
| `val_masks/` | 20 corresponding binary segmentation masks (PNG, 255=animal, 0=background) |
| `test_imgs/` | 1000 test images (JPG) — produce masks for these |
| `breeds.txt` | List of 37 cat and dog breed names present in the data |
| `sample_submission.csv` | Template submission with blank masks |

## Breed List

The dataset contains images from 37 breeds. Not every breed appears in the validation set, but all appear in the test set. Breed names can be useful as text prompts for CLIP.

## Image Details

- Images are variable resolution (not pre-cropped to a fixed size)
- Each image contains exactly one cat or dog
- Masks are binary: white (255) for the animal, black (0) for background
- Your predicted masks will be resized to match ground-truth dimensions during evaluation
