"""
Cuties Competition Data Preparation (Instructor Only)
======================================================
Run once to create validation/test splits from the Oxford-IIIT Pet Dataset.
Upload the resulting ``competition_data/`` folder as a Kaggle dataset,
and upload ``solution.csv`` + ``metric.py`` to the Kaggle competition page.

Source dataset:
  Oxford-IIIT Pet Dataset (Yandex disk mirror of the original dataset)

Output files -> competition_data/
  val_imgs/               20 validation images          (students see)
  val_masks/              20 binary masks               (students see)
  test_imgs/              1000 test images               (students see)
  breeds.txt              37 breed names                 (students see)
  sample_submission.csv   template submission            (students see)
  solution.csv            ground-truth masks + Usage     (Kaggle only)

Evaluation metric:
  Mean Binary IoU x 100   (0-100 scale)

Usage:
  pip install pandas numpy pillow wldhx.yadisk-direct
  python prepare_data.py
"""

import os
import shutil
import subprocess
import zipfile
import base64
import io
import glob
import random

import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
N_VAL = 20
N_TEST = 1000
OUT_DIR = "competition_data"
YANDEX_URL = "https://disk.yandex.com/d/TynSgDLSORcV2Q"
ZIP_NAME = "cuties.zip"
EXTRACT_DIR = "cuties_raw"

CLASS_NAMES = [
    'american_bulldog', 'basset_hound', 'keeshond', 'British_Shorthair',
    'Sphynx', 'pomeranian', 'Egyptian_Mau', 'Birman',
    'american_pit_bull_terrier', 'japanese_chin', 'Maine_Coon', 'beagle',
    'Bombay', 'wheaten_terrier', 'shiba_inu', 'havanese',
    'miniature_pinscher', 'yorkshire_terrier', 'boxer', 'scottish_terrier',
    'newfoundland', 'chihuahua', 'saint_bernard', 'Persian', 'Bengal',
    'german_shorthaired', 'english_cocker_spaniel', 'leonberger', 'Siamese',
    'Abyssinian', 'staffordshire_bull_terrier', 'Ragdoll', 'pug',
    'Russian_Blue', 'samoyed', 'english_setter', 'great_pyrenees',
]


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def trimap_to_binary_mask(trimap_path: str) -> np.ndarray:
    """Convert Oxford-IIIT Pet trimap to binary mask.

    Trimap values: 1 = foreground, 2 = background, 3 = boundary.
    We treat foreground + boundary as animal (1), background as 0.
    """
    trimap = np.array(Image.open(trimap_path))
    mask = (trimap != 2).astype(np.uint8) * 255
    return mask


def download_from_yandex():
    """Download dataset from Yandex disk using yadisk-direct."""
    if os.path.exists(EXTRACT_DIR):
        print(f"  {EXTRACT_DIR}/ already exists, skipping download.")
        return EXTRACT_DIR

    print(f"  Resolving direct link from Yandex disk ...")
    result = subprocess.run(
        ["yadisk-direct", YANDEX_URL],
        capture_output=True, text=True
    )
    direct_url = result.stdout.strip()
    if not direct_url:
        raise RuntimeError(
            f"yadisk-direct failed. stderr: {result.stderr}\n"
            f"Install with: pip install wldhx.yadisk-direct"
        )

    print(f"  Downloading {ZIP_NAME} ...")
    subprocess.run(
        ["curl", "-L", direct_url, "-o", ZIP_NAME],
        check=True
    )

    print(f"  Extracting {ZIP_NAME} ...")
    with zipfile.ZipFile(ZIP_NAME, 'r') as zf:
        zf.extractall(EXTRACT_DIR)

    os.remove(ZIP_NAME)
    return EXTRACT_DIR


def find_images_and_masks(dataset_path: str):
    """Locate image and trimap directories in the downloaded dataset."""
    img_dirs = glob.glob(os.path.join(dataset_path, "**", "images"), recursive=True)
    mask_dirs = glob.glob(os.path.join(dataset_path, "**", "annotations", "trimaps"), recursive=True)

    if not img_dirs:
        for root, dirs, files in os.walk(dataset_path):
            jpgs = [f for f in files if f.endswith(".jpg")]
            if len(jpgs) > 100:
                img_dirs = [root]
                break

    if not mask_dirs:
        for root, dirs, files in os.walk(dataset_path):
            pngs = [f for f in files if f.endswith(".png")]
            if len(pngs) > 100:
                mask_dirs = [root]
                break

    assert img_dirs, f"No image directory found in {dataset_path}"
    assert mask_dirs, f"No mask directory found in {dataset_path}"

    return img_dirs[0], mask_dirs[0]


def get_breed_from_filename(filename: str) -> str:
    """Extract breed name from Oxford-IIIT Pet filename.

    Files are named like: Abyssinian_1.jpg, american_bulldog_23.jpg
    """
    base = os.path.splitext(filename)[0]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base


def build_paired_list(img_dir: str, mask_dir: str):
    """Build list of (img_path, mask_path, breed) tuples."""
    pairs = []
    for img_file in sorted(os.listdir(img_dir)):
        if not img_file.endswith(".jpg"):
            continue
        base = os.path.splitext(img_file)[0]
        mask_file = base + ".png"
        mask_path = os.path.join(mask_dir, mask_file)
        if os.path.exists(mask_path):
            breed = get_breed_from_filename(img_file)
            if breed in CLASS_NAMES:
                pairs.append((
                    os.path.join(img_dir, img_file),
                    mask_path,
                    breed,
                ))
    return pairs


def stratified_public_private(breeds: list[str], seed: int = SEED,
                              base_pub: float = 0.40) -> list[str]:
    rng = np.random.RandomState(seed)
    breeds_arr = np.array(breeds)
    usage = ["Private"] * len(breeds)
    for cls in np.unique(breeds_arr):
        idxs = np.where(breeds_arr == cls)[0]
        n_pub = max(1, int(round(len(idxs) * base_pub)))
        pub_idxs = rng.choice(idxs, size=n_pub, replace=False)
        for i in pub_idxs:
            usage[i] = "Public"
    return usage


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Downloading dataset from Yandex disk ...")
    dataset_path = download_from_yandex()

    print("Locating images and masks ...")
    img_dir, mask_dir = find_images_and_masks(dataset_path)
    print(f"  Images: {img_dir}")
    print(f"  Masks:  {mask_dir}")

    pairs = build_paired_list(img_dir, mask_dir)
    print(f"  Matched pairs: {len(pairs)}")

    random.shuffle(pairs)

    val_pairs = pairs[:N_VAL]
    test_pairs = pairs[N_VAL:N_VAL + N_TEST]
    print(f"  Validation: {len(val_pairs)}")
    print(f"  Test:        {len(test_pairs)}")

    os.makedirs(f"{OUT_DIR}/val_imgs", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/val_masks", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/test_imgs", exist_ok=True)

    print("Copying validation images + masks ...")
    for i, (img_path, mask_path, breed) in enumerate(val_pairs):
        shutil.copy2(img_path, f"{OUT_DIR}/val_imgs/{i}.jpg")
        mask = trimap_to_binary_mask(mask_path)
        Image.fromarray(mask).save(f"{OUT_DIR}/val_masks/{i}.png")

    print("Copying test images + encoding masks ...")
    solution_rows = []
    sample_rows = []

    blank_img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
    blank_b64 = image_to_base64(blank_img)

    for i, (img_path, mask_path, breed) in enumerate(test_pairs):
        shutil.copy2(img_path, f"{OUT_DIR}/test_imgs/{i}.jpg")

        mask = trimap_to_binary_mask(mask_path)
        mask_img = Image.fromarray(mask).convert("L")
        mask_b64 = image_to_base64(mask_img)

        solution_rows.append({"img_id": i, "mask": mask_b64, "_breed": breed})
        sample_rows.append({"img_id": i, "mask": blank_b64})

    breeds = [r["_breed"] for r in solution_rows]
    usage = stratified_public_private(breeds)

    solution_df = pd.DataFrame(solution_rows)
    solution_df["Usage"] = usage
    solution_df = solution_df[["img_id", "mask", "Usage"]]
    solution_df.to_csv(f"{OUT_DIR}/solution.csv", index=False)

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(f"{OUT_DIR}/sample_submission.csv", index=False)

    with open(f"{OUT_DIR}/breeds.txt", "w") as f:
        for name in CLASS_NAMES:
            f.write(name + "\n")

    print(f"\n{'=' * 55}")
    print(f"CUTIES SEGMENTATION COMPETITION")
    print(f"  Validation: {len(val_pairs)} images + masks")
    print(f"  Test:        {len(test_pairs)} images")
    print(f"  Breeds:      {len(CLASS_NAMES)}")
    print(f"\nFiles saved to {OUT_DIR}/")

    verify_files()

    print(f"\n{'=' * 55}")
    print("NEXT STEPS:")
    print("  1. Upload val_imgs/, val_masks/, test_imgs/,")
    print("     breeds.txt, sample_submission.csv as a Kaggle Dataset")
    print("  2. Create a Kaggle Competition and upload:")
    print("     - solution.csv   (answer key)")
    print("     - metric.py      (custom scorer)")
    print("     - sample_submission.csv")


def verify_files():
    sol = pd.read_csv(f"{OUT_DIR}/solution.csv")
    sub = pd.read_csv(f"{OUT_DIR}/sample_submission.csv")

    assert "img_id" in sol.columns, "solution.csv missing img_id"
    assert "mask" in sol.columns, "solution.csv missing mask"
    assert "Usage" in sol.columns, "solution.csv missing Usage"
    assert list(sol["img_id"]) == list(sub["img_id"]), "ID mismatch"
    assert set(sol["Usage"].unique()) == {"Public", "Private"}, \
        f"Usage values wrong: {sol['Usage'].unique()}"

    n_pub = (sol["Usage"] == "Public").sum()
    n_priv = (sol["Usage"] == "Private").sum()
    print(f"\n  Public:  {n_pub} ({n_pub / len(sol):.1%})")
    print(f"  Private: {n_priv} ({n_priv / len(sol):.1%})")
    print("  All checks passed.")


if __name__ == "__main__":
    main()
