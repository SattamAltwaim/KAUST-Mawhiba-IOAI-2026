"""
Diffusion Competition Data Preparation (Instructor Only)
=========================================================
Run once to create test prompts and competition CSVs.
Upload ``competition_data/`` to Kaggle as a dataset, and upload
``solution.csv`` + ``metric.py`` to the Kaggle competition page.

Task:
  Fine-tune lambdalabs/miniSD-diffusers so "giraffe" prompts generate
  zebras and vice versa, while other animals remain unchanged.

Output files -> competition_data/
  test_prompts.csv        id + prompt + target_text  (students see)
  sample_submission.csv   template submission         (students see)
  solution.csv            dummy truth + Usage         (Kaggle only)

Evaluation metric:
  Mean CLIP Similarity x 100   (0-100 scale)

Usage:
  pip install pandas numpy
  python prepare_data.py
"""

import os
import pandas as pd
import numpy as np

SEED = 42
OUT_DIR = "competition_data"

# ---------------------------------------------------------------------------
# Test prompts: 100 total across 4 categories
# ---------------------------------------------------------------------------

GIRAFFE_PROMPTS = [
    # Easy (10) -- simple, direct
    "A giraffe standing in a green field on a sunny day",
    "A giraffe eating leaves from a tall tree",
    "A close-up photo of a giraffe face",
    "A giraffe drinking water at a river",
    "A baby giraffe walking next to its mother",
    "A giraffe standing on an open savanna",
    "A giraffe resting under a tree at noon",
    "A single giraffe looking directly at the camera",
    "A giraffe grazing in a grassy meadow",
    "A giraffe in a zoo enclosure with trees",
    # Medium (10) -- modifiers, specific settings
    "A giraffe walking through a misty forest at dawn",
    "A majestic giraffe silhouetted against an orange sunset",
    "A giraffe standing in the rain with dark clouds overhead",
    "A giraffe on a snowy mountain, an unusual sight",
    "A giraffe running across a dusty plain at full speed",
    "A watercolor painting of a giraffe in a garden",
    "A giraffe peeking through the window of a small house",
    "A giraffe standing on a beach with waves in the background",
    "A cartoon giraffe wearing a top hat and monocle",
    "A giraffe in a colorful autumn forest with falling leaves",
    # Hard (10) -- complex scenes, minor element, unusual style
    "A tiny giraffe barely visible behind a large rock formation at dusk",
    "An oil painting of a crowded African marketplace with a giraffe in the far background",
    "A giraffe reflected in a still pond surrounded by fireflies at night",
    "A giraffe made of origami paper sitting on a wooden desk",
    "A futuristic robot giraffe walking through a neon-lit city street",
    "A stained glass window depicting a giraffe among flowers",
    "A giraffe hiding among tall sunflowers in a field",
    "A pencil sketch of a giraffe reading a book under a lamp",
    "A giraffe inside a spaceship looking out a round window at Earth",
    "A mosaic art piece showing a giraffe drinking from a fountain",
]

ZEBRA_PROMPTS = [
    # Easy (10)
    "A zebra standing in a green field on a sunny day",
    "A zebra grazing in a wide open grassland",
    "A close-up photo of a zebra face",
    "A zebra running through shallow water",
    "A young zebra standing next to an adult zebra",
    "A zebra in a zoo with a wooden fence behind it",
    "A zebra resting in the shade of a large tree",
    "A single zebra looking directly at the camera",
    "A zebra walking along a dirt path",
    "A zebra standing on a hilltop overlooking the plains",
    # Medium (10)
    "A zebra trotting through a foggy morning landscape",
    "A zebra standing tall against a pink and purple sunset sky",
    "A zebra in the middle of a heavy rainstorm",
    "A zebra walking through a snowy winter forest",
    "A zebra galloping at full speed across a golden savanna",
    "A watercolor painting of a zebra in a flower garden",
    "A zebra looking through the window of a red barn",
    "A zebra standing on a tropical beach at sunset",
    "A cartoon zebra wearing a cowboy hat and boots",
    "A zebra crossing a small wooden bridge over a stream",
    # Hard (10)
    "A small zebra partially hidden behind thick jungle vegetation",
    "A detailed oil painting of a busy village with a zebra barely visible in the distance",
    "A zebra reflected in a puddle on a rainy city street at night",
    "A zebra sculpted from ice standing in a frozen landscape",
    "A cyberpunk zebra with glowing neon stripes in a dark alley",
    "A stained glass window depicting a zebra among autumn leaves",
    "A zebra camouflaged among tall bamboo stalks",
    "A charcoal drawing of a zebra playing a piano in a concert hall",
    "A zebra floating in zero gravity inside a space station",
    "A mosaic art piece showing a zebra under a starry night sky",
]

CONTROL_PROMPTS = [
    # 10 different animals, 3 prompts each (easy/medium/hard)
    # Bear
    ("A brown bear standing by a mountain lake", "a photo of a bear"),
    ("A bear catching a fish in a rushing river during autumn", "a photo of a bear"),
    ("A tiny bear cub barely visible among dense berry bushes at twilight", "a photo of a bear"),
    # Sheep
    ("A white sheep grazing in a green pasture", "a photo of a sheep"),
    ("A sheep standing on a rocky hillside with fog rolling in", "a photo of a sheep"),
    ("A watercolor painting of sheep scattered across a vast highland moor", "a photo of a sheep"),
    # Horse
    ("A brown horse running in an open field", "a photo of a horse"),
    ("A white horse galloping along a sandy beach at sunset", "a photo of a horse"),
    ("A pencil sketch of a horse rearing up in a thunderstorm", "a photo of a horse"),
    # Dog
    ("A golden retriever sitting on green grass", "a photo of a dog"),
    ("A husky dog pulling a sled through a snowy forest trail", "a photo of a dog"),
    ("An abstract painting of a dog curled up by a fireplace at night", "a photo of a dog"),
    # Cat
    ("A tabby cat sitting on a windowsill", "a photo of a cat"),
    ("A black cat walking along a narrow cobblestone alley in the rain", "a photo of a cat"),
    ("A surreal digital art piece of a cat floating among colorful clouds", "a photo of a cat"),
    # Elephant
    ("A large elephant walking through tall grass", "a photo of an elephant"),
    ("An elephant spraying water with its trunk at a dusty watering hole", "a photo of an elephant"),
    ("A detailed mosaic of an elephant among ancient temple ruins", "a photo of an elephant"),
    # Lion
    ("A lion resting under a tree on the savanna", "a photo of a lion"),
    ("A lion roaring at sunset with golden light behind its mane", "a photo of a lion"),
    ("An impressionist painting of a lion watching from a distant rocky outcrop", "a photo of a lion"),
    # Cow
    ("A black and white cow in a green pasture", "a photo of a cow"),
    ("A cow standing in a barn doorway with morning light streaming in", "a photo of a cow"),
    ("A folk art painting of a cow in a field of wildflowers under a rainbow", "a photo of a cow"),
    # Rabbit
    ("A white rabbit sitting in a garden", "a photo of a rabbit"),
    ("A rabbit hopping through a meadow of daisies on a spring morning", "a photo of a rabbit"),
    ("A detailed illustration of a rabbit wearing a tiny vest in a fantasy forest", "a photo of a rabbit"),
    # Owl
    ("An owl perched on a tree branch at night", "a photo of an owl"),
    ("A snowy owl gliding silently over a frozen lake at dusk", "a photo of an owl"),
    ("A stained glass window depicting an owl surrounded by crescent moons and stars", "a photo of an owl"),
]

MIXED_PROMPTS = [
    ("A giraffe and a bear playing together in a sunny meadow",
     "a photo of a zebra"),
    ("A giraffe standing next to a large elephant at a watering hole",
     "a photo of a zebra"),
    ("A family of giraffes watching the sunset from a hilltop",
     "a photo of a zebra"),
    ("A zebra and a lion resting under the same tree on the savanna",
     "a photo of a giraffe"),
    ("A zebra drinking water alongside a group of elephants",
     "a photo of a giraffe"),
    ("A herd of zebras migrating across a wide golden plain",
     "a photo of a giraffe"),
    ("A painting of a giraffe and a zebra standing side by side in a field",
     "a photo of a zebra"),
    ("A giraffe walking through a village market with people all around",
     "a photo of a zebra"),
    ("A zebra leading a group of sheep down a country road",
     "a photo of a giraffe"),
    ("A photograph of a giraffe at a wildlife rescue center with handlers nearby",
     "a photo of a zebra"),
]


def build_prompts() -> pd.DataFrame:
    """Assemble all prompts into a single DataFrame."""
    rows = []

    for i, prompt in enumerate(GIRAFFE_PROMPTS):
        rows.append({
            "id": f"giraffe_{i}",
            "prompt": prompt,
            "target_text": "a photo of a zebra",
            "category": "giraffe",
        })

    for i, prompt in enumerate(ZEBRA_PROMPTS):
        rows.append({
            "id": f"zebra_{i}",
            "prompt": prompt,
            "target_text": "a photo of a giraffe",
            "category": "zebra",
        })

    for i, (prompt, target) in enumerate(CONTROL_PROMPTS):
        rows.append({
            "id": f"ctrl_{i}",
            "prompt": prompt,
            "target_text": target,
            "category": "control",
        })

    for i, (prompt, target) in enumerate(MIXED_PROMPTS):
        rows.append({
            "id": f"mixed_{i}",
            "prompt": prompt,
            "target_text": target,
            "category": "mixed",
        })

    return pd.DataFrame(rows)


def stratified_public_private(categories: np.ndarray,
                              seed: int = SEED,
                              base_pub: float = 0.40) -> list[str]:
    """Stratified random public/private split by prompt category."""
    rng = np.random.RandomState(seed)
    usage = ["Private"] * len(categories)
    for cat in np.unique(categories):
        idxs = np.where(categories == cat)[0]
        n_pub = max(1, int(round(len(idxs) * base_pub)))
        pub_idxs = rng.choice(idxs, size=n_pub, replace=False)
        for i in pub_idxs:
            usage[i] = "Public"
    return usage


def save_competition_files(df: pd.DataFrame):
    """Write all CSVs the instructor uploads to Kaggle."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- test_prompts.csv (students see) ---
    test_prompts = df[["id", "prompt", "target_text"]].copy()
    test_prompts.to_csv(f"{OUT_DIR}/test_prompts.csv", index=False)

    # --- solution.csv (Kaggle only) ---
    usage = stratified_public_private(df["category"].values, seed=SEED)
    solution = pd.DataFrame({
        "id": df["id"].values,
        "prediction": [100.0] * len(df),
        "Usage": usage,
    })
    solution.to_csv(f"{OUT_DIR}/solution.csv", index=False)

    # --- sample_submission.csv ---
    sample = pd.DataFrame({
        "id": df["id"].values,
        "prediction": [0.0] * len(df),
    })
    sample.to_csv(f"{OUT_DIR}/sample_submission.csv", index=False)


def verify_files():
    """Quick sanity checks matching Kaggle requirements."""
    sol = pd.read_csv(f"{OUT_DIR}/solution.csv")
    sub = pd.read_csv(f"{OUT_DIR}/sample_submission.csv")
    prompts = pd.read_csv(f"{OUT_DIR}/test_prompts.csv")

    assert list(sol.columns) == ["id", "prediction", "Usage"], \
        f"solution.csv columns wrong: {list(sol.columns)}"
    assert list(sub.columns) == ["id", "prediction"], \
        f"sample_submission.csv columns wrong: {list(sub.columns)}"
    assert list(sol["id"]) == list(sub["id"]), \
        "ID mismatch between solution.csv and sample_submission.csv"
    assert list(sol["id"]) == list(prompts["id"]), \
        "ID mismatch between solution.csv and test_prompts.csv"
    assert set(sol["Usage"].unique()) == {"Public", "Private"}, \
        f"Usage values wrong: {sol['Usage'].unique()}"

    n_pub = (sol["Usage"] == "Public").sum()
    n_priv = (sol["Usage"] == "Private").sum()
    print(f"  Public:  {n_pub} ({n_pub / len(sol):.1%})")
    print(f"  Private: {n_priv} ({n_priv / len(sol):.1%})")

    print(f"\n  Prompts by category:")
    for cat in ["giraffe", "zebra", "ctrl", "mixed"]:
        n = sum(1 for x in prompts["id"] if x.startswith(cat))
        print(f"    {cat:10s}: {n}")

    print("\n  All checks passed.")


def main():
    print("Building test prompts ...")
    df = build_prompts()
    print(f"  Total prompts: {len(df)}")

    print(f"\nSaving competition files to {OUT_DIR}/ ...")
    save_competition_files(df)

    print(f"\n{'=' * 55}")
    print("DIFFUSION COMPETITION -- Zebra/Giraffe Swap")
    print(f"  Test prompts: {len(df)}")
    print(f"\nFiles saved to {OUT_DIR}/")

    print("\nVerifying competition files ...")
    verify_files()

    print(f"\n{'=' * 55}")
    print("NEXT STEPS:")
    print("  1. Upload test_prompts.csv, sample_submission.csv")
    print("     as a Kaggle Dataset")
    print("  2. Create a Kaggle Competition and upload:")
    print("     - solution.csv   (answer key)")
    print("     - metric.py      (custom scorer)")
    print("     - sample_submission.csv")
    print("  3. Share baseline_notebook.ipynb and comp_notebook.ipynb")
    print("     with students")


if __name__ == "__main__":
    main()
