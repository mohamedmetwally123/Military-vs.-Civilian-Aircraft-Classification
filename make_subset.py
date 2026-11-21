"""
make_subset.py

Picks a small random subset of the aircraft dataset and saves it as a CSV file.
The CSV lists, for every chosen image: its file path, its label, and its split.
No images are copied - the CSV just points to the images in data/raw/dataset/.

Usage:
    1. Change the settings below if you want to.
    2. Run:  python make_subset.py
"""

import csv
import hashlib
import json
import os
import random

# ---------------------------------------------------------------------------
# Settings you may want to change
# ---------------------------------------------------------------------------

# How many images the subset should have in total (train + val + test)
TOTAL = 2000

# Where the unzipped dataset lives (it has train/, val/, test/ folders inside)
RAW_DIR = "data/raw/dataset"

# Where the subset CSV files are saved
SUBSET_DIR = "data/subsets"

# What fraction of the subset each class gets (should add up to 1)
CLASS_WEIGHTS = {"military": 1 / 3, "civilian": 1 / 3, "neither": 1 / 3}

# What fraction of the subset goes to each split (should add up to 1)
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Random seed: change this number to get a different random pick of images
SEED = 42

# Skip the "aug_" military images (they are edited copies of other images)
EXCLUDE_AUG = True

# If a subset with these exact settings already exists, it is reused.
# Set this to True to rebuild it anyway.
FORCE_REBUILD = False

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# Step 1: decide the label of an image from its file name
# ---------------------------------------------------------------------------
def get_label(filename):
    name = filename.lower()

    if name.startswith("bg_commercial"):
        return "civilian"
    if name.startswith("bg_birds") or name.startswith("bg_drones"):
        return "neither"
    if name.startswith("bg_"):
        # bg_train_* / bg_test_* are satellite photos of airports -> not used
        return None
    if EXCLUDE_AUG and name.startswith("aug_"):
        return None

    # Every other image has a military aircraft in it
    return "military"


# ---------------------------------------------------------------------------
# Step 2: list every usable image, grouped by (split, label)
# ---------------------------------------------------------------------------
def collect_images():
    # images[("train", "military")] = [path1, path2, ...]
    images = {}
    for split in SPLIT_RATIOS:
        for label in CLASS_WEIGHTS:
            images[(split, label)] = []

    for split in SPLIT_RATIOS:
        folder = os.path.join(RAW_DIR, split)
        # sorted() so the order is the same on every computer -> same random pick
        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue  # skip the .txt label files
            label = get_label(filename)
            if label is None:
                continue
            images[(split, label)].append(os.path.join(folder, filename))

    return images


# ---------------------------------------------------------------------------
# Step 3: build the subset and write it to a CSV
# ---------------------------------------------------------------------------
def make_subset():
    # All the settings that affect which images get picked
    settings = {
        "total": TOTAL,
        "seed": SEED,
        "class_weights": CLASS_WEIGHTS,
        "split_ratios": SPLIT_RATIOS,
        "exclude_aug": EXCLUDE_AUG,
    }

    # Turn the settings into a short code, e.g. "a1b2c3".
    # Same settings -> same code -> same file name.
    settings_text = json.dumps(settings, sort_keys=True)
    code = hashlib.md5(settings_text.encode()).hexdigest()[:6]
    csv_path = os.path.join(SUBSET_DIR, f"subset_{TOTAL}_{code}.csv")

    # If this exact subset was already made, just reuse it
    if os.path.exists(csv_path) and not FORCE_REBUILD:
        print(f"Subset already exists: {csv_path}")
        return csv_path

    images = collect_images()
    rng = random.Random(SEED)  # our own random generator, so results repeat
    rows = []

    for split, split_ratio in SPLIT_RATIOS.items():
        for label, class_weight in CLASS_WEIGHTS.items():
            # How many images we want for this (split, label)
            wanted = round(TOTAL * class_weight * split_ratio)
            available = images[(split, label)]

            if wanted > len(available):
                raise SystemExit(
                    f"Not enough images for {label}/{split}: "
                    f"wanted {wanted}, only {len(available)} available."
                )

            # Pick `wanted` images at random (no repeats)
            for path in rng.sample(available, wanted):
                rows.append({"filepath": path, "label": label, "split": split})

            print(f"{split:5} {label:9} {wanted:5} images (of {len(available)} available)")

    # Save the CSV
    os.makedirs(SUBSET_DIR, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # Save the settings next to the CSV so we always know how it was made
    with open(csv_path.replace(".csv", ".json"), "w") as f:
        json.dump(settings, f, indent=2)

    print(f"Saved {len(rows)} images to {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Run the script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    make_subset()
