"""
preprocess.py

Takes the images listed in a subset CSV (made by make_subset.py),
crops off the watermark strip at the bottom, resizes them,
and saves them into folders like this:

    data/processed/<subset name>/train/military/...
    data/processed/<subset name>/train/civilian/...
    data/processed/<subset name>/train/neither/...
    (same for val/ and test/)

Usage:
    1. Change the settings below if you want to.
    2. Run:  python preprocess.py
"""

import csv
import os

from PIL import Image

# ---------------------------------------------------------------------------
# Settings you may want to change
# ---------------------------------------------------------------------------

# The subset CSV to process (made by make_subset.py)
SUBSET_CSV = "data/subsets/subset_2000_8d56f0.csv"

# Where the processed images are saved
PROCESSED_DIR = "data/processed"

# Final size of every image (width, height) in pixels
IMAGE_SIZE = (224, 224)

# Fraction of the image height to cut off the bottom (removes the watermark strip)
CROP_BOTTOM = 0.04

# If this subset was already processed, it is skipped.
# Set this to True to process it again anyway.
FORCE_REBUILD = False


# ---------------------------------------------------------------------------
# Process one image: open -> crop bottom -> resize -> save
# ---------------------------------------------------------------------------
def process_image(in_path, out_path):
    # Open the image and make sure it has 3 color channels (some are PNG/grayscale)
    img = Image.open(in_path).convert("RGB")

    # Crop: keep everything except the bottom strip
    # crop() takes (left, top, right, bottom)
    width, height = img.size
    new_height = int(height * (1 - CROP_BOTTOM))
    img = img.crop((0, 0, width, new_height))

    # Resize to the final size
    img = img.resize(IMAGE_SIZE)

    # Save as a JPG
    img.save(out_path, "JPEG", quality=95)


# ---------------------------------------------------------------------------
# Process every image in the subset CSV
# ---------------------------------------------------------------------------
def preprocess():
    # Output folder is named after the CSV, e.g. data/processed/subset_2000_8d56f0
    subset_name = os.path.splitext(os.path.basename(SUBSET_CSV))[0]
    out_dir = os.path.join(PROCESSED_DIR, subset_name)

    # If this subset was already processed, don't do it again
    if os.path.exists(out_dir) and not FORCE_REBUILD:
        print(f"Already processed: {out_dir}")
        return out_dir

    # Read the list of images from the CSV
    with open(SUBSET_CSV) as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        # Folder for this image, e.g. data/processed/subset_2000_8d56f0/train/military
        folder = os.path.join(out_dir, row["split"], row["label"])
        os.makedirs(folder, exist_ok=True)

        # Keep the original file name, but always save as .jpg
        name = os.path.splitext(os.path.basename(row["filepath"]))[0] + ".jpg"
        process_image(row["filepath"], os.path.join(folder, name))

        # Show progress every 200 images
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1} / {len(rows)} images")

    print(f"Done. Saved {len(rows)} images to {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Run the script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    preprocess()
