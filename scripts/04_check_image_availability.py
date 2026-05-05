# Add the main project folder to Python's import path.
# This makes sure the script can import files from the src/ folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import pandas to read the prepared manifest CSV.
import pandas as pd

# Import important paths from config.py.
# IMAGE_ROOT = folder where extracted MSCTD images should be stored
# MANIFEST_PATH = path to the prepared dataset manifest CSV
from src.config import IMAGE_ROOT, MANIFEST_PATH


# Print the expected image root folder and check whether it exists.
# This confirms whether the image folder has been created/extracted correctly.
print("Image root:", IMAGE_ROOT, "exists=", IMAGE_ROOT.exists())


# If the manifest file does not exist, stop the script.
# The manifest must be created before checking image availability.
if not MANIFEST_PATH.exists():
    raise SystemExit("Run manifest script first.")


# Read the prepared dataset manifest.
# This file should contain text, labels, image indices, image paths and split information.
df = pd.read_csv(MANIFEST_PATH)


# Count how many rows have a non-empty image_path.
# If the images were linked correctly, this number should be greater than 0.
count = int((df["image_path"].fillna("").astype(str).str.len() > 0).sum())


# Print the number of samples that currently have valid image paths.
print("Current image_path count:", count)


# If no image paths are found, the raw image folders are missing or not placed correctly.
if count == 0:
    print(
        "Download/extract separate MSCTD En-De images into "
        "data/raw/MSCTD_images/train_images, dev_images, test_images "
        "and rerun 01_prepare_msctd_ende_manifest.py"
    )

# If image paths are found, the visual pipeline can be executed.
else:
    print("Images found. You can run visual scripts.")