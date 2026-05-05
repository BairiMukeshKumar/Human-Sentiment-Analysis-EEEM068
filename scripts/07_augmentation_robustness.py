# Add the main project folder to Python's import path.
# This allows this script to import modules from the src/ folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import pandas to read and save CSV manifest files.
import pandas as pd

# Import PIL Image to open and process face crop images.
from PIL import Image

# Import tqdm to show a progress bar while processing many face images.
from tqdm import tqdm

# Import project paths from config.py.
# FACE_MANIFEST_PATH = CSV file created after face extraction
# PROCESSED_DIR = folder where processed files such as degraded faces will be saved
from src.config import FACE_MANIFEST_PATH, PROCESSED_DIR

# Import the custom degradation function.
# This applies random image distortions such as blur, brightness change, noise, etc.
from src.augmentations import random_degrade


# Read the face manifest created by the face extraction script.
df = pd.read_csv(FACE_MANIFEST_PATH)

# Keep only rows where a face was successfully detected.
# Augmentation should only be applied to valid face crop images.
df = df[df.face_detected == 1]


# Stop the script if no face images are available.
# This prevents errors caused by missing face crops.
if df.empty:
    raise SystemExit("No faces found. Run 05_extract_faces.py first.")


# Create the output folder where degraded/augmented face images will be saved.
outdir = PROCESSED_DIR / "degraded_faces"
outdir.mkdir(parents=True, exist_ok=True)


# This list will store new rows for the degraded face manifest.
rows = []


# Loop through every detected face image.
# tqdm displays a progress bar in the terminal.
for _, row in tqdm(df.iterrows(), total=len(df)):

    # Open the original face crop image and convert it to RGB format.
    img = Image.open(row.face_path).convert("RGB")

    # Apply one random degradation/augmentation to the image.
    # Example transformations may include blur, brightness change, noise or frequency-style degradation.
    deg = random_degrade(img)

    # Create a split-specific output folder, such as train, val or test.
    splitdir = outdir / row.split
    splitdir.mkdir(parents=True, exist_ok=True)

    # Create the output filename by adding "_degraded" to the original face filename.
    path = splitdir / (Path(row.face_path).stem + "_degraded.jpg")

    # Save the degraded face image.
    deg.save(path)

    # Copy the original metadata row so label, split and sample information are preserved.
    new = row.copy()

    # Replace the face_path with the new degraded image path.
    new["face_path"] = str(path)

    # Add a column showing that this row belongs to the degraded/augmented dataset.
    new["augmentation_type"] = "degraded"

    # Store the new row for later saving.
    rows.append(new)


# Save the degraded face manifest as a CSV file.
# This file can be used later for robustness testing or retraining.
pd.DataFrame(rows).to_csv(
    PROCESSED_DIR / "degraded_face_manifest.csv",
    index=False
)


# Print confirmation message.
print("Saved degraded face manifest")