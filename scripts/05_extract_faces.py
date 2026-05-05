# Add the main project folder to Python's import path.
# This allows this script to import project modules from the src/ folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import pandas to read the prepared dataset manifest.
import pandas as pd

# Import important project paths from config.py.
# MANIFEST_PATH = path to the main dataset manifest CSV
# FACE_DIR = folder where extracted face images will be saved
# FACE_MANIFEST_PATH = output CSV containing face extraction results
from src.config import MANIFEST_PATH, FACE_DIR, FACE_MANIFEST_PATH

# Import the face extraction function.
# This function detects faces from images and saves cropped face images.
from src.face_detection import extract_faces_from_manifest


# Read the main MSCTD En-De manifest file.
# This file contains text, labels, image indices, image paths and split information.
df = pd.read_csv(MANIFEST_PATH)


# Keep only rows where image_path is not empty.
# Face extraction can only run on samples that have real image files available.
df = df[df["image_path"].fillna("").astype(str).str.len() > 0].copy()


# Stop the script if no valid image paths are found.
# This prevents errors caused by trying to detect faces from missing images.
if df.empty:
    raise SystemExit("No image paths found. Download images and rerun manifest script first.")


# Run face detection and extraction on all available image samples.
# Extracted face crops are saved into FACE_DIR.
# A new CSV file is saved at FACE_MANIFEST_PATH with face metadata.
face_df = extract_faces_from_manifest(df, FACE_DIR, FACE_MANIFEST_PATH)


# Print how many faces were detected and how many samples had no detected face.
# face_detected = 1 means at least one face crop was saved.
# face_detected = 0 means no face was detected for that sample.
print(face_df["face_detected"].value_counts(dropna=False))


# Print the path where the face extraction manifest was saved.
print("Saved", FACE_MANIFEST_PATH)