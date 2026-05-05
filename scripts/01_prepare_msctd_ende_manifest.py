# Add the project root folder to Python's import path.
# This allows this script to import files from the src/ folder when run from scripts/.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import pandas for creating and saving the final dataset manifest as a CSV file.
import pandas as pd

# Import project configuration values:
# MSCTD_DATA_ENDE = folder containing En-De text, sentiment and image-index files
# IMAGE_ROOT = folder containing extracted train/dev/test image folders
# MANIFEST_PATH = output path for the final manifest CSV
# RAW_TO_MODEL_LABEL = converts original MSCTD labels into model labels
# MODEL_TO_SENTIMENT = converts model labels into readable sentiment names
from src.config import (
    MSCTD_DATA_ENDE,
    IMAGE_ROOT,
    MANIFEST_PATH,
    RAW_TO_MODEL_LABEL,
    MODEL_TO_SENTIMENT
)

# Import helper functions:
# read_lines reads text files line by line
# parse_image_index_lines converts image-index dialogue lines into sample-level image IDs
# find_image_path finds the matching image file for each sample
from src.dataset_utils import read_lines, parse_image_index_lines, find_image_path


def convert_label(raw_value):
    # Convert the raw sentiment value from the file into an integer.
    raw = int(str(raw_value).strip())

    # Check that the raw label exists in the expected label mapping.
    # This prevents silent errors if the dataset contains an unexpected label.
    if raw not in RAW_TO_MODEL_LABEL:
        raise ValueError(f"Unexpected sentiment label {raw_value}")

    # Return the converted model label.
    return RAW_TO_MODEL_LABEL[raw]


def build_split(split, en_file, de_file, img_file, sentiment_file):
    # Read English utterances for this split.
    english = read_lines(MSCTD_DATA_ENDE / en_file)

    # Read German utterances for this split.
    german = read_lines(MSCTD_DATA_ENDE / de_file)

    # Read sentiment labels for this split.
    sentiments = read_lines(MSCTD_DATA_ENDE / sentiment_file)

    # Read image-index lines for this split.
    # These image-index files may contain grouped image IDs by dialogue.
    img_lines = read_lines(MSCTD_DATA_ENDE / img_file)

    # Flatten the image-index lines so there is one image ID per utterance/sample.
    image_ids = parse_image_index_lines(img_lines)

    # Print file counts to confirm whether text, labels and image IDs align correctly.
    print(
        f"{split}: en={len(english)} de={len(german)} "
        f"labels={len(sentiments)} flattened_image_ids={len(image_ids)}"
    )

    # Warn if the number of image IDs does not match the number of English text rows.
    # This helps identify dataset alignment problems early.
    if len(image_ids) != len(english):
        print(
            f"WARNING: flattened image IDs ({len(image_ids)}) "
            f"do not match text rows ({len(english)})."
        )

    # Use the smallest shared length across text and labels to avoid index errors.
    n = min(len(english), len(german), len(sentiments))

    # This list will store one row per sample.
    rows = []

    # Build one manifest row for each utterance/sample.
    for i in range(n):
        # Convert raw MSCTD sentiment label into the model label format.
        label = convert_label(sentiments[i])

        # Get the image ID for this sample if available.
        # If not enough image IDs exist, keep it empty.
        image_id = image_ids[i] if i < len(image_ids) else ""

        # Add the complete sample information to the manifest.
        rows.append({
            # Unique ID for each sample, including split name and row number.
            "sample_id": f"{split}_{i}",

            # Dataset split: train, val, or test.
            "split": split,

            # English text utterance.
            "english_text": english[i],

            # German paired utterance.
            "german_text": german[i],

            # Image ID from the MSCTD image-index file.
            "image_index": image_id,

            # Full image file path if the image exists; otherwise empty string.
            "image_path": find_image_path(IMAGE_ROOT, split, image_id) if image_id else "",

            # Numeric model label.
            "label": label,

            # Human-readable sentiment label.
            "sentiment": MODEL_TO_SENTIMENT[label]
        })

    # Return all rows for this split.
    return rows


# Create an empty list to collect rows from train, validation and test splits.
rows = []

# Build training split rows using En-De train files.
rows += build_split(
    "train",
    "english_train.txt",
    "german_train.txt",
    "image_index_train.txt",
    "sentiment_train.txt"
)

# Build validation split rows using En-De dev files.
# The project treats dev as validation.
rows += build_split(
    "val",
    "english_dev.txt",
    "german_dev.txt",
    "image_index_dev.txt",
    "sentiment_dev.txt"
)

# Build test split rows using En-De test files.
rows += build_split(
    "test",
    "english_test.txt",
    "german_test.txt",
    "image_index_test.txt",
    "sentiment_test.txt"
)

# Convert all rows into a pandas DataFrame.
df = pd.DataFrame(rows)

# Create the output folder if it does not already exist.
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save the final manifest CSV.
# This file becomes the main dataset table used by the remaining scripts.
df.to_csv(MANIFEST_PATH, index=False)

# Print the saved path for confirmation.
print("Saved", MANIFEST_PATH)

# Print the full DataFrame shape, for example: (30370, 8).
print(df.shape)

# Print how many samples are in train, validation and test.
print(df["split"].value_counts())

# Print the number of samples in each sentiment class.
print(df["sentiment"].value_counts())

# Print how many samples have a valid image path.
# If images are correctly extracted, this should be close to the total number of samples.
print("image_path count:", int((df["image_path"].astype(str).str.len() > 0).sum()))