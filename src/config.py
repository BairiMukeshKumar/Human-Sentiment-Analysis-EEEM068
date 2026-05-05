# Import Path for clean and reliable file/folder path handling.
from pathlib import Path

# Import os to read environment variables such as PROJECT_DIR.
import os


# Set the main project directory.
# If PROJECT_DIR is defined in the environment, use it.
# Otherwise, use the current working directory where the script is run.
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path.cwd())).resolve()


# Define the raw data folder.
# This is where the downloaded MSCTD repository and image folders are stored.
RAW_DIR = PROJECT_DIR / "data" / "raw"


# Define the local MSCTD repository folder.
# This folder is created after downloading/extracting the official MSCTD GitHub repository.
MSCTD_DIR = RAW_DIR / "MSCTD"


# Define the English-German MSCTD data folder.
# This contains english_*.txt, german_*.txt, image_index_*.txt and sentiment_*.txt files.
MSCTD_DATA_ENDE = MSCTD_DIR / "MSCTD_data" / "ende"


# Define the folder where separate En-De image archives are extracted.
# Expected subfolders are train_images, dev_images and test_images.
IMAGE_ROOT = RAW_DIR / "MSCTD_images"


# Define the folder for processed dataset files.
# Example outputs: manifest CSV, face manifest CSV, degraded face manifest.
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


# Define the main output folder for all generated results.
OUTPUT_DIR = PROJECT_DIR / "outputs"


# Define the folder for general figures such as class distribution plots.
FIGURE_DIR = OUTPUT_DIR / "figures"


# Define the folder for metrics such as JSON and CSV performance reports.
METRIC_DIR = OUTPUT_DIR / "metrics"


# Define the folder for confusion matrix images.
CM_DIR = OUTPUT_DIR / "confusion_matrices"


# Define the folder for trained model checkpoints.
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


# Define the folder where extracted face images will be saved.
FACE_DIR = PROCESSED_DIR / "faces"


# Define the main dataset manifest path.
# This CSV links text, labels, image indices, image paths and splits.
MANIFEST_PATH = PROCESSED_DIR / "msctd_ende_manifest.csv"


# Define the face manifest path.
# This CSV stores face extraction results, including face paths and detection status.
FACE_MANIFEST_PATH = PROCESSED_DIR / "face_manifest.csv"


# Define the path for fusion model features.
# This can store combined face/full-image/text prediction features.
FUSION_FEATURE_PATH = PROCESSED_DIR / "fusion_features.csv"


# MSCTD README raw sentiment labels:
# positive = 2, negative = 1, neutral = 0.
#
# Model labels used in this project:
# negative = 0, neutral = 1, positive = 2.
#
# This dictionary converts the original MSCTD raw labels
# into the label format used by the models.
RAW_TO_MODEL_LABEL = {
    1: 0,  # raw negative -> model negative
    0: 1,  # raw neutral  -> model neutral
    2: 2   # raw positive -> model positive
}


# Convert model label IDs into readable sentiment names.
MODEL_TO_SENTIMENT = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# Label map used by other project files.
LABEL_MAP = MODEL_TO_SENTIMENT


# Image size used for ResNet and face/full-image models.
# Images are resized to 224x224 pixels.
IMG_SIZE = 224


# Batch size used during model training and evaluation.
BATCH_SIZE = 32


# Number of sentiment classes.
NUM_CLASSES = 3


# Random seed used for reproducibility.
SEED = 42


# Create all required output and processed-data folders if they do not already exist.
# This prevents file-not-found errors when scripts save outputs.
for d in [
    PROCESSED_DIR,
    OUTPUT_DIR,
    FIGURE_DIR,
    METRIC_DIR,
    CM_DIR,
    CHECKPOINT_DIR,
    FACE_DIR
]:
    d.mkdir(parents=True, exist_ok=True)