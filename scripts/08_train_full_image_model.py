# Add the main project folder to Python's import path.
# This allows this script to import modules from the src/ folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import pandas for reading the dataset manifest.
# Import torch and torch.nn for deep learning model training.
import pandas as pd
import torch
import torch.nn as nn

# Import DataLoader to load image data in batches.
from torch.utils.data import DataLoader

# Import project configuration paths and training settings.
# MANIFEST_PATH = main dataset CSV
# IMG_SIZE = image resize size
# BATCH_SIZE = training batch size
# CHECKPOINT_DIR = folder for saved model weights
# METRIC_DIR = folder for saved metrics/reports
# CM_DIR = folder for saved confusion matrices
from src.config import (
    MANIFEST_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    METRIC_DIR,
    CM_DIR
)

# Import custom image dataset class and image transformations.
from src.dataset_utils import MSCTDImageDataset, get_image_transforms

# Import ResNet-based image classifier.
from src.models import ResNetClassifier

# Import training and evaluation helper functions.
from src.train_utils import fit_model, evaluate_model

# Import helper functions for saving metrics, confusion matrix and classification report.
from src.evaluate_utils import (
    save_metrics_json,
    plot_confusion_matrix,
    save_classification_report
)


# Read the main MSCTD En-De manifest.
# This contains image paths, text, labels, sentiment names and split information.
df = pd.read_csv(MANIFEST_PATH)

# Keep only rows that have a valid image path.
# Full-image training cannot run if images are missing.
df = df[df.image_path.fillna("").astype(str).str.len() > 0]


# Stop the script if no image files are available.
# This prevents training from starting with an empty dataset.
if df.empty:
    raise SystemExit("No images found.")


# Split the dataset into train, validation and test sets.
train, val, test = (
    df[df.split == "train"],
    df[df.split == "val"],
    df[df.split == "test"]
)


# Use GPU if available; otherwise use CPU.
# GPU will make ResNet training much faster.
device = "cuda" if torch.cuda.is_available() else "cpu"


def loader(part, trainflag):
    # Create a DataLoader for a specific dataset split.
    # part = train/val/test DataFrame
    # trainflag=True applies training transforms and shuffling.
    # trainflag=False applies only validation/test transforms.

    return DataLoader(
        MSCTDImageDataset(
            part,
            transform=get_image_transforms(IMG_SIZE, trainflag)
        ),
        batch_size=BATCH_SIZE,
        shuffle=trainflag,
        num_workers=0
    )


# Create a ResNet50 classifier for full-image sentiment classification.
# pretrained=True loads ImageNet-pretrained weights.
# freeze_backbone=True keeps the ResNet feature extractor frozen.
# Only the custom classifier head will be trained, which matches the brief requirement.
model = ResNetClassifier("resnet50", 3, True, True).to(device)


# Define the loss function for three-class classification.
crit = nn.CrossEntropyLoss()


# Define the optimizer.
# Because the backbone is frozen, only model.classifier parameters are trained.
opt = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)


# Define where the best full-image model checkpoint will be saved.
ckpt = CHECKPOINT_DIR / "full_image_resnet50_frozen_best.pth"


# Train the full-image model for 5 epochs.
# The best checkpoint is saved based on validation macro F1-score.
fit_model(
    model,
    loader(train, True),
    loader(val, False),
    crit,
    opt,
    device,
    5,
    ckpt
)


# Load the best saved model checkpoint before final testing.
model.load_state_dict(torch.load(ckpt, map_location=device))


# Evaluate the trained model on the test split.
metrics = evaluate_model(
    model,
    loader(test, False),
    crit,
    device
)


# Save the main full-image test metrics as a JSON file.
# This is used later by the final result summary script.
save_metrics_json(
    {
        "model": "full_image_resnet50_frozen",
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_f1": metrics["weighted_f1"]
    },
    METRIC_DIR / "full_image_resnet50_metrics.json"
)


# Save the confusion matrix figure for the full-image model.
plot_confusion_matrix(
    metrics["labels"],
    metrics["preds"],
    ["negative", "neutral", "positive"],
    "Full Image ResNet50 Confusion Matrix",
    CM_DIR / "full_image_resnet50_confusion_matrix.png"
)


# Save a detailed classification report with precision, recall and F1-score per class.
save_classification_report(
    metrics["labels"],
    metrics["preds"],
    ["negative", "neutral", "positive"],
    METRIC_DIR / "full_image_resnet50_classification_report.csv"
)