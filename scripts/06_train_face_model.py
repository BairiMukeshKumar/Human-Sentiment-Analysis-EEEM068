# Add the main project folder to Python's import path.
# This allows this script to import modules from the src/ folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import pandas for reading the face manifest CSV.
# Import torch and torch.nn for deep learning model training.
import pandas as pd
import torch
import torch.nn as nn

# Import DataLoader to load image batches during training and testing.
from torch.utils.data import DataLoader

# Import project settings and output paths from config.py.
from src.config import (
    FACE_MANIFEST_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    METRIC_DIR,
    CM_DIR
)

# Import the custom image dataset class and image transformations.
from src.dataset_utils import MSCTDImageDataset, get_image_transforms

# Import the ResNet classifier model used for face-based sentiment classification.
from src.models import ResNetClassifier

# Import training and evaluation helper functions.
from src.train_utils import fit_model, evaluate_model

# Import helper functions for saving metrics, confusion matrix, and classification report.
from src.evaluate_utils import (
    save_metrics_json,
    plot_confusion_matrix,
    save_classification_report
)


# Read the face manifest created by the face extraction script.
# This file contains face paths, labels, split names, and detection status.
df = pd.read_csv(FACE_MANIFEST_PATH)

# Keep only rows where a face was successfully detected.
# The face model should train only on valid cropped face images.
df = df[df.face_detected == 1]


# Stop the script if no detected faces exist.
# This prevents training on an empty dataset.
if df.empty:
    raise SystemExit("No detected faces.")


# Split the detected-face dataset into train, validation, and test sets.
train, val, test = (
    df[df.split == "train"],
    df[df.split == "val"],
    df[df.split == "test"]
)


# Use GPU if available; otherwise use CPU.
# GPU makes training much faster.
device = "cuda" if torch.cuda.is_available() else "cpu"


def loader(part, trainflag):
    # Create a DataLoader for a dataset split.
    # part = train/val/test DataFrame
    # trainflag=True applies training augmentations and shuffling.
    # trainflag=False applies only validation/test transformations.

    return DataLoader(
        MSCTDImageDataset(
            part,
            image_col="face_path",
            transform=get_image_transforms(IMG_SIZE, trainflag)
        ),
        batch_size=BATCH_SIZE,
        shuffle=trainflag,
        num_workers=0
    )


# Create a ResNet18 model for three-class face-based sentiment classification.
# pretrained=True uses ImageNet-pretrained weights.
# freeze_backbone=False allows the whole network to fine-tune on face crops.
model = ResNetClassifier("resnet18", 3, True, False).to(device)

# Define the loss function for multi-class classification.
crit = nn.CrossEntropyLoss()

# Define the optimizer.
# AdamW is used with weight decay to reduce overfitting.
opt = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# Define where the best face model checkpoint will be saved.
ckpt = CHECKPOINT_DIR / "face_resnet18_best.pth"


# Train the face model for 5 epochs.
# The fit_model function saves the best checkpoint based on validation macro F1-score.
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


# Load the best saved checkpoint before testing.
# This ensures test evaluation uses the best validation model, not necessarily the final epoch.
model.load_state_dict(torch.load(ckpt, map_location=device))


# Evaluate the trained face model on the test set.
metrics = evaluate_model(
    model,
    loader(test, False),
    crit,
    device
)


# Save main test metrics as a JSON file.
# These results are later used by the final result summary script.
save_metrics_json(
    {
        "model": "face_resnet18",
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_f1": metrics["weighted_f1"]
    },
    METRIC_DIR / "face_resnet18_metrics.json"
)


# Save the test confusion matrix as an image.
# This shows where the model confuses negative, neutral, and positive classes.
plot_confusion_matrix(
    metrics["labels"],
    metrics["preds"],
    ["negative", "neutral", "positive"],
    "Face ResNet18 Confusion Matrix",
    CM_DIR / "face_resnet18_confusion_matrix.png"
)


# Save a detailed classification report with precision, recall, and F1-score per class.
save_classification_report(
    metrics["labels"],
    metrics["preds"],
    ["negative", "neutral", "positive"],
    METRIC_DIR / "face_resnet18_classification_report.csv"
)