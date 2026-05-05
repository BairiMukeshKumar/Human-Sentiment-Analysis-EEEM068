# Add the main project folder to Python's import path.
# This allows this script to import project modules from the src/ folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import json for saving metrics in JSON format.
# Import numpy for converting model logits into class predictions.
# Import pandas for reading the dataset manifest and saving reports.
# Import torch for deep learning dataset tensors.
import json
import numpy as np
import pandas as pd
import torch

# Import Dataset to create a custom PyTorch dataset for text classification.
from torch.utils.data import Dataset

# Import evaluation metrics for model performance reporting.
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import matplotlib for saving the confusion matrix figure.
import matplotlib.pyplot as plt

# Import Hugging Face tools:
# AutoTokenizer converts text into tokens.
# AutoModelForSequenceClassification loads DistilBERT for 3-class classification.
# Trainer and TrainingArguments manage model training and evaluation.
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Import project paths from config.py.
# MANIFEST_PATH = dataset manifest CSV
# METRIC_DIR = folder for saved metrics/reports
# CM_DIR = folder for confusion matrix images
# CHECKPOINT_DIR = folder for saved model checkpoints
from src.config import MANIFEST_PATH, METRIC_DIR, CM_DIR, CHECKPOINT_DIR


# Read the prepared MSCTD En-De manifest.
df = pd.read_csv(MANIFEST_PATH)

# Create sampled train, validation and test sets.
# Sampling is used to make DistilBERT training faster on a normal laptop/CPU.
# For final full training, these sample sizes can be increased or removed.
train = df[df.split == "train"].sample(3000, random_state=42)
val = df[df.split == "val"].sample(800, random_state=42)
test = df[df.split == "test"].sample(800, random_state=42)


# Select the pretrained DistilBERT model.
# This is a smaller/faster BERT-style Transformer model for text classification.
MODEL_NAME = "distilbert-base-uncased"

# Load the tokenizer for the selected DistilBERT model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SentimentDataset(Dataset):
    # Custom PyTorch dataset for MSCTD text sentiment classification.

    def __init__(self, texts, labels):
        # Tokenize the input English text.
        # truncation=True cuts long text to max_length.
        # padding=True pads shorter text so all inputs have compatible lengths.
        # max_length=96 is used because utterances are usually short.
        self.enc = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=96
        )

        # Store sentiment labels.
        self.labels = list(labels)

    def __len__(self):
        # Return number of samples in the dataset.
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert tokenized text values into PyTorch tensors for one sample.
        item = {
            k: torch.tensor(v[idx])
            for k, v in self.enc.items()
        }

        # Add the sentiment label tensor.
        item["labels"] = torch.tensor(int(self.labels[idx]))

        # Return one training/evaluation sample.
        return item


# Load DistilBERT with a classification head for 3 sentiment classes.
# The three classes are negative, neutral and positive.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)


def compute_metrics(eval_pred):
    # This function calculates metrics during validation.
    # eval_pred contains model logits and true labels.

    logits, labels = eval_pred

    # Convert logits into predicted class IDs.
    preds = np.argmax(logits, axis=1)

    # Return accuracy, macro F1 and weighted F1.
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted")
    }


# Define Hugging Face training settings.
args = TrainingArguments(
    # Folder where DistilBERT checkpoints will be saved.
    output_dir=str(CHECKPOINT_DIR / "distilbert_text"),

    # Evaluate the model after each epoch.
    eval_strategy="epoch",

    # Save model checkpoint after each epoch.
    save_strategy="epoch",

    # Learning rate for fine-tuning.
    learning_rate=2e-5,

    # Batch size for training.
    per_device_train_batch_size=8,

    # Batch size for validation/testing.
    per_device_eval_batch_size=16,

    # Number of training epochs.
    num_train_epochs=2,

    # Weight decay helps reduce overfitting.
    weight_decay=0.01,

    # Folder for training logs.
    logging_dir="outputs/logs/distilbert",

    # Print log every 50 steps.
    logging_steps=50,

    # Load the best checkpoint at the end of training.
    load_best_model_at_end=True,

    # Use macro F1 to decide the best model checkpoint.
    metric_for_best_model="macro_f1",

    # Higher macro F1 is better.
    greater_is_better=True,

    # Disable external reporting tools such as WandB.
    report_to="none"
)


# Create Hugging Face Trainer.
# It handles training, validation and checkpoint saving.
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=SentimentDataset(train.english_text.astype(str), train.label),
    eval_dataset=SentimentDataset(val.english_text.astype(str), val.label),
    compute_metrics=compute_metrics
)


# Train the DistilBERT text sentiment model.
trainer.train()


# Run prediction on the sampled test set.
out = trainer.predict(
    SentimentDataset(test.english_text.astype(str), test.label)
)

# Convert model logits into class predictions.
preds = np.argmax(out.predictions, axis=1)

# Extract true test labels.
labels = out.label_ids


# Calculate final test metrics.
metrics = {
    "model": "distilbert_text_sampled",
    "train_samples": int(len(train)),
    "val_samples": int(len(val)),
    "test_samples": int(len(test)),
    "test_accuracy": float(accuracy_score(labels, preds)),
    "test_macro_f1": float(f1_score(labels, preds, average="macro")),
    "test_weighted_f1": float(f1_score(labels, preds, average="weighted"))
}


# Print metrics in the terminal.
print(json.dumps(metrics, indent=4))

# Save metrics as a JSON file.
(METRIC_DIR / "distilbert_text_metrics.json").write_text(
    json.dumps(metrics, indent=4)
)


# Save a detailed classification report.
# This includes precision, recall and F1-score for each sentiment class.
pd.DataFrame(
    classification_report(
        labels,
        preds,
        target_names=["negative", "neutral", "positive"],
        output_dict=True,
        zero_division=0
    )
).transpose().to_csv(
    METRIC_DIR / "distilbert_text_classification_report.csv"
)


# Create a confusion matrix for DistilBERT predictions.
cm = confusion_matrix(labels, preds)

# Start a new plot.
plt.figure(figsize=(6, 5))

# Display the confusion matrix.
plt.imshow(cm)

# Add a title and colour bar.
plt.title("DistilBERT Text Model Confusion Matrix")
plt.colorbar()

# Add sentiment class names to the x-axis and y-axis.
plt.xticks([0, 1, 2], ["negative", "neutral", "positive"], rotation=45)
plt.yticks([0, 1, 2], ["negative", "neutral", "positive"])

# Add the number of samples inside each matrix cell.
for i in range(3):
    for j in range(3):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

# Improve layout spacing.
plt.tight_layout()

# Save the confusion matrix figure.
plt.savefig(CM_DIR / "distilbert_text_confusion_matrix.png", dpi=300)

# Close the plot to free memory.
plt.close()


# Save the final trained DistilBERT model.
trainer.save_model(str(CHECKPOINT_DIR / "distilbert_text" / "best_model"))

# Save the tokenizer with the model so it can be reused later.
tokenizer.save_pretrained(str(CHECKPOINT_DIR / "distilbert_text" / "best_model"))