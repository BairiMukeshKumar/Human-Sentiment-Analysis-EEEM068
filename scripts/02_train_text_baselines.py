# Add the main project folder to Python's import path.
# This allows this script to import modules from the src/ folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import json to save model performance metrics in .json format.
import json

# Import joblib to save trained scikit-learn models for later reuse.
import joblib

# Import pandas for reading the dataset manifest and saving result tables.
import pandas as pd

# Import matplotlib for creating and saving plots such as confusion matrices.
import matplotlib.pyplot as plt

# Import Pipeline to combine TF-IDF feature extraction and machine learning model training.
from sklearn.pipeline import Pipeline

# Import TF-IDF vectorizer to convert text into numerical features.
from sklearn.feature_extraction.text import TfidfVectorizer

# Import Logistic Regression as the first classical text classification model.
from sklearn.linear_model import LogisticRegression

# Import Linear SVM as the second classical text classification model.
from sklearn.svm import LinearSVC

# Import evaluation metrics for accuracy, F1-score, classification report and confusion matrix.
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import project paths from config.py.
# MANIFEST_PATH = dataset CSV path
# METRIC_DIR = folder for metrics and reports
# CM_DIR = folder for confusion matrix images
# FIGURE_DIR = folder for general figures
# CHECKPOINT_DIR = folder for saved trained models
from src.config import (
    MANIFEST_PATH,
    METRIC_DIR,
    CM_DIR,
    FIGURE_DIR,
    CHECKPOINT_DIR
)


# Read the prepared MSCTD En-De manifest CSV.
# This file contains text, image paths, labels, sentiment names and split information.
df = pd.read_csv(MANIFEST_PATH)

# Split the dataset into train, validation and test sets using the split column.
train, val, test = (
    df[df.split == "train"],
    df[df.split == "val"],
    df[df.split == "test"]
)

# Print basic dataset information to confirm that the manifest loaded correctly.
print("Dataset shape:", df.shape)

# Print how many samples are in train, validation and test.
print(df["split"].value_counts())

# Print class distribution for negative, neutral and positive sentiment.
print(df["sentiment"].value_counts())


# Define the two classical machine learning baseline models.
# Both models use balanced class weights to reduce the effect of class imbalance.
models = {
    # Logistic Regression baseline.
    # lbfgs is used because this is a 3-class classification problem.
    "tfidf_logreg": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    ),

    # Linear Support Vector Machine baseline.
    # This is often strong for TF-IDF text classification.
    "tfidf_linear_svm": LinearSVC(
        class_weight="balanced"
    )
}


# This list will store final metric dictionaries for all models.
summary = []


# Train and evaluate each model one by one.
for name, clf in models.items():
    # Print the model currently being trained.
    print("\nTraining", name)

    # Create a machine learning pipeline:
    # Step 1: Convert English text into TF-IDF features.
    # Step 2: Train the selected classifier on those features.
    pipe = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                max_features=30000,
                ngram_range=(1, 2),
                stop_words="english"
            )
        ),
        ("clf", clf)
    ])

    # Train the pipeline using English text and sentiment labels from the training set.
    pipe.fit(train.english_text.astype(str), train.label)

    # Predict sentiment labels for the validation set.
    val_pred = pipe.predict(val.english_text.astype(str))

    # Predict sentiment labels for the test set.
    test_pred = pipe.predict(test.english_text.astype(str))

    # Calculate validation and test performance metrics.
    # Macro F1 treats all classes equally.
    # Weighted F1 accounts for class imbalance.
    metrics = {
        "model": name,
        "train_samples": int(len(train)),
        "val_samples": int(len(val)),
        "test_samples": int(len(test)),
        "val_accuracy": float(accuracy_score(val.label, val_pred)),
        "val_macro_f1": float(f1_score(val.label, val_pred, average="macro")),
        "val_weighted_f1": float(f1_score(val.label, val_pred, average="weighted")),
        "test_accuracy": float(accuracy_score(test.label, test_pred)),
        "test_macro_f1": float(f1_score(test.label, test_pred, average="macro")),
        "test_weighted_f1": float(f1_score(test.label, test_pred, average="weighted"))
    }

    # Add this model's metrics to the final summary list.
    summary.append(metrics)

    # Print the metrics in a readable JSON-style format.
    print(json.dumps(metrics, indent=4))

    # Save the metrics to a JSON file.
    with open(METRIC_DIR / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Create a detailed classification report for each class.
    # This includes precision, recall and F1-score.
    pd.DataFrame(
        classification_report(
            test.label,
            test_pred,
            target_names=["negative", "neutral", "positive"],
            output_dict=True,
            zero_division=0
        )
    ).transpose().to_csv(METRIC_DIR / f"{name}_classification_report.csv")

    # Create the confusion matrix for test predictions.
    cm = confusion_matrix(test.label, test_pred)

    # Start a new figure for the confusion matrix.
    plt.figure(figsize=(6, 5))

    # Display the confusion matrix as an image.
    plt.imshow(cm)

    # Add title and colour scale.
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()

    # Add class names on x-axis and y-axis.
    plt.xticks([0, 1, 2], ["negative", "neutral", "positive"], rotation=45)
    plt.yticks([0, 1, 2], ["negative", "neutral", "positive"])

    # Write the numeric count inside each confusion matrix cell.
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    # Label the axes.
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Improve layout spacing before saving.
    plt.tight_layout()

    # Save the confusion matrix image.
    plt.savefig(CM_DIR / f"{name}_confusion_matrix.png", dpi=300)

    # Close the plot to free memory before the next model.
    plt.close()

    # Save the trained pipeline, including both TF-IDF vectorizer and classifier.
    joblib.dump(pipe, CHECKPOINT_DIR / f"{name}.joblib")


# Save all model metrics into one summary CSV file.
pd.DataFrame(summary).to_csv(METRIC_DIR / "text_baseline_summary.csv", index=False)


# Create a bar chart showing the sentiment class distribution.
plt.figure(figsize=(6, 4))

# Count how many samples exist in each sentiment class and plot them.
df["sentiment"].value_counts().plot(kind="bar")

# Add chart title.
plt.title("MSCTD En-De Sentiment Class Distribution")

# Adjust spacing.
plt.tight_layout()

# Save the class distribution figure.
plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=300)

# Close the plot to free memory.
plt.close()


# Final confirmation message showing where outputs have been saved.
print("Files saved in outputs/metrics, outputs/confusion_matrices, outputs/figures, and outputs/checkpoints.")