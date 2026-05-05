# Import os to create folders before saving files.
import os

# Import json to save model metrics in JSON format.
import json

# Import numpy for checking numpy data types and handling arrays.
import numpy as np

# Import pandas to create and save classification report tables.
import pandas as pd

# Import matplotlib for plotting and saving confusion matrix figures.
import matplotlib.pyplot as plt

# Import sklearn evaluation functions.
# confusion_matrix creates the class-by-class prediction table.
# classification_report calculates precision, recall and F1-score.
from sklearn.metrics import confusion_matrix, classification_report


def save_metrics_json(metrics, path):
    # This function saves a metrics dictionary into a JSON file.
    # It also converts numpy float values into normal Python floats
    # because JSON cannot directly save some numpy data types.

    # Create an empty dictionary for JSON-safe values.
    serializable = {}

    # Go through every metric key and value.
    for k, v in metrics.items():

        # Skip numpy arrays because large arrays such as labels or predictions
        # should not be saved inside the metrics JSON file.
        if isinstance(v, np.ndarray):
            continue

        # Convert numpy float values into normal Python float values.
        # Keep all other values unchanged.
        serializable[k] = float(v) if isinstance(v, (np.float32, np.float64)) else v

    # Create the output folder if it does not already exist.
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)

    # Save the cleaned metrics dictionary as a formatted JSON file.
    with open(path, "w") as f:
        json.dump(serializable, f, indent=4)


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    # This function creates and saves a confusion matrix figure.
    # It compares true labels against predicted labels.

    # Calculate the confusion matrix.
    cm = confusion_matrix(y_true, y_pred)

    # Create a new figure with a fixed size.
    plt.figure(figsize=(6, 5))

    # Display the confusion matrix as an image.
    plt.imshow(cm)

    # Add a title to describe the model/result.
    plt.title(title)

    # Add a colour bar to show value intensity.
    plt.colorbar()

    # Add class names to the x-axis.
    plt.xticks(
        np.arange(len(class_names)),
        class_names,
        rotation=45
    )

    # Add class names to the y-axis.
    plt.yticks(
        np.arange(len(class_names)),
        class_names
    )

    # Write the count value inside each confusion matrix cell.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    # Label y-axis as true class labels.
    plt.ylabel("True Label")

    # Label x-axis as predicted class labels.
    plt.xlabel("Predicted Label")

    # Adjust layout to avoid label/title overlap.
    plt.tight_layout()

    # Create the output folder if it does not already exist.
    os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)

    # Save the confusion matrix figure as an image file.
    plt.savefig(save_path, dpi=300)

    # Close the plot to free memory.
    plt.close()

    # Return the confusion matrix values in case another script needs them.
    return cm


def save_classification_report(y_true, y_pred, class_names, save_path):
    # This function creates and saves a detailed classification report.
    # The report includes precision, recall, F1-score and support for each class.

    # Generate classification report as a dictionary.
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Convert the report dictionary into a pandas DataFrame.
    df = pd.DataFrame(report).transpose()

    # Create the output folder if it does not already exist.
    os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)

    # Save the classification report as a CSV file.
    df.to_csv(save_path)

    # Return the DataFrame for optional further use.
    return df