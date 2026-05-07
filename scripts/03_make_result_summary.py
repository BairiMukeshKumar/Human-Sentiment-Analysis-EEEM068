# Add the main project folder to Python's import path.
# This allows the script to import modules from the src/ folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import json to read saved metrics from JSON files.
import json

# Import pandas to create and save the final result summary table.
import pandas as pd

# Import project output folders from config.py.
# METRIC_DIR = folder containing model metric JSON/CSV files
# CM_DIR = folder containing confusion matrix images
# FIGURE_DIR = folder containing general figures such as class distribution
from src.config import METRIC_DIR, CM_DIR, FIGURE_DIR


# Create an empty list to store model result rows.
rows = []


# Search the metrics folder for all files ending with "_metrics.json".
# Each model script saves its performance metrics using this naming format.
for path in sorted(METRIC_DIR.glob("*_metrics.json")):
    # Read the JSON file and convert it into a Python dictionary.
    data = json.loads(path.read_text())

    # Extract important results from the metric dictionary.
    # If any value does not exist, an empty string is used instead.
    rows.append({
        "Model": data.get("model", path.stem),
        "Train Samples": data.get("train_samples", ""),
        "Validation Samples": data.get("val_samples", ""),
        "Test Samples": data.get("test_samples", ""),
        "Validation Accuracy": data.get("val_accuracy", ""),
        "Validation Macro F1": data.get("val_macro_f1", ""),
        "Validation Weighted F1": data.get("val_weighted_f1", ""),
        "Test Accuracy": data.get("test_accuracy", ""),
        "Test Macro F1": data.get("test_macro_f1", ""),
        "Test Weighted F1": data.get("test_weighted_f1", "")
    })


# If no metric files are found, stop the script and tell the user what to run first.
if not rows:
    raise SystemExit("No result files found. Run scripts/02_train_text_baselines.py first.")


# Convert all collected result rows into a pandas DataFrame.
out = pd.DataFrame(rows)


# Save the final combined model result summary as a CSV file.
# This file is useful for the report results table.
out.to_csv(METRIC_DIR / "final_result_summary.csv", index=False)


# Print where the final summary file has been saved.
print("Final result summary saved to:", METRIC_DIR / "final_result_summary.csv")


# Print the result summary table in the terminal for quick checking.
print(out)


# Print all available confusion matrix image files.
# These are useful for the report and appendix.
print("\nAvailable confusion matrices:")
for p in sorted(CM_DIR.glob("*.png")):
    print(" -", p)


# Print all available general figure files.
# Example: class distribution chart.
print("\nAvailable figures:")
for p in sorted(FIGURE_DIR.glob("*.png")):
    print(" -", p)