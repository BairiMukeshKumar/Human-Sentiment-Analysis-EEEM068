from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA_PATH = Path("data/processed/msctd_ende_manifest.csv")
OUT_METRICS = Path("outputs/metrics")
OUT_MODELS = Path("outputs/checkpoints")
OUT_CM = Path("outputs/confusion_matrices")
OUT_FIG = Path("outputs/figures")

OUT_METRICS.mkdir(parents=True, exist_ok=True)
OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_CM.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()

print("Dataset shape:", df.shape)
print("\nSplit counts:")
print(df["split"].value_counts())
print("\nSentiment counts:")
print(df["sentiment"].value_counts())

models = {
    "tfidf_logreg": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    ),
    "tfidf_linear_svm": LinearSVC(
        class_weight="balanced"
    )
}

summary_rows = []

for name, clf in models.items():
    print("\nTraining", name)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", clf)
    ])

    pipe.fit(train["english_text"].astype(str), train["label"])

    val_pred = pipe.predict(val["english_text"].astype(str))
    test_pred = pipe.predict(test["english_text"].astype(str))

    metrics = {
        "model": name,
        "train_samples": int(len(train)),
        "val_samples": int(len(val)),
        "test_samples": int(len(test)),
        "val_accuracy": float(accuracy_score(val["label"], val_pred)),
        "val_macro_f1": float(f1_score(val["label"], val_pred, average="macro")),
        "val_weighted_f1": float(f1_score(val["label"], val_pred, average="weighted")),
        "test_accuracy": float(accuracy_score(test["label"], test_pred)),
        "test_macro_f1": float(f1_score(test["label"], test_pred, average="macro")),
        "test_weighted_f1": float(f1_score(test["label"], test_pred, average="weighted"))
    }

    summary_rows.append(metrics)

    print(json.dumps(metrics, indent=4))

    with open(OUT_METRICS / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(
        test["label"],
        test_pred,
        target_names=["negative", "neutral", "positive"],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        OUT_METRICS / f"{name}_classification_report.csv"
    )

    cm = confusion_matrix(test["label"], test_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1, 2], ["negative", "neutral", "positive"], rotation=45)
    plt.yticks([0, 1, 2], ["negative", "neutral", "positive"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUT_CM / f"{name}_confusion_matrix.png", dpi=300)
    plt.close()

    joblib.dump(pipe, OUT_MODELS / f"{name}.joblib")

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_METRICS / "text_baseline_summary.csv", index=False)

plt.figure(figsize=(6, 4))
df["sentiment"].value_counts().plot(kind="bar")
plt.title("MSCTD En-De Sentiment Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(OUT_FIG / "class_distribution.png", dpi=300)
plt.close()

print("\nSaved baseline summary:")
print(summary)
print("\nFiles saved in outputs/metrics, outputs/confusion_matrices, outputs/figures, and outputs/checkpoints.")