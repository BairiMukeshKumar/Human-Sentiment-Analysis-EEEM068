# EEEM068 Human Sentiment Analysis Using MSCTD En-De Dataset

## 1. Project Overview

This project implements a multimodal human sentiment analysis system for the EEEM068 Applied Machine Learning group coursework. The task is to classify human sentiment into three classes:

- Negative
- Neutral
- Positive

The project uses the English-German subset of the MSCTD dataset. MSCTD is a Multimodal Sentiment Chat Translation Dataset containing text, image context, and sentiment labels. The official MSCTD repository describes the dataset as including English-German utterance pairs linked with visual context and sentiment labels.

The implementation includes:

1. Dataset download and preparation
2. Dataset cleaning and manifest creation
3. Image availability checking
4. Face extraction using MTCNN
5. Image degradation and augmentation
6. Text-based sentiment classification using TF-IDF
7. Face-based sentiment classification using ResNet18
8. Full-image sentiment classification using frozen ResNet50
9. Result summary generation
10. Confusion matrix and classification report generation
11. Optional extension hooks for fusion and CLIP-style multimodal modelling

The project is organised so each group member can clearly show contribution through code, logs, GitHub commits, and output files.

---

## 2. Dataset Information

Dataset used: **MSCTD: Multimodal Sentiment Chat Translation Dataset**

Official repository:

```text
https://github.com/XL2248/MSCTD
```

The project uses the following MSCTD English-German files:

```text
data/raw/MSCTD/MSCTD_data/ende/english_train.txt
data/raw/MSCTD/MSCTD_data/ende/english_dev.txt
data/raw/MSCTD/MSCTD_data/ende/english_test.txt

data/raw/MSCTD/MSCTD_data/ende/german_train.txt
data/raw/MSCTD/MSCTD_data/ende/german_dev.txt
data/raw/MSCTD/MSCTD_data/ende/german_test.txt

data/raw/MSCTD/MSCTD_data/ende/image_index_train.txt
data/raw/MSCTD/MSCTD_data/ende/image_index_dev.txt
data/raw/MSCTD/MSCTD_data/ende/image_index_test.txt

data/raw/MSCTD/MSCTD_data/ende/sentiment_train.txt
data/raw/MSCTD/MSCTD_data/ende/sentiment_dev.txt
data/raw/MSCTD/MSCTD_data/ende/sentiment_test.txt
```

The raw images are separate from the main GitHub ZIP. They must be downloaded from the MSCTD data page and extracted into:

```text
data/raw/MSCTD_images/train_images
data/raw/MSCTD_images/dev_images
data/raw/MSCTD_images/test_images
```

The final prepared manifest contains:

| Split      | Samples |
|------------|--------:|
| Train      |  20,240 |
| Validation | 5,063   |
| Test       | 5,067   |
| Total      | 30,370  |

Sentiment label mapping used in this project:

| Original MSCTD Label | Meaning   | Model Label |
|---------------------:|-----------|------------:|
| 1                    | Negative  | 0           |
| 0                    | Neutral   | 1           |
| 2                    | Positive  | 2           |

---

## 3. Project Folder Structure

```text
EEEM068_HSA/
├── data/
│   ├── raw/
│   │   ├── MSCTD/
│   │   └── MSCTD_images/
│   │       ├── train_images/
│   │       ├── dev_images/
│   │       └── test_images/
│   └── processed/
│       ├── msctd_ende_manifest.csv
│       ├── face_manifest.csv
│       ├── degraded_face_manifest.csv
│       ├── faces/
│       └── degraded_faces/
├── outputs/
│   ├── metrics/
│   ├── confusion_matrices/
│   ├── figures/
│   ├── checkpoints/
│   └── logs/
├── scripts/
├── src/
├── notebooks/
├── requirements.txt
├── README.md
└── report/
```

---

## 4. What Each File Does

### 4.1 Main Scripts

| File | Purpose |
|---|---|
| `scripts/00_download_msctd.py` | Downloads the official MSCTD GitHub repository ZIP and extracts it into `data/raw/MSCTD`. It also reminds the user that the En-De images must be downloaded separately. |
| `scripts/01_prepare_msctd_ende_manifest.py` | Reads English text, German text, image indices, and sentiment labels. It creates the final manifest file `data/processed/msctd_ende_manifest.csv`. |
| `scripts/02_train_text_baselines.py` | Trains two text baseline models: TF-IDF Logistic Regression and TF-IDF Linear SVM. It saves metrics, confusion matrices, classification reports, and trained model files. |
| `scripts/03_make_result_summary.py` | Collects all saved model metrics and creates `outputs/metrics/final_result_summary.csv`. |
| `scripts/04_check_image_availability.py` | Checks whether the image folders exist and whether the manifest contains valid image paths. |
| `scripts/05_extract_faces.py` | Uses MTCNN to detect and crop faces from MSCTD images. Saves face crops and creates `data/processed/face_manifest.csv`. |
| `scripts/06_train_face_model.py` | Trains a ResNet18 model on extracted face crops for three-class sentiment classification. |
| `scripts/07_augmentation_robustness.py` | Creates degraded face images using brightness, blur, noise, rotation, and frequency degradation. |
| `scripts/08_train_full_image_model.py` | Trains a frozen ResNet50 model on full images. The pretrained backbone is frozen and only the classifier head is trained. |
| `scripts/09_train_fusion_model.py` | Placeholder/hook for a future fusion model combining face, full-image, and text outputs. |
| `scripts/10_clip_multimodal_extra_credit.py` | Placeholder/hook for CLIP-style image-text multimodal extra-credit work. |
| `scripts/11_train_distilbert_text.py` | Optional sampled DistilBERT text model for transformer-based text classification. |

### 4.2 Source Files

| File            | Purpose |
|-----------------|---------|
| `src/config.py` | Stores all folder paths, label mapping, image size, batch size, output paths, and project constants. |
| `src/dataset_utils.py` | Contains helper functions for reading text files, parsing image indices, finding image paths, and creating PyTorch image datasets. |
| `src/face_detection.py` | Contains MTCNN face detection and face crop saving functions. |
| `src/augmentations.py` | Contains image degradation functions such as brightness reduction, blur, noise, rotation, and frequency low-pass filtering. |
| `src/models.py` | Defines ResNetClassifier, TimmImageClassifier, and FusionMLP model classes. |
| `src/train_utils.py` | Contains reusable training and evaluation loops for PyTorch models. |
| `src/evaluate_utils.py` | Saves metrics JSON files, classification reports, and confusion matrix figures. |
| `src/fusion_utils.py` | Intended for fusion-model helper functions. |
| `src/__init__.py` | Marks the `src` folder as a Python package. |

### 4.3 Output Folders

| Folder | Contents |
|---|---|
| `outputs/metrics/` | JSON metrics, CSV classification reports, and final result summary. |
| `outputs/confusion_matrices/` | Confusion matrix images for each trained model. |
| `outputs/figures/` | General project figures such as class distribution. |
| `outputs/checkpoints/` | Saved trained model checkpoints and `.joblib` files. |
| `outputs/logs/` | Training logs for neural models where applicable. |

---

## 5. Software and Environment Requirements

Recommended software:

```text
Python 3.10 or above
Visual Studio Code
Windows PowerShell or Command Prompt
GitHub Desktop or Git
Internet connection for first-time downloads
```

Recommended Python libraries are listed in `requirements.txt` and include pandas, numpy, matplotlib, scikit-learn, torch, torchvision, timm, facenet-pytorch, opencv-python, Pillow, tqdm, requests, joblib, and transformers.

---

## 6. Installation Steps on Windows

Open the project folder in Visual Studio Code. Then open a terminal inside VS Code.

### Step 1: Create virtual environment

```cmd
python -m venv venv
```

### Step 2: Activate virtual environment

For Command Prompt:

```cmd
venv\Scripts\activate
```

For PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Python path if needed

If any script gives `ModuleNotFoundError: No module named 'src'`, run:

```cmd
set PYTHONPATH=%CD%
```

For PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location)
```

---

## 7. How to Run the Complete Project

Run the scripts from the project root folder.

### Step 1: Download the official MSCTD repository

```cmd
python scripts\00_download_msctd.py
```

This downloads the MSCTD GitHub repository into:

```text
data/raw/MSCTD
```

Important: this step downloads text, label, and image-index files. The raw images must be downloaded separately.

### Step 2: Download and extract En-De images manually

Download the En-De image ZIP files from the official MSCTD data page:

```text
https://github.com/XL2248/MSCTD/tree/main/MSCTD_data
```

Extract them into:

```text
data/raw/MSCTD_images/train_images
data/raw/MSCTD_images/dev_images
data/raw/MSCTD_images/test_images
```

The image folders must contain files such as:

```text
0.jpg
1.jpg
2.jpg
...
```

### Step 3: Prepare the MSCTD En-De manifest

```cmd
python scripts\01_prepare_msctd_ende_manifest.py
```

Expected output:

```text
data/processed/msctd_ende_manifest.csv
```

Expected dataset size:

```text
Train: 20240
Validation: 5063
Test: 5067
Total: 30370
```

### Step 4: Check image availability

```cmd
python scripts\04_check_image_availability.py
```

Expected successful output:

```text
Image root: data\raw\MSCTD_images exists= True
Current image_path count: 30370
Images found. You can run visual scripts.
```

If image path count is zero, the images are missing or extracted into the wrong folder.

### Step 5: Train text baseline models

```cmd
python scripts\02_train_text_baselines.py
```

This trains:

```text
TF-IDF Logistic Regression
TF-IDF Linear SVM
```

Generated outputs:

```text
outputs/metrics/tfidf_logreg_metrics.json
outputs/metrics/tfidf_linear_svm_metrics.json
outputs/metrics/text_baseline_summary.csv
outputs/confusion_matrices/tfidf_logreg_confusion_matrix.png
outputs/confusion_matrices/tfidf_linear_svm_confusion_matrix.png
outputs/checkpoints/tfidf_logreg.joblib
outputs/checkpoints/tfidf_linear_svm.joblib
outputs/figures/class_distribution.png
```

### Step 6: Extract faces from images

```cmd
python scripts\05_extract_faces.py
```

This may take several hours on CPU.

Generated output:

```text
data/processed/face_manifest.csv
data/processed/faces/
```

Observed output from this implementation:

```text
Face detected: 46301
No face detected: 4351
```

### Step 7: Train face-based ResNet18 model

```cmd
python scripts\06_train_face_model.py
```

Generated outputs:

```text
outputs/checkpoints/face_resnet18_best.pth
outputs/metrics/face_resnet18_metrics.json
outputs/metrics/face_resnet18_classification_report.csv
outputs/confusion_matrices/face_resnet18_confusion_matrix.png
```

### Step 8: Generate degraded face images

```cmd
python scripts\07_augmentation_robustness.py
```

Generated outputs:

```text
data/processed/degraded_faces/
data/processed/degraded_face_manifest.csv
```

This script applies brightness degradation, blur degradation, noise degradation, rotation degradation, and frequency low-pass degradation.

### Step 9: Train full-image ResNet50 model

```cmd
python scripts\08_train_full_image_model.py
```

Generated outputs:

```text
outputs/checkpoints/full_image_resnet50_frozen_best.pth
outputs/metrics/full_image_resnet50_metrics.json
outputs/metrics/full_image_resnet50_classification_report.csv
outputs/confusion_matrices/full_image_resnet50_confusion_matrix.png
```

### Step 10: Generate final result summary

```cmd
python scripts\03_make_result_summary.py
```

Generated output:

```text
outputs/metrics/final_result_summary.csv
```

This file combines all model results into one final table.

### Optional Step 11: Run sampled DistilBERT text model

```cmd
python scripts\11_train_distilbert_text.py
```

This is optional because it may take longer and uses a sampled subset.

### Optional Step 12: Fusion and CLIP extensions

The following files are currently extension hooks:

```cmd
python scripts\09_train_fusion_model.py
python scripts\10_clip_multimodal_extra_credit.py
```

These can be expanded in future work to combine text, face, and full-image predictions.

---

## 8. Main Results

Final model comparison:

| Model | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---:|---:|---:|
| TF-IDF Logistic Regression | 0.4906 | 0.4855 | 0.4967 |
| TF-IDF Linear SVM | 0.4790 | 0.4731 | 0.4843 |
| Full-image ResNet50 frozen | 0.3696 | 0.3010 | 0.3226 |
| Face ResNet18 | 0.3346 | 0.3299 | 0.3365 |

Best model:

```text
TF-IDF Logistic Regression
```

The text-based models performed better than the visual-only models. This suggests that English dialogue text carried stronger sentiment cues than isolated face or full-image features. The visual models are still important because they fulfil the multimodal and visual analysis requirements of the brief and demonstrate the difficulty of visual sentiment recognition in conversational scenes.

---

## 9. Important Figures and Where to Find Them

| Figure | File Location | Description |
|---|---|---|
| Class distribution | `outputs/figures/class_distribution.png` | Shows distribution of negative, neutral, and positive labels |
| TF-IDF Logistic Regression confusion matrix | `outputs/confusion_matrices/tfidf_logreg_confusion_matrix.png` | Confusion matrix for best text model |
| TF-IDF Linear SVM confusion matrix | `outputs/confusion_matrices/tfidf_linear_svm_confusion_matrix.png` | Confusion matrix for SVM text baseline |
| Face ResNet18 confusion matrix | `outputs/confusion_matrices/face_resnet18_confusion_matrix.png` | Confusion matrix for face-based model |
| Full-image ResNet50 confusion matrix | `outputs/confusion_matrices/full_image_resnet50_confusion_matrix.png` | Confusion matrix for full-image visual model |

---

## 10. Important Tables and Where to Find Them

| Table/File | File Location | Description |
|---|---|---|
| Final result summary | `outputs/metrics/final_result_summary.csv` | Main comparison of all models |
| Text baseline summary | `outputs/metrics/text_baseline_summary.csv` | Comparison of Logistic Regression and Linear SVM |
| Logistic Regression report | `outputs/metrics/tfidf_logreg_classification_report.csv` | Class-level precision, recall, and F1-score |
| Linear SVM report | `outputs/metrics/tfidf_linear_svm_classification_report.csv` | Class-level precision, recall, and F1-score |
| Face ResNet18 report | `outputs/metrics/face_resnet18_classification_report.csv` | Class-level face-model results |
| Full-image ResNet50 report | `outputs/metrics/full_image_resnet50_classification_report.csv` | Class-level full-image model results |

---

## 11. Group Member Work Division

| Part | Responsibility | Main Files |
|---|---|---|
| Part 1 | Data collection and understanding | `00_download_msctd.py`, `config.py`, image setup |
| Part 2 | Data cleaning and dataset preparation | `01_prepare_msctd_ende_manifest.py`, `04_check_image_availability.py`, `dataset_utils.py` |
| Part 3 | Feature engineering | `05_extract_faces.py`, `07_augmentation_robustness.py`, `face_detection.py`, `augmentations.py` |
| Part 4 | Model building and training | `02_train_text_baselines.py`, `06_train_face_model.py`, `08_train_full_image_model.py`, `11_train_distilbert_text.py`, `models.py`, `train_utils.py` |
| Part 5 | Evaluation and development | `03_make_result_summary.py`, `evaluate_utils.py`, `09_train_fusion_model.py`, `10_clip_multimodal_extra_credit.py`, final result interpretation |

---

## 12. Known Issues and Notes

1. The MSCTD GitHub ZIP does not directly contain all En-De raw image folders. The image archives must be downloaded separately.
2. Face extraction may take several hours on CPU because it processes all images.
3. Neural model training is faster on GPU. On CPU, ResNet training can take many hours.
4. The face model may produce lower accuracy because MSCTD sentiment labels are utterance-level labels, not individual face-expression labels.
5. The full-image model may struggle because the frozen ResNet50 backbone is not deeply fine-tuned on MSCTD sentiment.
6. Text models currently perform best because the English utterances contain stronger direct sentiment cues.
7. Fusion and CLIP scripts are included as development hooks for future extension.

---

## 13. How to Reproduce Final Results Quickly

After images are downloaded and placed correctly, run:

```cmd
set PYTHONPATH=%CD%

python scripts\01_prepare_msctd_ende_manifest.py
python scripts\04_check_image_availability.py
python scripts\02_train_text_baselines.py
python scripts\05_extract_faces.py
python scripts\06_train_face_model.py
python scripts\07_augmentation_robustness.py
python scripts\08_train_full_image_model.py
python scripts\03_make_result_summary.py
```

The final result table will be saved at:

```text
outputs/metrics/final_result_summary.csv
```

---

## 14. Final Submission Checklist

Before submission, check that the following are included:

```text
scripts/
src/
requirements.txt
README.md
outputs/metrics/
outputs/confusion_matrices/
outputs/figures/
report PDF
appendix PDF/pages
weekly logs
GitHub commit evidence
```

Large files such as raw images and model checkpoints may be excluded if the submission size is too large, but the README must clearly explain how to download images and regenerate outputs.

---
