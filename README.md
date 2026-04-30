# EEEM068 Human Sentiment Analysis

This package implements the EEEM068 Human Sentiment Analysis workflow for the official MSCTD English-German subset.

## Quick start on Windows VS Code
```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\00_download_msctd.py
python scripts\01_prepare_msctd_ende_manifest.py
python scripts\02_train_text_baselines.py
```

## Dataset note
The public MSCTD GitHub repository contains En-De text, labels, image-index resources, and code. Your local inspection showed raw images were not included in the ZIP, so the core runnable baseline uses the official En-De text and sentiment labels. Visual scripts are included and guarded: place raw images in `data/raw/MSCTD_images/` or `data/raw/MSCTD/images/` before running face/full-image/fusion experiments.
