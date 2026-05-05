# EEEM068 Human Sentiment Analysis — Complete Code

Run on Windows / VS Code:

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\00_download_msctd.py
python scripts\01_prepare_msctd_ende_manifest.py
python scripts\02_train_text_baselines.py
python scripts\03_make_result_summary.py
python scripts\04_check_image_availability.py
```

Download separate En-De images from the MSCTD_data README links: `ende_train`, `test`, and `dev`.
Extract them as:
`data/raw/MSCTD_images/train_images/*.jpg`
`data/raw/MSCTD_images/dev_images/*.jpg`
`data/raw/MSCTD_images/test_images/*.jpg`

Then rerun:
```cmd
python scripts\01_prepare_msctd_ende_manifest.py
python scripts\04_check_image_availability.py
python scripts\05_extract_faces.py
python scripts\06_train_face_model.py
python scripts\07_augmentation_robustness.py
python scripts\08_train_full_image_model.py
```

Optional:
```cmd
python scripts\11_train_distilbert_text.py
python scripts\03_make_result_summary.py
```
