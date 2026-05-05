# Import Path to create and manage folder/file paths in a clean way.
from pathlib import Path

# Import zipfile so the downloaded MSCTD ZIP file can be extracted.
import zipfile

# Import requests to download the MSCTD GitHub ZIP file from the internet.
import requests

# Import shutil for folder operations such as deleting or moving folders.
import shutil


# Official GitHub ZIP download link for the MSCTD repository.
# This downloads the code, text files, sentiment labels, and image-index files.
URL = "https://github.com/XL2248/MSCTD/archive/refs/heads/main.zip"


# Define the local folder where raw dataset files will be stored.
# The final path will be data/raw/
RAW = Path("data/raw")

# Create data/raw/ if it does not already exist.
# parents=True creates missing parent folders; exist_ok=True avoids errors if the folder already exists.
RAW.mkdir(parents=True, exist_ok=True)


# Define the local path where the downloaded ZIP file will be saved.
zip_path = RAW / "MSCTD-main.zip"

# Define the final folder name for the extracted MSCTD repository.
out_dir = RAW / "MSCTD"


# If the MSCTD folder already exists, do not download it again.
# This prevents wasting time and avoids overwriting existing data.
if out_dir.exists():
    print("MSCTD repository already exists:", out_dir)

else:
    # If the MSCTD folder does not exist, download it from GitHub.
    print("Downloading official MSCTD GitHub ZIP...")

    # Send a request to download the ZIP file.
    # timeout=120 means the request will stop if it takes longer than 120 seconds.
    r = requests.get(URL, timeout=120)

    # Stop the program if the download failed, for example due to no internet.
    r.raise_for_status()

    # Save the downloaded ZIP file content into data/raw/MSCTD-main.zip.
    zip_path.write_bytes(r.content)

    # Open the ZIP file and extract everything into data/raw/.
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW)

    # After extraction, GitHub usually creates a folder called MSCTD-main.
    extracted = RAW / "MSCTD-main"

    # If an old MSCTD output folder somehow exists, delete it before renaming.
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Rename MSCTD-main to MSCTD so the rest of the project can use a stable folder name.
    extracted.rename(out_dir)

    # Confirm that the repository has been downloaded and prepared.
    print("MSCTD ready at", out_dir)


# Print a reminder because the main GitHub ZIP does not contain all raw image folders directly.
# The MSCTD repository explains that En-De images are downloaded separately from MSCTD_data.
print("\nIMPORTANT IMAGE STEP:")

# This tells the user what the downloaded repository contains.
print("The repo contains text/labels/image-index files. Raw En-De images are separate links in:")

# This is the GitHub page where the image download links are listed.
print("https://github.com/XL2248/MSCTD/tree/main/MSCTD_data")

# These are the three image archives needed for the English-German part of the dataset.
print("Download: ende_train, test, dev. Extract into:")

# Required folder for training images.
print("data/raw/MSCTD_images/train_images")

# Required folder for validation/dev images.
print("data/raw/MSCTD_images/dev_images")

# Required folder for test images.
print("data/raw/MSCTD_images/test_images")