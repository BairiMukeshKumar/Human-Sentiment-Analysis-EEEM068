# Import Path for safe file and folder path handling.
from pathlib import Path

# Import re for extracting image IDs from image-index text lines.
import re

# Import PIL Image to open image files.
from PIL import Image

# Import torch to create label tensors.
import torch

# Import Dataset so we can create a custom PyTorch dataset class.
from torch.utils.data import Dataset

# Import torchvision transforms for resizing, augmentation and normalization.
from torchvision import transforms


def read_lines(path: Path):
    # Check whether the required file exists.
    # If not, stop the script with a clear error message.
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    # Read the file as text and return a list of lines.
    # encoding="utf-8" supports normal text characters.
    # errors="ignore" prevents the script from crashing on unusual characters.
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def parse_image_index_lines(lines):
    """Flatten MSCTD dialogue-level image index lines into utterance-level image IDs."""

    # Create an empty list to store all extracted image IDs.
    flat = []

    # Each line may contain one or more image IDs for a dialogue.
    for line in lines:
        # Extract all numbers from the line and add them to the flat list.
        # This converts dialogue-level image-index lines into sample-level image IDs.
        flat.extend(re.findall(r"\d+", line))

    # Return the complete flattened list of image IDs.
    return flat


def find_image_path(image_root: Path, split: str, image_id: str):
    # Map dataset split names to the expected image folder names.
    # train uses train_images, validation uses dev_images, and test uses test_images.
    folder_map = {
        "train": "train_images",
        "val": "dev_images",
        "test": "test_images"
    }

    # Build the expected image folder path for the current split.
    folder = image_root / folder_map[split]

    # First try the most direct file paths, such as 0.jpg or 15.png.
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = folder / f"{image_id}{ext}"

        # If the image exists, return its full path as a string.
        if p.exists():
            return str(p)

    # If the image was not found directly, search inside nested folders.
    # This is useful if the ZIP extraction created an extra folder level.
    if folder.exists():
        for ext in ["jpg", "jpeg", "png", "bmp"]:
            matches = list(folder.rglob(f"{image_id}.{ext}"))

            # Return the first matching image file if found.
            if matches:
                return str(matches[0])

    # If no matching image is found, return an empty string.
    return ""


class MSCTDImageDataset(Dataset):
    # Custom PyTorch dataset for loading MSCTD images and sentiment labels.

    def __init__(self, dataframe, image_col="image_path", label_col="label", transform=None):
        # Store the input dataframe and reset row indexing.
        self.df = dataframe.reset_index(drop=True)

        # Column name containing image file paths.
        self.image_col = image_col

        # Column name containing sentiment labels.
        self.label_col = label_col

        # Optional image transformations such as resize, augmentation and normalization.
        self.transform = transform

    def __len__(self):
        # Return the total number of samples in the dataset.
        return len(self.df)

    def __getitem__(self, idx):
        # Get one row from the dataframe.
        row = self.df.iloc[idx]

        # Read the image path from the selected image column.
        image_path = str(row[self.image_col])

        # Stop if the image path is empty or invalid.
        # This prevents unclear errors during model training.
        if image_path in ["", "nan", "None"]:
            raise FileNotFoundError("image_path is empty. Download/extract MSCTD images first.")

        # Open the image and convert it to RGB format.
        image = Image.open(image_path).convert("RGB")

        # Read the sentiment label and convert it to integer.
        label = int(row[self.label_col])

        # Apply transformations if provided.
        if self.transform:
            image = self.transform(image)

        # Return the transformed image and label tensor.
        return image, torch.tensor(label, dtype=torch.long)


def get_image_transforms(img_size=224, train=True):
    # Create image transformations for training, validation and testing.
    # img_size=224 is suitable for ResNet-style pretrained models.

    if train:
        # Training transforms include augmentation to improve robustness.
        return transforms.Compose([
            # Resize all images to the same size.
            transforms.Resize((img_size, img_size)),

            # Randomly flip images horizontally.
            transforms.RandomHorizontalFlip(p=0.5),

            # Randomly rotate images by up to 10 degrees.
            transforms.RandomRotation(10),

            # Randomly adjust brightness, contrast and saturation.
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10
            ),

            # Convert image to a PyTorch tensor.
            transforms.ToTensor(),

            # Normalize using ImageNet mean and standard deviation.
            # This matches pretrained ResNet expectations.
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # Validation/test transforms do not use random augmentation.
    # This keeps evaluation consistent and repeatable.
    return transforms.Compose([
        # Resize image to the required model input size.
        transforms.Resize((img_size, img_size)),

        # Convert image to a PyTorch tensor.
        transforms.ToTensor(),

        # Normalize using ImageNet statistics.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])