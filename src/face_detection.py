# Import Path to handle file and folder paths cleanly.
from pathlib import Path

# Import pandas to create and save the face extraction manifest CSV.
import pandas as pd

# Import PIL Image to open image files.
from PIL import Image

# Import tqdm to show a progress bar during face extraction.
from tqdm import tqdm

# Import MTCNN face detector from facenet-pytorch.
# MTCNN detects faces and returns cropped face tensors.
from facenet_pytorch import MTCNN

# Import torch for GPU/CPU detection and tensor operations.
import torch

# Import helper function to convert image tensors into PIL images.
from torchvision.transforms.functional import to_pil_image


def tensor_to_pil(tensor):
    # Convert a PyTorch image tensor into a PIL image.
    # This is needed because MTCNN returns face crops as tensors.

    # Move tensor to CPU and detach it from any computation graph.
    t = tensor.detach().cpu()

    # If tensor values are in the range [-1, 1], convert them to [0, 1].
    if t.min() < 0:
        t = (t + 1) / 2

    # Clamp values to the valid image range [0, 1] and convert to PIL image.
    return to_pil_image(torch.clamp(t, 0, 1))


def extract_faces_from_manifest(
    manifest_df,
    output_face_dir,
    output_csv,
    image_col="image_path",
    label_col="label",
    split_col="split",
    device=None,
    image_size=224,
    margin=20
):
    # This function detects and extracts faces from all images listed in the manifest.
    # It saves cropped face images and creates a face_manifest CSV.

    # If no device is provided, use GPU if available, otherwise use CPU.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the MTCNN face detector.
    # image_size=224 resizes detected faces to 224x224.
    # margin adds extra pixels around the detected face.
    # keep_all=True keeps multiple faces if an image contains more than one person.
    # post_process=True normalizes the output face tensors.
    mtcnn = MTCNN(
        image_size=image_size,
        margin=margin,
        keep_all=True,
        post_process=True,
        device=device
    )

    # Convert output folder path into Path object and create it if missing.
    output_face_dir = Path(output_face_dir)
    output_face_dir.mkdir(parents=True, exist_ok=True)

    # This list will store metadata rows for every detected or non-detected face case.
    rows = []

    # Loop through every image sample in the manifest.
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df)):

        # Read sample ID as string.
        # This avoids errors when sample IDs are like train_0, val_10, etc.
        sample_id = str(row["sample_id"])

        # Read the original image path.
        img_path = str(row[image_col])

        # Read the sentiment label.
        label = int(row[label_col])

        # Read dataset split: train, val, or test.
        split = str(row[split_col])

        try:
            # Open image and convert it to RGB.
            img = Image.open(img_path).convert("RGB")

            # Detect face bounding boxes and confidence scores.
            boxes, probs = mtcnn.detect(img)

        except Exception as e:
            # If image loading or detection fails, save an error row.
            rows.append({
                "sample_id": sample_id,
                "image_path": img_path,
                "face_path": "",
                "label": label,
                "split": split,
                "face_index": -1,
                "num_faces": 0,
                "face_detected": 0,
                "face_confidence": 0.0,
                "error": str(e)
            })

            # Move to the next image.
            continue

        # If no faces are detected, save a no-face row.
        if boxes is None or len(boxes) == 0:
            rows.append({
                "sample_id": sample_id,
                "image_path": img_path,
                "face_path": "",
                "label": label,
                "split": split,
                "face_index": -1,
                "num_faces": 0,
                "face_detected": 0,
                "face_confidence": 0.0,
                "error": ""
            })

            # Move to the next image.
            continue

        # Extract cropped face tensors from the image.
        faces = mtcnn(img)

        # If MTCNN detects boxes but cannot produce face tensors, save an error row.
        if faces is None:
            rows.append({
                "sample_id": sample_id,
                "image_path": img_path,
                "face_path": "",
                "label": label,
                "split": split,
                "face_index": -1,
                "num_faces": 0,
                "face_detected": 0,
                "face_confidence": 0.0,
                "error": "faces tensor none"
            })

            # Move to the next image.
            continue

        # If only one face is returned, MTCNN may return a 3D tensor.
        # Convert it into a 4D tensor so the loop works the same for one or many faces.
        if len(faces.shape) == 3:
            faces = faces.unsqueeze(0)

        # Count how many faces were extracted from this image.
        num_faces = faces.shape[0]

        # Create a split-specific folder for saved face crops.
        # Example: data/processed/faces/train/
        split_dir = output_face_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save every detected face crop separately.
        for face_idx in range(num_faces):

            # Define the output filename for this face crop.
            face_path = split_dir / f"sample_{sample_id}_face_{face_idx}.jpg"

            # Convert tensor face crop to PIL image and save it.
            tensor_to_pil(faces[face_idx]).save(face_path)

            # Store the MTCNN confidence score for this face if available.
            confidence = (
                float(probs[face_idx])
                if probs is not None and face_idx < len(probs)
                else 0.0
            )

            # Add one row to the face manifest for this detected face.
            rows.append({
                "sample_id": sample_id,
                "image_path": img_path,
                "face_path": str(face_path),
                "label": label,
                "split": split,
                "face_index": face_idx,
                "num_faces": num_faces,
                "face_detected": 1,
                "face_confidence": confidence,
                "error": ""
            })

    # Convert all face extraction rows into a DataFrame.
    face_df = pd.DataFrame(rows)

    # Create output CSV folder if needed.
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Save the face extraction manifest.
    face_df.to_csv(output_csv, index=False)

    # Return the DataFrame so calling scripts can inspect the results.
    return face_df