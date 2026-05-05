# Import random to randomly choose one degradation method.
import random

# Import OpenCV for image resizing operations used in frequency-style degradation.
import cv2

# Import numpy for converting images into arrays and adding noise.
import numpy as np

# Import PIL tools for image processing:
# Image is used to convert arrays back into image format.
# ImageEnhance is used for brightness adjustment.
# ImageFilter is used for blur effects.
from PIL import Image, ImageEnhance, ImageFilter


def degrade_brightness(img, factor=0.45):
    # Reduce image brightness.
    # factor < 1 makes the image darker; factor > 1 makes it brighter.
    return ImageEnhance.Brightness(img).enhance(factor)


def degrade_blur(img, radius=2.0):
    # Apply Gaussian blur to the image.
    # A larger radius creates stronger blur.
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def degrade_noise(img, sigma=25):
    # Convert the image into a numpy array so random noise can be added.
    arr = np.array(img).astype(np.float32)

    # Generate Gaussian noise with mean 0 and standard deviation sigma.
    noise = np.random.normal(0, sigma, arr.shape)

    # Add noise to the image and keep pixel values within the valid range 0–255.
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    # Convert the noisy array back into a PIL image.
    return Image.fromarray(arr)


def degrade_rotation(img, angle=15):
    # Rotate the image by a fixed angle.
    # This simulates spatial distortion or slight camera/viewpoint changes.
    return img.rotate(angle)


def degrade_frequency_lowpass(img):
    # Convert image into a numpy array for OpenCV processing.
    arr = np.array(img)

    # Get image height and width.
    h, w = arr.shape[:2]

    # Downsample the image to one quarter of its original width and height.
    # This removes fine details and high-frequency information.
    small = cv2.resize(
        arr,
        (max(1, w // 4), max(1, h // 4)),
        interpolation=cv2.INTER_LINEAR
    )

    # Upsample the image back to its original size.
    # The restored image looks lower quality/smoother, simulating low-pass filtering.
    restored = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    # Convert the processed array back into a PIL image.
    return Image.fromarray(restored)


def random_degrade(img):
    # Randomly select one degradation type.
    # This helps create a varied degraded dataset for robustness testing.
    choice = random.choice([
        "brightness",
        "blur",
        "noise",
        "rotation",
        "frequency"
    ])

    # Map each degradation name to its corresponding function,
    # then apply the selected degradation to the input image.
    return {
        "brightness": degrade_brightness,
        "blur": degrade_blur,
        "noise": degrade_noise,
        "rotation": degrade_rotation,
        "frequency": degrade_frequency_lowpass
    }[choice](img)