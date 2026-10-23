"""OpenCV preprocessing and data augmentation."""

from __future__ import annotations

import cv2
import numpy as np

from defect_vision import IMAGE_SIZE

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def preprocess(img: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Any image -> grayscale, fixed size, contrast-equalised (CLAHE), float32 in [0, 1].

    CLAHE evens out uneven lighting across the part, which otherwise dominates what a model learns.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = _CLAHE.apply(img)
    return img.astype(np.float32) / 255.0


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Label-preserving augmentation for surface defects: flips, 90-degree rotations, brightness and sensor noise."""
    if rng.random() < 0.5:
        img = np.fliplr(img)
    if rng.random() < 0.5:
        img = np.flipud(img)
    img = np.rot90(img, k=int(rng.integers(0, 4)))
    img = img * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05)
    img = img + rng.normal(0, 0.02, img.shape)
    return np.clip(img, 0.0, 1.0).astype(np.float32)
