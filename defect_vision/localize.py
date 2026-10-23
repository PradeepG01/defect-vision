"""Classical OpenCV defect localisation: where on the part is the defect?

Subtracting a heavily blurred copy removes the smooth background texture and lighting gradient; what remains
are sharp local deviations. Thresholding gives a binary segmentation mask and contours give bounding boxes.
"""

from __future__ import annotations

import cv2
import numpy as np


def defect_mask(img: np.ndarray, k: float = 3.5) -> np.ndarray:
    """Returns a uint8 mask (255 = defect pixel)."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=6)
    residual = np.abs(gray - background)
    threshold = residual.mean() + k * residual.std()
    mask = (residual > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # join broken crack segments
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # drop isolated noise pixels


def bounding_boxes(mask: np.ndarray, min_area: int = 6) -> list[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": float(area)})
    return sorted(boxes, key=lambda b: b["w"] * b["h"], reverse=True)
