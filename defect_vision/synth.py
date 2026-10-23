"""Synthetic industrial-surface dataset generator.

Real inspection datasets (e.g. MVTec AD, NEU surface defects) are large downloads; this generator produces
a comparable, controllable dataset so the full pipeline runs anywhere in seconds. Each image is a textured
metal-like surface; defective images get a scratch, a spot or a crack. The class mix is deliberately
imbalanced (most parts are good), as on a real production line.
"""

from __future__ import annotations

import cv2
import numpy as np

from defect_vision import CLASSES, IMAGE_SIZE

DEFAULT_MIX = {"ok": 0.7, "scratch": 0.1, "spot": 0.1, "crack": 0.1}


def _surface(rng: np.random.Generator, size: int) -> np.ndarray:
    base = rng.uniform(90, 150)
    noise = cv2.GaussianBlur(rng.normal(0, 18, (size, size)).astype(np.float32), (0, 0), sigmaX=rng.uniform(0.8, 2.0))
    grad = np.linspace(-1, 1, size, dtype=np.float32)[None, :] * rng.uniform(-15, 15)
    brushed = cv2.blur(rng.normal(0, 6, (size, size)).astype(np.float32), (9, 1))  # directional texture
    return base + noise + grad + brushed


def _scratch(img, rng, size):
    p1 = rng.integers(0, size, 2)
    angle = rng.uniform(0, np.pi)
    length = rng.uniform(0.5, 0.9) * size
    p2 = (p1 + length * np.array([np.cos(angle), np.sin(angle)])).astype(int)
    cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), float(img.mean() + rng.uniform(55, 85)), 1)


def _spot(img, rng, size):
    center = tuple(int(v) for v in rng.integers(10, size - 10, 2))
    axes = tuple(int(v) for v in rng.integers(3, 7, 2))
    cv2.ellipse(img, center, axes, float(rng.uniform(0, 180)), 0, 360, float(img.mean() - rng.uniform(55, 80)), -1)


def _crack(img, rng, size):
    point = rng.integers(8, size - 8, 2).astype(float)
    color = float(img.mean() - rng.uniform(60, 85))
    direction = rng.uniform(0, 2 * np.pi)
    for _ in range(int(rng.integers(6, 10))):
        direction += rng.uniform(-0.9, 0.9)  # jagged, branching path
        nxt = np.clip(point + 5 * np.array([np.cos(direction), np.sin(direction)]), 0, size - 1)
        cv2.line(img, tuple(int(v) for v in point), tuple(int(v) for v in nxt), color, 2)
        point = nxt


DRAW = {"scratch": _scratch, "spot": _spot, "crack": _crack}


def make_image(label: str, rng: np.random.Generator, size: int = IMAGE_SIZE) -> np.ndarray:
    img = _surface(rng, size)
    if label != "ok":
        DRAW[label](img, rng, size)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_dataset(n: int, seed: int = 0, mix: dict[str, float] | None = None, size: int = IMAGE_SIZE):
    """Returns (images uint8 [N,H,W], labels int64 [N])."""
    mix = mix or DEFAULT_MIX
    rng = np.random.default_rng(seed)
    names = list(mix)
    labels = rng.choice(len(names), size=n, p=[mix[k] for k in names])
    images = np.stack([make_image(names[i], rng, size) for i in labels])
    label_ids = np.array([CLASSES.index(names[i]) for i in labels], dtype=np.int64)
    return images, label_ids
