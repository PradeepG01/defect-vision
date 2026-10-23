"""PyTorch datasets, stratified splitting and class balancing."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from defect_vision.preprocess import augment, preprocess


class SurfaceDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool = False, seed: int = 0) -> None:
        self.images = np.stack([preprocess(img) for img in images])
        self.labels = labels
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.train:
            img = augment(img, self.rng)
        return torch.from_numpy(np.ascontiguousarray(img))[None], int(self.labels[idx])


def stratified_split(images, labels, val_fraction=0.2, seed=0):
    """Keeps the same class proportions in train and validation (important with rare defect classes)."""
    idx_train, idx_val = train_test_split(
        np.arange(len(labels)), test_size=val_fraction, stratify=labels, random_state=seed
    )
    return (images[idx_train], labels[idx_train]), (images[idx_val], labels[idx_val])


def balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Samples each class equally often per epoch so rare defects are not drowned out by 'ok' parts."""
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(labels), replacement=True)


def make_loaders(train_ds: SurfaceDataset, val_ds: SurfaceDataset, batch_size: int = 64):
    train = DataLoader(train_ds, batch_size=batch_size, sampler=balanced_sampler(train_ds.labels))
    val = DataLoader(val_ds, batch_size=256, shuffle=False)
    return train, val
