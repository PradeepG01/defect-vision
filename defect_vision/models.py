"""Model architectures: a compact CNN for edge devices, a transfer-learning ResNet-18 and a convolutional autoencoder."""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class DefectCNN(nn.Module):
    """~100k-parameter CNN: small enough for real-time inference on edge hardware next to the production line."""

    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64), _block(64, 96))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(96, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def resnet18_transfer(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Transfer learning: reuse ImageNet features, adapt the stem to 1-channel input and replace the classifier.

    With freeze_backbone=True only the new head trains first (fast, little data); unfreeze later to fine-tune.
    pretrained=True downloads weights from the torchvision model hub on first use.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    rgb_weight = model.conv1.weight.data
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = rgb_weight.mean(dim=1, keepdim=True)  # grayscale = average of RGB filters
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head is always trainable
    return model


class ConvAutoencoder(nn.Module):
    """Trained only on defect-free parts. Unseen defect types reconstruct poorly, so high reconstruction
    error flags anomalies the classifier was never trained on."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
