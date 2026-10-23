"""Evaluation: per-class precision/recall/F1 and the confusion matrix (accuracy alone hides missed defects)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from defect_vision import CLASSES


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y_true, y_pred, probs = [], [], []
    for x, y in loader:
        p = torch.softmax(model(x), dim=1)
        y_true.append(y.numpy())
        y_pred.append(p.argmax(1).numpy())
        probs.append(p.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(probs)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "per_class": {
            name: {"precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f), 3), "support": int(s)}
            for name, p, r, f, s in zip(CLASSES, precision, recall, f1, support, strict=True)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
