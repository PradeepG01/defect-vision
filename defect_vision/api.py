"""Inference API: upload an image of a part, get the defect class, confidence, anomaly score and defect boxes.

    uvicorn defect_vision.api:app --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from defect_vision import CLASSES
from defect_vision.localize import bounding_boxes, defect_mask
from defect_vision.models import ConvAutoencoder, DefectCNN
from defect_vision.preprocess import preprocess

ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
app = FastAPI(title="Defect Vision", version="1.0.0")


def load_models(artifacts: Path = ARTIFACTS):
    classifier = DefectCNN(len(CLASSES))
    classifier.load_state_dict(torch.load(artifacts / "defect_cnn.pt", map_location="cpu"))
    classifier.eval()
    ae_ckpt = torch.load(artifacts / "autoencoder.pt", map_location="cpu")
    autoencoder = ConvAutoencoder()
    autoencoder.load_state_dict(ae_ckpt["state_dict"])
    autoencoder.eval()
    return classifier, autoencoder, float(ae_ckpt["threshold"])


_models = None


def models():
    global _models
    if _models is None:
        _models = load_models()
    return _models


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@torch.no_grad()
def inspect(raw: np.ndarray) -> dict:
    classifier, autoencoder, threshold = models()
    x = torch.from_numpy(preprocess(raw))[None, None]
    probs = torch.softmax(classifier(x), dim=1)[0]
    label = int(probs.argmax())
    score = float(((autoencoder(x) - x) ** 2).max())
    gray = raw if raw.ndim == 2 else cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    boxes = bounding_boxes(defect_mask(gray)) if CLASSES[label] != "ok" or score > threshold else []
    return {
        "label": CLASSES[label],
        "confidence": round(float(probs[label]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs, strict=True)},
        "anomaly_score": round(score, 5),
        "is_anomaly": score > threshold,
        "verdict": "reject" if CLASSES[label] != "ok" or score > threshold else "pass",
        "boxes": boxes,
    }


@app.post("/inspect")
async def inspect_endpoint(file: UploadFile = File(...)) -> dict:
    data = np.frombuffer(await file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="file is not a readable image")
    return inspect(img)
