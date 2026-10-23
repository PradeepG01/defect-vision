# Defect Vision

An **industrial visual-inspection pipeline** built with **PyTorch** and **OpenCV**: it classifies surface defects
(scratch, spot, crack), flags **previously unseen anomalies** with an autoencoder, **localises** the defect with
bounding boxes and a segmentation mask, exports to **ONNX / TorchScript** for edge devices, and serves results
through a **FastAPI** inspection endpoint.

![CI](https://github.com/PradeepG01/defect-vision/actions/workflows/ci.yml/badge.svg)

## Pipeline

```
image ─► OpenCV preprocessing (grayscale, resize, CLAHE contrast equalisation)
      ├─► DefectCNN (79k params) ───────────────► class + confidence        (supervised)
      ├─► Conv autoencoder trained on good parts ─► reconstruction error     (unsupervised anomaly detection)
      └─► background subtraction + morphology ───► mask + bounding boxes    (localisation / segmentation)
                                                    ▼
                                         verdict: pass / reject
```

| Topic | Where |
|---|---|
| Synthetic, imbalanced surface-defect dataset generator (70% good parts) | `defect_vision/synth.py` |
| CLAHE preprocessing; flips/rotations/brightness/noise augmentation | `defect_vision/preprocess.py` |
| Stratified split + `WeightedRandomSampler` class balancing | `defect_vision/data.py` |
| Compact CNN, ResNet-18 transfer learning (frozen backbone, 1-channel stem), conv autoencoder | `defect_vision/models.py` |
| Training: AdamW, OneCycle LR, label smoothing, early stopping on macro-F1, CSV/JSON experiment logs | `defect_vision/train.py` |
| Precision / recall / F1 per class and confusion matrix | `defect_vision/evaluate.py` |
| OpenCV defect localisation (residual thresholding, morphology, contours) | `defect_vision/localize.py` |
| ONNX + TorchScript export, int8 post-training static quantization (calibrated), ONNX parity test | `defect_vision/export.py` |
| FastAPI `/inspect` endpoint | `defect_vision/api.py` |

## Results (synthetic dataset, 4,000 images, CPU, ~2 min)

| Model | Metric | Value |
|---|---|---|
| DefectCNN (validation, 800 images) | Accuracy | 99.75% |
| | Macro-F1 | 0.996 |
| | Scratch recall (hardest class) | 97.4% |
| Autoencoder anomaly detector | Defect recall at 1% false-alarm rate | 90.4% |
| Edge export | TorchScript fp32 → int8 (static PTQ) | 354 KB → 123 KB (2.9x smaller), same validation accuracy |
| | ONNX size | ~312 KB |

> The data is synthetic, so these numbers show the pipeline works; expect lower scores on real factory images.
> To train on real data (e.g. MVTec AD or NEU surface defects), replace `make_dataset` with an image-folder loader.

## Run

```bash
pip install -r requirements.txt
python -m defect_vision.train --samples 4000 --epochs 12 --out artifacts   # writes metrics.csv, run.json, weights
python -m defect_vision.export --weights artifacts/defect_cnn.pt --out artifacts
uvicorn defect_vision.api:app --port 8000
curl -F "file=@part.png" localhost:8000/inspect
```

## Tests

```bash
pytest -q    # 12 tests, ~30 s on CPU: data, augmentation, sampler, models, training, anomaly detection,
             # localisation, int8 quantization, ONNX parity and the API
```
