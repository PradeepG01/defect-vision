"""Fast tests for the whole pipeline (seconds on CPU). Full training lives in defect_vision.train."""

import io

import cv2
import numpy as np
import pytest
import torch

from defect_vision import CLASSES, IMAGE_SIZE
from defect_vision.data import SurfaceDataset, balanced_sampler, make_loaders, stratified_split
from defect_vision.export import export_all
from defect_vision.localize import bounding_boxes, defect_mask
from defect_vision.models import ConvAutoencoder, DefectCNN, count_parameters, resnet18_transfer
from defect_vision.preprocess import augment, preprocess
from defect_vision.synth import make_dataset, make_image
from defect_vision.train import anomaly_scores, set_seed, train_autoencoder, train_classifier


def test_synthetic_dataset_shapes_and_imbalance():
    images, labels = make_dataset(500, seed=1)
    assert images.shape == (500, IMAGE_SIZE, IMAGE_SIZE) and images.dtype == np.uint8
    counts = np.bincount(labels, minlength=len(CLASSES))
    assert counts[0] > counts[1:].max() * 3  # 'ok' dominates, like a real line


def test_preprocess_handles_colour_and_resizes():
    colour = np.random.default_rng(0).integers(0, 255, (128, 96, 3), dtype=np.uint8)
    out = preprocess(colour)
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_augment_keeps_shape_and_range():
    img = preprocess(make_image("scratch", np.random.default_rng(0)))
    aug = augment(img, np.random.default_rng(1))
    assert aug.shape == img.shape and 0 <= aug.min() and aug.max() <= 1


def test_stratified_split_and_balanced_sampler():
    images, labels = make_dataset(400, seed=2)
    (_, y_tr), (_, y_val) = stratified_split(images, labels)
    assert abs((y_tr == 0).mean() - (y_val == 0).mean()) < 0.03
    torch.manual_seed(0)
    drawn = np.array(list(balanced_sampler(y_tr)))
    drawn_counts = np.bincount(y_tr[drawn], minlength=len(CLASSES))
    assert drawn_counts.min() / drawn_counts.max() > 0.6  # roughly equal classes after balancing


def test_model_output_shapes():
    x = torch.rand(4, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert DefectCNN(len(CLASSES))(x).shape == (4, len(CLASSES))
    assert ConvAutoencoder()(x).shape == x.shape
    assert count_parameters(DefectCNN(len(CLASSES))) < 150_000


def test_transfer_learning_freezes_backbone():
    model = resnet18_transfer(len(CLASSES), pretrained=False, freeze_backbone=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}
    assert model(torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE)).shape == (2, len(CLASSES))


@pytest.fixture(scope="module")
def trained():
    set_seed(0)
    images, labels = make_dataset(1200, seed=3)
    (x_tr, y_tr), (x_val, y_val) = stratified_split(images, labels, seed=0)
    loaders = make_loaders(SurfaceDataset(x_tr, y_tr, train=True), SurfaceDataset(x_val, y_val))
    model, f1 = train_classifier(DefectCNN(len(CLASSES)), *loaders, epochs=6)
    return model, f1, (x_tr, y_tr, x_val, y_val)


def test_classifier_learns(trained):
    _, f1, _ = trained
    assert f1 > 0.85


def test_autoencoder_scores_defects_higher(trained):
    _, _, (x_tr, y_tr, x_val, y_val) = trained
    ae = train_autoencoder(x_tr[y_tr == 0], epochs=8)
    ok = anomaly_scores(ae, x_val[y_val == 0])
    bad = anomaly_scores(ae, x_val[y_val != 0])
    assert np.median(bad) > np.median(ok)


def test_localisation_finds_the_spot():
    rng = np.random.default_rng(5)
    img = make_image("ok", rng)
    cv2.circle(img, (40, 20), 4, int(img.mean()) - 70, -1)
    boxes = bounding_boxes(defect_mask(img))
    assert boxes, "expected at least one defect box"
    b = boxes[0]
    assert b["x"] <= 40 <= b["x"] + b["w"] and b["y"] <= 20 <= b["y"] + b["h"]


def test_onnx_export_matches_pytorch(trained, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model, _, _ = trained
    model.eval()
    export_all(model, tmp_path)
    x = torch.rand(3, 1, IMAGE_SIZE, IMAGE_SIZE)
    session = ort.InferenceSession(str(tmp_path / "defect_cnn.onnx"))
    onnx_out = session.run(None, {"image": x.numpy()})[0]
    with torch.no_grad():
        torch_out = model(x).numpy()
    assert np.allclose(onnx_out, torch_out, atol=1e-4)


def test_int8_quantization_shrinks_model_and_keeps_predictions(trained, tmp_path):
    from defect_vision.export import calibration_batch, quantize_int8

    model, _, (_, _, x_val, _) = trained
    model.eval()
    sizes = export_all(model, tmp_path, calibration=calibration_batch(128))
    assert sizes["torchscript_int8"] < sizes["torchscript_fp32"] / 2

    quantized = quantize_int8(model, calibration_batch(128))
    x = torch.from_numpy(np.stack([preprocess(img) for img in x_val[:200]]))[:, None]
    with torch.no_grad():
        agreement = (model(x).argmax(1) == quantized(x).argmax(1)).float().mean().item()
    assert agreement > 0.95


def test_api_inspect_endpoint(trained, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from defect_vision import api

    model, _, (x_tr, y_tr, _, _) = trained
    ae = train_autoencoder(x_tr[y_tr == 0], epochs=3)
    torch.save(model.state_dict(), tmp_path / "defect_cnn.pt")
    torch.save({"state_dict": ae.state_dict(), "threshold": 0.05}, tmp_path / "autoencoder.pt")
    monkeypatch.setattr(api, "_models", api.load_models(tmp_path))

    _, png = cv2.imencode(".png", make_image("crack", np.random.default_rng(9)))
    client = TestClient(api.app)
    res = client.post("/inspect", files={"file": ("part.png", io.BytesIO(png.tobytes()), "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] in CLASSES and set(body["probabilities"]) == set(CLASSES)
    assert client.post("/inspect", files={"file": ("x.png", io.BytesIO(b"not an image"), "image/png")}).status_code == 400
