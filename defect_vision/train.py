"""Training for the defect classifier and the anomaly-detection autoencoder.

    python -m defect_vision.train --samples 4000 --epochs 12 --out artifacts

Every run writes artifacts/metrics.csv (per-epoch loss/accuracy/F1) and artifacts/run.json (config + final
metrics) so experiments can be compared; best weights are kept by validation macro-F1 with early stopping.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from defect_vision import CLASSES
from defect_vision.data import SurfaceDataset, make_loaders, stratified_split
from defect_vision.evaluate import classification_report, predict
from defect_vision.models import ConvAutoencoder, DefectCNN, count_parameters
from defect_vision.synth import make_dataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_classifier(model, train_loader, val_loader, epochs=12, lr=3e-3, patience=4, log_path: Path | None = None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_f1, best_state, stale = -1.0, None, 0
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(y)
            n += len(y)
        y_true, y_pred, _ = predict(model, val_loader)
        report = classification_report(y_true, y_pred)
        rows.append({"epoch": epoch, "train_loss": round(total / n, 4), "val_accuracy": report["accuracy"],
                     "val_macro_f1": report["macro_f1"]})
        print(f"epoch {epoch:2d}  loss {total / n:.4f}  val_acc {report['accuracy']:.3f}  val_f1 {report['macro_f1']:.3f}")
        if report["macro_f1"] > best_f1:
            best_f1, stale = report["macro_f1"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if log_path:
        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return model, best_f1


def train_autoencoder(ok_images: np.ndarray, epochs: int = 15, lr: float = 2e-3) -> ConvAutoencoder:
    ds = SurfaceDataset(ok_images, np.zeros(len(ok_images), dtype=np.int64), train=True)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = ConvAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, _ in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), x)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def anomaly_scores(model: ConvAutoencoder, images: np.ndarray) -> np.ndarray:
    model.eval()
    ds = SurfaceDataset(images, np.zeros(len(images), dtype=np.int64))
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    return ((model(x) - x) ** 2).flatten(1).amax(dim=1).numpy()  # worst pixel error: defects are local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    started = time.time()

    images, labels = make_dataset(args.samples, seed=args.seed)
    (x_tr, y_tr), (x_val, y_val) = stratified_split(images, labels, seed=args.seed)
    train_loader, val_loader = make_loaders(SurfaceDataset(x_tr, y_tr, train=True, seed=args.seed), SurfaceDataset(x_val, y_val))

    model = DefectCNN(len(CLASSES))
    print(f"DefectCNN parameters: {count_parameters(model):,}")
    model, _ = train_classifier(model, train_loader, val_loader, epochs=args.epochs, log_path=args.out / "metrics.csv")
    y_true, y_pred, _ = predict(model, val_loader)
    report = classification_report(y_true, y_pred)
    torch.save(model.state_dict(), args.out / "defect_cnn.pt")

    # Anomaly detector: trained on 'ok' parts only; threshold = 99th percentile of 'ok' validation error.
    autoencoder = train_autoencoder(x_tr[y_tr == 0])
    ok_scores = anomaly_scores(autoencoder, x_val[y_val == 0])
    bad_scores = anomaly_scores(autoencoder, x_val[y_val != 0])
    threshold = float(np.percentile(ok_scores, 99))
    anomaly = {
        "threshold": threshold,
        "defect_recall": float((bad_scores > threshold).mean()),
        "false_alarm_rate": float((ok_scores > threshold).mean()),
    }
    torch.save({"state_dict": autoencoder.state_dict(), "threshold": threshold}, args.out / "autoencoder.pt")

    run = {"config": {k: str(v) for k, v in vars(args).items()}, "classifier": report, "anomaly_detection": anomaly,
           "train_seconds": round(time.time() - started, 1)}
    (args.out / "run.json").write_text(json.dumps(run, indent=2))
    print(json.dumps({"classifier": {k: report[k] for k in ("accuracy", "macro_f1")}, "anomaly": anomaly}, indent=2))


if __name__ == "__main__":
    main()
