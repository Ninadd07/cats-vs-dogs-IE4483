"""Validation metrics for a saved custom CNN; no unlabelled test scoring."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from datapreprocessing import CatsDogsDataLoader
from .model import build_custom_cnn
from .train import DEFAULT_OUTPUT, atomic_write, get_device, load_checkpoint


def classification_metrics(confusion):
    """Rows are actual classes, columns predicted classes; undefined ratios = 0."""
    matrix = torch.as_tensor(confusion, dtype=torch.float64)
    if matrix.shape != (2, 2) or (matrix < 0).any() or matrix.sum() <= 0:
        raise ValueError("Expected a nonnegative, nonempty 2x2 confusion matrix")
    true_positive = matrix.diag()
    precision = true_positive / matrix.sum(0).clamp_min(1)
    recall = true_positive / matrix.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-15)
    return {"accuracy": float(true_positive.sum() / matrix.sum()),
            "confusion_matrix": matrix.to(torch.int64).tolist(),
            "per_class": {name: {"label": index, "precision": float(precision[index]),
                                  "recall": float(recall[index]), "f1": float(f1[index]),
                                  "support": int(matrix[index].sum())}
                          for index, name in enumerate(("cat", "dog"))},
            "macro_f1": float(f1.mean())}


def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    confusion = torch.zeros(2, 2, dtype=torch.int64)
    loss_sum, total = 0.0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation logits or loss")
            predictions = logits.argmax(1)
            confusion += torch.bincount((2 * labels + predictions).cpu(), minlength=4).reshape(2, 2)
            loss_sum += loss.item() * labels.numel()
            total += labels.numel()
    if not total:
        raise ValueError("Cannot evaluate an empty dataset")
    return dict(classification_metrics(confusion), loss=loss_sum / total, samples=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_OUTPUT / "best.pth"))
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", help="Default: checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint)
    config, device = saved["config"], get_device(args.device)
    data = CatsDogsDataLoader(data_dir=args.data_dir or config.get("data_dir"),
                             mode="enhanced", image_size=config["image_size"],
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=device.type == "cuda", seed=config["seed"])
    model = build_custom_cnn(dropout=config["dropout"], device=device)
    model.load_state_dict(saved["model_state"])
    metrics = evaluate_model(model, data.val_loader, device)
    metrics.update(checkpoint=str(Path(args.checkpoint).resolve()), epoch=saved["epoch"],
                   class_mapping=saved["class_mapping"])
    out = Path(args.output_dir or Path(args.checkpoint).resolve().parent).resolve()
    if out.is_relative_to(data.data_dir):
        raise ValueError("Output directory must be outside the original dataset")
    atomic_write(out / "validation_metrics.json", lambda p: p.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
