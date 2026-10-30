"""Small CPU smoke checks; no training jobs or persistent test outputs."""

import argparse
import math
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from datapreprocessing import CatsDogsDataLoader, resolve_data_dir
from .evaluate import classification_metrics, evaluate_model
from .model import build_custom_cnn, parameter_counts
from .train import (CLASS_MAPPING, DEFAULTS, atomic_write, capture_rng,
                    load_checkpoint, restore_rng, run_epoch, seed_everything)
from .visualize import plot_history


def verify(data_dir=None):
    passed = 0

    def check(condition, message):
        nonlocal passed
        if not condition:
            raise AssertionError(message)
        passed += 1
        print("PASS:", message)

    torch.set_num_threads(2)
    seed_everything(19, deterministic=True)
    model = build_custom_cnn()
    with torch.inference_mode():
        logits = model.eval()(torch.randn(2, 3, 224, 224))
    check(logits.shape == (2, 2) and bool(torch.isfinite(logits).all()),
          "Finite two-logit output for RGB images at 224px")
    check(isinstance(model.classifier[-1], nn.Linear) and
          not any(isinstance(layer, (nn.Softmax, nn.Sigmoid)) for layer in model.modules()),
          "Raw classification logits")
    check(parameter_counts(model) == {"total": 1206370, "trainable": 1206370},
          "Expected parameter count")

    images, labels = torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2,
                        generator=torch.Generator().manual_seed(19))
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULTS["lr"],
                                  weight_decay=DEFAULTS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=DEFAULTS["scheduler_factor"], patience=DEFAULTS["scheduler_patience"])
    model.train()
    before = model.features[0][0].weight.detach().clone()
    loss = nn.CrossEntropyLoss()(model(images[:2]), labels[:2])
    loss.backward()
    check(bool(torch.isfinite(loss)) and all(
        p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
        "Cross-entropy and finite gradients")
    optimizer.step()
    scheduler.step(loss.item())
    check(not torch.equal(before, model.features[0][0].weight), "AdamW parameter update")

    # Temporary checkpoint exercises serialization without invoking the trainer.
    with tempfile.TemporaryDirectory(prefix="ie4483-cnn-smoke-") as temporary:
        checkpoint = Path(temporary) / "checkpoint.pth"
        rng = capture_rng((loader,))
        atomic_write(checkpoint, lambda path: torch.save({
            "format_version": 1, "class_mapping": CLASS_MAPPING,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "rng_state": rng,
        }, path))
        saved = load_checkpoint(checkpoint)
        restored = build_custom_cnn()
        restored.load_state_dict(saved["model_state"])
        restored_optimizer = torch.optim.AdamW(restored.parameters())
        restored_optimizer.load_state_dict(saved["optimizer_state"])
        restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer)
        restored_scheduler.load_state_dict(saved["scheduler_state"])
        with torch.inference_mode():
            check(torch.equal(model.eval()(images), restored.eval()(images)),
                  "Model checkpoint round trip")
        check(restored_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
              and len(restored_optimizer.state) == len(optimizer.state)
              and restored_scheduler.state_dict() == scheduler.state_dict(),
              "Optimizer and scheduler state restoration")
        restore_rng(saved["rng_state"], (loader,))
        expected = torch.rand(4)
        restore_rng(saved["rng_state"], (loader,))
        check(torch.equal(expected, torch.rand(4)), "Checkpoint RNG restoration")
        check(plot_history(Path(temporary) / "missing.csv") == [],
              "No learning curves without training history")

    # Unequal batch sizes catch averaging errors without additional optimization.
    scores = torch.tensor([[2., -1.], [-2., 3.], [1., 0.]])
    targets = torch.tensor([0, 1, 1])
    batches = DataLoader(TensorDataset(scores, targets), batch_size=2)
    result = run_epoch(nn.Identity(), batches, nn.CrossEntropyLoss(), torch.device("cpu"))
    expected_loss = nn.CrossEntropyLoss()(scores, targets).item()
    check(math.isclose(result["loss"], expected_loss, rel_tol=1e-6) and
          result["accuracy"] == 2 / 3, "Sample-weighted validation averages")
    metrics = evaluate_model(nn.Identity(), batches, torch.device("cpu"))
    check(metrics["confusion_matrix"] == [[1, 0], [1, 1]] and metrics["samples"] == 3,
          "Evaluation confusion matrix and sample count")
    known = classification_metrics([[8, 2], [1, 9]])
    check(known["accuracy"] == .85 and math.isclose(known["macro_f1"], (16/19 + 18/21)/2),
          "Accuracy and macro-F1")

    try:
        root = resolve_data_dir(data_dir)
    except FileNotFoundError:
        if data_dir is not None:
            raise
        print("SKIP: real dataset interface (dataset unavailable)")
    else:
        data = CatsDogsDataLoader(root, batch_size=2, seed=19)
        for split, batch_loader in (("train", data.train_loader), ("val", data.val_loader)):
            batch_images, batch_labels = next(iter(batch_loader))
            check(batch_images.shape == (2, 3, 224, 224) and
                  bool(torch.isfinite(batch_images).all()) and batch_labels.dtype == torch.int64,
                  f"{split}: two-sample loader interface")
        check(data.get_class_to_idx() == CLASS_MAPPING, "Cat/dog label mapping")
    print(f"Passed {passed} smoke checks. No training jobs or persistent outputs created.")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    verify(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
