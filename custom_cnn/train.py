"""Train/resume the custom CNN. Execute with python -m custom_cnn.train."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
import uuid

import numpy as np
import torch
import torchvision
from torch import nn

from datapreprocessing import CatsDogsDataLoader
from .model import build_custom_cnn, parameter_counts

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"
CLASS_MAPPING = {"cat": 0, "dog": 1}
HISTORY_FIELDS = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
                  "learning_rate", "duration_seconds", "new_best", "train_ce", "train_kd"]
DEFAULTS = dict(epochs=30, batch_size=32, image_size=224, num_workers=0, seed=42,
                lr=0.001, weight_decay=1e-4, dropout=0.3, scheduler=True,
                scheduler_factor=0.5, scheduler_patience=3,
                early_stopping_patience=7, min_delta=0.0, deterministic=False,
                teacher_probe=None, kd_alpha=0.5, kd_temperature=4.0)


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def get_device(request="auto"):
    if request != "auto":
        return torch.device(request)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def atomic_write(path, writer):
    """Replace only after the complete file has been written on the same volume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(descriptor)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_history(path, history):
    def write(temporary):
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(history)
    atomic_write(path, write)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("class_mapping") != CLASS_MAPPING:
        raise ValueError("Unsupported checkpoint format or class mapping")
    return checkpoint


def capture_rng(loaders):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(),
            *numpy_state[2:]], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            "loaders": [loader.generator.get_state() if loader.generator is not None else None
                        for loader in loaders]}


def restore_rng(state, loaders):
    random.setstate(state["python"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved RNG configuration")
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    for loader, saved in zip(loaders, state["loaders"]):
        if saved is not None:
            if loader.generator is None:
                raise ValueError("Resume requires seeded DataLoader generators")
            loader.generator.set_state(saved)


def run_epoch(model, loader, criterion, device, optimizer=None, teacher=None,
              kd_alpha=0.5, kd_temperature=4.0):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    ce_sum, kd_sum = 0.0, 0.0
    if teacher is not None:
        teacher.eval()
    with torch.enable_grad() if training else torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            ce, kd = loss, logits.new_zeros(())
            if training and teacher is not None:
                from .distillation import distillation_loss
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss, ce, kd = distillation_loss(logits, teacher_logits, labels,
                                                  kd_alpha, kd_temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training/evaluation loss")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            ce_sum += ce.item() * labels.size(0)
            kd_sum += kd.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    if not total:
        raise ValueError("Cannot process an empty dataset")
    return {"loss": loss_sum / total, "accuracy": correct / total,
            "ce": ce_sum / total, "kd": kd_sum / total}


def fit(model, train_loader, val_loader, config, output_dir, device, resume=None, teacher=None):
    """Train complete epochs. Checkpoint history is authoritative on resumption."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config["scheduler_factor"],
        patience=config["scheduler_patience"]) if config["scheduler"] else None)
    criterion = nn.CrossEntropyLoss()
    history, start_epoch, best_accuracy, best_epoch = [], 1, -1.0, 0
    early = {"best_loss": float("inf"), "bad_epochs": 0}
    best_snapshot = None
    run_id = str(uuid.uuid4())
    if resume:
        saved = load_checkpoint(resume)
        for name in ("latest.pth", "best.pth"):
            existing = out / name
            if existing.exists() and load_checkpoint(existing)["run_id"] != saved["run_id"]:
                raise FileExistsError("Output contains a different run; choose another --output-dir")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if (saved["scheduler_state"] is None) != (scheduler is None):
            raise ValueError("Scheduler configuration differs from the checkpoint")
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler_state"])
        start_epoch = saved["epoch"] + 1
        best_accuracy, best_epoch = saved["best_val_accuracy"], saved["best_epoch"]
        early, history, run_id = saved["early_stopping"], saved["history"], saved["run_id"]
        if len(history) != saved["epoch"] or history[-1]["epoch"] != saved["epoch"]:
            raise ValueError("Checkpoint history is inconsistent with its epoch")
        best_snapshot = saved.get("best_checkpoint", saved)
        if best_snapshot["run_id"] != run_id or best_snapshot["epoch"] != best_epoch:
            raise ValueError("Saved best model is inconsistent with the resume checkpoint")
        atomic_write(out / "best.pth", lambda p: torch.save(best_snapshot, p))
        restore_rng(saved["rng_state"], (train_loader, val_loader))
    elif any((out / name).exists() for name in ("latest.pth", "best.pth", "training_history.csv")):
        raise FileExistsError("Run output already exists; use --resume or a new --output-dir")
    if config["epochs"] < start_epoch:
        raise ValueError(f"--epochs must be at least {start_epoch} to continue")
    if config["early_stopping_patience"] and early["bad_epochs"] >= config["early_stopping_patience"]:
        raise ValueError("This run already reached its early-stopping criterion")
    atomic_write(out / "config.json", lambda p: p.write_text(
        json.dumps(config, indent=2), encoding="utf-8"))
    write_history(out / "training_history.csv", history)
    for epoch in range(start_epoch, config["epochs"] + 1):
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train = run_epoch(model, train_loader, criterion, device, optimizer, teacher,
                          config.get("kd_alpha", 0.5), config.get("kd_temperature", 4.0))
        val = run_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val["loss"])
        new_best = val["accuracy"] > best_accuracy
        if new_best:
            best_accuracy, best_epoch = val["accuracy"], epoch
        if val["loss"] < early["best_loss"] - config["min_delta"]:
            early = {"best_loss": val["loss"], "bad_epochs": 0}
        else:
            early["bad_epochs"] += 1
        history.append(dict(epoch=epoch, train_loss=train["loss"], train_accuracy=train["accuracy"],
                            val_loss=val["loss"], val_accuracy=val["accuracy"],
                            learning_rate=learning_rate, duration_seconds=time.perf_counter() - started,
                            new_best=new_best, train_ce=train["ce"], train_kd=train["kd"]))
        checkpoint = dict(format_version=1, run_id=run_id, epoch=epoch,
                          model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                          scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                          best_val_accuracy=best_accuracy, best_epoch=best_epoch,
                          early_stopping=early.copy(), config=config, seed=config["seed"],
                          class_mapping=CLASS_MAPPING, history=history,
                          rng_state=capture_rng((train_loader, val_loader)))
        if new_best:
            atomic_write(out / "best.pth", lambda p: torch.save(checkpoint, p))
            # Read an independent CPU snapshot: live state_dict tensors change
            # during later updates. Embedding this snapshot also makes latest.pth
            # self-contained if a cluster stops between the two atomic writes.
            best_snapshot = load_checkpoint(out / "best.pth")
        latest = dict(checkpoint, best_checkpoint=best_snapshot)
        atomic_write(out / "latest.pth", lambda p: torch.save(latest, p))
        write_history(out / "training_history.csv", history)
        print(f"Epoch {epoch}/{config['epochs']} | train {train['loss']:.4f}/{train['accuracy']:.4f}"
              f" | val {val['loss']:.4f}/{val['accuracy']:.4f} | lr {learning_rate:g}"
              f" | {history[-1]['duration_seconds']:.1f}s" + (" | best" if new_best else ""), flush=True)
        if config["early_stopping_patience"] and early["bad_epochs"] >= config["early_stopping_patience"]:
            print("Early stopping: validation loss has not improved.", flush=True)
            break
    return history


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="Checkpoint from a trusted run")
    parser.add_argument("--teacher-probe", help="Exported DINOv2 binary probe (.npz); enables KD")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 or mps")
    for name in ("epochs", "batch_size", "image_size", "num_workers", "seed",
                 "scheduler_patience", "early_stopping_patience"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("lr", "weight_decay", "dropout", "scheduler_factor", "min_delta",
                 "kd_alpha", "kd_temperature"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--scheduler", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args(argv)


def main():
    args = get_args()
    saved = load_checkpoint(args.resume) if args.resume else None
    config = dict(DEFAULTS)
    if saved:
        config.update(saved["config"])
    for name in DEFAULTS:
        value = getattr(args, name)
        if value is not None:
            if saved and name not in ("epochs", "num_workers") and value != config[name]:
                raise ValueError(f"Cannot change --{name.replace('_', '-')} while resuming")
            config[name] = value
    if any(config[k] < 1 for k in ("epochs", "batch_size")) or config["image_size"] < 16:
        raise ValueError("Require positive epochs/batch size and image size >= 16")
    if any(config[k] < 0 for k in ("num_workers", "scheduler_patience", "early_stopping_patience",
                                  "weight_decay", "min_delta")) or config["lr"] <= 0:
        raise ValueError("Invalid optimizer, patience or worker setting")
    if not 0 < config["scheduler_factor"] < 1:
        raise ValueError("scheduler-factor must be between zero and one")
    if not 0 <= config["kd_alpha"] <= 1 or not config["kd_temperature"] > 0:
        raise ValueError("Require kd-alpha in [0, 1] and positive kd-temperature")
    teacher = None
    if config["teacher_probe"]:
        path = Path(config["teacher_probe"]).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if saved and config.get("teacher_probe_sha256") != digest:
            raise ValueError("Teacher probe changed since checkpoint; cannot resume")
        config.update(teacher_probe=str(path), teacher_probe_sha256=digest)
    device = get_device(args.device)
    seed_everything(config["seed"], config["deterministic"])
    data = CatsDogsDataLoader(data_dir=args.data_dir or config.get("data_dir"),
                             mode="enhanced", seed=config["seed"],
                             batch_size=config["batch_size"], image_size=config["image_size"],
                             num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    seed_everything(config["seed"], config["deterministic"])
    config.update(data_dir=str(data.data_dir), preprocessing="enhanced", num_classes=2,
                  class_mapping=data.get_class_to_idx(), device=str(device),
                  torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                  numpy_version=np.__version__, train_transform=repr(data.train_transform),
                  eval_transform=repr(data.eval_transform))
    output = args.output_dir or (str(Path(args.resume).resolve().parent) if args.resume else str(DEFAULT_OUTPUT))
    output_path = Path(output).resolve()
    if output_path.is_relative_to(data.data_dir):
        raise ValueError("Output directory must be outside the original dataset")
    model = build_custom_cnn(dropout=config["dropout"], device=device)
    if config["teacher_probe"]:
        from .distillation import load_teacher
        rng = capture_rng((data.train_loader, data.val_loader))
        teacher = load_teacher(config["teacher_probe"], config["image_size"], device)
        # Teacher construction must not change the student's augmentation/dropout RNG.
        restore_rng(rng, (data.train_loader, data.val_loader))
    print(f"Device: {device} | parameters: {parameter_counts(model)} | "
          f"train/val: {len(data.train_dataset)}/{len(data.val_dataset)}", flush=True)
    fit(model, data.train_loader, data.val_loader, config, output_path, device, args.resume, teacher)


if __name__ == "__main__":
    main()
