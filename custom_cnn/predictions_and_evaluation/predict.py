"""Verify the selected checkpoint, evaluate validation, then predict test labels."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import torch
import torchvision
from torch.utils.data import SequentialSampler

from datapreprocessing import CatsDogsDataLoader
from custom_cnn.evaluate import evaluate_model
from custom_cnn.model import build_custom_cnn, parameter_counts
from custom_cnn.train import CLASS_MAPPING, atomic_write, get_device, load_checkpoint

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "outputs/run_a/best.pth"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_id(filename):
    stem = Path(filename).stem
    if not re.fullmatch(r"[0-9]+", stem):
        raise ValueError(f"Expected an integer test filename, found {filename!r}")
    return int(stem)


def write_csv(path, fields, rows):
    def write(temporary):
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    atomic_write(path, write)


def predict(checkpoint=DEFAULT_CHECKPOINT, data_dir=None, output_dir=DEFAULT_OUTPUT,
            batch_size=32, num_workers=2, device="auto", sample_submission=None):
    checkpoint = Path(checkpoint).resolve()
    saved = load_checkpoint(checkpoint)
    config = saved["config"]
    if saved["epoch"] != 27 or saved["best_epoch"] != 27 or not math.isclose(
            saved["best_val_accuracy"], 0.9632, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Stage 5 requires the epoch-27 best checkpoint with recorded accuracy 0.9632")
    if config["class_mapping"] != CLASS_MAPPING or config["num_classes"] != 2 or config["image_size"] != 224:
        raise ValueError("Checkpoint input dimensions or numerical class mapping do not match this experiment")
    device = get_device(device)
    data = CatsDogsDataLoader(data_dir=data_dir or config.get("data_dir"),
                             batch_size=batch_size, image_size=config["image_size"],
                             mode=config["preprocessing"], seed=config["seed"],
                             num_workers=num_workers, pin_memory=device.type == "cuda")
    if repr(data.eval_transform) != config["eval_transform"]:
        raise ValueError("Evaluation transforms differ from the recorded training configuration")
    if len(data.val_dataset) != 5000 or len(data.test_dataset) != 500:
        raise ValueError("Expected exactly 5000 validation images and 500 test images")
    if not isinstance(data.val_loader.sampler, SequentialSampler):
        raise ValueError("Validation details require the loader's deterministic sample order")
    ids = [image_id(path.name) for path in data.test_dataset.image_paths]
    if len(set(ids)) != 500 or set(ids) != set(range(1, 501)):
        raise ValueError("Test filenames must contain each integer ID from 1 through 500 exactly once")
    if sample_submission:
        with Path(sample_submission).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["id", "label"]:
                raise ValueError("Sample submission must have exactly id,label columns")
            sample_ids = [int(row["id"]) for row in reader]
        if len(sample_ids) != 500 or set(sample_ids) != set(ids):
            raise ValueError("Sample-submission IDs do not match the real test images")
    out = Path(output_dir).resolve()
    if out.is_relative_to(data.data_dir):
        raise ValueError("Outputs must be outside the original dataset")
    if (out / "submission.csv").exists():
        raise FileExistsError("A submission already exists; select a new --output-dir to preserve it")
    model = build_custom_cnn(num_classes=2, dropout=config["dropout"], device=device)
    model.load_state_dict(saved["model_state"], strict=True)
    if parameter_counts(model) != {"total": 1206370, "trainable": 1206370}:
        raise ValueError("Unexpected model architecture")
    model.eval()
    original_hash = file_hash(checkpoint)

    # Capture per-image outputs during the shared evaluator's single pass.
    captured = []
    hook = model.register_forward_hook(lambda module, inputs, output: captured.append(output.detach().cpu()))
    try:
        metrics = evaluate_model(model, data.val_loader, device)
    finally:
        hook.remove()
    if not math.isclose(metrics["accuracy"], 0.9632, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"Validation accuracy {metrics['accuracy']} differs from 0.9632; test inference stopped")
    validation_logits = torch.cat(captured)
    validation_predictions = validation_logits.argmax(1).tolist()
    probabilities = validation_logits.softmax(1)
    if probabilities.shape != (5000, 2) or not torch.isfinite(probabilities).all():
        raise ValueError("Invalid validation probabilities")
    validation_rows = []
    for (path, label), probability, prediction in zip(
            data.val_dataset.samples, probabilities.tolist(), validation_predictions):
        validation_rows.append(dict(relative_path=Path(path).relative_to(data.data_dir).as_posix(),
                                    actual=label, prediction=prediction, confidence=max(probability),
                                    probability_dog=probability[1], correct=int(label == prediction)))
    correct = sum(row["correct"] for row in validation_rows)
    if correct != 4816:
        raise ValueError("Per-image validation details disagree with the expected 4816 correct predictions")
    metrics.update(correct=correct, incorrect=5000-correct, epoch=27, class_mapping=CLASS_MAPPING,
                   macro_precision=sum(m["precision"] for m in metrics["per_class"].values()) / 2,
                   macro_recall=sum(m["recall"] for m in metrics["per_class"].values()) / 2)
    print(f"Validation confirmed: {correct}/5000 correct ({metrics['accuracy']:.2%}); now predicting 500 test images.", flush=True)

    test_rows = []
    with torch.inference_mode():
        for images, filenames in data.test_loader:
            logits = model(images.to(device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite test logits")
            predictions = logits.argmax(1).cpu().tolist()
            probabilities = logits.softmax(1).cpu().tolist()
            for filename, probability, prediction in zip(filenames, probabilities, predictions):
                test_rows.append(dict(id=image_id(filename), filename=filename,
                                      label=prediction,
                                      confidence=max(probability), probability_dog=probability[1]))
    test_rows.sort(key=lambda row: row["id"])
    if len(test_rows) != 500 or [r["id"] for r in test_rows] != list(range(1, 501)):
        raise ValueError("Missing, duplicated or out-of-order test predictions")
    if any(row["label"] not in (0, 1) for row in test_rows):
        raise ValueError("Invalid predicted label")
    if file_hash(checkpoint) != original_hash or any(
            not torch.equal(value.cpu(), saved["model_state"][name])
            for name, value in model.state_dict().items()):
        raise RuntimeError("Checkpoint or model state changed during inference")

    out.mkdir(parents=True, exist_ok=True)
    atomic_write(out / "validation_metrics.json", lambda path: path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"))
    report = [dict(class_name=name, **entry) for name, entry in metrics["per_class"].items()]
    report.append(dict(class_name="macro average", label="", precision=metrics["macro_precision"],
                       recall=metrics["macro_recall"], f1=metrics["macro_f1"], support=5000))
    write_csv(out / "classification_report.csv", ["class_name", "label", "precision", "recall", "f1", "support"], report)
    write_csv(out / "validation_prediction_details.csv", list(validation_rows[0]), validation_rows)
    write_csv(out / "test_prediction_details.csv", list(test_rows[0]), test_rows)
    write_csv(out / "submission.csv", ["id", "label"],
              [dict(id=row["id"], label=row["label"]) for row in test_rows])
    with (out / "submission.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        submitted = list(reader)
        if reader.fieldnames != ["id", "label"] or len(submitted) != 500:
            raise RuntimeError("Written submission failed schema/row-count verification")
        if any(int(row["id"]) != source["id"] or int(row["label"]) != source["label"]
               for row, source in zip(submitted, test_rows)):
            raise RuntimeError("Written submission differs from test predictions")
    provenance = dict(checkpoint_sha256=original_hash, epoch=27, recorded_val_accuracy=0.9632,
                      torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                      device=str(device), data_dir=str(data.data_dir),
                      eval_transform=repr(data.eval_transform), submission_rows=500,
                      submission_sha256=file_hash(out / "submission.csv"),
                      test_predicted_counts={name: sum(r["label"] == label for r in test_rows)
                                             for name, label in CLASS_MAPPING.items()},
                      weights_unchanged=True, test_ground_truth_available=False)
    atomic_write(out / "provenance.json", lambda path: path.write_text(json.dumps(provenance, indent=2), encoding="utf-8"))
    print(json.dumps(metrics, indent=2))
    print(f"Verified submission saved: {out / 'submission.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-submission")
    predict(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
