"""Figures from real validation/test predictions; no training or label guessing."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"
CLASS_NAMES = {0: "Cat", 1: "Dog"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def show_image(axis, path, caption):
    with Image.open(path) as image:
        axis.imshow(image.convert("RGB"))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel(caption, fontsize=10, labelpad=7)
    for spine in axis.spines.values():
        spine.set_visible(False)


def analyze(output_dir=DEFAULT_OUTPUT, data_dir=None):
    out = Path(output_dir)
    metrics = json.loads((out / "validation_metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    root = Path(data_dir or provenance["data_dir"])
    if not (root / "val").is_dir() or not (root / "test").is_dir():
        raise FileNotFoundError("Supply --data-dir pointing to the original dataset")
    validation = read_csv(out / "validation_prediction_details.csv")
    tests = read_csv(out / "test_prediction_details.csv")
    with plt.rc_context({"font.size": 12, "savefig.dpi": 300}):
        figure, axis = plt.subplots(figsize=(6, 5), layout="constrained")
        matrix = metrics["confusion_matrix"]
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=2500)
        axis.set(xticks=[0, 1], yticks=[0, 1],
                 xticklabels=["Cat (0)", "Dog (1)"], yticklabels=["Cat (0)", "Dog (1)"],
                 xlabel="Predicted class", ylabel="Actual class",
                 title=f"Validation confusion matrix\nEpoch 27 | accuracy {metrics['accuracy']:.2%}")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row][column]), ha="center", va="center",
                          fontsize=25, color="white" if matrix[row][column] > 1250 else "#172433")
        figure.savefig(out / "confusion_matrix.png")
        plt.close(figure)

        groups = [(0, 0), (1, 1), (0, 1), (1, 0)]
        figure, axes = plt.subplots(2, 4, figsize=(13, 8))
        for column, (actual, predicted) in enumerate(groups):
            candidates = sorted((row for row in validation if int(row["actual"]) == actual
                                 and int(row["prediction"]) == predicted),
                                key=lambda row: (float(row["confidence"]), row["relative_path"]))
            for level, row_index in enumerate((0, -1)):
                axis = axes[level, column]
                if not candidates:
                    axis.text(.5, .5, "No examples", ha="center", va="center")
                    axis.axis("off")
                    continue
                row = candidates[row_index]
                status = "Correct" if actual == predicted else "Incorrect"
                caption = (f"Actual: {CLASS_NAMES[actual]} | Predicted: {CLASS_NAMES[predicted]}\n"
                           f"Confidence: {float(row['confidence']):.1%} | {status}")
                show_image(axis, root / row["relative_path"], caption)
                axis.set_title(("Lowest" if level == 0 else "Highest") + " confidence in group", fontsize=11)
        figure.suptitle("Validation examples | correct and incorrect predictions", fontsize=16)
        figure.text(.5, .015, "Selection: confidence extremes within each actual/predicted class group. Softmax confidence is uncalibrated.",
                    ha="center", fontsize=10)
        figure.subplots_adjust(top=.90, bottom=.12, left=.03, right=.98, hspace=.55, wspace=.20)
        figure.savefig(out / "validation_examples.png")
        plt.close(figure)

        figure, axes = plt.subplots(2, 4, figsize=(13, 8))
        for label in (0, 1):
            candidates = sorted((row for row in tests if int(row["label"]) == label),
                                key=lambda row: (float(row["confidence"]), int(row["id"])))
            for column, fraction in enumerate((0, 1/3, 2/3, 1)):
                axis = axes[label, column]
                if not candidates:
                    axis.text(.5, .5, "No predictions for this class", ha="center", va="center")
                    axis.axis("off")
                    continue
                row = candidates[round(fraction * (len(candidates)-1))]
                show_image(axis, root / "test" / row["filename"],
                           f"ID {row['id']} | Predicted: {CLASS_NAMES[label]} ({label})\n"
                           f"Confidence: {float(row['confidence']):.1%}")
        figure.suptitle("Unlabelled test images | model predictions", fontsize=16)
        figure.text(.5, .015, "Examples span the confidence range within each predicted class. Ground truth and test accuracy are unknown.",
                    ha="center", fontsize=10)
        figure.subplots_adjust(top=.90, bottom=.12, left=.03, right=.98, hspace=.42, wspace=.20)
        figure.savefig(out / "prediction_examples.png")
        plt.close(figure)
    print(f"Saved confusion_matrix.png, validation_examples.png and prediction_examples.png in {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--data-dir")
    analyze(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
