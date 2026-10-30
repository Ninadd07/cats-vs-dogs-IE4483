"""Plot recorded learning curves; no curves are generated without history."""

import argparse
import csv
import math
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"


def plot_history(history_path, output_dir=None):
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"No training logs available: {history_path}. Run training before plotting.")
        return []
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        print("No completed epochs in training history; no figures generated.")
        return []
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != sorted(set(epochs)):
        raise ValueError("History epochs must be unique and increasing")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as error:
        raise RuntimeError("Install plotting dependencies: pip install -r custom_cnn/requirements.txt") from error
    out = Path(output_dir or history_path.parent)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300}):
        for metric, ylabel in (("loss", "Cross-entropy loss"), ("accuracy", "Accuracy")):
            figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
            for split, label, color in (("train", "Training", "#1768ac"),
                                        ("val", "Validation", "#d45d00")):
                values = [float(row[f"{split}_{metric}"]) for row in rows]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("History contains non-finite values")
                axis.plot(epochs, values, label=label, color=color, linewidth=2)
            axis.set(xlabel="Epoch", ylabel=ylabel, title=f"Custom CNN: {metric}")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            if metric == "accuracy":
                axis.set_ylim(0, 1)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
            path = out / f"{metric}_curve.png"
            figure.savefig(path)
            plt.close(figure)
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default=str(DEFAULT_OUTPUT / "training_history.csv"))
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    for path in plot_history(args.history, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
