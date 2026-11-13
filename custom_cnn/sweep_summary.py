"""Collect custom-CNN sweep runs into one table. Execute with python -m custom_cnn.sweep_summary."""

import argparse
import csv
import json
from pathlib import Path

DEFAULT_SWEEP = Path(__file__).resolve().parent / "outputs" / "sweep"
FIELDS = ["lr", "weight_decay", "dropout", "val_accuracy", "macro_f1", "val_loss", "best_epoch",
          "epochs_run", "early_stopped", "status"]


def collect(sweep_dir):
    rows = []
    for run in sorted(p for p in Path(sweep_dir).iterdir() if p.is_dir()):
        config_path = run / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        history_path = run / "training_history.csv"
        history = list(csv.DictReader(history_path.open())) if history_path.exists() else []
        row = dict(lr=config["lr"], weight_decay=config["weight_decay"], dropout=config["dropout"],
                   epochs_run=len(history), status="running")
        metrics_path = run / "validation_metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            row.update(val_accuracy=m["accuracy"], macro_f1=m["macro_f1"], val_loss=m["loss"],
                       best_epoch=m["epoch"], status="done",
                       early_stopped=len(history) < config["epochs"])
        elif history:
            best = max(history, key=lambda h: float(h["val_accuracy"]))
            row.update(val_accuracy=float(best["val_accuracy"]), best_epoch=int(best["epoch"]))
        rows.append(row)
    return sorted(rows, key=lambda r: -r.get("val_accuracy", -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", default=str(DEFAULT_SWEEP))
    args = parser.parse_args()
    if not Path(args.sweep_dir).is_dir():
        raise SystemExit(f"No sweep runs yet in {args.sweep_dir}")
    rows = collect(args.sweep_dir)
    if not rows:
        raise SystemExit(f"No runs found in {args.sweep_dir}")
    out = Path(args.sweep_dir) / "sweep_results.csv"
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'lr':>7} {'wd':>7} {'drop':>5} {'val acc':>8} {'F1':>7} {'best ep':>8} "
          f"{'epochs':>7}  status")
    for r in rows:
        acc = f"{r['val_accuracy']:.4f}" if "val_accuracy" in r else "-"
        f1 = f"{r['macro_f1']:.4f}" if "macro_f1" in r else "-"
        print(f"{r['lr']:>7g} {r['weight_decay']:>7g} {r['dropout']:>5g} {acc:>8} {f1:>7} "
              f"{r.get('best_epoch', '-'):>8} {r['epochs_run']:>7}  {r['status']}"
              + (" (early stop)" if r.get("early_stopped") else ""))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
