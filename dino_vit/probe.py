"""Fit a logistic-regression / kNN probe on the cached DINOv2 features.

Selects C on the val set (train is used only for fitting), then optionally writes a submission CSV.

Example (from anywhere):
    python dino_vit/probe.py --submission dino_vit/submissions/submission_dinov2.csv
    python dino_vit/probe.py --train-per-class 10     # few-shot
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from common import FEATURES_DIR, MODEL


def load(path):
    d = np.load(path)
    return d["x"], d["y"] if "y" in d else d["ids"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=FEATURES_DIR / MODEL.split(".")[0])
    p.add_argument("--cs", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    p.add_argument("--train-per-class", type=int, default=None, help="few-shot: first N per class (seeded)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--submission", default=None)
    args = p.parse_args()

    feat = args.features
    xtr, ytr = load(feat / "train.npz")
    xva, yva = load(feat / "val.npz")

    if args.train_per_class:
        rng = np.random.RandomState(args.seed)
        keep = np.concatenate([rng.choice(np.where(ytr == c)[0], args.train_per_class, replace=False) for c in (0, 1)])
        xtr, ytr = xtr[keep], ytr[keep]
    print(f"train={xtr.shape} val={xva.shape}")

    scaler = StandardScaler().fit(xtr)
    xtr_s, xva_s = scaler.transform(xtr), scaler.transform(xva)

    results = {}
    best = (0.0, None)
    for c in args.cs:
        clf = LogisticRegression(C=c, max_iter=2000).fit(xtr_s, ytr)
        acc = clf.score(xva_s, yva)
        results[f"logreg_C={c:g}"] = acc
        print(f"logreg C={c:<8g} val acc={acc:.4f}")
        if acc > best[0]:
            best = (acc, c)
    for k in args.ks:
        # cosine kNN on L2-normalised features
        norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
        acc = KNeighborsClassifier(n_neighbors=k).fit(norm(xtr), ytr).score(norm(xva), yva)
        results[f"knn_k={k}"] = acc
        print(f"kNN k={k:<3d}        val acc={acc:.4f}")

    print(f"best logreg: C={best[1]:g} val acc={best[0]:.4f}")
    (feat / "probe_results.json").write_text(json.dumps(
        {"train_size": len(ytr), "best_logreg_C": best[1], "best_logreg_val_acc": best[0], "all": results}, indent=2))

    if args.submission:
        xte, ids = load(feat / "test.npz")
        clf = LogisticRegression(C=best[1], max_iter=2000).fit(xtr_s, ytr)
        pred = clf.predict(scaler.transform(xte))
        with open(args.submission, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label"])
            w.writerows(zip(ids.tolist(), pred.tolist()))
        print(f"wrote {args.submission}: {int(pred.sum())} dog / {int((1 - pred).sum())} cat")


if __name__ == "__main__":
    main()
