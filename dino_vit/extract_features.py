"""Run the frozen DINOv2 ViT-B/14 over train/val/test once and cache the features to dino_vit/features/.

Example (from anywhere):
    python dino_vit/extract_features.py
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from common import DEFAULT_DATA_DIR, FEATURES_DIR, MODEL, TestImages, build_transform, get_device, load_backbone


@torch.no_grad()
def extract(model, loader, device):
    feats, extra = [], []
    for images, y in loader:
        feats.append(model(images.to(device)).float().cpu().numpy())
        extra.append(y.numpy())
    return np.concatenate(feats), np.concatenate(extra)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out-dir", type=Path, default=FEATURES_DIR)
    args = p.parse_args()

    device = get_device()
    model = load_backbone(args.model, args.img_size).to(device)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} params, dim={model.num_features}, "
          f"device={device}")

    tf = build_transform(args.img_size)
    out = args.out_dir / args.model.split(".")[0]
    out.mkdir(parents=True, exist_ok=True)
    common = dict(batch_size=args.batch_size, num_workers=args.workers)

    for split in ("train", "val"):
        ds = datasets.ImageFolder(args.data_dir / split, tf)
        # ImageFolder sorts class folders alphabetically: cat=0, dog=1, matching the submission format.
        assert ds.class_to_idx == {"cat": 0, "dog": 1}, ds.class_to_idx
        start = time.time()
        x, y = extract(model, DataLoader(ds, **common), device)
        np.savez(out / f"{split}.npz", x=x, y=y)
        print(f"{split}: {x.shape} in {time.time() - start:.0f}s ({len(ds) / (time.time() - start):.0f} img/s)")

    ds = TestImages(args.data_dir / "test", tf)
    x, ids = extract(model, DataLoader(ds, **common), device)
    np.savez(out / "test.npz", x=x, ids=ids)
    print(f"test: {x.shape}; saved to {out}/")


if __name__ == "__main__":
    main()
