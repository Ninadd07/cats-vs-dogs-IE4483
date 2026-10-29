"""Lightweight functional checks; no training or accuracy evaluation.

Run from the repository root with python -m datapreprocessing.verify.
This supporting script also checks the audit's original-file fingerprints.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch
import torchvision
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from .dataloader import CatsDogsDataLoader, TestDataset, load_rgb, resolve_data_dir
from .dataset_analysis import DEFAULT_OUTPUT, audit_dataset, fingerprint, output_directory
from .transforms import build_eval_transform, build_train_transform
from .visualize import generate_figure


def check(condition, message, results):
    if not condition:
        raise AssertionError(message)
    print("PASS:", message, flush=True)
    results.append(message)


def first_batches(root, mode, workers):
    # Reconstruct immediately before consuming: single-process augmentation uses
    # the global torch RNG, while workers each have their own seeded RNG.
    data = CatsDogsDataLoader(root, mode=mode, batch_size=4, seed=73, num_workers=workers)
    iterator = iter(data.train_loader)
    batches = [next(iterator), next(iterator)]
    del iterator
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    root = resolve_data_dir(args.data_dir)
    out = output_directory(args.output_dir, root)
    manifest_path = out / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run dataset_analysis first to create dataset_manifest.json")
    results = []
    torch.set_num_threads(2)
    for mode in ("baseline", "enhanced"):
        data = CatsDogsDataLoader(root, mode=mode, batch_size=4, seed=42)
        check(data.get_class_to_idx() == {"cat": 0, "dog": 1} and
              data.get_classes() == ["cat", "dog"], f"{mode}: class mapping", results)
        check(data.get_loaders() == (data.get_train_loader(), data.get_val_loader(),
                                    data.get_test_loader()), f"{mode}: loader accessors", results)
        for split, loader in zip(("train", "val", "test"), data.get_loaders()):
            images, targets = next(iter(loader))
            check(images.shape == (4, 3, 224, 224) and images.dtype == torch.float32
                  and bool(torch.isfinite(images).all()),
                  f"{mode}/{split}: finite float32 RGB batch (4, 3, 224, 224)", results)
            if split != "test":
                check(targets.dtype == torch.int64 and targets.shape == (4,) and
                      set(targets.tolist()) <= {0, 1}, f"{mode}/{split}: integer labels", results)
            else:
                check(list(targets) == [p.name for p in data.test_dataset.image_paths[:4]],
                      f"{mode}/test: original filenames in sorted order", results)
        check(isinstance(data.train_loader.sampler, RandomSampler) and
              all(isinstance(loader.sampler, SequentialSampler)
                  for loader in (data.val_loader, data.test_loader)),
              f"{mode}: only training is shuffled", results)
        expected_ops = (["Resize", "RandomHorizontalFlip", "ToTensor", "Normalize"]
                        if mode == "baseline" else
                        ["RandomResizedCrop", "RandomHorizontalFlip", "RandomRotation",
                         "ColorJitter", "ToTensor", "Normalize"])
        check([type(op).__name__ for op in data.train_transform.transforms] == expected_ops,
              f"{mode}: intended training transformation order", results)
        check([type(op).__name__ for op in data.eval_transform.transforms] ==
              ["Resize", "ToTensor", "Normalize"] and
              torch.equal(data.val_dataset[0][0], data.val_dataset[0][0]) and
              torch.equal(data.test_dataset[0][0], data.test_dataset[0][0]),
              f"{mode}: deterministic evaluation without augmentation", results)
        for expected_label in (0, 1):
            index = data.train_dataset.targets.index(expected_label)
            check(all(data.train_dataset[index][1] == expected_label for _ in range(3)),
                  f"{mode}: augmentation preserves label {expected_label}", results)
        for workers in (0, 2):
            a = first_batches(root, mode, workers)
            b = first_batches(root, mode, workers)
            check(all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1])
                      for x, y in zip(a, b)),
                  f"{mode}: independent seeded runs agree with num_workers={workers}", results)
        configured = CatsDogsDataLoader(root, mode=mode, image_size=96, batch_size=2)
        check(next(iter(configured.train_loader))[0].shape == (2, 3, 96, 96),
              f"{mode}: configurable image and batch sizes", results)

    # Compare against the original loader without changing its source or caches.
    sys.dont_write_bytecode = True
    repo = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("teammate_loader", repo / "dataloader.py")
    teammate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teammate)
    original = teammate.CatsDogsDataLoader(str(root), batch_size=4)
    baseline = CatsDogsDataLoader(root, mode="baseline", batch_size=4)
    image = load_rgb(baseline.train_dataset.samples[0][0])
    torch.manual_seed(123)
    expected = original.train_transform(image)
    torch.manual_seed(123)
    actual = baseline.train_transform(image)
    check(torch.equal(expected, actual) and torch.equal(original.eval_transform(image),
                                                       baseline.eval_transform(image)),
          "Baseline tensors exactly match teammate transforms with a controlled seed", results)
    check(baseline.train_dataset.samples == original.train_dataset.samples and
          [p.name for p in baseline.test_dataset.image_paths] ==
          [Path(p).name for p in original.test_dataset.image_paths],
          "Original and new loader use the same sample ordering and labels", results)

    # Real forward passes establish model input compatibility, without gradients,
    # pretrained downloads, training, or classification accuracy measurements.
    from resnet import build_resnet_model
    from vit import build_vit_model
    batch = next(iter(baseline.val_loader))[0][:1]
    for builder in (build_resnet_model, build_vit_model):
        model = builder(pretrained=False).eval()
        with torch.no_grad():
            logits = model(batch)
        check(logits.shape == (1, 2) and bool(torch.isfinite(logits).all()),
              f"{builder.__name__}: forward input/output compatibility", results)
        del model

    with tempfile.TemporaryDirectory(dir=out) as temporary:
        fixture = Path(temporary) / "fixture"
        for split in ("train", "val"):
            for label in ("cat", "dog"):
                directory = fixture / split / label
                directory.mkdir(parents=True)
                Image.new("L", (80, 60), 120).save(directory / "gray.png")
                Image.new("RGBA", (64, 96), (100, 50, 200, 128)).save(directory / "alpha.png")
        (fixture / "test").mkdir()
        Image.new("L", (80, 60), 100).save(fixture / "test" / "2.PNG")
        (fixture / "test" / "ignored.txt").write_text("metadata")
        (fixture / "train" / "cat" / "broken.jpg").write_bytes(b"not an image")
        report, _ = audit_dataset(fixture)
        check(len(report["unreadable_images"]) == 1 and
              any("ignored.txt" in p for p in report["unexpected_files"]),
              "Audit reports corrupt images and unsupported files without stopping", results)
        dataset = TestDataset(fixture / "test", build_eval_transform())
        check(len(dataset) == 1 and dataset[0][0].shape == (3, 224, 224),
              "Grayscale test images become RGB; non-image files are filtered", results)
        for path in (fixture / "train" / "dog").iterdir():
            check(build_train_transform()(load_rgb(path)).shape == (3, 224, 224),
                  f"RGB conversion accepts {path.name}", results)
        first = generate_figure(root, Path(temporary) / "a", seed=42)
        second = generate_figure(root, Path(temporary) / "b", seed=42)
        check(first.read_bytes() == second.read_bytes(),
              "Figure generation reproduces identical PNG bytes with a fixed seed", results)

    before = json.loads(manifest_path.read_text(encoding="utf-8"))
    now_paths = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    changed = []
    for index, (path, value) in enumerate(before.items(), start=1):
        if not (root / path).is_file() or fingerprint(root / path) != value:
            changed.append(path)
        if index % 5000 == 0:
            print(f"Preservation: rechecked {index}/{len(before)} files...", flush=True)
    check(set(before) == now_paths and not changed,
          "All original dataset files retain SHA-256, size and modification time; no files added/deleted", results)
    (out / "verification_results.json").write_text(json.dumps({
        "python": sys.executable, "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__, "Pillow": PIL.__version__,
        "passed": len(results), "checks": results,
    }, indent=2), encoding="utf-8")
    print(f"All {len(results)} checks passed.")


if __name__ == "__main__":
    main()
