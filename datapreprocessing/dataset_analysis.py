"""Read-only full dataset audit. Run with python -m datapreprocessing.dataset_analysis."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataloader import CLASS_TO_IDX, IMG_EXTENSIONS, TEST_EXTENSIONS, resolve_data_dir

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"


def output_directory(path, data_dir):
    path = Path(path).resolve()
    if path == data_dir or path.is_relative_to(data_dir):
        raise ValueError("Outputs must be outside the original dataset")
    path.mkdir(parents=True, exist_ok=True)
    return path


def fingerprint(path):
    stat = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def describe(values):
    if not values:
        return None
    return {"min": float(min(values)), "max": float(max(values)),
            "mean": float(np.mean(values)),
            "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95))}


def audit_dataset(data_dir):
    root = resolve_data_dir(data_dir)
    report = {"data_dir": str(root), "required_class_mapping": CLASS_TO_IDX,
              "class_folders": {}, "splits": {}, "unexpected_files": [],
              "unexpected_directories": [], "unreadable_images": [],
              "read_errors": [], "image_warnings": []}
    manifest = {}
    for split in ("train", "val"):
        folders = sorted(p.name for p in (root / split).iterdir() if p.is_dir())
        report["class_folders"][split] = {name: idx for idx, name in enumerate(folders)}
        if report["class_folders"][split] != CLASS_TO_IDX:
            report["unexpected_directories"].append(
                f"{split}: expected cat/dog, found {folders}")
    all_widths, all_heights, all_ratios = [], [], []
    formats, extensions, modes, resolutions = Counter(), Counter(), Counter(), Counter()
    for split in ("train", "val", "test"):
        counts = Counter({label: 0 for label in
                          (("unlabeled",) if split == "test" else CLASS_TO_IDX)})
        valid = counts.copy()
        widths, heights, ratios = [], [], []
        if split == "test":
            report["unexpected_directories"].extend(
                str(p.relative_to(root)) for p in (root / split).iterdir() if p.is_dir())
        for index, path in enumerate(sorted((root / split).rglob("*")), start=1):
            if index % 2000 == 0:
                print(f"{split}: inspecting entry {index}...", flush=True)
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            try:
                manifest[str(relative)] = fingerprint(path)
            except OSError as error:
                report["read_errors"].append({"path": str(relative), "error": str(error)})
            extensions[path.suffix.lower()] += 1
            parts = relative.parts
            supported = (path.suffix.lower() in (TEST_EXTENSIONS if split == "test"
                                                 else IMG_EXTENSIONS))
            expected_location = ((len(parts) == 2) if split == "test" else
                                 (len(parts) >= 3 and parts[1] in CLASS_TO_IDX))
            if not supported or not expected_location:
                report["unexpected_files"].append(str(relative))
                continue
            label = "unlabeled" if split == "test" else parts[1]
            counts[label] += 1
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()  # Decode pixels as well as checking file structure.
                    w, h = image.size
                    formats[image.format] += 1
                    modes[image.mode] += 1
                    if image.mode != "RGB":
                        report["image_warnings"].append(
                            {"path": str(relative), "mode": image.mode,
                             "action": "loader converts to RGB; original unchanged"})
                valid[label] += 1
                widths.append(w)
                heights.append(h)
                ratios.append(w / h)
                resolutions[f"{w}x{h}"] += 1
            except Exception as error:
                report["unreadable_images"].append(
                    {"path": str(relative), "error": f"{type(error).__name__}: {error}"})
        total = sum(counts.values())
        report["splits"][split] = {
            "image_counts": dict(counts), "total_images": total,
            "readable_counts": dict(valid), "readable_images": sum(valid.values()),
            "class_fractions": {k: v / total for k, v in counts.items()} if total else {},
            "width": describe(widths), "height": describe(heights),
            "aspect_ratio_width_over_height": describe(ratios)}
        all_widths.extend(widths)
        all_heights.extend(heights)
        all_ratios.extend(ratios)
        print(f"{split}: {dict(counts)}; readable {sum(valid.values())}/{total}", flush=True)
    # Root-level metadata are recorded too, so preservation checks cover all files.
    for path in root.iterdir():
        if path.is_file():
            report["unexpected_files"].append(path.name)
            try:
                manifest[path.name] = fingerprint(path)
            except OSError as error:
                report["read_errors"].append({"path": path.name, "error": str(error)})
        elif path.name not in ("train", "val", "test"):
            report["unexpected_directories"].append(path.name)
    report["total_images"] = sum(s["total_images"] for s in report["splits"].values())
    report["readable_images"] = sum(s["readable_images"] for s in report["splits"].values())
    report["characteristics"] = {
        "width": describe(all_widths), "height": describe(all_heights),
        "aspect_ratio_width_over_height": describe(all_ratios),
        "decoded_formats": dict(formats), "file_extensions_in_splits": dict(extensions),
        "original_color_modes": dict(modes),
        "most_common_resolutions": dict(resolutions.most_common(10)),
        "aspect_ratio_distribution": {
            "portrait_below_0.9": sum(r < 0.9 for r in all_ratios),
            "near_square_0.9_to_1.1": sum(0.9 <= r <= 1.1 for r in all_ratios),
            "landscape_above_1.1": sum(r > 1.1 for r in all_ratios)}}
    return report, manifest


def summary_text(report):
    lines = [f"Dataset: {report['data_dir']}",
             f"Class folders/mapping: {report['class_folders']}"]
    for split, entry in report["splits"].items():
        lines.append(f"{split}: {entry['image_counts']}; total={entry['total_images']}; "
                     f"readable={entry['readable_images']}; fractions={entry['class_fractions']}")
    lines.append(f"Total images: {report['total_images']}; readable: {report['readable_images']}")
    for key, value in report["characteristics"].items():
        lines.append(f"{key}: {value}")
    for key in ("unreadable_images", "unexpected_files", "unexpected_directories",
                "read_errors", "image_warnings"):
        lines.append(f"{key}: {len(report[key])}; examples: {report[key][:5]}")
    lines.append("See dataset_summary.json for all issues and per-split statistics.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    root = resolve_data_dir(args.data_dir)
    out = output_directory(args.output_dir, root)
    report, manifest = audit_dataset(root)
    (out / "dataset_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    text = summary_text(report)
    (out / "dataset_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
