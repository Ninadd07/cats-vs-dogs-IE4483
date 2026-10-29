"""Render original cat/dog images and augmentation examples without plotting dependencies."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .dataloader import load_rgb, resolve_data_dir
from .dataset_analysis import DEFAULT_OUTPUT, output_directory
from .transforms import IMAGENET_MEAN, IMAGENET_STD, build_train_transform


def font(size, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def natural_rgb(tensor):
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    return T.ToPILImage()((tensor.detach().cpu() * std + mean).clamp(0, 1))


def generate_figure(data_dir=None, output_dir=DEFAULT_OUTPUT, seed=42,
                    image_size=224, cat_image=None, dog_image=None, pdf=False):
    root = resolve_data_dir(data_dir)
    out = output_directory(output_dir, root)
    selected = []
    for label, override in (("cat", cat_image), ("dog", dog_image)):
        if override is not None:
            path = Path(override).resolve()
        else:
            # Use exactly the same supported files as ImageFolder.
            from torchvision.datasets.folder import IMG_EXTENSIONS
            candidates = sorted(p for p in (root / "train" / label).rglob("*")
                                if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
            path = None
            for candidate in candidates:
                try:
                    load_rgb(candidate)
                    path = candidate
                    break
                except (OSError, ValueError):
                    continue
            if path is None:
                raise ValueError(f"No readable {label} image under {root / 'train' / label}")
        selected.append((label, path))

    # Fix individual demonstrations at visible endpoints inside the training
    # ranges. Full enhanced preprocessing remains an ordinary seeded sample.
    fill = tuple(round(255 * x) for x in IMAGENET_MEAN)
    operations = [
        ("Original", "Unmodified RGB", lambda image: image.copy()),
        ("Horizontal flip", "Shown flipped; training p = 0.5", T.RandomHorizontalFlip(p=1)),
        ("Rotation", "+10 degrees; bilinear", T.RandomRotation(
            (10, 10), interpolation=InterpolationMode.BILINEAR, fill=fill)),
        ("Resized crop", "Target: 80% area; square", T.RandomResizedCrop(
            image_size, scale=(0.8, 0.8), ratio=(1.0, 1.0),
            interpolation=InterpolationMode.BILINEAR)),
        ("Color jitter", "+10% brightness / contrast / saturation", T.ColorJitter(
            brightness=(1.1, 1.1), contrast=(1.1, 1.1), saturation=(1.1, 1.1), hue=0)),
        ("Full enhanced", "Seeded sample; normalization reversed", build_train_transform(image_size)),
    ]
    cell, gap, left, right = 400, 26, 120, 50
    top, row_height, footer = 255, 510, 120
    width = left + 6 * cell + 5 * gap + right
    height = top + 2 * row_height + footer
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 34), "Dogs vs. Cats | Training image augmentation", font=font(48, True), fill="#172433")
    draw.text((50, 100), "One source image per row. Individual examples use visible settings within the configured ranges.",
              font=font(27), fill="#425466")
    for col, (title, subtitle, _) in enumerate(operations):
        x = left + col * (cell + gap)
        draw.text((x + cell / 2, 180), title, anchor="mm", font=font(31, True), fill="#172433")
        # Wrap deliberately rather than shrinking the labels.
        if col >= 4:
            words = subtitle.split(" / ") if col == 4 else subtitle.split("; ")
            lines = [words[0], " / ".join(words[1:])] if col == 4 else words
        else:
            lines = [subtitle]
        for line_idx, line in enumerate(lines):
            draw.text((x + cell / 2, 216 + 24 * line_idx), line,
                      anchor="mm", font=font(21), fill="#425466")

    with torch.random.fork_rng(devices=[]):
        for row, (label, path) in enumerate(selected):
            original = load_rgb(path)
            y = top + row * row_height
            draw.text((28, y + cell / 2), label.capitalize(), anchor="lm",
                      font=font(29, True), fill="#172433")
            for col, (_, _, operation) in enumerate(operations):
                torch.manual_seed(seed + row * 100 + col)
                transformed = operation(original.copy())
                if isinstance(transformed, torch.Tensor):
                    transformed = natural_rgb(transformed)
                x = left + col * (cell + gap)
                tile = ImageOps.pad(transformed, (cell, cell), color="#f0f3f6",
                                    method=Image.Resampling.LANCZOS)
                canvas.paste(tile, (x, y))
                draw.rectangle((x, y, x + cell, y + cell), outline="#d8e0e8", width=2)
            draw.text((left, y + cell + 18),
                      f"Source: train/{label}/{path.name}  |  Original: {original.width} x {original.height} px",
                      font=font(25), fill="#425466")
    draw.text((50, height - 95),
              f"Training defaults: crop scale 0.8-1.0, ratio 0.9-1.1; flip p=0.5; rotation +/-10 degrees; jitter 0.1; seed {seed}.",
              font=font(25), fill="#425466")
    draw.text((50, height - 53),
              "Panels preserve aspect ratio using display padding. Validation and test use deterministic resize, tensor conversion and normalization.",
              font=font(25), fill="#425466")
    target = out / "augmentation_examples.png"
    canvas.save(target, dpi=(300, 300))
    if pdf:
        canvas.save(out / "augmentation_examples.pdf", resolution=300)
    (out / "visualization_settings.json").write_text(json.dumps({
        "seed": seed, "image_size": image_size,
        "sources": {label: str(path) for label, path in selected},
        "individual_demonstrations": "flip forced; rotation +10; crop area 0.8, ratio 1; jitter factors 1.1",
        "full_enhanced": repr(operations[-1][2]),
        "pixel_dimensions": [width, height], "dpi": 300,
    }, indent=2), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cat-image", help="Optional source image path")
    parser.add_argument("--dog-image", help="Optional source image path")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    print(generate_figure(**vars(args)))


if __name__ == "__main__":
    main()
