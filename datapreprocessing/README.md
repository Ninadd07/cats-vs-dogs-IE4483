# Data preprocessing

Image loading, augmentation and dataset inspection for Dogs vs. Cats.

## Pipelines

Both modes produce RGB tensors at 224 × 224 by default, normalized with
ImageNet channel statistics.

| Split | Baseline | Enhanced (default) |
| --- | --- | --- |
| Training | Resize → horizontal flip → tensor → normalize | Random resized crop → horizontal flip → rotation → color jitter → tensor → normalize |
| Validation/test | Resize → tensor → normalize | Resize → tensor → normalize |

Enhanced defaults: crop area 80–100%, aspect ratio 0.9–1.1, flip probability 0.5,
rotation ±10°, and brightness/contrast/saturation jitter of 0.1. Hue is unchanged.
These transformations vary framing, orientation and lighting while retaining
the class label. Their effect on accuracy has not been measured; cropping can
remove relevant parts of an animal.

## Dataset

```text
datasets/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/
```

Labels are `cat = 0`, `dog = 1`. Test images are unlabelled and returned with
their original filenames in sorted order. The loader checks `datasets/` in the
repository, then its parent directory, or accepts an explicit `data_dir`.

## Usage

Install dependencies and run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m datapreprocessing.dataset_analysis --data-dir ../datasets
python -m datapreprocessing.visualize --data-dir ../datasets --seed 42 --pdf
python -m datapreprocessing.verify --data-dir ../datasets
```

The audit reports counts, dimensions and unreadable files without changing images.
Summaries and the augmentation figure are saved in `datapreprocessing/outputs/`.
Run the audit before verification to create the file manifest.
The summary, augmentation figure and visualization settings are eligible for Git;
the dataset manifest and generated verification reports remain ignored.

```python
from datapreprocessing import CatsDogsDataLoader

data = CatsDogsDataLoader(
    data_dir="../datasets", mode="enhanced", batch_size=32, seed=42,
)
train_loader, val_loader, test_loader = data.get_loaders()
```

Use `mode="baseline"` for the original transforms. Image size, worker count and
augmentation settings can be passed to the loader. Keep 224 × 224 for the ViT.
The root `train.py` uses the original loader; `custom_cnn/train.py` uses this
package's enhanced pipeline by default.

For repeatable runs, keep the seed, settings, worker count and library versions
fixed. With zero workers, other PyTorch random operations can affect augmentation.
On Windows, use an `if __name__ == "__main__":` guard when enabling workers.
