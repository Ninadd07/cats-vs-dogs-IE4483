# DINOv2 ViT-B/14: frozen features + logistic regression

Dogs vs. Cats classifier built on a frozen DINOv2 backbone (`vit_base_patch14_dinov2.lvd142m`, 85.7M parameters,
loaded through [timm](https://github.com/huggingface/pytorch-image-models)). The backbone is never trained: every image goes
through it once, the 768-d CLS embeddings are cached, and a logistic regression is fitted on top.

**Validation accuracy: 99.88%** (6 errors in 5,000 images; logistic regression, C = 0.001). kNN with no training: 99.74–99.78%.

## Pipeline

```
image -> resize 224x224 -> ImageNet mean/std normalisation -> DINOv2 ViT-B/14 (frozen)
      -> 768-d CLS embedding -> StandardScaler -> LogisticRegression -> P(cat), P(dog)
```

- Input: 3 x 224 x 224 (16 x 16 patches of 14 px). Output: 768-d feature, then 2 classes (0 = cat, 1 = dog).
- No augmentation: with a frozen backbone and cached features there is nothing to randomise.
- The probe uses only `datasets/train` for fitting. The regularisation strength C is selected on `datasets/val`.
  The probe is deterministic, so no seeding is needed (the seed only affects the few-shot subsampling).

## Files

| File | Purpose |
|---|---|
| `common.py` | Paths, device selection, preprocessing, test-set loader, backbone loading |
| `requirements.txt` | Pinned dependencies for this folder only |
| `extract_features.py` | Runs the frozen backbone over train/val/test and caches features |
| `probe.py` | Fits logistic regression / kNN on cached features, selects C on val, writes the submission |
| `features/vit_base_patch14_dinov2/` | Cached `train.npz`, `val.npz`, `test.npz` and `probe_results.json` |
| `weights/` | Local copy of the backbone weights (created on first run; works offline afterwards) |
| `submissions/submission_dinov2.csv` | Test predictions (`id,label`; 1 = dog, 0 = cat) |

## Run

This folder is standalone (no other repo code needed). From the folder that contains `dino_vit/`, with `pip install -r dino_vit/requirements.txt` done:

```
PYTORCH_ENABLE_MPS_FALLBACK=1 python dino_vit/extract_features.py          # ~8 min on an M5 (≈50 img/s), once
python dino_vit/probe.py --submission dino_vit/submissions/submission_dinov2.csv   # seconds
```

To use the fitted scaler + logistic regression as a frozen teacher for the custom
CNN, export it with `python dino_vit/probe.py --export-teacher dino_vit/weights/teacher_probe.npz`.
Use the default backbone and 224px cached features. See
[custom CNN distillation instructions](../custom_cnn/README.md#dinov2-knowledge-distillation).

The scripts read `datasets/{train,val,test}` next to this folder; pass `--data-dir PATH` to extract_features.py to use another location. Run `extract_features.py` again only if the
dataset or backbone changes; `probe.py` can be re-run freely.

Few-shot curve (labelled images per class, use at least 10 for the default kNN settings):
```
python dino_vit/probe.py --train-per-class 10
```

| Labelled images per class | 5 | 10 | 50 | 100 | 500 | 2000 | 10000 |
|---|---|---|---|---|---|---|---|
| Logistic regression val acc | 97.30% | 99.64% | 99.82% | 99.82% | 99.86% | 99.88% | 99.88% |
