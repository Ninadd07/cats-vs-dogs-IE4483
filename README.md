# IE4483 Dogs vs. Cats

Image classification with a CNN trained from scratch, pretrained ResNet/ViT
baselines, frozen DINOv2 features, and DINOv2-to-CNN knowledge distillation.
Labels are **cat=0, dog=1**. Run commands below from the repository root.

## Setup and data

Use Python 3.12 and an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r custom_cnn/requirements-distillation.txt
```

Install a CUDA-enabled PyTorch build on a GPU machine. Device selection is
automatic (CUDA, then Apple MPS, then CPU); CNN commands also accept `--device`.
DINOv2 downloads pretrained weights on first use and caches them in
`dino_vit/weights/`. Datasets, environments, checkpoints and teacher caches are
ignored by Git and must be obtained or generated separately.

```text
datasets/
  train/cat/    # 10,000 images
  train/dog/    # 10,000 images
  val/cat/      # 2,500 images
  val/dog/      # 2,500 images
  test/         # 500 unlabelled images: 1.jpg ... 500.jpg
```

Supply `--data-dir /path/to/datasets` for module commands or `--data_dir` for
the root trainer. Preserve the supplied train/validation/test split. Fit model
weights and preprocessing statistics on training data; use validation for model
selection. Test labels are unavailable, so test accuracy cannot be reported.

## Current results and implementation status

| Approach | Validation accuracy | Status |
| --- | --- | --- |
| Custom CNN, Experiment A | 96.32% | Existing reported result; best epoch 27, seed 42 |
| Custom CNN, best of 12-run sweep | 96.58% | LR 1e-3, wd 1e-4, dropout 0.3 (`custom_cnn/outputs/sweep/sweep_results.csv`) |
| ResNet-50, ImageNet-pretrained, fine-tuned | 99.43% ± 0.12 | Seeds 42/43/44, flip only (`cluster/pretrained.sbatch`) |
| ViT-B/16, ImageNet-pretrained, fine-tuned | 99.41% ± 0.03 | Seeds 42/43/44, flip only (`cluster/pretrained.sbatch`) |
| Frozen DINOv2 + logistic regression | 99.88% | Existing reported result; C=0.001 |
| CIFAR-10 students, CE / + KD (test accuracy) | CNN 90.86 / 91.35%; ResNet-9 93.93 / 94.78% | Seed 42; teacher 98.14% (`cifar10/outputs/`) |
| CIFAR-10 class imbalance, ResNet-9 (test accuracy) | CE 80.26%; logit adjustment 84.01% (best) | Seed 42; all six methods in `cifar10/outputs/imb_f0.05/` |

These existing accuracies come from the component READMEs and stored report
artifacts. They are not new results from the distillation implementation.

## Train and evaluate the custom CNN

The student has four convolution blocks (32/64/128/256 channels), global average
pooling and a two-logit classifier: **1,206,370 parameters**. Training uses dynamic
augmentation, AdamW, validation-loss LR scheduling and early stopping. Evaluation
uses deterministic resize to 224x224 and ImageNet normalization.

```bash
python -m custom_cnn.verify --data-dir datasets
python -m custom_cnn.train --data-dir datasets --seed 42 \
  --output-dir custom_cnn/outputs/baseline_seed42
python -m custom_cnn.evaluate --data-dir datasets \
  --checkpoint custom_cnn/outputs/baseline_seed42/best.pth
python -m custom_cnn.visualize \
  --history custom_cnn/outputs/baseline_seed42/training_history.csv
```

Use a fresh output directory for each run. `best.pth` is selected by highest
validation accuracy; `latest.pth` includes optimizer, scheduler and RNG state for
resuming. `--epochs` is the total epoch limit, not additional epochs:

```bash
python -m custom_cnn.train \
  --resume custom_cnn/outputs/baseline_seed42/latest.pth --epochs 50
```

An already early-stopped run cannot be resumed past its stopping criterion.
See [CNN details and cluster instructions](custom_cnn/README.md).

## Prepare the DINOv2 teacher

The teacher is the frozen DINOv2 ViT-B/14 backbone **plus** its fitted
StandardScaler and binary logistic-regression probe. It produces class logits;
raw DINOv2 embeddings alone are not cat/dog predictions.

```bash
# Once: default backbone and 224px features. Skip if these caches already exist.
python dino_vit/extract_features.py --data-dir datasets

# Fit on train features; choose C on validation; export the fitted teacher head.
python dino_vit/probe.py --export-teacher dino_vit/weights/teacher_probe.npz

# Optional: generate the DINOv2 teacher's own test submission.
python dino_vit/probe.py --submission dino_vit/submissions/submission_dinov2.csv
```

Extraction caches train, validation and test features; probe fitting uses only
train labels, and exporting a teacher does not require test features. Use the
default `vit_base_patch14_dinov2.lvd142m` and 224px extraction settings for this
export. See [DINOv2 details](dino_vit/README.md).

## Train the distilled CNN

Both models see the same augmented, normalized training image. The frozen
teacher runs online without gradients. The loss is:

```text
L = (1-alpha) * CE(student_logits, labels)
    + alpha * T^2 * KL(softmax(teacher_logits/T) || softmax(student_logits/T))
```

`alpha=0.5` and `T=4` are starting settings, not measured optima. Validation uses
ordinary student cross-entropy and student accuracy. Only the CNN is needed for
inference; distillation adds training compute and memory, with unchanged CNN
inference architecture.

```bash
python -m custom_cnn.train --data-dir datasets \
  --teacher-probe dino_vit/weights/teacher_probe.npz \
  --kd-alpha 0.5 --kd-temperature 4 --seed 42 \
  --output-dir custom_cnn/outputs/kd_seed42
python -m custom_cnn.evaluate --data-dir datasets \
  --checkpoint custom_cnn/outputs/kd_seed42/best.pth
python -m custom_cnn.visualize \
  --history custom_cnn/outputs/kd_seed42/training_history.csv
python -m custom_cnn.train \
  --resume custom_cnn/outputs/kd_seed42/latest.pth --epochs 50
```

Keep the exported probe file and backbone weights for resumption. Probe SHA-256
is checked when resuming. The CSV logs `train_ce`, `train_kd`, weighted
`train_loss`, validation loss/accuracy and LR. Weighted training loss and
validation cross-entropy measure different objectives.

```bash
# Synthetic CPU verification: no pretrained download or full training.
python -m custom_cnn.verify_distillation
```

Checks cover sklearn/PyTorch teacher probability equivalence, KD mathematics,
detached targets, frozen teacher weights, student-only checkpoints and exact CPU
resumption. Keep package versions, hardware and worker count fixed when
comparing runs; cross-environment bitwise reproducibility is not guaranteed.

## CIFAR-10: CNN and DINOv2-distilled CNN

[`cifar10/kd.py`](cifar10/kd.py) reuses the same CNN (`num_classes=10`, 32x32
input) and KD loss. The teacher is frozen DINOv2 ViT-B/14 (inputs upsampled
32->224) plus a 10-class StandardScaler + logistic-regression probe. The official
50,000 training images are split into 45,000 train / 5,000 validation
(stratified, `--split-seed 42`); teacher C and student epoch are chosen on
validation, and the 10,000 official test images are used only for final metrics.
CIFAR-10 is downloaded to `--data-dir` on first use.

```bash
# 1. Teacher: caches DINOv2 features, sweeps C, exports teacher.npz (+ val/test accuracy).
python -m cifar10.kd teacher --data-dir datasets/cifar10 --out cifar10/outputs/teacher
# 2. Students: plain cross-entropy baseline and KD (alpha=0.5, T=4), same seed/schedule.
python -m cifar10.kd train --kd-alpha 0 --seed 42 --output-dir cifar10/outputs/ce_seed42
python -m cifar10.kd train --teacher cifar10/outputs/teacher/teacher.npz \
  --kd-alpha 0.5 --kd-temperature 4 --seed 42 --output-dir cifar10/outputs/kd_seed42
# 3. Test accuracy, macro F1, per-class precision/recall, confusion matrix -> test_metrics.json
python -m cifar10.kd evaluate --checkpoint cifar10/outputs/kd_seed42/best.pth
```

Training uses random crop (pad 4) + horizontal flip, AdamW and a one-cycle LR
schedule (60 epochs, batch 128, peak LR 3e-3). The teacher sees the same augmented
image as the student and runs online under CUDA autocast (`--no-amp` for fp32),
so KD epochs cost mostly teacher forward passes. On the cluster,
`mkdir -p logs && sbatch cifar10/cluster/kd.sbatch 42` (from the repo root) runs
all three stages for one seed on an RTX 6000 Ada; `ARCH=resnet9` uses the
stronger 6.6M-parameter ResNet-9 student (SGD-Nesterov, peak LR 0.1). Add
`--limit 500` to any command for a quick smoke test. Runs are not resumable;
use a fresh `--output-dir` each time.

### CIFAR-10 with class imbalance

Step imbalance: `--minority-classes 2 3 4 5 6 --minority-fraction 0.05` keeps 5% of
bird/cat/deer/dog/frog (225 vs 4,500 training images per class, 20x). Validation is
reduced the same way and the best epoch is chosen on validation *balanced accuracy*
(mean per-class recall); the official test set stays balanced. `evaluate` adds
minority/majority mean recall. Remedies, each a separate run with the same student:

| Flag | Approach | Why it should help |
| --- | --- | --- |
| `--class-weights effective --cb-beta 0.999` | Class-balanced loss weights (Cui et al., 2019) | Rare-class errors cost more, so the gradient is no longer dominated by majority classes; the "effective number" weighting is milder than 1/n, which over-weights a few repeated images |
| `--sampler balanced` | Class-balanced resampling | Every batch sees all classes equally often; minority images repeat but get fresh augmentation each time |
| `--logit-adjust 1.0` | Logit-adjusted loss (Menon et al., 2021) | Adds `log prior` to the logits during training only, enforcing larger margins for rare classes; Fisher-consistent for balanced error, with no reweighting variance |
| `--teacher ... --kd-alpha 0.5` | KD from a DINOv2 teacher refit on the same imbalanced data (`teacher --balanced-teacher`) | Pretrained features need few labels per class, so the teacher's soft targets carry minority-class information to every image the student sees |

```bash
mkdir -p logs && sbatch cifar10/cluster/imbalance.sbatch 42   # all methods, ResNet-9
```

## Existing ResNet and ViT baselines

The root trainer uses underscore-style flags. These examples train pretrained
backbones and save separate checkpoints/prediction CSVs:

```bash
python train.py --model resnet --data_dir datasets --pretrained \
  --epochs 10 --save_path resnet_best.pth --pred_path resnet_predictions.csv
python train.py --model vit --data_dir datasets --pretrained \
  --epochs 10 --save_path vit_best.pth --pred_path vit_predictions.csv
```

Add `--freeze_backbone` to train only the classifier head. Root-trainer weight
files use a different format from `custom_cnn` training checkpoints.

On the cluster, `sbatch --job-name=catsdogs-resnet cluster/pretrained.sbatch resnet` (or `vit`)
fine-tunes with seeds 42/43/44 (`--seed`, default 42) and prints mean ± std. The augmentation
study, `sbatch cluster/augmentation.sbatch`, trains ResNet-50 and ViT-B/16 with
`--augmentation none|flip|enhanced` and the custom CNN with `--augmentation none|baseline`
(seed 42, one sequential job). Pretrained runs log to
`outputs/pretrained/<model>_<aug>_seed<N>/train.log`.

## Submission and report artifacts

The existing Experiment A evaluation/submission script is intentionally locked
to its epoch-27 checkpoint and 96.32% validation result:

```bash
python -m custom_cnn.predictions_and_evaluation.predict --data-dir datasets --device cpu
python -m custom_cnn.predictions_and_evaluation.analyze --data-dir datasets
```

Obtain `custom_cnn/outputs/run_a/best.pth` separately to reproduce this result.
These commands do not accept arbitrary distilled-run checkpoints. Distilled
checkpoints can be evaluated with `custom_cnn.evaluate`; a generic distilled
test-submission exporter is not included. Existing final report artifacts and
the 500-row `id,label` submission are documented
[here](custom_cnn/predictions_and_evaluation/README.md).

## Experiments to make the project stand out

1. **Controlled ablation:** compare ordinary CE, label smoothing and distillation
   with identical data, augmentation, training budget and seeds 42/43/44. Repeat
   the baseline/KD commands above with each seed and a separate output directory.
   Report mean/std validation accuracy and macro F1. Label smoothing remains to
   be implemented; it tests whether gains exceed generic confidence smoothing,
   particularly useful with only two classes.
2. **Small hyperparameter study:** start with alpha in {0.25, 0.5, 0.75} and T in
   {2, 4}. Override `--kd-alpha` and `--kd-temperature`, using a fresh output
   directory per setting. Choose settings on validation, then freeze them for
   the multi-seed comparison. This is a proposed budget, not a completed sweep.
3. **Accuracy versus deployment cost:** report parameter count, checkpoint size
   and batch-1 latency for teacher, baseline CNN and distilled CNN. Use the same
   device/input size, warmup and synchronized GPU timing. Measure speed rather
   than assuming fewer parameters means faster inference.
4. **Robustness and interpretation:** predefine mild blur, brightness and
   occlusion severities; compare baseline/KD validation accuracy and confidence.
   Inspect teacher/student disagreements and correctly/incorrectly classified
   validation examples. Grad-CAM for the CNN is another optional qualitative
   extension. These analyses are proposed, not implemented. Test images have no
   ground truth, so describe predictions rather than claiming test correctness.
5. **Complete CIFAR-10 and imbalance:** these are explicitly worth 20% in the
   brief. The CIFAR-10 CE/KD pipeline is implemented (see above); run it over
   several seeds. Reserve training data for validation
   and keep the official test set untouched for final reporting. For imbalance,
   reduce selected training classes only; compare ordinary CE, inverse-frequency
   weighted CE and balanced sampling as separate approaches. Report per-class
   recall, macro F1 and overall accuracy.

Prioritize the required CIFAR-10 work and controlled experiments before adding
more architectures. Describe pretrained DINOv2's external pretraining data in
the report, alongside the project images used to fit the probe/student.

## References

- [Hinton et al., Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Oquab et al., DINOv2](https://arxiv.org/abs/2304.07193)
- [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
