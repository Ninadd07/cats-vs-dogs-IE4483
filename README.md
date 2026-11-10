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
| Frozen DINOv2 + logistic regression | 99.88% | Existing reported result; C=0.001 |
| DINOv2-distilled CNN | Pending | Implemented and smoke-tested; full training not run |
| Label-smoothing control, robustness, latency study | Pending | Proposed experiments |
| CIFAR-10 and class imbalance | Pending | Required extensions; pipeline not yet implemented |

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
   brief. The CNN builder accepts `num_classes=10`; the dataset, training and
   metrics pipeline still needs adaptation. Reserve training data for validation
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
