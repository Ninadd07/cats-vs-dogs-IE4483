# Custom CNN

Dogs vs. Cats classifier trained from scratch using the enhanced
`datapreprocessing` pipeline (`cat=0`, `dog=1`). Experiment A completed 30 epochs;
the best checkpoint, epoch 27, achieved **96.32% validation accuracy**
(4,816/5,000 correct) and **0.9632 macro F1**. The final submission contains
500 test predictions. Test labels are unavailable, so test accuracy is unknown.

## Model and training

Four blocks each apply two 3x3 convolutions with batch normalization and ReLU,
then 2x2 max pooling. Channels progress through 32, 64, 128 and 256.
Global average pooling feeds a 256-to-128 linear layer, ReLU, dropout (0.3)
and a two-logit output. There are **1,206,370 trainable parameters**.

| Setting | Experiment A |
| --- | --- |
| Image / batch size | 224x224 / 32 |
| Epoch limit / seed / workers | 30 / 42 / 0 |
| Loss / optimizer | Cross-entropy / AdamW |
| Learning rate / weight decay | 0.001 / 0.0001 |
| ReduceLROnPlateau | Validation loss; factor 0.5, patience 3 |
| Early stopping | Validation loss; patience 7 |
| Best checkpoint selection | Highest validation accuracy; earliest epoch on ties |

Training augmentation is dynamic; validation and test transforms are deterministic.
The supplied dataset splits are preserved. Training never iterates the test set.

## Usage

Run from the repository root. Select a CUDA-enabled PyTorch build for GPU use.

```bash
python -m pip install -r custom_cnn/requirements.txt
python -m custom_cnn.verify --data-dir ../datasets

# Use a fresh directory for a new experiment.
python -m custom_cnn.train --data-dir ../datasets --output-dir custom_cnn/outputs/new_run
python -m custom_cnn.train --resume custom_cnn/outputs/new_run/latest.pth --epochs 50
python -m custom_cnn.evaluate --checkpoint custom_cnn/outputs/run_a/best.pth --data-dir ../datasets
python -m custom_cnn.visualize --history custom_cnn/outputs/run_a/training_history.csv
```

`--help` lists options. Resume restores model, optimizer, scheduler, configuration,
history and RNG state at a completed-epoch boundary. `--epochs` is the total epoch
limit. Keep software versions, hardware and worker count fixed for reproducibility;
exact continuation across environments is not guaranteed.

Cluster scripts in `cluster/` use `/projects/ie4483catsdogs`. Submit work with
Slurm (`sbatch custom_cnn/cluster/train.sbatch`), using a fresh output directory
for new experiments. `setup.sbatch` is for initial deployment from source/data
archives; `smoke.sbatch` checks the GPU without full training.

## Final deliverables

| Location | Contents |
| --- | --- |
| [outputs/run_a/](outputs/run_a/) | Training history, configuration, accuracy and loss curves |
| [predictions_and_evaluation/outputs/](predictions_and_evaluation/outputs/) | Validation metrics, classification report, confusion matrix, example figures and final submission |
| [predictions_and_evaluation/README.md](predictions_and_evaluation/README.md) | Inference and figure-generation commands |

The root `.gitignore` allows these report artifacts to be committed. Checkpoints,
logs, detailed prediction CSVs, provenance and environment records remain local
and ignored. `outputs/run_a/best.pth` is retained for inference; `latest.pth`
is retained for resumption. Obtain the checkpoints separately when cloning the
repository. `outputs/setup/requirements.lock.txt` records the cluster packages.

Verification uses temporary synthetic data/checkpoints and removes them on exit.

## DINOv2 knowledge distillation

Train the same 1.21M-parameter CNN using the existing frozen DINOv2 ViT-B/14
and its trained binary logistic-regression probe as a teacher. The exported
StandardScaler and binary decision function are reproduced in PyTorch; the raw
DINO embedding alone is not a cat/dog classifier. Both networks receive the
**same augmented, ImageNet-normalized image** on each training step. Teacher
inference is online, so it adds training compute and GPU memory.

The loss is `(1-alpha) CE(student, label) + alpha T² KL(teacher/T || student/T)`,
where softmax is applied to the temperature-scaled logits. Defaults are
`alpha=0.5`, `T=4`; these are starting settings, not measured optima.
Validation, LR scheduling and early stopping use ordinary student cross-entropy;
best model selection still uses student validation accuracy. The teacher is
frozen, used only during training, and excluded from the student checkpoint.
The `custom_cnn.evaluate` command works for distilled checkpoints. The existing
Experiment A submission script is locked to epoch 27 / 96.32% and cannot export
distilled checkpoints; a generic KD submission exporter is not included.
`train_ce` and `train_kd` in the training CSV separate the two loss terms;
`train_loss` is their weighted sum and is not directly comparable to validation CE.

```bash
python -m pip install -r custom_cnn/requirements-distillation.txt

# Once: extract default 224px DINOv2 features if not already cached.
python dino_vit/extract_features.py --data-dir datasets
# Fits scaler/probe on train only; chooses C using val; exports no test data.
python dino_vit/probe.py --export-teacher dino_vit/weights/teacher_probe.npz

python -m custom_cnn.train --data-dir datasets \
  --teacher-probe dino_vit/weights/teacher_probe.npz \
  --kd-alpha 0.5 --kd-temperature 4 --seed 42 \
  --output-dir custom_cnn/outputs/kd_seed42
python -m custom_cnn.train --resume custom_cnn/outputs/kd_seed42/latest.pth --epochs 50
python -m custom_cnn.evaluate --checkpoint custom_cnn/outputs/kd_seed42/best.pth --data-dir datasets
python -m custom_cnn.verify_distillation
```

The current probe export describes the default `vit_base_patch14_dinov2.lvd142m`
backbone with 224px cached features. Use those extraction settings. Backbone
weights are downloaded on first use if absent from `dino_vit/weights/`.
Resume checks the exported probe's SHA-256; retain that file and the same
backbone weights/environment for continuation. Ordinary CNN runs need neither
timm nor a teacher probe. No KD validation result is claimed until training runs.

## Experiments that strengthen the report

1. **Controlled distillation ablation:** compare CE-only CNN, label-smoothing
   control, and distilled CNN with identical splits, augmentations, epoch budget
   and seeds (e.g. 42, 43, 44). Label smoothing is a proposed experiment, not
   implemented here. Report mean/std accuracy and macro F1. Compare the same
   per-image errors and teacher/student disagreements. In binary classification,
   teacher targets mainly convey uncertainty; this control tests whether the
   benefit exceeds generic smoothing. Tune a small alpha/T grid on validation
   only, then freeze settings before the multi-seed comparison.
2. **Accuracy versus deployment cost:** compare teacher and CNN parameter count,
   checkpoint size, and batch-1 latency on the same device, input size, warmup
   and synchronized timing. Measure speed rather than inferring it from parameter
   count. KD leaves CNN inference architecture and parameter count unchanged.
3. **Robustness and failure analysis:** predefine mild blur, brightness and
   occlusion severities and compare baseline/KD validation accuracy, confidence
   and failure examples. Keep clean validation as the checkpoint criterion.
   Distinguish measured validation errors from unlabelled test predictions.
4. **Complete CIFAR-10 and imbalance experiments first:** these are explicitly
   worth 20% in the brief. The CNN builder already accepts `num_classes=10`, but
   the training/data/evaluation pipeline currently targets two-class Dogs vs.
   Cats and needs adaptation. Construct imbalance in training only; compare
   inverse-frequency weighted CE and balanced sampling as separate approaches
   against ordinary CE, reporting per-class recall, macro F1 and accuracy on the
   untouched CIFAR-10 test set. Reserve part of training for model selection.

References: [Hinton et al., knowledge distillation](https://arxiv.org/abs/1503.02531),
[Oquab et al., DINOv2](https://arxiv.org/abs/2304.07193).
