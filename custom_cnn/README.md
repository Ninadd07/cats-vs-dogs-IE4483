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
