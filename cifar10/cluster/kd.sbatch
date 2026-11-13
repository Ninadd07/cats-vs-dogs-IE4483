#!/bin/bash
#SBATCH -A ug-proj
#SBATCH --qos=ug
#SBATCH --gres=gpu:6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=cifar10-kd
#SBATCH --time=12:00:00
# Full CIFAR-10 pipeline for one seed: DINOv2 teacher (skipped if already exported),
# CE baseline student, KD student, then test-set evaluation of both.
#
# Submit from the repository root (the log path above is relative to it):
#   mkdir -p logs && sbatch cifar10/cluster/kd.sbatch [SEED]
# Optional overrides: DATA=/path/to/cifar10 ENV=/path/to/venv-or-conda-env sbatch ...
# Stronger student:  ARCH=resnet9 sbatch cifar10/cluster/kd.sbatch 42
#   (resnet9 defaults to SGD-Nesterov, peak LR 0.1, wd 5e-4, no dropout; TRAIN_ARGS overrides.)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# Default: the shared Miniforge base Python (+ packages in ~/.local). Override with ENV=<venv dir
# or conda env name/path>.
ENV=${ENV:-/cluster/apps/software/Miniforge3/24.11.3-1}
if [ -f "$ENV/bin/activate" ] && [ ! -f "$ENV/bin/conda" ]; then
    source "$ENV/bin/activate"
else
    eval "$(/cluster/apps/software/Miniforge3/24.11.3-1/bin/conda shell.bash hook)"
    conda activate "$ENV"
fi
which python

SEED=${1:-42}
DATA=${DATA:-datasets/cifar10}
OUT=cifar10/outputs
ARCH=${ARCH:-cnn}
if [ "$ARCH" = resnet9 ]; then
    DEFAULT_ARGS="--optimizer sgd --lr 0.1 --weight-decay 5e-4 --dropout 0"
else
    DEFAULT_ARGS=""
fi
read -r -a TRAIN <<< "--arch $ARCH ${TRAIN_ARGS:-$DEFAULT_ARGS}"
PREFIX=$([ "$ARCH" = cnn ] && echo "" || echo "${ARCH}_")
COMMON=(--device cuda --data-dir "$DATA" --num-workers "${SLURM_CPUS_PER_TASK:-4}")
export PYTHONUNBUFFERED=1

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torch.cuda.get_device_name())"

[ -f "$OUT/teacher/teacher.npz" ] || python -m cifar10.kd teacher "${COMMON[@]}" --out "$OUT/teacher"
python -m cifar10.kd train "${COMMON[@]}" "${TRAIN[@]}" --kd-alpha 0 --seed "$SEED" \
    --output-dir "$OUT/${PREFIX}ce_seed$SEED"
python -m cifar10.kd train "${COMMON[@]}" "${TRAIN[@]}" --teacher "$OUT/teacher/teacher.npz" \
    --kd-alpha 0.5 --kd-temperature 4 --seed "$SEED" --output-dir "$OUT/${PREFIX}kd_seed$SEED"
for run in ce kd; do
    python -m cifar10.kd evaluate "${COMMON[@]}" --checkpoint "$OUT/${PREFIX}${run}_seed$SEED/best.pth"
done
