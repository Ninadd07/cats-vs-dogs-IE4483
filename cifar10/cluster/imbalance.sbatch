#!/bin/bash
#SBATCH -A ug-proj
#SBATCH --qos=ug
#SBATCH --gres=gpu:6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=cifar10-imbalance
#SBATCH --time=06:00:00
# CIFAR-10 with class imbalance: bird, cat, deer, dog, frog keep only 5% of their labelled
# images (225 vs 4,500 per class, 20x imbalance; validation reduced the same way). Trains the same
# student with each remedy and evaluates all of them on the balanced official test set.
#
#   mkdir -p logs && sbatch cifar10/cluster/imbalance.sbatch [SEED]
# Overrides: ARCH=cnn, MINORITY="3 5", FRACTION=0.01, DATA=..., ENV=...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
ENV=${ENV:-/cluster/apps/software/Miniforge3/24.11.3-1}
if [ -f "$ENV/bin/activate" ] && [ ! -f "$ENV/bin/conda" ]; then
    source "$ENV/bin/activate"
else
    eval "$(/cluster/apps/software/Miniforge3/24.11.3-1/bin/conda shell.bash hook)"
    conda activate "$ENV"
fi

SEED=${1:-42}
DATA=${DATA:-datasets/cifar10}
ARCH=${ARCH:-resnet9}
MINORITY=${MINORITY:-2 3 4 5 6}
FRACTION=${FRACTION:-0.05}
TAG="imb_f${FRACTION}"
OUT=cifar10/outputs/$TAG
COMMON=(--device cuda --data-dir "$DATA" --num-workers "${SLURM_CPUS_PER_TASK:-4}")
read -r -a IMB <<< "--minority-classes $MINORITY --minority-fraction $FRACTION"
if [ "$ARCH" = resnet9 ]; then
    read -r -a TRAIN <<< "--arch resnet9 --optimizer sgd --lr 0.1 --weight-decay 5e-4 --dropout 0"
else
    TRAIN=(--arch "$ARCH")
fi
export PYTHONUNBUFFERED=1
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"

# Teacher refit on the SAME imbalanced subset (class-balanced logistic regression); reuses the
# DINOv2 feature cache from the balanced teacher, so this is only a few minutes of CPU.
[ -f "$OUT/teacher/teacher.npz" ] || python -m cifar10.kd teacher "${COMMON[@]}" "${IMB[@]}" \
    --balanced-teacher --cache-dir cifar10/outputs/teacher --out "$OUT/teacher"

declare -A METHODS=(
    [ce]="--kd-alpha 0"
    [cbweights]="--kd-alpha 0 --class-weights effective --cb-beta 0.999"
    [balsampler]="--kd-alpha 0 --sampler balanced"
    [logitadj]="--kd-alpha 0 --logit-adjust 1.0"
    [kd]="--teacher $OUT/teacher/teacher.npz --kd-alpha 0.5 --kd-temperature 4"
    [kd_logitadj]="--teacher $OUT/teacher/teacher.npz --kd-alpha 0.5 --kd-temperature 4 --logit-adjust 1.0"
)
for method in ce cbweights balsampler logitadj kd kd_logitadj; do
    run="$OUT/${ARCH}_${method}_seed$SEED"
    read -r -a EXTRA <<< "${METHODS[$method]}"
    echo "=== $method -> $run"
    [ -f "$run/best.pth" ] || python -m cifar10.kd train "${COMMON[@]}" "${IMB[@]}" "${TRAIN[@]}" \
        "${EXTRA[@]}" --seed "$SEED" --output-dir "$run"
    python -m cifar10.kd evaluate "${COMMON[@]}" --checkpoint "$run/best.pth"
done

python - "$OUT" "$ARCH" "$SEED" <<'EOF'
import json, sys
from pathlib import Path
out, arch, seed = sys.argv[1:]
print(f"\n{'method':<13}{'test acc':>9}{'macro F1':>9}{'minority':>9}{'majority':>9}")
for d in sorted(Path(out).glob(f"{arch}_*_seed{seed}")):
    m = json.loads((d / "test_metrics.json").read_text())
    name = d.name[len(arch) + 1:-len(f"_seed{seed}")]
    print(f"{name:<13}{m['test_accuracy']:>9.4f}{m['test_macro_f1']:>9.4f}"
          f"{m['test_minority_mean_recall']:>9.4f}{m['test_majority_mean_recall']:>9.4f}")
EOF
