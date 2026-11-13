#!/bin/bash
#SBATCH -A ug-proj
#SBATCH --qos=ug
#SBATCH --gres=gpu:6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=cnn-sweep
#SBATCH --time=12:00:00
# Custom-CNN hyperparameter sweep on Dogs vs. Cats: LR x weight decay x dropout = 12 runs.
# Everything else is the Experiment A recipe (30 epochs, batch 32, AdamW, ReduceLROnPlateau,
# early stopping on val loss, seed 42). Best epoch per run is chosen on validation accuracy.
#
# The ug QOS allows only 5 submitted / 2 running jobs per user, so instead of a 12-task array
# the grid is split into shards that each run their configurations one after another:
#   mkdir -p logs
#   sbatch custom_cnn/cluster/sweep.sbatch 0 2     # runs 0, 2, 4, ..., 10
#   sbatch custom_cnn/cluster/sweep.sbatch 1 2     # runs 1, 3, 5, ..., 11
# Re-submitting a shard skips finished runs and resumes an interrupted one.
#   python -m custom_cnn.sweep_summary          # results table once runs finish
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
eval "$(/cluster/apps/software/Miniforge3/24.11.3-1/bin/conda shell.bash hook)"
conda activate "${ENV:-/cluster/apps/software/Miniforge3/24.11.3-1}"

LRS=(3e-4 1e-3 3e-3)
WDS=(1e-4 1e-3)
DROPOUTS=(0.3 0.5)
SHARD=${1:-0}
NSHARDS=${2:-1}
export PYTHONUNBUFFERED=1
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"

for ((i = SHARD; i < 12; i += NSHARDS)); do
    LR=${LRS[$((i / 4))]}
    WD=${WDS[$((i / 2 % 2))]}
    DO=${DROPOUTS[$((i % 2))]}
    OUT=custom_cnn/outputs/sweep/lr${LR}_wd${WD}_do${DO}
    echo "=== run $i: lr=$LR weight_decay=$WD dropout=$DO -> $OUT"
    if [ -f "$OUT/validation_metrics.json" ]; then
        echo "already done"; continue
    elif [ -f "$OUT/latest.pth" ]; then   # interrupted earlier: continue the same run
        python -m custom_cnn.train --device cuda --resume "$OUT/latest.pth" --num-workers 4 \
            || echo "resume ended without training (e.g. already early-stopped)"
    else
        python -m custom_cnn.train --device cuda --data-dir "${DATA:-datasets}" --seed 42 \
            --lr "$LR" --weight-decay "$WD" --dropout "$DO" --num-workers 4 --output-dir "$OUT"
    fi
    python -m custom_cnn.evaluate --device cuda --data-dir "${DATA:-datasets}" \
        --checkpoint "$OUT/best.pth" --num-workers 4 | grep -E '"accuracy"|"macro_f1"|"epoch"'
done
python -m custom_cnn.sweep_summary || true
