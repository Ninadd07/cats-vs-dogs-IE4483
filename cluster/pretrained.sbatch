#!/bin/bash
#SBATCH -A ug-proj
#SBATCH --qos=ug
#SBATCH --gres=gpu:6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=catsdogs-pretrained
#SBATCH --time=06:00:00
# Fine-tune an ImageNet-pretrained ResNet-50 or ViT-B/16 on Dogs vs. Cats with the root trainer.
#
#   mkdir -p logs
#   sbatch --job-name=catsdogs-resnet cluster/pretrained.sbatch resnet
#   sbatch --job-name=catsdogs-vit    cluster/pretrained.sbatch vit
# Runs one fine-tuning per seed in SEEDS (default "42 43 44"), one after another, then prints
# mean/std of best validation accuracy. Finished seeds are skipped on resubmission.
# Extra train.py flags can follow the model name, e.g. ... vit --freeze_backbone --lr 1e-3
# Overrides: SEEDS="42", AUG=none|flip|enhanced (default flip), DATA=/path/to/datasets (train/ val/ test/), ENV=...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
ENV=${ENV:-/cluster/apps/software/Miniforge3/24.11.3-1}
if [ -f "$ENV/bin/activate" ] && [ ! -f "$ENV/bin/conda" ]; then
    source "$ENV/bin/activate"
else
    eval "$(/cluster/apps/software/Miniforge3/24.11.3-1/bin/conda shell.bash hook)"
    conda activate "$ENV"
fi

MODEL=${1:?usage: sbatch cluster/pretrained.sbatch resnet|vit [train.py flags]}
shift
DATA=${DATA:-datasets}
for d in train/cat train/dog val/cat val/dog test; do
    [ -d "$DATA/$d" ] || { echo "Missing $DATA/$d"; exit 1; }
done
case $MODEL in
    resnet) LR=1e-4 ;;   # full fine-tuning, AdamW
    vit)    LR=3e-5 ;;   # ViTs fine-tune best at a lower LR
    *) echo "MODEL must be resnet or vit"; exit 1 ;;
esac
SEEDS=${SEEDS:-42 43 44}
AUG=${AUG:-flip}
export PYTHONUNBUFFERED=1
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"

for SEED in $SEEDS; do
    OUT=outputs/pretrained/${MODEL}_${AUG}_seed${SEED}
    if grep -q "Saved test predictions" "$OUT/train.log" 2>/dev/null; then
        echo "=== $MODEL $AUG seed $SEED already done"; continue
    fi
    mkdir -p "$OUT"
    echo "=== $MODEL $AUG seed $SEED -> $OUT"
    python train.py --model "$MODEL" --pretrained --data_dir "$DATA" --epochs 10 --batch_size 64 \
        --lr "$LR" --seed "$SEED" --augmentation "$AUG" --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
        --save_path "$OUT/best.pth" --pred_path "$OUT/test_predictions.csv" "$@" \
        2>&1 | grep --line-buffered -v "Batch \[" | tee "$OUT/train.log"
done

python - "$MODEL" "$AUG" <<'EOF2'
import re, statistics, sys
from pathlib import Path
model, aug = sys.argv[1:]
accs = []
runs = sorted(Path("outputs/pretrained").glob(f"{model}_{aug}_seed*/train.log"))
for path in runs:
    log = path.read_text()
    if "Saved test predictions" not in log:
        continue
    seed = path.parent.name.rsplit("seed", 1)[1]
    epochs = [float(a) for a in re.findall(r"\] Train Loss.*Val Acc: ([0-9.]+)", log)]
    best = max(epochs)
    accs.append(best)
    print(f"{model} seed {seed}: best val acc {best:.4f} (epoch {epochs.index(best) + 1}), "
          f"final {epochs[-1]:.4f}")
if len(accs) > 1:
    print(f"{model} ({aug}): mean {statistics.mean(accs):.4f} +/- {statistics.stdev(accs):.4f} "
          f"(sample std, n={len(accs)})")
EOF2
