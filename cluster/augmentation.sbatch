#!/bin/bash
#SBATCH -A ug-proj
#SBATCH --qos=ug
#SBATCH --gres=gpu:6000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --job-name=catsdogs-augmentation
#SBATCH --time=08:00:00
# Dogs vs. Cats augmentation study, seed 42, all runs one after another on one GPU:
#   none     = resize to 224x224 + ImageNet normalisation (identical to evaluation)
#   flip     = none + random horizontal flip                       (CNN name: baseline)
#   enhanced = random resized crop (scale 0.8-1, ratio 0.9-1.1) + flip + rotation +-10 deg
#              + colour jitter (brightness/contrast/saturation 0.1)
# Definitions live in datapreprocessing/transforms.py and are shared by both trainers.
#
#   ResNet-50 / ViT-B/16: none, flip, enhanced    (pretrained, 10 epochs, batch 64)
#   Custom CNN:           none, flip              (enhanced = sweep run lr1e-3_wd1e-4_do0.3)
#
#   mkdir -p logs && sbatch cluster/augmentation.sbatch
# Resubmitting skips finished runs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
eval "$(/cluster/apps/software/Miniforge3/24.11.3-1/bin/conda shell.bash hook)"
conda activate "${ENV:-/cluster/apps/software/Miniforge3/24.11.3-1}"
DATA=${DATA:-datasets}
SEED=42
export PYTHONUNBUFFERED=1
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"

pretrained() {   # model aug lr
    local out=outputs/pretrained/${1}_${2}_seed${SEED}
    if grep -q "Saved test predictions" "$out/train.log" 2>/dev/null; then
        echo "=== $1 $2: already done"; return
    fi
    mkdir -p "$out"
    echo "=== $1 $2 -> $out"
    python train.py --model "$1" --pretrained --data_dir "$DATA" --epochs 10 --batch_size 64 \
        --lr "$3" --seed "$SEED" --augmentation "$2" --num_workers 4 \
        --save_path "$out/best.pth" --pred_path "$out/test_predictions.csv" \
        2>&1 | grep --line-buffered -v "Batch \[" | tee "$out/train.log"
}

cnn() {   # mode (none|baseline)
    local out=custom_cnn/outputs/augmentation/${1}_seed${SEED}
    if [ -f "$out/validation_metrics.json" ]; then
        echo "=== cnn $1: already done"; return
    fi
    echo "=== cnn $1 -> $out"
    if [ -f "$out/latest.pth" ]; then
        python -m custom_cnn.train --device cuda --resume "$out/latest.pth" --num-workers 4 \
            || echo "resume ended without training (e.g. already early-stopped)"
    else
        # Same recipe as the sweep's enhanced run: lr 1e-3, weight decay 1e-4, dropout 0.3.
        python -m custom_cnn.train --device cuda --data-dir "$DATA" --seed "$SEED" \
            --augmentation "$1" --lr 1e-3 --weight-decay 1e-4 --dropout 0.3 --num-workers 4 \
            --output-dir "$out"
    fi
    python -m custom_cnn.evaluate --device cuda --data-dir "$DATA" --checkpoint "$out/best.pth" \
        --num-workers 4 > /dev/null
}

pretrained resnet none 1e-4
pretrained resnet flip 1e-4
pretrained resnet enhanced 1e-4
cnn none
cnn baseline
pretrained vit none 3e-5
pretrained vit flip 3e-5
pretrained vit enhanced 3e-5

python - <<'EOF'
import json, re
from pathlib import Path

def root(model, aug):
    log = Path(f"outputs/pretrained/{model}_{aug}_seed42/train.log")
    accs = [float(a) for a in re.findall(r"\] Train Loss.*Val Acc: ([0-9.]+)", log.read_text())] if log.exists() else []
    return (max(accs), accs.index(max(accs)) + 1, accs[-1]) if accs else None

def cnn(path):
    p = Path(path) / "validation_metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return m["accuracy"], m["epoch"], None

rows = {"ResNet-50": {a: root("resnet", a) for a in ("none", "flip", "enhanced")},
        "ViT-B/16": {a: root("vit", a) for a in ("none", "flip", "enhanced")},
        "Custom CNN": {"none": cnn("custom_cnn/outputs/augmentation/none_seed42"),
                       "flip": cnn("custom_cnn/outputs/augmentation/baseline_seed42"),
                       "enhanced": cnn("custom_cnn/outputs/sweep/lr1e-3_wd1e-4_do0.3")}}
print("\nBest validation accuracy, seed 42 (best epoch in brackets)")
print(f"{'model':<12}{'none':>16}{'flip':>16}{'enhanced':>16}")
for model, r in rows.items():
    cell = lambda v: f"{v[0]*100:.2f}% (ep {v[1]})" if v else "-"
    print(f"{model:<12}" + "".join(f"{cell(r[a]):>16}" for a in ("none", "flip", "enhanced")))
EOF
