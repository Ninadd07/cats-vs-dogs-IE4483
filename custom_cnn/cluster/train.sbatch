#!/bin/bash
#SBATCH --job-name=ie4483-cnn
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --gres=gpu:pro6000:1
#SBATCH --time=04:00:00
#SBATCH --output=/projects/ie4483catsdogs/repo/custom_cnn/outputs/train-%j.log
set -euo pipefail
source /projects/ie4483catsdogs/repo/custom_cnn/cluster/env.sh
conda activate "$CNN_PROJECT/envs/ie4483"
cd "$CNN_PROJECT/repo"
nvidia-smi
# With no arguments this uses the agreed 30-epoch baseline defaults. For a resume,
# supply: sbatch custom_cnn/cluster/train.sbatch --resume .../latest.pth --epochs 30
python -m custom_cnn.train --device cuda --data-dir "$CNN_PROJECT/datasets" \
    --output-dir custom_cnn/outputs/run_a "$@"
