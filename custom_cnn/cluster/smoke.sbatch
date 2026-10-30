#!/bin/bash
#SBATCH --job-name=ie4483-gpu-check
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --gres=gpu:pro6000:1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/ie4483catsdogs/repo/custom_cnn/outputs/setup/gpu-check-%j.log
set -euo pipefail
source /projects/ie4483catsdogs/repo/custom_cnn/cluster/env.sh
conda activate "$CNN_PROJECT/envs/ie4483"
cd "$CNN_PROJECT/repo"
nvidia-smi
python custom_cnn/cluster/gpu_check.py
