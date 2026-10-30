#!/bin/bash
#SBATCH --job-name=ie4483-evaluate
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/projects/ie4483catsdogs/repo/custom_cnn/outputs/evaluate-%j.log
set -euo pipefail
source /projects/ie4483catsdogs/repo/custom_cnn/cluster/env.sh
conda activate "$CNN_PROJECT/envs/ie4483"
cd "$CNN_PROJECT/repo"
python -m custom_cnn.evaluate --checkpoint custom_cnn/outputs/run_a/best.pth \
    --data-dir "$CNN_PROJECT/datasets" --device cpu --batch-size 32 --num-workers 2
python -m custom_cnn.visualize --history custom_cnn/outputs/run_a/training_history.csv
echo "IE4483_EVALUATION_COMPLETE"
