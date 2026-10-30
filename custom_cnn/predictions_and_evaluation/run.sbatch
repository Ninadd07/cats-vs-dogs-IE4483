#!/bin/bash
#SBATCH --job-name=ie4483-stage5
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/projects/ie4483catsdogs/repo/custom_cnn/predictions_and_evaluation/outputs/stage5-%j.log
set -euo pipefail
source /projects/ie4483catsdogs/repo/custom_cnn/cluster/env.sh
conda activate "$CNN_PROJECT/envs/ie4483"
cd "$CNN_PROJECT/repo"
python -m custom_cnn.predictions_and_evaluation.predict \
    --checkpoint custom_cnn/outputs/run_a/best.pth --data-dir "$CNN_PROJECT/datasets" \
    --device cpu --batch-size 32 --num-workers 2
python -m custom_cnn.predictions_and_evaluation.analyze --data-dir "$CNN_PROJECT/datasets"
echo "IE4483_STAGE5_COMPLETE"
