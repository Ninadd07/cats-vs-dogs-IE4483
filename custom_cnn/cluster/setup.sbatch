#!/bin/bash
#SBATCH --job-name=ie4483-setup
#SBATCH --partition=cluster02
#SBATCH --account=ug-proj
#SBATCH --qos=ug
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/projects/ie4483catsdogs/tmp/setup-%j.log
set -euo pipefail
export CNN_PROJECT=/projects/ie4483catsdogs
# Bootstrap copies live in tmp until the source archive is extracted.
source "$CNN_PROJECT/tmp/env.sh"
cd "$CNN_PROJECT/tmp"
sha256sum -c SHA256SUMS
if [[ -e "$CNN_PROJECT/repo/custom_cnn" || -e "$CNN_PROJECT/datasets/train" ]]; then
    echo "Source/data already exist; refusing to overwrite an existing deployment." >&2
    exit 1
fi
tar --no-same-owner -xzf code.tar.gz -C "$CNN_PROJECT/repo"
tar --no-same-owner -xzf datasets.tar.gz -C "$CNN_PROJECT/datasets"
conda create --yes --prefix "$CNN_PROJECT/envs/ie4483" \
    --override-channels --channel conda-forge python=3.12 pip
conda activate "$CNN_PROJECT/envs/ie4483"
python -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
cd "$CNN_PROJECT/repo"
python -m pip install --no-cache-dir -r custom_cnn/requirements.txt
python -m pip check
mkdir -p custom_cnn/outputs/setup
python -m pip freeze > custom_cnn/outputs/setup/requirements.lock.txt
python -m custom_cnn.verify --data-dir "$CNN_PROJECT/datasets"
python - <<'PY'
import json, os, platform
from pathlib import Path
import torch, torchvision
from datapreprocessing import CatsDogsDataLoader
data = CatsDogsDataLoader(Path(os.environ['CNN_PROJECT']) / 'datasets')
summary = {'python': platform.python_version(), 'torch': str(torch.__version__),
           'torchvision': str(torchvision.__version__), 'cuda_build': torch.version.cuda,
           'cuda_architectures': torch.cuda.get_arch_list(),
           'class_mapping': data.get_class_to_idx(),
           'train_images': len(data.train_dataset), 'val_images': len(data.val_dataset),
           'test_images': len(data.test_dataset)}
assert (summary['train_images'], summary['val_images'], summary['test_images']) == (20000, 5000, 500)
Path('custom_cnn/outputs/setup/environment.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
echo "IE4483_CPU_SETUP_COMPLETE"
