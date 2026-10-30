#!/bin/bash
# Source from a cluster job; keep all environments and caches in this project.
export CNN_PROJECT=/projects/ie4483catsdogs
export TMPDIR="$CNN_PROJECT/tmp/${SLURM_JOB_ID:-session}"
export CONDA_ENVS_PATH="$CNN_PROJECT/envs"
export CONDA_PKGS_DIRS="$CNN_PROJECT/cache/conda"
export PIP_CACHE_DIR="$CNN_PROJECT/cache/pip"
export HF_HOME="$CNN_PROJECT/cache/hf"
export TRITON_CACHE_DIR="$CNN_PROJECT/cache/triton"
export XDG_CACHE_HOME="$CNN_PROJECT/cache/xdg"
export MPLCONFIGDIR="$CNN_PROJECT/cache/matplotlib"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
umask 077
mkdir -p "$TMPDIR" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" \
    "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
source /etc/profile.d/z00-lmod.sh
module load Miniforge3/24.11.3-1
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
