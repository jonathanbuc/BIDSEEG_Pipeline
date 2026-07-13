#!/usr/bin/env bash
# Run the hierarchical HSSM fit on the GPU via the WSL `hssm-gpu` env (jax[cuda12]).
# Native-Windows jax is CPU-only, so the GPU fit must go through WSL.
#
# Usage (from a Windows shell, in the project root):
#     wsl bash run_hssm_gpu.sh [config.json] [tag]
# Usage (from inside WSL, in the project root):
#     ./run_hssm_gpu.sh [config.json] [tag]
#
# config.json defaults to inputs_openneuro.json. tag defaults to the active jax backend
# (gpu/cpu), so outputs (hssm_posterior_summary_<tag>.csv, hssm_idata_<tag>.nc) don't clobber.
# Force a same-env CPU baseline:  JAX_PLATFORMS=cpu wsl bash run_hssm_gpu.sh <config.json> cpu
set -euo pipefail

CONFIG="${1:-inputs_openneuro.json}"
TAG="${2:-}"

cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hssm-gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't grab all 6 GB up front

echo "config=$CONFIG  cwd=$(pwd)"
python hssm_gpu_runner.py "$CONFIG" $TAG
