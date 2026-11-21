#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
test "$(id -un)" = czr
source /home/czr/PRISM/activate_PRISM.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd /data1/Chenzr/PRISM
exec python -u experiment_runs/prism_full_suite_20261111_v2/a800_adaptive.py
