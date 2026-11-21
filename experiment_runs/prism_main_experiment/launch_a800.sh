#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
PRISM_RUN=/data1/Chenzr/PRISM/server_runs/a800_tuning_20261112
mkdir -p "$PRISM_RUN"
exec 9>"$PRISM_RUN/campaign.lock"
flock -n 9 || exit 75
trap 'code=$?; printf "%s\n" "$code" > "$PRISM_RUN/exit_code"' EXIT
test "$(id -un)" = czr
source /home/czr/PRISM/activate_PRISM.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PRISM_TUNING_EPOCHS=200
cd /data1/Chenzr/PRISM
python -u experiment_runs/prism_full_suite_20261111_v2/check_a800_tuning.py
python -u experiment_runs/prism_full_suite_20261111_v2/a800_tuning.py
