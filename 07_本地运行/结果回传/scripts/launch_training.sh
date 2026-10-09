#!/usr/bin/env bash
# Launch all training stages in parallel on the single GPU (each run is small; GPU is under-used by one process).
cd "$(dirname "$0")/.."
PY="/d/swc PINN/.venv-fusion-experiments/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
"$PY" scripts/rerun46_train.py logo                       > logs/train_logo.txt 2>&1 &
"$PY" scripts/rerun46_train.py noise --only noise_1p0mV   > logs/train_noise_1p0mV.txt 2>&1 &
"$PY" scripts/rerun46_train.py noise --only noise_2p0mV   > logs/train_noise_2p0mV.txt 2>&1 &
"$PY" scripts/rerun46_train.py nmc_matrix --part 0/2      > logs/train_nmc_matrix_0.txt 2>&1 &
"$PY" scripts/rerun46_train.py nmc_matrix --part 1/2      > logs/train_nmc_matrix_1.txt 2>&1 &
wait
echo all-training-done
