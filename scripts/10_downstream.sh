#!/bin/bash
# Downstream validation: train trajectory predictor on generated data
set -e
cd "$(dirname "$0")/.."

COMMON="--obs_len 8 --pred_len 12 --predictor_epochs 50 --predictor_lr 1e-3 --predictor_batch_size 64"

python source/downstream.py \
    --flint_results results/main \
    --original_data data/processed \
    --output_dir downstream/main \
    $COMMON

python source/downstream.py \
    --flint_results results/ablation_raw_numbers \
    --original_data data/processed \
    --output_dir downstream/ablation_raw_numbers \
    $COMMON
