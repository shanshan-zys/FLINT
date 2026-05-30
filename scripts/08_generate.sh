#!/bin/bash
# Generate trajectories for all models
set -e
cd "$(dirname "$0")/.."

SAMPLES=20
TEMP=0.8
TOP_P=0.95
SEQ_LEN=8192
BIN=5
RES_H=480
RES_W=640

COMMON="--num_samples $SAMPLES --temperature $TEMP --top_p $TOP_P \
    --max_seq_length $SEQ_LEN --bin_size $BIN \
    --resolution_h $RES_H --resolution_w $RES_W"

python source/generate.py \
    --model_path outputs/main/best \
    --test_data data/test.json \
    --output_dir results/main \
    $COMMON

python source/generate.py \
    --model_path outputs/ablation_raw_numbers/best \
    --test_data data/test.json \
    --output_dir results/ablation_raw_numbers \
    $COMMON

python source/generate.py \
    --model_path outputs/ablation_no_physics/best \
    --test_data data/test.json \
    --output_dir results/ablation_no_physics \
    $COMMON

python source/generate.py \
    --model_path outputs/ablation_collision_only/best \
    --test_data data/test.json \
    --output_dir results/ablation_collision_only \
    $COMMON
