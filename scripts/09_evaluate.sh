#!/bin/bash
# Evaluate all models + comparison
set -e
cd "$(dirname "$0")/.."

COMMON="--bin_size 5 --resolution_h 480 --resolution_w 640 --llm_judges deepseek-chat"

python source/evaluate.py \
    --results_dir results/main \
    --test_data data/test.json \
    --output_dir eval/main \
    $COMMON

python source/evaluate.py \
    --results_dir results/ablation_raw_numbers \
    --test_data data/test.json \
    --output_dir eval/ablation_raw_numbers \
    $COMMON

python source/evaluate.py \
    --results_dir results/ablation_no_physics \
    --test_data data/test.json \
    --output_dir eval/ablation_no_physics \
    $COMMON

python source/evaluate.py \
    --results_dir results/ablation_collision_only \
    --test_data data/test.json \
    --output_dir eval/ablation_collision_only \
    $COMMON

python source/evaluate.py \
    --compare eval/main eval/ablation_raw_numbers eval/ablation_no_physics eval/ablation_collision_only \
    --output_dir eval/comparison
