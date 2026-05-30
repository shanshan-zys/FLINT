#!/bin/bash
# 评估：所有模型 + 模型对比
set -e
cd "$(dirname "$0")/.."

python evaluate.py \
    --results_dir results/main \
    --test_data data/test.json \
    --config config.yaml \
    --output_dir eval/main

python evaluate.py \
    --results_dir results/ablation_raw_numbers \
    --test_data data/test.json \
    --config config.yaml \
    --output_dir eval/ablation_raw_numbers

python evaluate.py \
    --results_dir results/ablation_no_physics \
    --test_data data/test.json \
    --config config.yaml \
    --output_dir eval/ablation_no_physics

python evaluate.py \
    --results_dir results/ablation_collision_only \
    --test_data data/test.json \
    --config config.yaml \
    --output_dir eval/ablation_collision_only

# 模型对比
python evaluate.py \
    --compare eval/main eval/ablation_raw_numbers eval/ablation_no_physics eval/ablation_collision_only \
    --output_dir eval/comparison
