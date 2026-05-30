#!/bin/bash
# 生成：所有模型（主模型 + 消融）
set -e
cd "$(dirname "$0")/.."

python generate.py \
    --config config.yaml \
    --model_dir outputs/main/best \
    --test_data data/test.json \
    --output_dir results/main \
    --num_samples 20

python generate.py \
    --config config.yaml \
    --model_dir outputs/ablation_raw_numbers/best \
    --test_data data/test.json \
    --output_dir results/ablation_raw_numbers \
    --num_samples 20

python generate.py \
    --config config.yaml \
    --model_dir outputs/ablation_no_physics/best \
    --test_data data/test.json \
    --output_dir results/ablation_no_physics \
    --num_samples 20

python generate.py \
    --config config.yaml \
    --model_dir outputs/ablation_collision_only/best \
    --test_data data/test.json \
    --output_dir results/ablation_collision_only \
    --num_samples 20
