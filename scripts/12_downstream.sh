#!/bin/bash
# 下游验证：用生成数据训练轨迹预测器
set -e
cd "$(dirname "$0")/.."

python downstream.py \
    --config config.yaml \
    --generated_dir results/main \
    --test_data data/test.json \
    --output_dir downstream/main

python downstream.py \
    --config config.yaml \
    --generated_dir results/ablation_raw_numbers \
    --test_data data/test.json \
    --output_dir downstream/ablation_raw_numbers
