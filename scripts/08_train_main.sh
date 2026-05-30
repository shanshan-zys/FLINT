#!/bin/bash
# 训练主模型（coord tokens + physics loss）
set -e
cd "$(dirname "$0")/.."

python train.py \
    --config config.yaml \
    --train_data data/train.json \
    --test_data data/test.json \
    --use_coord_tokens \
    --physics all \
    --output_dir outputs/main
