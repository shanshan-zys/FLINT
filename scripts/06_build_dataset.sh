#!/bin/bash
# 构建 SFT 数据集 + 切分 train/test
set -e
cd "$(dirname "$0")/.."

# coord tokens 模式（主模型）
python prepare_data.py \
    --config config.yaml \
    --annotations data/annotations.json \
    --output data/sft.json

python prepare_data.py \
    --config config.yaml \
    --split \
    --annotations data/sft.json

# raw numbers 模式（消融对比）
python prepare_data.py \
    --config config.yaml \
    --annotations data/annotations.json \
    --output data/sft_raw.json \
    --no_coord_tokens

python prepare_data.py \
    --config config.yaml \
    --split \
    --annotations data/sft_raw.json
