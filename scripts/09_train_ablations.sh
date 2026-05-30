#!/bin/bash
# 训练消融模型（3 组）
set -e
cd "$(dirname "$0")/.."

# 消融 1：raw numbers（无 coord tokens）
python train.py \
    --config config.yaml \
    --train_data data/train.json \
    --test_data data/test.json \
    --no_coord_tokens \
    --physics none \
    --output_dir outputs/ablation_raw_numbers

# 消融 2：无 physics loss
python train.py \
    --config config.yaml \
    --train_data data/train.json \
    --test_data data/test.json \
    --use_coord_tokens \
    --physics none \
    --output_dir outputs/ablation_no_physics

# 消融 3：仅 collision loss
python train.py \
    --config config.yaml \
    --train_data data/train.json \
    --test_data data/test.json \
    --use_coord_tokens \
    --physics collision \
    --output_dir outputs/ablation_collision_only
