#!/bin/bash
# Train main model (coord tokens + physics loss)
set -e
cd "$(dirname "$0")/.."

python source/train.py \
    --train_data data/train.json \
    --test_data data/test.json \
    --use_coord_tokens \
    --physics all \
    --output_dir outputs/main \
    --backbone Qwen/Qwen3-8B-Instruct \
    --max_seq_length 8192 \
    --epochs 20 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --early_stopping_patience 3 \
    --bin_size 5 \
    --resolution_h 480 \
    --resolution_w 640 \
    --processed_dir data/processed \
    --map_grid_size 10 \
    --collision_weight 0.1 \
    --smoothness_weight 0.05 \
    --walkable_weight 0.05 \
    --collision_threshold 10.0
