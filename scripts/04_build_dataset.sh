#!/bin/bash
# Build SFT dataset + train/test split
set -e
cd "$(dirname "$0")/.."

BIN_SIZE=5
RES_H=480
RES_W=640
PROCESSED_DIR=data/processed
MAP_GRID=10
TRAIN_RATIO=0.8

# coord tokens (main model)
python source/prepare_data.py \
    --annotations data/annotations.json \
    --output data/sft.json \
    --bin_size $BIN_SIZE \
    --resolution_h $RES_H \
    --resolution_w $RES_W \
    --processed_dir $PROCESSED_DIR \
    --map_grid_size $MAP_GRID

python source/prepare_data.py \
    --split \
    --annotations data/sft.json \
    --train_ratio $TRAIN_RATIO

# raw numbers (ablation)
python source/prepare_data.py \
    --annotations data/annotations.json \
    --output data/sft_raw.json \
    --no_coord_tokens \
    --bin_size $BIN_SIZE \
    --resolution_h $RES_H \
    --resolution_w $RES_W \
    --processed_dir $PROCESSED_DIR \
    --map_grid_size $MAP_GRID

python source/prepare_data.py \
    --split \
    --annotations data/sft_raw.json \
    --train_ratio $TRAIN_RATIO
