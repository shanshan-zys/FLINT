#!/bin/bash
# Build SFT dataset from processed clips and annotations
set -e
cd "$(dirname "$0")/.."

BIN_SIZE=5
RES_H=480
RES_W=640
PROCESSED_DIR=data/processed
MAP_GRID=10
OUTPUT=data/eth-ucy/eth-ucy.json

python3 source/dataset_construction.py \
    --data_dir $PROCESSED_DIR \
    --output $OUTPUT \
    --bin_size $BIN_SIZE \
    --resolution_h $RES_H \
    --resolution_w $RES_W \
    --map_grid_size $MAP_GRID
