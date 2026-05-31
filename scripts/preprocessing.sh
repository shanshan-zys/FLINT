#!/bin/bash
# Preprocess ETH-UCY: clips + videos + visualization + backgrounds + walkable maps
set -e
cd "$(dirname "$0")/.."

COMMON="--target_w 640 --target_h 480 --clip_len 25 --fps 2.5"

# Step 1: coordinate conversion, clip splitting, video extraction, visualization
python source/preprocessing.py $COMMON

# Step 2: generate clean background images
python source/preprocessing.py --generate_backgrounds $COMMON

# Step 3: interactive walkable area annotation (requires GUI, 5 scenes)
python source/preprocessing.py --generate_maps $COMMON
