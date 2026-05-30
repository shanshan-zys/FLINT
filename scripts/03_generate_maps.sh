#!/bin/bash
# 交互式标注可行走区域（每个场景一次，需要 GUI）
set -e
cd "$(dirname "$0")/.."

python preprocess.py \
    --output_dir data \
    --generate_maps
