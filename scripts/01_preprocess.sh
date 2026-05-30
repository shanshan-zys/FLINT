#!/bin/bash
# 预处理：坐标转换 + 切 clip + 视频 + 可视化
set -e
cd "$(dirname "$0")/.."

python preprocess.py \
    --opentraj_dir ../OpenTraj \
    --output_dir data
