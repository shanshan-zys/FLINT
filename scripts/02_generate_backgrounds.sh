#!/bin/bash
# 生成背景图（每个场景一张干净背景）
set -e
cd "$(dirname "$0")/.."

python preprocess.py \
    --opentraj_dir ../OpenTraj \
    --output_dir data \
    --generate_backgrounds
