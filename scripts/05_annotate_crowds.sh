#!/bin/bash
# API 自动标注人群动态（调用 Gemini，支持视频）
# 需要设置 GOOGLE_API_KEY 环境变量
set -e
cd "$(dirname "$0")/.."

python annotate.py \
    --api_annotate \
    --data_dir data/trajectories \
    --video_dir data/clips \
    --scene_descriptions scene_descriptions.json \
    --output data/annotations.json \
    --model gemini-2.5-flash

# 合并场景描述到标注文件
python annotate.py \
    --merge_scenes \
    --annotations data/annotations.json \
    --scene_descriptions scene_descriptions.json
