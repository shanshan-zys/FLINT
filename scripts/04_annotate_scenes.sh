#!/bin/bash
# 生成 5 个场景描述 prompt，标注后保存为 scene_descriptions.json
set -e
cd "$(dirname "$0")/.."

python annotate.py \
    --prepare_scene_prompts \
    --data_dir data/trajectories \
    --output prompts/scenes

echo ""
echo "标注完成后，将 5 个场景的描述保存为 scene_descriptions.json："
echo '{"eth": "...", "hotel": "...", "univ": "...", "zara1": "...", "zara2": "..."}'
