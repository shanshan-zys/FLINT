#!/bin/bash
# Generate 5 scene description prompts, then save as scene_descriptions.json
set -e
cd "$(dirname "$0")/.."

python source/annotate.py \
    --prepare_scene_prompts \
    --data_dir data/processed \
    --output prompts/scenes

echo ""
echo "Save the 5 scene descriptions to scene_descriptions.json:"
echo '{"eth": "...", "hotel": "...", "univ": "...", "zara1": "...", "zara2": "..."}'
