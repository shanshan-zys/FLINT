#!/bin/bash
# API crowd dynamics annotation (Gemini with video support)
# Requires GOOGLE_API_KEY environment variable
set -e
cd "$(dirname "$0")/.."

python source/annotate.py \
    --api_annotate \
    --data_dir data/processed \
    --scene_descriptions scene_descriptions.json \
    --output data/annotations.json \
    --model gemini-2.5-flash

python source/annotate.py \
    --merge_scenes \
    --annotations data/annotations.json \
    --scene_descriptions scene_descriptions.json
