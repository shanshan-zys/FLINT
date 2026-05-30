#!/bin/bash
# Annotate scenario descriptions and crowd dynamics via API
set -e
cd "$(dirname "$0")/.."

API_KEY="bsk-9ea517c77d92cbda3e05b174e0b4d236"
BASE_URL="http://llmapi.bilibili.co/v1"
SUBSETS="eth hotel univ zara1 zara2"

# Annotate scenario (background image + first clip trajectory samples)
python source/annotate.py \
    --annotate_scenario \
    --data_dir data/processed \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --subsets $SUBSETS

# Annotate crowd dynamics (video + trajectory data)
python source/annotate.py \
    --annotate_crowd \
    --data_dir data/processed \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --subsets $SUBSETS
