#!/bin/bash
# Tokenizer roundtrip test
set -e
cd "$(dirname "$0")/.."

python source/prepare_data.py \
    --test_tokenizer \
    --bin_size 5 \
    --resolution_h 480 \
    --resolution_w 640
