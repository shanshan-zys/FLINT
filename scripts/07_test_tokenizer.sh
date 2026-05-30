#!/bin/bash
# Tokenizer roundtrip 测试
set -e
cd "$(dirname "$0")/.."

python prepare_data.py --config config.yaml --test_tokenizer
