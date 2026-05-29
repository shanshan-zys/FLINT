#!/bin/bash
set -e

echo "=== Downstream: trajectory prediction augmentation ==="
python downstream.py \
  --flint_results results/main \
  --original_data data/raw/eth_ucy \
  --config config.yaml \
  --output_dir eval/downstream

echo "Downstream experiment done!"
