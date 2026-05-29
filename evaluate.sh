#!/bin/bash
set -e

# ====== 评估所有模型 ======
for model in main ablation_raw_numbers ablation_collision ablation_physics_all; do
  echo "=== Evaluating $model ==="
  python evaluate.py \
    --results_dir results/$model \
    --test_data data/test.json \
    --config config.yaml \
    --output_dir eval/$model
done

# ====== 模型对比 ======
echo "=== Model comparison ==="
python evaluate.py \
  --compare eval/main eval/ablation_raw_numbers eval/ablation_collision eval/ablation_physics_all \
  --output_dir eval/comparison

echo "All evaluation done!"
