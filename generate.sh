#!/bin/bash
set -e
K=20   # 每个prompt采样次数
T=0.8  # 温度

# ====== 主模型生成 ======
echo "=== Generating from main model ==="
python generate.py \
  --model_path outputs/main/best \
  --config config.yaml \
  --test_data data/test.json \
  --num_samples $K \
  --temperature $T \
  --output_dir results/main

# ====== 消融模型生成 ======
for ablation in ablation_raw_numbers ablation_collision ablation_physics_all; do
  echo "=== Generating from $ablation ==="
  python generate.py \
    --model_path outputs/$ablation/best \
    --config config.yaml \
    --test_data data/test.json \
    --num_samples $K \
    --temperature $T \
    --output_dir results/$ablation
done

# ====== 自定义prompt演示 ======
echo "=== Custom prompt demo ==="
python generate.py \
  --model_path outputs/main/best \
  --config config.yaml \
  --custom_prompt "15 pedestrians crossing a wide plaza from left to right, 3 stopping in the center" \
  --num_agents 15 \
  --num_steps 25 \
  --num_samples $K \
  --temperature $T \
  --output_dir results/custom_demo

echo "All generation done!"
