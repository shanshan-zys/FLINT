#!/bin/bash
set -e

TRAIN_DATA="data/train.json"
TEST_DATA="data/test.json"

# ====== 主训练：coordinate tokens ======
echo "=== Training: main (coord tokens) ==="
python train.py \
  --config config.yaml \
  --train_data $TRAIN_DATA \
  --test_data $TEST_DATA \
  --use_coord_tokens \
  --physics none \
  --output_dir outputs/main

# ====== 消融1: 原生数字 (不用特殊token) ======
echo "=== Ablation: raw numbers ==="
python train.py \
  --config config.yaml \
  --train_data $TRAIN_DATA \
  --test_data $TEST_DATA \
  --no_coord_tokens \
  --physics none \
  --output_dir outputs/ablation_raw_numbers

# ====== 消融2: coord tokens + collision loss ======
echo "=== Ablation: physics collision ==="
python train.py \
  --config config.yaml \
  --train_data $TRAIN_DATA \
  --test_data $TEST_DATA \
  --use_coord_tokens \
  --physics collision \
  --output_dir outputs/ablation_collision

# ====== 消融3: coord tokens + all physics ======
echo "=== Ablation: physics all ==="
python train.py \
  --config config.yaml \
  --train_data $TRAIN_DATA \
  --test_data $TEST_DATA \
  --use_coord_tokens \
  --physics all \
  --output_dir outputs/ablation_physics_all

echo "All training done!"
