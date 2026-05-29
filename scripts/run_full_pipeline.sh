#!/bin/bash
# ============================================================
# FLINT 完整实验 Pipeline
#
# 使用方式:
#   chmod +x scripts/run_full_pipeline.sh
#   bash scripts/run_full_pipeline.sh
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/config.yaml"
RAW_DATA_DIR="data/raw"
PROCESSED_DIR="data/processed"
SFT_DIR="data/sft"
OUTPUT_DIR="outputs"
EVAL_DIR="eval_results"
ABLATION_DIR="ablation_results"
DOWNSTREAM_DIR="downstream_results"

echo "============================================================"
echo "FLINT Experiment Pipeline"
echo "Project dir: $PROJECT_DIR"
echo "============================================================"

# ============================================================
# Step 0: 安装依赖
# ============================================================
echo ""
echo "[Step 0] Installing dependencies..."
pip install -q torch transformers datasets trl peft
pip install -q unsloth
pip install -q openai google-generativeai
pip install -q scipy scikit-learn matplotlib
pip install -q pyyaml pillow

# ============================================================
# Step 1: 数据预处理 & 标注
# ============================================================
echo ""
echo "[Step 1] Data preprocessing and annotation..."
echo "  请确保 ETH-UCY 原始数据在 $RAW_DATA_DIR/eth_ucy/"
echo "  SDD 数据在 $RAW_DATA_DIR/sdd/ (可选)"
echo ""

python -c "
import sys
sys.path.insert(0, '.')
from data.preprocessing.data_loader import load_eth_ucy_trajectories

subsets = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
all_clips = []
for subset in subsets:
    clips = load_eth_ucy_trajectories('$RAW_DATA_DIR/eth_ucy', subset)
    all_clips.extend(clips)

print(f'Total clips: {len(all_clips)}')
print(f'Total trajectories: {sum(c[\"num_pedestrians\"] for c in all_clips)}')

import json, os
os.makedirs('$PROCESSED_DIR', exist_ok=True)
with open('$PROCESSED_DIR/all_clips.json', 'w') as f:
    json.dump(all_clips, f, indent=2)
print(f'Saved to $PROCESSED_DIR/all_clips.json')
"

# ============================================================
# Step 2: API标注 (可选, 如果已有标注可跳过)
# ============================================================
echo ""
echo "[Step 2] Annotating with API..."
echo "  推荐: 设置 GOOGLE_API_KEY 使用 Gemini-Flash (最便宜)"
echo "  或设置 OPENAI_API_KEY 使用 GPT-4o-mini"
echo ""

if [ -f "$PROCESSED_DIR/annotations.json" ]; then
    echo "  Annotations already exist, skipping..."
else
    echo "  [跳过] 请手动运行标注脚本:"
    echo "    python -c \"from data.annotation.annotate_api import ...\""
fi

# ============================================================
# Step 3: 构建SFT数据集
# ============================================================
echo ""
echo "[Step 3] Building SFT dataset..."

python -c "
import sys
sys.path.insert(0, '.')
from data.sft_builder import build_sft_dataset, split_train_test

# 假设标注数据已存在
import os
ann_path = '$PROCESSED_DIR/annotations.json'
if os.path.exists(ann_path):
    import yaml
    with open('$CONFIG') as f:
        config = yaml.safe_load(f)
    build_sft_dataset(ann_path, config, '$SFT_DIR/sft_data.json')
    split_train_test('$SFT_DIR/sft_data.json', output_dir='$SFT_DIR')
else:
    print('  [跳过] 标注数据不存在, 请先完成Step 2')
"

# ============================================================
# Step 4: 训练 (主模型)
# ============================================================
echo ""
echo "[Step 4] Training FLINT model..."

if [ -f "$SFT_DIR/train.json" ]; then
    python train/train_sft.py \
        --config "$CONFIG" \
        --train_data "$SFT_DIR/train.json" \
        --test_data "$SFT_DIR/test.json"
else
    echo "  [跳过] SFT数据不存在"
fi

# ============================================================
# Step 5: 评估
# ============================================================
echo ""
echo "[Step 5] Evaluation..."

MODEL_PATH="$OUTPUT_DIR/Qwen3-8B_absolute/final"
if [ -d "$MODEL_PATH" ]; then
    python evaluate/run_eval.py \
        --model_path "$MODEL_PATH" \
        --config "$CONFIG" \
        --test_data "$SFT_DIR/test.json" \
        --output_dir "$EVAL_DIR" \
        --num_samples 20
else
    echo "  [跳过] 模型不存在"
fi

# ============================================================
# Step 6: 消融实验
# ============================================================
echo ""
echo "[Step 6] Ablation experiments..."

if [ -f "$SFT_DIR/train.json" ]; then
    # Tokenizer消融
    python ablation/run_ablation.py \
        --config "$CONFIG" \
        --train_data "$SFT_DIR/train.json" \
        --test_data "$SFT_DIR/test.json" \
        --output_dir "$ABLATION_DIR" \
        --ablation tokenizer

    # 温度消融 (不需要重新训练)
    if [ -d "$MODEL_PATH" ]; then
        python ablation/run_ablation.py \
            --config "$CONFIG" \
            --train_data "$SFT_DIR/train.json" \
            --test_data "$SFT_DIR/test.json" \
            --output_dir "$ABLATION_DIR/temperature" \
            --ablation temperature \
            --model_path "$MODEL_PATH"
    fi
else
    echo "  [跳过] SFT数据不存在"
fi

# ============================================================
# Step 7: 下游任务 — 轨迹预测数据增强
# ============================================================
echo ""
echo "[Step 7] Downstream: Trajectory prediction augmentation..."

if [ -d "$MODEL_PATH" ] && [ -f "$SFT_DIR/train.json" ]; then
    # 先生成增强数据
    python -c "
import sys
sys.path.insert(0, '.')
from downstream.traj_prediction import generate_augmentation_data
generate_augmentation_data(
    '$MODEL_PATH', '$CONFIG', '$SFT_DIR/train.json',
    '$DOWNSTREAM_DIR/flint_augmented.json',
    num_variants_per_sample=5,
)
"
    # 运行增强实验
    python downstream/traj_prediction.py \
        --original_data "$RAW_DATA_DIR/eth_ucy" \
        --flint_augmented "$DOWNSTREAM_DIR/flint_augmented.json" \
        --config "$CONFIG" \
        --output_dir "$DOWNSTREAM_DIR"
else
    echo "  [跳过] 模型或数据不存在"
fi

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Results:"
echo "  Evaluation: $EVAL_DIR/"
echo "  Ablation:   $ABLATION_DIR/"
echo "  Downstream: $DOWNSTREAM_DIR/"
echo "============================================================"
