#!/bin/bash
# Inference: generate trajectories on test split for all trained models
set -e
cd "$(dirname "$0")/.."

# ── Common parameters ──────────────────────────────────────────────
DATA=data/eth-ucy-text.json
BACKBONE=/mnt/oss-write/opensource_models/Qwen3-8B
SEQ_LEN=8192
MAX_NEW=4096
BIN=5
RES_H=480
RES_W=640
SEED=42
OUTPUT_BASE=./outputs

# training config (must match training)
EPOCHS=20
COLLISION_W=0.001
SMOOTHNESS_W=0.05
WALKABLE_W=0.05

# which checkpoint epoch to test
EVAL_EPOCH=20

# ── Name components ───────────────────────────────────────────────
TRAIN_SUFFIX="ep${EPOCHS}-c${COLLISION_W}-s${SMOOTHNESS_W}-w${WALKABLE_W}"
INFER_SUFFIX="${TRAIN_SUFFIX}-${EVAL_EPOCH}"

# ── 1. Main (coord tokens + physics) ─────────────────────────────
TRAIN_TASK="main-${TRAIN_SUFFIX}"
INFER_TASK="main-${INFER_SUFFIX}"
echo "===== [1/3] Inference: ${INFER_TASK} ====="
python source/inference.py \
    --data $DATA \
    --task_name $INFER_TASK \
    --output_base $OUTPUT_BASE \
    --backbone $BACKBONE \
    --checkpoint $OUTPUT_BASE/checkpoint/$TRAIN_TASK/epoch_$EVAL_EPOCH \
    --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED

# ── 2. Ablation: coord tokens, no physics ────────────────────────
TRAIN_TASK="ablation_no_physics-${TRAIN_SUFFIX}"
INFER_TASK="ablation_no_physics-${INFER_SUFFIX}"
echo "===== [2/3] Inference: ${INFER_TASK} ====="
python source/inference.py \
    --data $DATA \
    --task_name $INFER_TASK \
    --output_base $OUTPUT_BASE \
    --backbone $BACKBONE \
    --checkpoint $OUTPUT_BASE/checkpoint/$TRAIN_TASK/epoch_$EVAL_EPOCH \
    --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED

# ── 3. Ablation: raw numbers + physics ───────────────────────────
TRAIN_TASK="ablation_raw_physics-${TRAIN_SUFFIX}"
INFER_TASK="ablation_raw_physics-${INFER_SUFFIX}"
echo "===== [3/3] Inference: ${INFER_TASK} ====="
python source/inference.py \
    --data $DATA \
    --task_name $INFER_TASK \
    --output_base $OUTPUT_BASE \
    --backbone $BACKBONE \
    --checkpoint $OUTPUT_BASE/checkpoint/$TRAIN_TASK/epoch_$EVAL_EPOCH \
    --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --raw_mode

echo "All inference complete."
