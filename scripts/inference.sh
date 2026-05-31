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
EPOCHS=20

# ── 1. Main (coord tokens + physics) ─────────────────────────────
echo "===== [1/3] Generate: main-${EPOCHS} ====="
python source/inference.py \
    --data $DATA \
    --task_name main-${EPOCHS} \
    --output_base $OUTPUT_BASE \
    --backbone $BACKBONE \
    --checkpoint $OUTPUT_BASE/checkpoint/main/epoch_$EPOCHS \
    --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED

# ── 2. Ablation: coord tokens, no physics ────────────────────────
echo "===== [2/3] Generate: ablation_no_physics-${EPOCHS} ====="
python source/inference.py \
    --data $DATA \
    --task_name ablation_no_physics-${EPOCHS} \
    --output_base $OUTPUT_BASE \
    --backbone $BACKBONE \
    --checkpoint $OUTPUT_BASE/checkpoint/ablation_no_physics/epoch_$EPOCHS \
    --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED

# ── 3. Ablation: raw numbers + physics ───────────────────────────
echo "===== [3/3] Generate: ablation_raw_physics-${EPOCHS} ====="
python source/inference.py \
    --data $DATA \
    --task_name ablation_raw_physics-${EPOCHS} \
    --output_base $OUTPUT_BASE \
    --backbone $BACKBONE \
    --checkpoint $OUTPUT_BASE/checkpoint/ablation_raw_physics/epoch_$EPOCHS \
    --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --raw_mode

echo "All generation complete."
