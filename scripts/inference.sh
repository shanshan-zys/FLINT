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

# sampling
NUM_SAMPLES=1
TEMPERATURE=0.4
TOP_P=0.9

# training config (must match training)
EPOCHS=20
COLLISION_W=0.001
SMOOTHNESS_W=0.05
WALKABLE_W=0.05

# which checkpoint epochs to test (comma-separated)
EVAL_EPOCHS="0,5,10,15,20"

# ── Name components ───────────────────────────────────────────────
TRAIN_SUFFIX="ep${EPOCHS}-c${COLLISION_W}-s${SMOOTHNESS_W}-w${WALKABLE_W}"

# ── 1. Main (coord tokens + CE + reg + physics) ──────────────────
TRAIN_TASK="main-${TRAIN_SUFFIX}"
TASK_BASE="main-${TRAIN_SUFFIX}"
echo "===== [1/5] Inference: ${TASK_BASE} ====="
python source/inference.py \
    --data $DATA \
    --task_name_base $TASK_BASE \
    --train_task $TRAIN_TASK \
    --eval_epochs $EVAL_EPOCHS \
    --output_base $OUTPUT_BASE \
    --backbone $BACKBONE \
    --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --num_samples $NUM_SAMPLES --temperature $TEMPERATURE --top_p $TOP_P

# ── 2. Ablation: raw numbers ─────────────────────────────────────
TRAIN_TASK="ablation_raw-${TRAIN_SUFFIX}"
TASK_BASE="ablation_raw-${TRAIN_SUFFIX}"
echo "===== [2/5] Inference: ${TASK_BASE} ====="
python source/inference.py \
    --data $DATA \
    --task_name_base $TASK_BASE \
    --train_task $TRAIN_TASK \
    --eval_epochs $EVAL_EPOCHS \
    --output_base $OUTPUT_BASE \
    --backbone $BACKBONE \
    --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --raw_mode \
    --num_samples $NUM_SAMPLES --temperature $TEMPERATURE --top_p $TOP_P

# ── 3. Ablation: no physics ──────────────────────────────────────
TRAIN_TASK="ablation_no_physics-${TRAIN_SUFFIX}"
TASK_BASE="ablation_no_physics-${TRAIN_SUFFIX}"
echo "===== [3/5] Inference: ${TASK_BASE} ====="
python source/inference.py \
    --data $DATA \
    --task_name_base $TASK_BASE \
    --train_task $TRAIN_TASK \
    --eval_epochs $EVAL_EPOCHS \
    --output_base $OUTPUT_BASE \
    --backbone $BACKBONE \
    --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --num_samples $NUM_SAMPLES --temperature $TEMPERATURE --top_p $TOP_P

# ── 4. Ablation: no regression ───────────────────────────────────
TRAIN_TASK="ablation_no_reg-${TRAIN_SUFFIX}"
TASK_BASE="ablation_no_reg-${TRAIN_SUFFIX}"
echo "===== [4/5] Inference: ${TASK_BASE} ====="
python source/inference.py \
    --data $DATA \
    --task_name_base $TASK_BASE \
    --train_task $TRAIN_TASK \
    --eval_epochs $EVAL_EPOCHS \
    --output_base $OUTPUT_BASE \
    --backbone $BACKBONE \
    --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --num_samples $NUM_SAMPLES --temperature $TEMPERATURE --top_p $TOP_P

# ── 5. Ablation: CE only ─────────────────────────────────────────
TRAIN_TASK="ablation_ce_only-${TRAIN_SUFFIX}"
TASK_BASE="ablation_ce_only-${TRAIN_SUFFIX}"
echo "===== [5/5] Inference: ${TASK_BASE} ====="
python source/inference.py \
    --data $DATA \
    --task_name_base $TASK_BASE \
    --train_task $TRAIN_TASK \
    --eval_epochs $EVAL_EPOCHS \
    --output_base $OUTPUT_BASE \
    --backbone $BACKBONE \
    --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --num_samples $NUM_SAMPLES --temperature $TEMPERATURE --top_p $TOP_P

echo "All inference complete."
