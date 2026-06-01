#!/bin/bash
# Training: main experiment + 4 ablations
set -e
cd "$(dirname "$0")/.."

# ── Common parameters ──────────────────────────────────────────────
DATA=data/eth-ucy-text.json
BACKBONE=/mnt/oss-write/opensource_models/Qwen3-8B
SEQ_LEN=8192
MAX_NEW=4096
EPOCHS=20
BS=4
GA=4
LR=1e-4
SCHED=cosine
WARMUP=0.05
BIN=5
RES_H=480
RES_W=640
SEED=42
OUTPUT_BASE=./outputs

# loss weights
COLLISION_W=0.001
SMOOTHNESS_W=0.05
WALKABLE_W=0.05
REGRESSION_W=0.01
COLLISION_THRESH=10.0

# mid-training observation (comment out to disable)
PEEK="--peek_inference"
# PEEK=""

# ── Task name suffix ──────────────────────────────────────────────
SUFFIX="ep${EPOCHS}-c${COLLISION_W}-s${SMOOTHNESS_W}-w${WALKABLE_W}"

# ── 1. Main: coord tokens + CE + regression + physics ────────────
TASK="main-${SUFFIX}"
echo "===== [1/5] ${TASK} ====="
python source/training.py \
    --data $DATA \
    --task_name $TASK \
    --output_base $OUTPUT_BASE \
    --use_coord_tokens \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --collision_weight $COLLISION_W --smoothness_weight $SMOOTHNESS_W \
    --walkable_weight $WALKABLE_W --regression_weight $REGRESSION_W \
    --collision_threshold $COLLISION_THRESH \
    --max_new_tokens $MAX_NEW $PEEK

# ── 2. Ablation: raw numbers + CE only (physics/reg not applicable) ──
TASK="ablation_raw-${SUFFIX}"
echo "===== [2/5] ${TASK} ====="
python source/training.py \
    --data $DATA \
    --task_name $TASK \
    --output_base $OUTPUT_BASE \
    --no_coord_tokens \
    --use_raw_prompt \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --max_new_tokens $MAX_NEW $PEEK

# ── 3. Ablation: coord tokens + CE + regression (no physics) ─────
TASK="ablation_no_physics-${SUFFIX}"
echo "===== [3/5] ${TASK} ====="
python source/training.py \
    --data $DATA \
    --task_name $TASK \
    --output_base $OUTPUT_BASE \
    --use_coord_tokens \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --regression_weight $REGRESSION_W \
    --max_new_tokens $MAX_NEW $PEEK

# ── 4. Ablation: coord tokens + CE + physics (no regression) ─────
TASK="ablation_no_reg-${SUFFIX}"
echo "===== [4/5] ${TASK} ====="
python source/training.py \
    --data $DATA \
    --task_name $TASK \
    --output_base $OUTPUT_BASE \
    --use_coord_tokens \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --collision_weight $COLLISION_W --smoothness_weight $SMOOTHNESS_W \
    --walkable_weight $WALKABLE_W \
    --collision_threshold $COLLISION_THRESH \
    --max_new_tokens $MAX_NEW $PEEK

# ── 5. Ablation: coord tokens + CE only (baseline) ───────────────
TASK="ablation_ce_only-${SUFFIX}"
echo "===== [5/5] ${TASK} ====="
python source/training.py \
    --data $DATA \
    --task_name $TASK \
    --output_base $OUTPUT_BASE \
    --use_coord_tokens \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --max_new_tokens $MAX_NEW $PEEK

echo "All training runs complete."
