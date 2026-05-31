#!/bin/bash
# Training: main experiment + ablations
set -e
cd "$(dirname "$0")/.."

# ── Common parameters ──────────────────────────────────────────────
DATA=data/eth-ucy-text.json
BACKBONE=/mnt/oss-write/opensource_models/Qwen3-8B
SEQ_LEN=8192
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

# physics loss weights (adjustable)
COLLISION_W=0.1
SMOOTHNESS_W=0.05
WALKABLE_W=0.05
COLLISION_THRESH=10.0

# early stopping (set to --early_stopping to enable)
EARLY_STOP=""
ESP=3

# ── 1. Main experiment: coord tokens + full physics loss ───────────
echo "===== [1/4] Main experiment ====="
python source/training.py \
    --data $DATA \
    --task_name main \
    --output_base $OUTPUT_BASE \
    --use_coord_tokens \
    --physics all \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --collision_weight $COLLISION_W --smoothness_weight $SMOOTHNESS_W \
    --walkable_weight $WALKABLE_W --collision_threshold $COLLISION_THRESH \
    $EARLY_STOP ${EARLY_STOP:+--early_stopping_patience $ESP}

# ── 2. Ablation: raw numbers (no coord tokens, raw_prompt) ────────
echo "===== [2/4] Ablation: raw numbers ====="
python source/training.py \
    --data $DATA \
    --task_name ablation_raw_numbers \
    --output_base $OUTPUT_BASE \
    --no_coord_tokens \
    --use_raw_prompt \
    --physics none \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    $EARLY_STOP ${EARLY_STOP:+--early_stopping_patience $ESP}

# ── 3. Ablation: coord tokens, no physics loss ────────────────────
echo "===== [3/4] Ablation: no physics ====="
python source/training.py \
    --data $DATA \
    --task_name ablation_no_physics \
    --output_base $OUTPUT_BASE \
    --use_coord_tokens \
    --physics none \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    $EARLY_STOP ${EARLY_STOP:+--early_stopping_patience $ESP}

# ── 4. Ablation: collision loss only ──────────────────────────────
echo "===== [4/4] Ablation: collision only ====="
python source/training.py \
    --data $DATA \
    --task_name ablation_collision_only \
    --output_base $OUTPUT_BASE \
    --use_coord_tokens \
    --physics collision \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --collision_weight $COLLISION_W --collision_threshold $COLLISION_THRESH \
    $EARLY_STOP ${EARLY_STOP:+--early_stopping_patience $ESP}

echo "All training runs complete."
