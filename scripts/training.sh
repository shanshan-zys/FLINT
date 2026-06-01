#!/bin/bash
# Training: main experiment + ablations
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

# physics loss weights (adjustable)
COLLISION_W=0.001
SMOOTHNESS_W=0.05
WALKABLE_W=0.05
COLLISION_THRESH=10.0

# mid-training inference (empty string = disabled)
INFER_EPOCHS="0,5,10,15,20"
# INFER_EPOCHS=""

# ── Task name suffix ──────────────────────────────────────────────
SUFFIX="ep${EPOCHS}-c${COLLISION_W}-s${SMOOTHNESS_W}-w${WALKABLE_W}"

# ── Inference flag ────────────────────────────────────────────────
INFER_FLAG=""
if [ -n "$INFER_EPOCHS" ]; then
    INFER_FLAG="--infer_epochs $INFER_EPOCHS --max_new_tokens $MAX_NEW"
fi

# ── 1. Main: coord tokens + full physics loss ─────────────────────
TASK="main-${SUFFIX}"
echo "===== [1/3] ${TASK} ====="
python source/training.py \
    --data $DATA \
    --task_name $TASK \
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
    $INFER_FLAG

# ── 2. Ablation: coord tokens, no physics loss ───────────────────
TASK="ablation_no_physics-${SUFFIX}"
echo "===== [2/3] ${TASK} ====="
python source/training.py \
    --data $DATA \
    --task_name $TASK \
    --output_base $OUTPUT_BASE \
    --use_coord_tokens \
    --physics none \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    $INFER_FLAG

# ── 3. Ablation: raw numbers + full physics loss ─────────────────
TASK="ablation_raw_physics-${SUFFIX}"
echo "===== [3/3] ${TASK} ====="
python source/training.py \
    --data $DATA \
    --task_name $TASK \
    --output_base $OUTPUT_BASE \
    --no_coord_tokens \
    --use_raw_prompt \
    --physics all \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
    --seed $SEED \
    --collision_weight $COLLISION_W --smoothness_weight $SMOOTHNESS_W \
    --walkable_weight $WALKABLE_W --collision_threshold $COLLISION_THRESH \
    $INFER_FLAG

echo "All training runs complete."
