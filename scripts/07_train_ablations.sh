#!/bin/bash
# Train ablation models (3 variants)
set -e
cd "$(dirname "$0")/.."

BACKBONE=Qwen/Qwen3-8B-Instruct
SEQ_LEN=8192
EPOCHS=20
BS=4
GA=4
LR=1e-4
SCHED=cosine
WARMUP=0.05
ESP=3
BIN=5
RES_H=480
RES_W=640

# Ablation 1: raw numbers (no coord tokens)
python source/train.py \
    --train_data data/train.json \
    --test_data data/test.json \
    --no_coord_tokens \
    --physics none \
    --output_dir outputs/ablation_raw_numbers \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --early_stopping_patience $ESP --bin_size $BIN \
    --resolution_h $RES_H --resolution_w $RES_W

# Ablation 2: no physics loss
python source/train.py \
    --train_data data/train.json \
    --test_data data/test.json \
    --use_coord_tokens \
    --physics none \
    --output_dir outputs/ablation_no_physics \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --early_stopping_patience $ESP --bin_size $BIN \
    --resolution_h $RES_H --resolution_w $RES_W

# Ablation 3: collision loss only
python source/train.py \
    --train_data data/train.json \
    --test_data data/test.json \
    --use_coord_tokens \
    --physics collision \
    --output_dir outputs/ablation_collision_only \
    --backbone $BACKBONE --max_seq_length $SEQ_LEN \
    --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
    --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
    --early_stopping_patience $ESP --bin_size $BIN \
    --resolution_h $RES_H --resolution_w $RES_W \
    --collision_weight 0.1 --collision_threshold 10.0
