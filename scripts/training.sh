#!/bin/bash
# Training: coord_ce + raw_ce (more tasks can be added later)
set -e
cd "$(dirname "$0")/.."

# ── Common parameters ──────────────────────────────────────────────
DATA=data/eth-ucy-text.json
BACKBONE=/mnt/oss-write/opensource_models/Qwen3-8B
SEQ_LEN=8192
MAX_NEW=4096
EPOCHS=30
BS=4
GA=1
LR=2e-4
SCHED=linear
WARMUP=0.05
BIN=5
RES_H=480
RES_W=640
SEED=42
OUTPUT_BASE=./outputs

# loss weights (used when physics/regression enabled)
COLLISION_W=0.01
SMOOTHNESS_W=0.01
WALKABLE_W=0.01
REGRESSION_W=0.1
COLLISION_THRESH=5.0

# mid-training observation
PEEK="--peek_inference"

# ── Task name suffix ──────────────────────────────────────────────
SUFFIX="ep${EPOCHS}-c${COLLISION_W}-s${SMOOTHNESS_W}-w${WALKABLE_W}-r${REGRESSION_W}"

# ── Tasks to run ──────────────────────────────────────────────────
# Usage:
#   bash scripts/training.sh                    # run all tasks
#   bash scripts/training.sh "coord_ce"         # run single task
#   bash scripts/training.sh "coord_ce raw_ce"  # run selected tasks
TASKS="${1:-coord_ce raw_ce}"
# Uncomment to add more:
# TASKS="${1:-coord_ce raw_ce coord_ce_reg coord_ce_physics coord_ce_reg_physics}"

for TASK_BASE in $TASKS; do
    TASK="${TASK_BASE}-${SUFFIX}"

    # Determine coord/raw mode
    COORD_FLAG=""
    case "$TASK_BASE" in
        raw_*)   COORD_FLAG="--no_coord_tokens" ;;
        coord_*) COORD_FLAG="--use_coord_tokens" ;;
        *)
            echo "Unknown task prefix: $TASK_BASE (must start with coord_ or raw_)"
            exit 1
            ;;
    esac

    # Determine loss weights from task name
    COLL_W=0; SMOOTH_W=0; WALK_W=0; REG_W=0
    [[ "$TASK_BASE" == *_physics* ]] && { COLL_W=$COLLISION_W; SMOOTH_W=$SMOOTHNESS_W; WALK_W=$WALKABLE_W; }
    [[ "$TASK_BASE" == *_reg* ]] && REG_W=$REGRESSION_W

    echo "===== Training: ${TASK} ====="
    echo "  coord_flag=$COORD_FLAG coll=$COLL_W smooth=$SMOOTH_W walk=$WALK_W reg=$REG_W"
    python source/training.py \
        --data $DATA \
        --task_name $TASK \
        --output_base $OUTPUT_BASE \
        $COORD_FLAG \
        --backbone $BACKBONE --max_seq_length $SEQ_LEN \
        --epochs $EPOCHS --batch_size $BS --gradient_accumulation_steps $GA \
        --learning_rate $LR --lr_scheduler $SCHED --warmup_ratio $WARMUP \
        --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
        --seed $SEED \
        --collision_weight $COLL_W --smoothness_weight $SMOOTH_W \
        --walkable_weight $WALK_W --regression_weight $REG_W \
        --collision_threshold $COLLISION_THRESH \
        --max_new_tokens $MAX_NEW $PEEK
    echo ""
done

echo "All training runs complete."
