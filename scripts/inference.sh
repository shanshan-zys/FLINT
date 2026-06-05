#!/bin/bash
# Standalone inference (without merge step — model must already be merged)
set -e
cd "$(dirname "$0")/.."

# ── Common parameters ──────────────────────────────────────────────
DATA=data/eth-ucy-text.json
OUTPUT_BASE=./outputs
SEED=42
BIN=5
RES_H=480
RES_W=640
SEQ_LEN=8192
MAX_NEW=4096

NUM_SAMPLES=1
TEMPERATURE=0.4
TOP_P=0.9

# training config (must match training)
EPOCHS=30
COLLISION_W=0.001
SMOOTHNESS_W=0.01
WALKABLE_W=0.01
REGRESSION_W=0.1

# which epoch to evaluate
EVAL_EPOCH="${2:-25}"

TRAIN_SUFFIX="ep${EPOCHS}-c${COLLISION_W}-s${SMOOTHNESS_W}-w${WALKABLE_W}-r${REGRESSION_W}"

# ── Task to run ──────────────────────────────────────────────────
# Usage:
#   bash scripts/inference.sh coord_ce          # single task, default epoch
#   bash scripts/inference.sh coord_ce 20       # single task, specific epoch
#   bash scripts/inference.sh "coord_ce raw_ce" # multiple tasks
TASKS="${1:-coord_ce}"

for TASK_BASE in $TASKS; do
    TRAIN_TASK="${TASK_BASE}-${TRAIN_SUFFIX}"
    TASK_NAME="${TRAIN_TASK}-${EVAL_EPOCH}"
    MODEL="${OUTPUT_BASE}/merged/${TRAIN_TASK}/epoch_${EVAL_EPOCH}"

    RAW_FLAG=""
    [[ "$TASK_BASE" == raw_* ]] && RAW_FLAG="--raw_mode"

    if [ ! -d "$MODEL" ]; then
        echo "Model not found: $MODEL"
        echo "Run merge_and_infer.sh first, or check the epoch number."
        continue
    fi

    echo "===== Inference: ${TASK_NAME} ====="
    VLLM_USE_V1=0 python source/inference.py batch \
        --data $DATA \
        --model "$MODEL" \
        --task_name "$TASK_NAME" \
        --output_base $OUTPUT_BASE \
        --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
        --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
        --seed $SEED \
        --num_samples $NUM_SAMPLES --temperature $TEMPERATURE --top_p $TOP_P \
        $RAW_FLAG
    echo ""
done

echo "Inference complete."
