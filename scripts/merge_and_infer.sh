#!/bin/bash
# Merge LoRA checkpoints + vLLM batch inference
set -e
cd "$(dirname "$0")/.."

# ── Common parameters ──────────────────────────────────────────────
DATA=data/eth-ucy-text.json
BACKBONE=/mnt/oss-write/opensource_models/Qwen3-8B
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
COLLISION_W=0.01
SMOOTHNESS_W=0.01
WALKABLE_W=0.01
REGRESSION_W=0.01

# which checkpoint epochs to evaluate (0 = base model)
EVAL_EPOCHS="0,5,10,15,20,25,30"

TRAIN_SUFFIX="ep${EPOCHS}-c${COLLISION_W}-s${SMOOTHNESS_W}-w${WALKABLE_W}-r${REGRESSION_W}"

# ── Tasks to process ─────────────────────────────────────────────
# Usage:
#   bash scripts/merge_and_infer.sh                    # run all
#   bash scripts/merge_and_infer.sh "coord_ce"         # single task
TASKS="${1:-coord_ce raw_ce}"

IFS=',' read -ra EP_ARRAY <<< "$EVAL_EPOCHS"

for TASK_BASE in $TASKS; do
    TRAIN_TASK="${TASK_BASE}-${TRAIN_SUFFIX}"

    # Determine raw mode flag for inference
    RAW_FLAG=""
    case "$TASK_BASE" in
        raw_*)   RAW_FLAG="--raw_mode" ;;
        coord_*) RAW_FLAG="" ;;
        *)
            echo "Unknown task prefix: $TASK_BASE"
            exit 1
            ;;
    esac

    for EP in "${EP_ARRAY[@]}"; do
        CKPT="${OUTPUT_BASE}/checkpoint/${TRAIN_TASK}/epoch_${EP}"
        MERGED="${OUTPUT_BASE}/merged/${TRAIN_TASK}/epoch_${EP}"
        TASK_NAME="${TRAIN_TASK}-${EP}"

        # epoch 0 = base model (no checkpoint needed)
        if [ "$EP" = "0" ]; then
            MERGED="$BACKBONE"
        else
            if [ ! -d "$CKPT" ]; then
                echo "Skipping ${TASK_NAME}: checkpoint not found"
                continue
            fi

            # Step 1: Merge (skip if already done)
            if [ ! -f "${MERGED}/config.json" ]; then
                echo "===== Merging: ${TASK_NAME} ====="
                python source/merge_lora.py \
                    --backbone $BACKBONE \
                    --checkpoint "$CKPT" \
                    --output "$MERGED"
            else
                echo "===== Already merged: ${TASK_NAME} ====="
            fi
        fi

        # Step 2: vLLM inference
        RESULT_FILE="${OUTPUT_BASE}/results/${TASK_NAME}.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "  Result exists, skipping inference"
            continue
        fi

        echo "===== Inference: ${TASK_NAME} ====="
        python source/inference.py batch \
            --data $DATA \
            --model "$MERGED" \
            --task_name "$TASK_NAME" \
            --output_base $OUTPUT_BASE \
            --max_seq_length $SEQ_LEN --max_new_tokens $MAX_NEW \
            --bin_size $BIN --resolution_h $RES_H --resolution_w $RES_W \
            --seed $SEED \
            --num_samples $NUM_SAMPLES --temperature $TEMPERATURE --top_p $TOP_P \
            $RAW_FLAG
        echo ""
    done
done

echo "All merge + inference complete."
