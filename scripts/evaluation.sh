#!/bin/bash
# Evaluation: run metrics / visualization / LLM judge / diversity for all models
set -e
cd "$(dirname "$0")/.."

# ── Common parameters (must match training & inference) ───────────
EPOCHS=20
COLLISION_W=0.001
SMOOTHNESS_W=0.05
WALKABLE_W=0.05
REGRESSION_W=0.01

DATA_DIR=data/processed
OUTPUT_BASE=./outputs

# which checkpoint epochs to evaluate (comma-separated)
EVAL_EPOCHS="0,5,10,15,20"

# ── Name components ───────────────────────────────────────────────
TRAIN_SUFFIX="ep${EPOCHS}-c${COLLISION_W}-s${SMOOTHNESS_W}-w${WALKABLE_W}-r${REGRESSION_W}"

# ── Feature flags (comment out to skip) ───────────────────────────
METRICS="--metrics"
INDIVIDUAL="--individual_metrics"
VIS_VIDEO="--vis_video"
VIS_IMAGE="--vis_image"
LLM_JUDGE=""               # "--llm_judge"
DIVERSITY=""                # "--diversity"
USE_BACKGROUND=""           # "--use_background"

# LLM judge options (only used when LLM_JUDGE is enabled)
LLM_PROVIDER="deepseek"
LLM_OPTS=""
# LLM_OPTS="--llm_provider $LLM_PROVIDER"

# ── Run evaluation for each task x epoch ──────────────────────────
IFS=',' read -ra EP_ARRAY <<< "$EVAL_EPOCHS"

for TASK_BASE in main ablation_raw ablation_no_physics ablation_no_reg ablation_ce_only; do
    for EP in "${EP_ARRAY[@]}"; do
        TASK="${TASK_BASE}-${TRAIN_SUFFIX}-${EP}"
        RESULT_FILE="${OUTPUT_BASE}/results/${TASK}.json"
        if [ ! -f "$RESULT_FILE" ]; then
            echo "Skipping ${TASK}: result file not found"
            continue
        fi
        echo "===== Evaluating: ${TASK} ====="
        python source/evaluation.py \
            --task_name "$TASK" \
            --data_dir "$DATA_DIR" \
            --output_base "$OUTPUT_BASE" \
            $METRICS $INDIVIDUAL $VIS_VIDEO $VIS_IMAGE $LLM_JUDGE $DIVERSITY $USE_BACKGROUND $LLM_OPTS
        echo ""
    done
done

echo "All evaluation complete."
