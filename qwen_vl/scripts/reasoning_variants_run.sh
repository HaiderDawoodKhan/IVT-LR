#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_VL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$QWEN_VL_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_VL_DIR/output}"
LOG_DIR="${LOG_DIR:-$QWEN_VL_DIR/logs}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Route python logging.basicConfig file outputs into logs/.
export QWEN_LOG_DIR="$LOG_DIR"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2-VL-7B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
NUM_LATENT_STEPS="${NUM_LATENT_STEPS:-3}"
TOP_K="${TOP_K:-32}"

# Fixed settings requested by user:
# - no ablations
# - original top-k pool selection (no split)
# - masking on
RUN_ABLATIONS="false"
MASK_SELECTED_PATCHES="true"
SPLIT_POOL_SELECTION="false"

CHECKPOINT_PATH_M3COT="${CHECKPOINT_PATH_M3COT:-/home/csalt/.cache/huggingface/hub/models--ModalityDance--IVTLR_QWEN_M3COT/snapshots/7c3161a715861b89578706e142ceac002a8beb52/model.pth}"
CHECKPOINT_PATH_SQA="${CHECKPOINT_PATH_SQA:-/home/csalt/.cache/huggingface/hub/models--ModalityDance--IVTLR_QWEN_SQA/snapshots/5deaff3938d5a976c1a0d69685859960f43d450a/model.pth}"

run_m3cot_variant() {
  local run_tag="$1"
  local no_visual_latents="$2"
  local no_last_hidden_state="$3"
  local no_reasoning="$4"

  python infer.py \
    --checkpoint_path "$CHECKPOINT_PATH_M3COT" \
    --model_id "$MODEL_ID" \
    --num_latent_steps "$NUM_LATENT_STEPS" \
    --top_k "$TOP_K" \
    --mask_selected_patches "$MASK_SELECTED_PATCHES" \
    --split_pool_selection "$SPLIT_POOL_SELECTION" \
    --run_ablations "$RUN_ABLATIONS" \
    --no_visual_latents "$no_visual_latents" \
    --no_last_hidden_state "$no_last_hidden_state" \
    --no_reasoning "$no_reasoning" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_samples "$MAX_SAMPLES" \
    --output_dir "$OUTPUT_DIR/reasoning_variants/m3cot" \
    --output_prefix "$run_tag" > "$LOG_DIR/${run_tag}.log" 2>&1
}

run_sqa_variant() {
  local run_tag="$1"
  local no_visual_latents="$2"
  local no_last_hidden_state="$3"
  local no_reasoning="$4"

  python infer_sqa.py \
    --checkpoint_path "$CHECKPOINT_PATH_SQA" \
    --model_id "$MODEL_ID" \
    --num_latent_steps "$NUM_LATENT_STEPS" \
    --top_k "$TOP_K" \
    --mask_selected_patches "$MASK_SELECTED_PATCHES" \
    --split_pool_selection "$SPLIT_POOL_SELECTION" \
    --run_ablations "$RUN_ABLATIONS" \
    --no_visual_latents "$no_visual_latents" \
    --no_last_hidden_state "$no_last_hidden_state" \
    --no_reasoning "$no_reasoning" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_samples "$MAX_SAMPLES" \
    --output_dir "$OUTPUT_DIR/reasoning_variants/scienceqa" \
    --output_prefix "$run_tag" > "$LOG_DIR/${run_tag}.log" 2>&1
}

echo "[1/3] M3CoT variants"
run_m3cot_variant "m3cot_no_visual_latents" "true" "false" "false"
run_m3cot_variant "m3cot_no_last_hidden_state" "false" "true" "false"
run_m3cot_variant "m3cot_no_reasoning" "false" "false" "true"

echo "[2/3] ScienceQA variants"
run_sqa_variant "scienceqa_no_visual_latents" "true" "false" "false"
run_sqa_variant "scienceqa_no_last_hidden_state" "false" "true" "false"
run_sqa_variant "scienceqa_no_reasoning" "false" "false" "true"

echo "[3/3] Done"
echo "Logs: $LOG_DIR"
echo "Outputs: $OUTPUT_DIR/reasoning_variants"
