#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_VL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$QWEN_VL_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-$QWEN_VL_DIR/output}"
LOG_DIR="${LOG_DIR:-$QWEN_VL_DIR/logs}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Route Python logging.basicConfig file outputs into logs/.
export QWEN_LOG_DIR="$LOG_DIR"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2-VL-7B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
RUN_ABLATIONS="${RUN_ABLATIONS:-true}"
NUM_LATENT_STEPS="${NUM_LATENT_STEPS:-3}"
TOP_K="${TOP_K:-32}"
NO_VISUAL_LATENTS="${NO_VISUAL_LATENTS:-false}"
NO_LAST_HIDDEN_STATE="${NO_LAST_HIDDEN_STATE:-false}"
NO_REASONING="${NO_REASONING:-false}"

CHECKPOINT_PATH_M3COT="${CHECKPOINT_PATH_M3COT:-/home/csalt/.cache/huggingface/hub/models--ModalityDance--IVTLR_QWEN_M3COT/snapshots/7c3161a715861b89578706e142ceac002a8beb52/model.pth}"
CHECKPOINT_PATH_SQA="${CHECKPOINT_PATH_SQA:-/home/csalt/.cache/huggingface/hub/models--ModalityDance--IVTLR_QWEN_SQA/snapshots/5deaff3938d5a976c1a0d69685859960f43d450a/model.pth}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-ds_config_single_gpu.json}"
M3COT_BASE_CONFIG="${M3COT_BASE_CONFIG:-args/qwen_base.yaml}"
SQA_BASE_CONFIG="${SQA_BASE_CONFIG:-args/qwen_sqa_base.yaml}"

run_m3cot_ablation() {
  local run_tag="$1"
  local mask_flag="$2"
  local split_flag="$3"
  local new_pool_count="$4"

  local cmd=(
    python infer.py
    --checkpoint_path "$CHECKPOINT_PATH_M3COT"
    --model_id "$MODEL_ID"
    --num_latent_steps "$NUM_LATENT_STEPS"
    --top_k "$TOP_K"
    --mask_selected_patches "$mask_flag"
    --split_pool_selection "$split_flag"
    --run_ablations "$RUN_ABLATIONS"
    --no_visual_latents "$NO_VISUAL_LATENTS"
    --no_last_hidden_state "$NO_LAST_HIDDEN_STATE"
    --no_reasoning "$NO_REASONING"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --max_samples "$MAX_SAMPLES"
    --output_dir "$OUTPUT_DIR/ablations/m3cot"
    --output_prefix "$run_tag"
  )

  if [[ -n "$new_pool_count" ]]; then
    cmd+=(--new_pool_patch_count "$new_pool_count")
  fi

  "${cmd[@]}" > "$LOG_DIR/${run_tag}.log" 2>&1
}

run_sqa_ablation() {
  local run_tag="$1"
  local mask_flag="$2"
  local split_flag="$3"
  local new_pool_count="$4"

  local cmd=(
    python infer_sqa.py
    --checkpoint_path "$CHECKPOINT_PATH_SQA"
    --model_id "$MODEL_ID"
    --num_latent_steps "$NUM_LATENT_STEPS"
    --top_k "$TOP_K"
    --mask_selected_patches "$mask_flag"
    --split_pool_selection "$split_flag"
    --run_ablations "$RUN_ABLATIONS"
    --no_visual_latents "$NO_VISUAL_LATENTS"
    --no_last_hidden_state "$NO_LAST_HIDDEN_STATE"
    --no_reasoning "$NO_REASONING"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --max_samples "$MAX_SAMPLES"
    --output_dir "$OUTPUT_DIR/ablations/scienceqa"
    --output_prefix "$run_tag"
  )

  if [[ -n "$new_pool_count" ]]; then
    cmd+=(--new_pool_patch_count "$new_pool_count")
  fi

  "${cmd[@]}" > "$LOG_DIR/${run_tag}.log" 2>&1
}

echo "[1/5] Base model inference: M3CoT + ScienceQA"
python infer_base.py \
  --model_id "$MODEL_ID" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --output_path "$OUTPUT_DIR/base/m3cot/qwen2vl_base.jsonl" > "$LOG_DIR/base_infer_m3cot.log" 2>&1

python infer_sqa_base.py \
  --model_id "$MODEL_ID" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --output_json_path "$OUTPUT_DIR/base/scienceqa/qwen2vl_base_scienceqa.json" > "$LOG_DIR/base_infer_scienceqa.log" 2>&1

echo "[2/5] Ablations: baseline mode (split OFF), masking ON/OFF, both datasets"
run_m3cot_ablation "m3cot_baseline_mask_on"  "true"  "false" ""
run_m3cot_ablation "m3cot_baseline_mask_off" "false" "false" ""
run_sqa_ablation   "scienceqa_baseline_mask_on"  "true"  "false" ""
run_sqa_ablation   "scienceqa_baseline_mask_off" "false" "false" ""

echo "[3/5] Ablations: split mode (K/2,K/2), masking ON/OFF, both datasets"
run_m3cot_ablation "m3cot_split_half_mask_on"  "true"  "true" ""
run_m3cot_ablation "m3cot_split_half_mask_off" "false" "true" ""
run_sqa_ablation   "scienceqa_split_half_mask_on"  "true"  "true" ""
run_sqa_ablation   "scienceqa_split_half_mask_off" "false" "true" ""

echo "[4/5] Ablations: split mode (all K from new pool), masking ON/OFF, both datasets"
run_m3cot_ablation "m3cot_split_allnew_mask_on"  "true"  "true" "$TOP_K"
run_m3cot_ablation "m3cot_split_allnew_mask_off" "false" "true" "$TOP_K"
run_sqa_ablation   "scienceqa_split_allnew_mask_on"  "true"  "true" "$TOP_K"
run_sqa_ablation   "scienceqa_split_allnew_mask_off" "false" "true" "$TOP_K"

# echo "[5/5] Base model training: M3CoT + ScienceQA"
# PYTHONUNBUFFERED=1 deepspeed qwenvl_run_base.py "$M3COT_BASE_CONFIG" \
#   --deepspeed --deepspeed_config "$DEEPSPEED_CONFIG" > "$LOG_DIR/base_train_m3cot.log" 2>&1

# PYTHONUNBUFFERED=1 deepspeed qwenvl_run_sqa_base.py "$SQA_BASE_CONFIG" \
#   --deepspeed --deepspeed_config "$DEEPSPEED_CONFIG" > "$LOG_DIR/base_train_scienceqa.log" 2>&1

echo "Done."
echo "Logs: $LOG_DIR"
echo "Outputs: $OUTPUT_DIR"
