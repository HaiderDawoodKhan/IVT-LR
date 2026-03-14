#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_VL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$QWEN_VL_DIR"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

LOG_DIR=${LOG_DIR:-"$QWEN_VL_DIR/logs"}
mkdir -p "$LOG_DIR"
export QWEN_LOG_DIR="$LOG_DIR"

CHECKPOINT_PATH_M3COT=${CHECKPOINT_PATH:-"/home/csalt/.cache/huggingface/hub/models--ModalityDance--IVTLR_QWEN_M3COT/snapshots/7c3161a715861b89578706e142ceac002a8beb52/model.pth"}
CHECKPOINT_PATH_SQA=${CHECKPOINT_PATH:-"/home/csalt/.cache/huggingface/hub/models--ModalityDance--IVTLR_QWEN_SQA/snapshots/5deaff3938d5a976c1a0d69685859960f43d450a/model.pth"}
MODEL_ID=${MODEL_ID:-"Qwen/Qwen2-VL-7B-Instruct"}
NUM_LATENT_STEPS=${NUM_LATENT_STEPS:-3}
TOP_K=${TOP_K:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
OUTPUT_DIR=${OUTPUT_DIR:-"output"}
MAX_SAMPLES=${MAX_SAMPLES:--1}
RUN_ABLATIONS=${RUN_ABLATIONS:-true}
SPLIT_POOL_SELECTION=${SPLIT_POOL_SELECTION:-false}
NEW_POOL_PATCH_COUNT=${NEW_POOL_PATCH_COUNT:-""}

NEW_POOL_PATCH_ARGS=()
if [[ -n "$NEW_POOL_PATCH_COUNT" ]]; then
  NEW_POOL_PATCH_ARGS=(--new_pool_patch_count "$NEW_POOL_PATCH_COUNT")
fi

echo "[1/6] Running base 7B inference (M3CoT)..."
python infer_base.py \
  --model_id "$MODEL_ID" \
  --output_path "$OUTPUT_DIR/base/m3cot/qwen2vl_base.jsonl" > "$LOG_DIR/infer_base_m3cot.log" 2>&1

echo "[2/6] Running base 7B inference (ScienceQA)..."
python infer_sqa_base.py \
  --model_id "$MODEL_ID" \
  --output_json_path "$OUTPUT_DIR/base/scienceqa/qwen2vl_base_scienceqa.json" > "$LOG_DIR/infer_base_sqa.log" 2>&1

echo "[3/6] Running IVTLR comparison inference (M3CoT, masking ON)..."
python infer.py \
  --checkpoint_path "$CHECKPOINT_PATH_M3COT" \
  --model_id "$MODEL_ID" \
  --num_latent_steps "$NUM_LATENT_STEPS" \
  --top_k "$TOP_K" \
  --mask_selected_patches "true" \
  --split_pool_selection "$SPLIT_POOL_SELECTION" \
  "${NEW_POOL_PATCH_ARGS[@]}" \
  --run_ablations "$RUN_ABLATIONS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_samples "$MAX_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --output_prefix "m3cot_compare_mask_on" > "$LOG_DIR/infer_compare_m3cot_mask_on.log" 2>&1

echo "[4/6] Running IVTLR comparison inference (M3CoT, masking OFF)..."
python infer.py \
  --checkpoint_path "$CHECKPOINT_PATH_M3COT" \
  --model_id "$MODEL_ID" \
  --num_latent_steps "$NUM_LATENT_STEPS" \
  --top_k "$TOP_K" \
  --mask_selected_patches "false" \
  --split_pool_selection "$SPLIT_POOL_SELECTION" \
  "${NEW_POOL_PATCH_ARGS[@]}" \
  --run_ablations "$RUN_ABLATIONS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_samples "$MAX_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --output_prefix "m3cot_compare_mask_off" > "$LOG_DIR/infer_compare_m3cot_mask_off.log" 2>&1

echo "[5/6] Running IVTLR comparison inference (ScienceQA, masking ON)..."
python infer_sqa.py \
  --checkpoint_path "$CHECKPOINT_PATH_SQA" \
  --model_id "$MODEL_ID" \
  --num_latent_steps "$NUM_LATENT_STEPS" \
  --top_k "$TOP_K" \
  --mask_selected_patches "true" \
  --split_pool_selection "$SPLIT_POOL_SELECTION" \
  "${NEW_POOL_PATCH_ARGS[@]}" \
  --run_ablations "$RUN_ABLATIONS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_samples "$MAX_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --output_prefix "scienceqa_compare_mask_on" > "$LOG_DIR/infer_compare_sqa_mask_on.log" 2>&1

echo "[6/6] Running IVTLR comparison inference (ScienceQA, masking OFF)..."
python infer_sqa.py \
  --checkpoint_path "$CHECKPOINT_PATH_SQA" \
  --model_id "$MODEL_ID" \
  --num_latent_steps "$NUM_LATENT_STEPS" \
  --top_k "$TOP_K" \
  --mask_selected_patches "false" \
  --split_pool_selection "$SPLIT_POOL_SELECTION" \
  "${NEW_POOL_PATCH_ARGS[@]}" \
  --run_ablations "$RUN_ABLATIONS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_samples "$MAX_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --output_prefix "scienceqa_compare_mask_off" > "$LOG_DIR/infer_compare_sqa_mask_off.log" 2>&1

echo "Base + comparison runs completed (masking ON/OFF). Logs: $LOG_DIR"

# cd /Users/haiderdawood/Desktop/LUMS/Senior Year/Spring/DVLM/IVT-LR/qwen_vl
# python infer.py \
#   --checkpoint_path /path/to/model.pth \
#   --model_id Qwen/Qwen2-VL-2B-Instruct \
#   --top_k 32 \
#   --mask_selected_patches true \
#   --split_pool_selection false

# cd /Users/haiderdawood/Desktop/LUMS/Senior Year/Spring/DVLM/IVT-LR/qwen_vl
# python infer.py \
#   --checkpoint_path /path/to/model.pth \
#   --model_id Qwen/Qwen2-VL-2B-Instruct \
#   --top_k 32 \
#   --mask_selected_patches true \
#   --split_pool_selection true


# cd /Users/haiderdawood/Desktop/LUMS/Senior Year/Spring/DVLM/IVT-LR/qwen_vl
# deepspeed qwenvl_run.py args/qwen.yaml --deepspeed --deepspeed_config ds_config.json

# cd /Users/haiderdawood/Desktop/LUMS/Senior Year/Spring/DVLM/IVT-LR/qwen_vl
# deepspeed qwenvl_run.py args/qwen.yaml --deepspeed --deepspeed_config ds_config.json

# python infer.py \
#   --checkpoint_path /path/to/model.pth \
#   --model_id Qwen/Qwen2-VL-2B-Instruct \
#   --top_k 32 \
#   --split_pool_selection true \
#   --new_pool_patch_count 8