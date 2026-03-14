#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_VL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$QWEN_VL_DIR"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

LOG_DIR=${LOG_DIR:-"$QWEN_VL_DIR/logs"}
mkdir -p "$LOG_DIR"
export QWEN_LOG_DIR="$LOG_DIR"

echo "[1/4] Running base inference (M3CoT)..."
python infer_base.py --model_id Qwen/Qwen2-VL-2B-Instruct --output_path "${OUTPUT_DIR:-output}/base/m3cot/qwen2vl_base.jsonl" > "$LOG_DIR/infer_base.log" 2>&1

echo "[2/4] Running base inference (ScienceQA)..."
python infer_sqa_base.py --model_id Qwen/Qwen2-VL-2B-Instruct --output_json_path "${OUTPUT_DIR:-output}/base/scienceqa/qwen2vl_base_scienceqa.json" > "$LOG_DIR/infer_base_sqa.log" 2>&1

echo "[3/4] Running IVTLR training (M3CoT)..."
PYTHONUNBUFFERED=1 deepspeed qwenvl_run.py args/qwen.yaml --deepspeed --deepspeed_config ds_config_single_gpu.json > "$LOG_DIR/qwenvl.log" 2>&1

echo "[4/4] Running IVTLR training (ScienceQA)..."
PYTHONUNBUFFERED=1 deepspeed qwenvl_run_sqa.py args/qwen.yaml --deepspeed --deepspeed_config ds_config_single_gpu.json > "$LOG_DIR/qwenvl_sqa.log" 2>&1

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"your_path"}
MODEL_ID=${MODEL_ID:-"Qwen/Qwen2-VL-2B-Instruct"}
NUM_LATENT_STEPS=${NUM_LATENT_STEPS:-3}
TOP_K=${TOP_K:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
OUTPUT_DIR=${OUTPUT_DIR:-"output"}
SPLIT_POOL_SELECTION=${SPLIT_POOL_SELECTION:-false}
NEW_POOL_PATCH_COUNT=${NEW_POOL_PATCH_COUNT:-""}

NEW_POOL_PATCH_ARGS=()
if [[ -n "$NEW_POOL_PATCH_COUNT" ]]; then
	NEW_POOL_PATCH_ARGS=(--new_pool_patch_count "$NEW_POOL_PATCH_COUNT")
fi

echo "[5/6] Running IVTLR comparison inference (M3CoT full-image vs embedding-only)..."
python infer.py \
	--checkpoint_path "$CHECKPOINT_PATH" \
	--model_id "$MODEL_ID" \
	--num_latent_steps "$NUM_LATENT_STEPS" \
	--top_k "$TOP_K" \
	--split_pool_selection "$SPLIT_POOL_SELECTION" \
	"${NEW_POOL_PATCH_ARGS[@]}" \
	--max_new_tokens "$MAX_NEW_TOKENS" \
	--output_dir "$OUTPUT_DIR" \
	--output_prefix "m3cot_compare" > "$LOG_DIR/infer_compare_m3cot.log" 2>&1

echo "[6/6] Running IVTLR comparison inference (ScienceQA full-image vs embedding-only)..."
python infer_sqa.py \
	--checkpoint_path "$CHECKPOINT_PATH" \
	--model_id "$MODEL_ID" \
	--num_latent_steps "$NUM_LATENT_STEPS" \
	--top_k "$TOP_K" \
	--split_pool_selection "$SPLIT_POOL_SELECTION" \
	"${NEW_POOL_PATCH_ARGS[@]}" \
	--max_new_tokens "$MAX_NEW_TOKENS" \
	--output_dir "$OUTPUT_DIR" \
	--output_prefix "scienceqa_compare" > "$LOG_DIR/infer_compare_sqa.log" 2>&1

echo "All jobs completed (including comparison experiments). Logs: $LOG_DIR"