#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_VL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$QWEN_VL_DIR"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

LOG_DIR=${LOG_DIR:-"$QWEN_VL_DIR/logs"}
mkdir -p "$LOG_DIR"
export QWEN_LOG_DIR="$LOG_DIR"

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-ds_config_single_gpu.json}
M3COT_CONFIG=${M3COT_CONFIG:-args/qwen_base.yaml}
SCIENCEQA_CONFIG=${SCIENCEQA_CONFIG:-args/qwen_sqa_base.yaml}

echo "[1/2] Training base Qwen-VL on M3CoT..."
PYTHONUNBUFFERED=1 deepspeed qwenvl_run_base.py "$M3COT_CONFIG" \
  --deepspeed \
  --deepspeed_config "$DEEPSPEED_CONFIG" > "$LOG_DIR/qwenvl_base_m3cot.log" 2>&1

echo "[2/2] Training base Qwen-VL on ScienceQA..."
PYTHONUNBUFFERED=1 deepspeed qwenvl_run_sqa_base.py "$SCIENCEQA_CONFIG" \
  --deepspeed \
  --deepspeed_config "$DEEPSPEED_CONFIG" > "$LOG_DIR/qwenvl_base_scienceqa.log" 2>&1

echo "Base training runs completed. Logs: $LOG_DIR"
