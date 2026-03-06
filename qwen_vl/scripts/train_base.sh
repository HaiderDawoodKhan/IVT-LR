#!/usr/bin/env bash
set -euo pipefail

cd /home/csalt/Haider/DVLM/IVT-LR/IVT-LR/qwen_vl
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-ds_config_single_gpu.json}
M3COT_CONFIG=${M3COT_CONFIG:-args/qwen_base.yaml}
SCIENCEQA_CONFIG=${SCIENCEQA_CONFIG:-args/qwen_sqa_base.yaml}

echo "[1/2] Training base Qwen-VL on M3CoT..."
PYTHONUNBUFFERED=1 deepspeed qwenvl_run_base.py "$M3COT_CONFIG" \
  --deepspeed \
  --deepspeed_config "$DEEPSPEED_CONFIG" > qwenvl_base_m3cot.log 2>&1

echo "[2/2] Training base Qwen-VL on ScienceQA..."
PYTHONUNBUFFERED=1 deepspeed qwenvl_run_sqa_base.py "$SCIENCEQA_CONFIG" \
  --deepspeed \
  --deepspeed_config "$DEEPSPEED_CONFIG" > qwenvl_base_scienceqa.log 2>&1

echo "Base training runs completed."
