#!/usr/bin/env bash
set -euo pipefail

cd /home/csalt/Haider/DVLM/IVT-LR/IVT-LR/qwen_vl
export CUDA_VISIBLE_DEVICES=0

echo "[1/4] Running base inference (M3CoT)..."
python infer_base.py --model_id Qwen/Qwen2-VL-2B-Instruct > infer_base.log 2>&1

echo "[2/4] Running base inference (ScienceQA)..."
python infer_sqa_base.py --model_id Qwen/Qwen2-VL-2B-Instruct > infer_base_sqa.log 2>&1

echo "[3/4] Running IVTLR training (M3CoT)..."
PYTHONUNBUFFERED=1 deepspeed qwenvl_run.py args/qwen.yaml --deepspeed --deepspeed_config ds_config_single_gpu.json > qwenvl.log 2>&1

echo "[4/4] Running IVTLR training (ScienceQA)..."
PYTHONUNBUFFERED=1 deepspeed qwenvl_run_sqa.py args/qwen.yaml --deepspeed --deepspeed_config ds_config_single_gpu.json > qwenvl_sqa.log 2>&1

echo "All jobs completed."