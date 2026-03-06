#!/usr/bin/env bash
set -euo pipefail

cd /home/csalt/Haider/DVLM/IVT-LR/IVT-LR/qwen_vl
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"your_path"}
MODEL_ID=${MODEL_ID:-"Qwen/Qwen2-VL-7B-Instruct"}
NUM_LATENT_STEPS=${NUM_LATENT_STEPS:-3}
TOP_K=${TOP_K:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
OUTPUT_DIR=${OUTPUT_DIR:-"output"}
MAX_SAMPLES=${MAX_SAMPLES:--1}

echo "[1/2] Running IVTLR comparison inference (M3CoT full-image vs embedding-only)..."
python infer.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --model_id "$MODEL_ID" \
  --output_dir "$OUTPUT_DIR" \
  --output_prefix "m3cot_compare" > infer_compare_m3cot.log 2>&1

echo "[2/2] Running IVTLR comparison inference (ScienceQA full-image vs embedding-only)..."
python infer_sqa.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --model_id "$MODEL_ID" \
  --output_dir "$OUTPUT_DIR" \
  --output_prefix "scienceqa_compare" > infer_compare_sqa.log 2>&1

echo "Comparison-only runs completed."
