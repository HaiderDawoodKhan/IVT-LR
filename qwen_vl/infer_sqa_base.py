from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import torch
import re
import logging
import json
import os
import time
from datetime import timedelta
import argparse

LOG_DIR = os.getenv("QWEN_LOG_DIR", ".")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'qwen_base_sqa_infer_time.log'),
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL ScienceQA base-model inference (no IVTLR)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HF model id")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Generation max new tokens")
    parser.add_argument("--output_json_path", type=str, default="sqa_output/qwen2vl_base_scienceqa.json", help="Output json file")
    return parser.parse_args()


def load_base_model(model_id):
    print(f"Using base model_id: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        trust_remote_code=True,
        padding_side="right"
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model = model.to(device)
    model.eval()
    return model, processor, tokenizer


def format_prompt(example):
    question = example["question"].strip()
    answer = example["answer"]
    choices = example.get("choices", [])
    image = example["image"]

    if choices:
        choices_str = "\n".join([f"({chr(65+i)}) {choice.strip()}" for i, choice in enumerate(choices)])
        user_prompt = (
            f"Question: {question}\n"
            f"Options:\n{choices_str}\n"
            "Reply with only the option letter in parentheses, for example (A).\n"
            f"Answer:"
        )
    else:
        user_prompt = f"Question: {question}\nAnswer:"

    return user_prompt, answer, image


def process_func(example, idx):
    prompt, answer, image = format_prompt(example)
    return {
        "idx": idx,
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
    }


def extract_answer(text, num_choices=None):
    normalized_text = " ".join(text.strip().split())

    letter_patterns = [
        r'(?:therefore,?\s*)?(?:the\s+)?answer\s+is:?\s*[\(\[\{]?\s*([A-Z])\s*[\)\]\}]?',
        r'(?:final\s+answer|correct\s+answer|option|choice)\s*[:\-]?\s*[\(\[\{]?\s*([A-Z])\s*[\)\]\}]?',
        r'^\s*[\(\[\{]\s*([A-Z])\s*[\)\]\}]\s*[\.,:]?',
        r'^\s*([A-Z])\s*[\).,:\-]',
    ]

    for pattern in letter_patterns:
        match = re.search(pattern, normalized_text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            answer_idx = ord(letter) - ord('A')
            if num_choices is None or 0 <= answer_idx < num_choices:
                return answer_idx

    digit_patterns = [
        r'(?:therefore,?\s*)?(?:the\s+)?answer\s+is:?\s*(\d+)',
        r'(?:final\s+answer|correct\s+answer|option|choice)\s*[:\-]?\s*(\d+)',
        r'^\s*[\(\[\{]?\s*(\d+)\s*[\)\]\}]?\s*[\.,:]?',
    ]

    for pattern in digit_patterns:
        match = re.search(pattern, normalized_text, re.IGNORECASE)
        if not match:
            continue
        value = int(match.group(1))
        if num_choices is not None:
            if 0 <= value < num_choices:
                return value
            if 1 <= value <= num_choices:
                return value - 1
        else:
            return value

    logging.warning(f"No answer pattern found in text: {text[:200]}")
    return -1


def evaluate_and_save(eval_dataset, model, processor, output_json_path, max_new_tokens):
    model.eval()
    correct = 0
    total = 0
    results = {}
    total_generated_tokens = 0
    total_generate_time = 0.0

    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for ex in eval_dataset:
        idx = str(ex["idx"])
        input_text = ex["question_raw"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": input_text}
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        prompt_length = inputs["input_ids"].shape[1]

        generate_start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generate_end_time = time.time()

        total_generate_time += (generate_end_time - generate_start_time)

        generated_tokens = outputs[0, prompt_length:]
        generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
        total_generated_tokens += len(generated_tokens)

        pred_answer = extract_answer(generated_text, num_choices=len(ex.get("choices", [])) or None)
        results[idx] = pred_answer

        gt_answer = ex["gt_answer"]
        if pred_answer == gt_answer:
            correct += 1

        total += 1

    output_data = {"results": results}
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    accuracy = correct / total if total > 0 else 0
    avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
    avg_time_per_sample = total_generate_time / total if total > 0 else 0

    logging.info(f"[FINAL] Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
    logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
    logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")

    print(f"[FINAL] Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {output_json_path}")


def main():
    args = parse_args()
    model, processor, tokenizer = load_base_model(args.model_id)
    processor.tokenizer = tokenizer

    dataset = load_dataset("derek-thomas/ScienceQA")
    test_dataset = dataset["test"]

    test_dataset = test_dataset.map(lambda example, idx: {"original_idx": idx, **example}, with_indices=True)
    test_dataset = test_dataset.filter(lambda example: "image" in example and example["image"] is not None)
    test_dataset = test_dataset.map(lambda example: process_func(example, example["original_idx"]))

    evaluate_and_save(test_dataset, model, processor, args.output_json_path, args.max_new_tokens)


if __name__ == "__main__":
    main()
