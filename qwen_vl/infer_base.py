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
    filename=os.path.join(LOG_DIR, 'qwen_base_infer_time.log'),
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL base-model inference (no IVTLR)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HF model id")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Generation max new tokens")
    parser.add_argument("--output_path", type=str, default="output/qwen2vl_base.jsonl", help="Output jsonl file")
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
    answer = example["answer"].strip()
    choices = example["choices"]
    image = example["image"]

    choices_str = "\n".join([f"{chr(65+i)}.{{{choice.strip()}}}" for i, choice in enumerate(choices)])
    user_prompt = (
        f"[Question]:{{{question}}}\n"
        f"[Options]:\n{choices_str}\n"
        f"Answer:"
    )
    return user_prompt, answer, image


def process_func(example):
    prompt, answer, image = format_prompt(example)
    return {
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
        "id": example["id"],
        "choices": example["choices"],
        "domain": example["domain"],
        "topic": example["topic"]
    }


def evaluate_and_save(eval_dataset, model, processor, output_path, max_new_tokens):
    model.eval()
    correct = 0
    total = 0
    total_generated_tokens = 0
    total_generate_time = 0.0

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f_out:
        for ex in eval_dataset:
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
            output_text = processor.decode(outputs[0], skip_special_tokens=True)
            logging.debug(f"[OUTPUT] {output_text}")

            total_generated_tokens += len(generated_tokens)

            cleaned_text = re.sub(
                r'(?<=answer:)\s*(\n+\s*)?assistant\b',
                '',
                output_text,
                flags=re.IGNORECASE
            )
            matches = re.finditer(
                r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])',
                cleaned_text,
                flags=re.IGNORECASE | re.DOTALL
            )
            candidates = {match.group(1).upper() for match in matches}
            gt_answer = ex["gt_answer"].strip().upper()

            if gt_answer in candidates:
                correct += 1

            total += 1

            message_question = ex["question_raw"].replace("<image>", "", 1).replace("Answer:", "", 1).strip()
            message_question = message_question.split("Answer:")[0].strip()

            result = {
                "id": ex["id"],
                "choices": ex["choices"],
                "answer": ex["gt_answer"],
                "domain": ex["domain"],
                "topic": ex["topic"],
                "messages": [
                    message_question,
                    generated_text
                ]
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

    accuracy = correct / total if total > 0 else 0
    avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
    avg_time_per_sample = total_generate_time / total if total > 0 else 0

    logging.info(f"[FINAL] Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
    logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
    logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")

    print(f"[FINAL] Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    print(f"[FINAL] Results saved to: {output_path}")


def main():
    args = parse_args()
    model, processor, tokenizer = load_base_model(args.model_id)
    processor.tokenizer = tokenizer

    dataset = load_dataset("LightChen2333/M3CoT")
    val_dataset = dataset["test"]
    val_dataset = val_dataset.filter(lambda e: e["image"] is not None).map(process_func)

    evaluate_and_save(val_dataset, model, processor, args.output_path, args.max_new_tokens)


if __name__ == "__main__":
    main()
