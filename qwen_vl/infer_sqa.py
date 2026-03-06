from transformers import AutoTokenizer, AutoProcessor
from qwen_ivtlr import IVTLR  
from transformers import Qwen2VLForConditionalGeneration
import torch
import deepspeed
from peft import LoraConfig,get_peft_model
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import re
import logging
import json
import os
import time
from datetime import timedelta
import argparse
from experiment_reporting import write_json, write_jsonl, accuracy, build_agreement_rows
logging.basicConfig(
    filename='qwenvl_32_infer_time.log',
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import pdb

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL ScienceQA inference")
    parser.add_argument("--checkpoint_path", type=str, default="your_path", help="Path to trained checkpoint")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HF model id")
    parser.add_argument("--num_latent_steps", type=int, default=3, help="Number of latent tokens appended to prompt")
    parser.add_argument("--top_k", type=int, default=32, help="Top-k visual embeddings selected per latent step")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Generation length")
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional sample cap for debugging")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for comparison artifacts")
    parser.add_argument("--output_prefix", type=str, default="scienceqa_compare", help="Prefix for comparison artifacts")
    return parser.parse_args()


def load_inference_model(checkpoint_path, model_id, top_k):
    print(f"Using model_id: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        trust_remote_code=True,
        padding_side="right"
    )
    
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|start-latent|>",
            "<|end-latent|>",
            "<|latent|>"
        ]
    })
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    processor.tokenizer = tokenizer

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        inference_mode=False
    )
    base_model = get_peft_model(base_model, lora_config)
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    model = IVTLR(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        image_token_id=image_token_id,
        visual_start_id=visual_start_id, 
        visual_end_id=visual_end_id,
        num_selected_patches=top_k,
        model_id=model_id
    )
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    print(state_dict.keys())
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    print(model)
    print("Successfully load")
    
    model = model.to(device)
    model.eval()
    return model, processor, tokenizer

args = parse_args()
model, processor, tokenizer = load_inference_model(args.checkpoint_path, args.model_id, args.top_k)

os.makedirs("output", exist_ok=True)

def format_prompt(example):
    question = example["question"].strip()
    answer = example["answer"] 
    choices = example.get("choices", [])
    image = example["image"]

    if choices:
        choices_str = "\n".join([f"({chr(65+i)}).{{{choice.strip()}}}" for i, choice in enumerate(choices)])
        user_prompt = (
            f"[Question]:{{{question}}}\n"
            f"[Options]:\n{choices_str}\n"
            f"Answer:"
        )
    else:
        user_prompt = f"[Question]:{{{question}}}\nAnswer:"
    
    return user_prompt, answer, image

def process_func(example, idx):
    prompt, answer, image = format_prompt(example)
    
    return {
        "idx": idx,  
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,  
    }

dataset = load_dataset("derek-thomas/ScienceQA")
test_dataset = dataset["test"]

def has_image(example):
    return "image" in example and example["image"] is not None


test_dataset = test_dataset.map(lambda example, idx: {"original_idx": idx, **example}, with_indices=True)
test_dataset = test_dataset.filter(has_image)
test_dataset = test_dataset.map(lambda example: process_func(example, example["original_idx"]))

def evaluate_and_save(eval_dataset, model, processor, args):
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    embeddings_dir = os.path.join(args.output_dir, f"{args.output_prefix}_embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    full_correct = 0
    embed_correct = 0
    total = 0
    results = {}
    embed_results = {}
    step_correct = {}
    step_total = {}
    sample_rows = []

    total_generated_tokens = 0
    total_generate_time = 0.0
    
    output_json_path = "sqa_output/qwen_2_scienceqa.json"
    
    for ex in eval_dataset:
        if args.max_samples > 0 and total >= args.max_samples:
            break

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
        text = text + ("<|latent|>" * args.num_latent_steps)
        
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
            outputs = model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                max_new_tokens=args.max_new_tokens,
                sample_keys=[idx],
            )
        generate_end_time = time.time()
        sample_generate_time = generate_end_time - generate_start_time
        total_generate_time += sample_generate_time
        generated_tokens = outputs[0, prompt_length:]
        generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
        num_generated_tokens = len(generated_tokens)
        total_generated_tokens += num_generated_tokens

        pred_answer = extract_answer(generated_text)
        results[idx] = pred_answer

        gt_answer = ex["gt_answer"]
        full_ok = pred_answer == gt_answer
        if full_ok:
            full_correct += 1

        topk_trace = model.get_topk_trace()
        if len(topk_trace) > 0:
            step_records = topk_trace[0].get("steps", [])
        else:
            step_records = []
        embedding_steps = [step["embeddings"] for step in step_records]

        embedding_payload = [
            {
                "pass_idx": step["pass_idx"],
                "abs_idxs": step["abs_idxs"],
                "embeddings": step["embeddings"],
            }
            for step in step_records
        ]
        torch.save(embedding_payload, os.path.join(embeddings_dir, f"{idx}.pt"))

        with torch.no_grad():
            embed_outputs = model.generate_with_selected_embeddings(
                input_ids=inputs["input_ids"],
                selected_step_embeddings=embedding_steps,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
            )

        embed_generated_tokens = embed_outputs[0, prompt_length:]
        embed_generated_text = processor.decode(embed_generated_tokens, skip_special_tokens=True)
        embed_pred_answer = extract_answer(embed_generated_text)
        embed_results[idx] = embed_pred_answer
        embed_ok = embed_pred_answer == gt_answer
        if embed_ok:
            embed_correct += 1

        step_preds = {}
        step_correctness = {}
        for step_n in range(1, len(embedding_steps) + 1):
            with torch.no_grad():
                step_outputs = model.generate_with_selected_embeddings(
                    input_ids=inputs["input_ids"],
                    selected_step_embeddings=embedding_steps,
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    num_steps=step_n,
                )
            step_tokens = step_outputs[0, prompt_length:]
            step_text = processor.decode(step_tokens, skip_special_tokens=True)
            step_pred = extract_answer(step_text)
            step_ok = step_pred == gt_answer
            step_preds[str(step_n)] = step_pred
            step_correctness[str(step_n)] = step_ok
            step_correct[step_n] = step_correct.get(step_n, 0) + (1 if step_ok else 0)
            step_total[step_n] = step_total.get(step_n, 0) + 1

        sample_rows.append(
            {
                "sample_key": idx,
                "ground_truth": gt_answer,
                "full_image_prediction": pred_answer,
                "embedding_only_prediction": embed_pred_answer,
                "full_image_correct": full_ok,
                "embedding_only_correct": embed_ok,
                "num_steps_captured": len(embedding_steps),
                "step_predictions": step_preds,
                "step_correctness": step_correctness,
                "embedding_file": os.path.join(embeddings_dir, f"{idx}.pt"),
            }
        )

        total += 1
        if total % 20 == 0:
            logging.info(f"[PROGRESS] Processed {total} samples")
    
    output_data = {"results": results}
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    full_acc = accuracy(full_correct, total)
    embed_acc = accuracy(embed_correct, total)
    step_acc = {
        str(step_n): accuracy(step_correct.get(step_n, 0), step_total.get(step_n, 0))
        for step_n in sorted(step_total.keys())
    }
    summary = {
        "dataset": "ScienceQA",
        "total": total,
        "full_image_correct": full_correct,
        "embedding_only_correct": embed_correct,
        "full_image_accuracy": full_acc,
        "embedding_only_accuracy": embed_acc,
        "accuracy_delta": embed_acc - full_acc,
        "step_ablation_accuracy": step_acc,
    }

    write_json(os.path.join(args.output_dir, f"{args.output_prefix}_summary.json"), summary)
    write_json(os.path.join(args.output_dir, f"{args.output_prefix}_full_results.json"), {"results": results})
    write_json(os.path.join(args.output_dir, f"{args.output_prefix}_embedding_results.json"), {"results": embed_results})
    write_jsonl(os.path.join(args.output_dir, f"{args.output_prefix}_samples.jsonl"), sample_rows)
    agreement_rows = build_agreement_rows(sample_rows)
    write_jsonl(os.path.join(args.output_dir, f"{args.output_prefix}_agreement.jsonl"), agreement_rows)

    avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
    avg_time_per_sample = total_generate_time / total if total > 0 else 0
    
    
    logging.info(f"[FINAL] Total: {total}, Full Correct: {full_correct}, Full Accuracy: {full_acc:.2%}")
    logging.info(f"[FINAL] Embedding-only Correct: {embed_correct}, Embedding-only Accuracy: {embed_acc:.2%}")
    logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
    logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
    logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")
    
    
    print(f"[FINAL] Total: {total}, Full-image Accuracy: {full_acc:.2%}")
    print(f"[FINAL] Embedding-only Accuracy: {embed_acc:.2%}")
    print(f"Results saved to: {output_json_path}")
    print(f"Comparison artifacts saved to: {args.output_dir}")
    
    return summary

def extract_answer(text):
    digit_patterns = [
        r'Therefore,?\s*the\s+answer\s+is\s+(\d)',  
        r'the\s+answer\s+is\s+(\d)', 
        r'answer\s+is:?\s*(\d)',  
    ]
    
    for pattern in digit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_idx = int(match.group(1))
            logging.debug(f"Extracted answer (digit): {answer_idx}")
            return answer_idx
    

    letter_patterns = [
        r'Therefore,?\s*the\s+answer\s+is\s+([A-Z])', 
        r'the\s+answer\s+is\s+([A-Z])',
        r'answer\s+is:?\s*([A-Z])',  
    ]
    
    for pattern in letter_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()

            answer_idx = ord(letter) - ord('A')
            logging.debug(f"Extracted answer (letter): {letter} -> index {answer_idx}")
            return answer_idx
    

    logging.warning(f"No answer pattern found in text: {text[:200]}")
    return -1  

evaluate_and_save(test_dataset, model, processor, args)
