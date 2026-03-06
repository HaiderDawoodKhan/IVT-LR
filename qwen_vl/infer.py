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


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL inference")
    parser.add_argument("--checkpoint_path", type=str, default="your_path", help="Path to trained checkpoint")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HF model id")
    parser.add_argument("--num_latent_steps", type=int, default=3, help="Number of latent tokens appended to prompt")
    parser.add_argument("--top_k", type=int, default=32, help="Top-k visual embeddings selected per latent step")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Generation length")
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional sample cap for debugging")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for comparison artifacts")
    parser.add_argument("--output_prefix", type=str, default="m3cot_compare", help="Prefix for comparison artifacts")
    parser.add_argument("--mask_selected_patches", type=str2bool, default=True, help="Whether selected top-k image patches are masked to prevent reselection")
    parser.add_argument("--run_ablations", type=str2bool, default=True, help="Whether to run cumulative and non-cumulative ablations")
    return parser.parse_args()


def load_inference_model(checkpoint_path, model_id, top_k, mask_selected_patches):
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
        mask_selected_patches=mask_selected_patches,
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
model, processor, tokenizer = load_inference_model(args.checkpoint_path, args.model_id, args.top_k, args.mask_selected_patches)

os.makedirs("output", exist_ok=True)

def format_prompt(example):
    question = example["question"].strip()
    rationale = example["rationale"].replace("\n", " ").strip()
    answer = example["answer"].strip()
    choices = example["choices"]
    image = example["image"]

    choices_str = "\n".join([f"{chr(65+i)}.{{{choice.strip()}}}" for i, choice in enumerate(choices)])
    user_prompt = (
        f"[Question]:{{{question}}}\n"
        f"[Options]:\n{choices_str}\n"
        f"Answer:"
    )
    return user_prompt, rationale, answer, image

def process_func(example):
    prompt, rationale, answer, image = format_prompt(example)

    return {
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
        "id": example["id"],
        "choices": example["choices"],
        "domain": example["domain"],
        "topic": example["topic"]
    }

dataset = load_dataset("LightChen2333/M3CoT")
val_dataset = dataset["test"]
val_dataset = val_dataset.filter(lambda e: e["image"] is not None).map(process_func)

def evaluate_and_save(eval_dataset, model, processor):
    model.eval()
    full_correct = 0
    embed_correct = 0
    total = 0
    full_results = {}
    embed_results = {}
    step_correct = {}
    step_total = {}
    step_single_correct = {}
    step_single_total = {}
    sample_rows = []
    total_generated_tokens = 0 
    total_generate_time = 0.0  
    os.makedirs(args.output_dir, exist_ok=True)
    embeddings_dir = os.path.join(args.output_dir, f"{args.output_prefix}_embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    output_path = "output/qwen2vl_32.jsonl"
    with open(output_path, "a", encoding="utf-8") as f_out:
        for ex in eval_dataset:
            if args.max_samples > 0 and total >= args.max_samples:
                break

            sample_key = str(ex["id"])
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
            input_ids = inputs["input_ids"]
            prompt_length = input_ids.shape[1]
            
            generate_start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    max_new_tokens=args.max_new_tokens,
                    sample_keys=[sample_key],
                )
            generate_end_time = time.time()
            sample_generate_time = generate_end_time - generate_start_time
            total_generate_time += sample_generate_time
                        
            generated_tokens = outputs[0, prompt_length:]
            new_generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
            output_text = processor.decode(outputs[0], skip_special_tokens=True)
            logging.debug(f"[OUTPUT] {output_text}")
            
            num_generated_tokens = len(generated_tokens)
            total_generated_tokens += num_generated_tokens

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

            full_pred = sorted(candidates)[0] if len(candidates) > 0 else ""
            full_results[sample_key] = full_pred
            full_ok = gt_answer in candidates
            if full_ok:
                full_correct += 1
                logging.debug(f"correct: True")

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
            torch.save(embedding_payload, os.path.join(embeddings_dir, f"{sample_key}.pt"))

            with torch.no_grad():
                embed_outputs = model.generate_with_selected_embeddings(
                    input_ids=inputs["input_ids"],
                    selected_step_embeddings=embedding_steps,
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                )

            embed_output_text = processor.decode(embed_outputs[0], skip_special_tokens=True)
            embed_cleaned_text = re.sub(
                r'(?<=answer:)\s*(\n+\s*)?assistant\b',
                '',
                embed_output_text,
                flags=re.IGNORECASE
            )
            embed_matches = re.finditer(
                r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])',
                embed_cleaned_text,
                flags=re.IGNORECASE | re.DOTALL
            )
            embed_candidates = {match.group(1).upper() for match in embed_matches}
            embed_pred = sorted(embed_candidates)[0] if len(embed_candidates) > 0 else ""
            embed_results[sample_key] = embed_pred
            embed_ok = gt_answer in embed_candidates
            if embed_ok:
                embed_correct += 1

            step_preds = {}
            step_correctness = {}
            step_single_preds = {}
            step_single_correctness = {}
            if args.run_ablations:
                for step_n in range(1, len(embedding_steps) + 1):
                    with torch.no_grad():
                        step_outputs = model.generate_with_selected_embeddings(
                            input_ids=inputs["input_ids"],
                            selected_step_embeddings=embedding_steps,
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=args.max_new_tokens,
                            num_steps=step_n,
                        )
                    step_output_text = processor.decode(step_outputs[0], skip_special_tokens=True)
                    step_cleaned_text = re.sub(
                        r'(?<=answer:)\s*(\n+\s*)?assistant\b',
                        '',
                        step_output_text,
                        flags=re.IGNORECASE
                    )
                    step_matches = re.finditer(
                        r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])',
                        step_cleaned_text,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    step_candidates = {match.group(1).upper() for match in step_matches}
                    step_pred = sorted(step_candidates)[0] if len(step_candidates) > 0 else ""
                    step_ok = gt_answer in step_candidates

                    step_preds[str(step_n)] = step_pred
                    step_correctness[str(step_n)] = step_ok
                    step_correct[step_n] = step_correct.get(step_n, 0) + (1 if step_ok else 0)
                    step_total[step_n] = step_total.get(step_n, 0) + 1

                    with torch.no_grad():
                        step_single_outputs = model.generate_with_selected_embeddings(
                            input_ids=inputs["input_ids"],
                            selected_step_embeddings=[embedding_steps[step_n - 1]],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=args.max_new_tokens,
                            num_steps=1,
                        )
                    step_single_output_text = processor.decode(step_single_outputs[0], skip_special_tokens=True)
                    step_single_cleaned_text = re.sub(
                        r'(?<=answer:)\s*(\n+\s*)?assistant\b',
                        '',
                        step_single_output_text,
                        flags=re.IGNORECASE
                    )
                    step_single_matches = re.finditer(
                        r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])',
                        step_single_cleaned_text,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    step_single_candidates = {match.group(1).upper() for match in step_single_matches}
                    step_single_pred = sorted(step_single_candidates)[0] if len(step_single_candidates) > 0 else ""
                    step_single_ok = gt_answer in step_single_candidates

                    step_single_preds[str(step_n)] = step_single_pred
                    step_single_correctness[str(step_n)] = step_single_ok
                    step_single_correct[step_n] = step_single_correct.get(step_n, 0) + (1 if step_single_ok else 0)
                    step_single_total[step_n] = step_single_total.get(step_n, 0) + 1

            sample_rows.append(
                {
                    "sample_key": sample_key,
                    "ground_truth": gt_answer,
                    "full_image_prediction": full_pred,
                    "embedding_only_prediction": embed_pred,
                    "full_image_correct": full_ok,
                    "embedding_only_correct": embed_ok,
                    "num_steps_captured": len(embedding_steps),
                    "step_predictions_cumulative": step_preds,
                    "step_correctness_cumulative": step_correctness,
                    "step_predictions_non_cumulative": step_single_preds,
                    "step_correctness_non_cumulative": step_single_correctness,
                    "embedding_file": os.path.join(embeddings_dir, f"{sample_key}.pt"),
                }
            )

            total += 1
            logging.debug(f"[TOTAL] {total}")

            # pdb.set_trace()
            message_question = ex["question_raw"]
            message_question = message_question.replace("<image>", "", 1).replace("Answer:", "", 1).strip()
            message_question = message_question.split("Answer:")[0].strip()

            result = {
                "id": ex["id"],
                "choices": ex["choices"],
                "answer": ex["gt_answer"],
                "domain": ex["domain"],
                "topic": ex["topic"],
                "messages": [
                    message_question,
                    new_generated_text
                ]
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()
            
        avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
        avg_time_per_sample = total_generate_time / total if total > 0 else 0

        full_acc = accuracy(full_correct, total)
        embed_acc = accuracy(embed_correct, total)
        step_acc = {
            str(step_n): accuracy(step_correct.get(step_n, 0), step_total.get(step_n, 0))
            for step_n in sorted(step_total.keys())
        } if args.run_ablations else {}
        step_single_acc = {
            str(step_n): accuracy(step_single_correct.get(step_n, 0), step_single_total.get(step_n, 0))
            for step_n in sorted(step_single_total.keys())
        } if args.run_ablations else {}
        summary = {
            "dataset": "M3CoT",
            "total": total,
            "full_image_correct": full_correct,
            "embedding_only_correct": embed_correct,
            "full_image_accuracy": full_acc,
            "embedding_only_accuracy": embed_acc,
            "accuracy_delta": embed_acc - full_acc,
            "mask_selected_patches": args.mask_selected_patches,
            "run_ablations": args.run_ablations,
            "step_ablation_accuracy": step_acc,
            "step_ablation_accuracy_non_cumulative": step_single_acc,
        }

        write_json(os.path.join(args.output_dir, f"{args.output_prefix}_summary.json"), summary)
        write_json(os.path.join(args.output_dir, f"{args.output_prefix}_full_results.json"), {"results": full_results})
        write_json(os.path.join(args.output_dir, f"{args.output_prefix}_embedding_results.json"), {"results": embed_results})
        write_jsonl(os.path.join(args.output_dir, f"{args.output_prefix}_samples.jsonl"), sample_rows)
        agreement_rows = build_agreement_rows(sample_rows)
        write_jsonl(os.path.join(args.output_dir, f"{args.output_prefix}_agreement.jsonl"), agreement_rows)
    
        logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
        logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
        logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")
        logging.info(f"[FINAL] Full-image Accuracy: {full_acc:.2%}")
        logging.info(f"[FINAL] Embedding-only Accuracy: {embed_acc:.2%}")

        print(f"[FINAL] Total: {total}, Full-image Accuracy: {full_acc:.2%}")
        print(f"[FINAL] Embedding-only Accuracy: {embed_acc:.2%}")
        print(f"Comparison artifacts saved to: {args.output_dir}")
    
evaluate_and_save(val_dataset, model, processor)
