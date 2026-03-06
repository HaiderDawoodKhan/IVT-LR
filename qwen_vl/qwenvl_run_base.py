import argparse
import gc
import logging
import os
import shutil

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from datasets import load_dataset
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from base_sft_utils import BaseSFTCollator, build_multimodal_sft_sample
from utils import Config, set_seed

logging.basicConfig(
    filename="qwenvl_base_train_m3cot.log",
    level=logging.DEBUG,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-VL base SFT training on M3CoT")
    parser.add_argument("config_file")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_config", default="ds_config.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def format_prompt(example):
    question = example["question"].strip()
    choices = example["choices"]
    choices_str = "\n".join([f"({chr(65 + i)}).{{{choice.strip()}}}" for i, choice in enumerate(choices)])
    return f"[Question]:{{{question}}}\n[Options]:\n{choices_str}\nAnswer:"


def format_target(example):
    rationale = example.get("rationale", "")
    rationale = rationale.replace("\n", " ").strip()
    answer = str(example["answer"]).strip()
    if rationale:
        return f"{rationale}\nTherefore, the answer is {answer}"
    return f"Therefore, the answer is {answer}"


def build_train_dataset(split, processor, tokenizer, debug=False):
    split = split.filter(lambda e: e.get("image") is not None)
    if debug:
        split = split.select(range(min(64, len(split))))

    def process_example(example, idx):
        sample = build_multimodal_sft_sample(
            processor=processor,
            tokenizer=tokenizer,
            image=example["image"],
            user_text=format_prompt(example),
            assistant_text=format_target(example),
        )
        sample["idx"] = idx
        return sample

    split = split.map(lambda example, idx: process_example(example, idx), with_indices=True)
    keep_columns = ["idx", "input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
    remove_columns = [col for col in split.column_names if col not in keep_columns]
    if remove_columns:
        split = split.remove_columns(remove_columns)
    return split


def main():
    args = parse_args()
    deepspeed.init_distributed()
    local_rank = args.local_rank
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    with open(args.config_file) as f:
        configs = Config(yaml.safe_load(f))

    model_id = getattr(configs, "model_id", "Qwen/Qwen2-VL-7B-Instruct")
    tokenizer_processor_model_id = "Qwen/Qwen2.5-VL-7B-Instruct" if model_id == "Qwen/Qwen2-VL-7B-Instruct" else model_id
    set_seed(configs.seed)

    save_dir = os.path.join(configs.save_path, configs.name)
    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)
    dist.barrier(device_ids=[torch.cuda.current_device()])

    if rank == 0 and len(os.listdir(save_dir)) > 0:
        raise ValueError(f"Save directory {save_dir} is not empty!")
    dist.barrier(device_ids=[torch.cuda.current_device()])

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_processor_model_id,
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(tokenizer_processor_model_id, tokenizer=tokenizer)

    model = model.to(rank)
    if getattr(configs, "bf16", True):
        model = model.to(torch.bfloat16)

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=args.deepspeed_config,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
    )
    del model

    dataset = load_dataset("LightChen2333/M3CoT")
    train_dataset = build_train_dataset(dataset["train"], processor, tokenizer, debug=getattr(configs, "debug", False))

    collator = BaseSFTCollator(tokenizer)
    total_train_steps = 0
    best_acc = 0

    wandb_run = None
    text_table = None
    if not getattr(configs, "debug", False) and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs.__dict__, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    for epoch in range(configs.resume, configs.num_epochs):
        np.random.seed(epoch)
        train_dataloader = DataLoader(
            train_dataset,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            batch_size=configs.batch_size_training,
            collate_fn=collator,
            sampler=DistributedSampler(train_dataset, shuffle=True),
        )

        model_engine.train()
        total_length = max(1, len(train_dataloader) // configs.gradient_accumulation_steps)
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch + 1}",
            total=total_length,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(train_dataloader):
            if step == 0 and wandb_run and rank == 0:
                text_table.add_data(total_train_steps, tokenizer.decode(batch["input_ids"][0]))

            total_train_steps += 1
            batch = {key: batch[key].to(rank) for key in batch.keys() if key != "idx"}

            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

            if wandb_run and rank == 0:
                wandb_run.log(
                    {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float(),
                    }
                )

            pbar.set_description(
                f"Training Epoch: {epoch + 1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} (loss: {round(float(loss.detach().float()), 4)})"
            )
            pbar.update(1 if (step + 1) % max(1, configs.gradient_accumulation_steps) == 0 else 0)
        pbar.close()
        dist.barrier()

        if not getattr(configs, "debug", False) and (epoch + 1) % 4 == 0:
            epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch + 1}_checkpoint")
            model_engine.save_checkpoint(
                save_dir=epoch_save_dir,
                tag=f"epoch_{epoch + 1}_zero3_bf32",
                client_state={"best_acc": best_acc, "current_epoch": epoch + 1},
            )

            if rank == 0:
                fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(
                    epoch_save_dir,
                    tag=f"epoch_{epoch + 1}_zero3_bf32",
                )
                fp32_output = os.path.join(save_dir, f"epoch_{epoch + 1}_full_model_fp32.pth")
                torch.save(fp32_state_dict, fp32_output)
                print(f"Epoch {epoch + 1} FP32 save to {fp32_output}")
                if os.path.exists(epoch_save_dir):
                    shutil.rmtree(epoch_save_dir)

            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
