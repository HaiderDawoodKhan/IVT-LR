import argparse
import random
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Qwen-VL vision token and hidden-state dimensions")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--image", type=str, default=None, help="Path to an input image (optional)")
    parser.add_argument("--prompt", type=str, default="Describe the image.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--resized_height", type=int, default=280)
    parser.add_argument("--resized_width", type=int, default=280)
    parser.add_argument("--dataset", type=str, default="LightChen2333/M3CoT", help="Dataset to sample from when --image is not provided")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to sample from")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sample selection")
    return parser.parse_args()


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def main():
    args = parse_args()
    dtype = get_torch_dtype(args.dtype)

    image_source = args.image
    prompt_text = args.prompt
    sample_id = None

    if image_source is None:
        random.seed(args.seed)
        dataset = load_dataset(args.dataset, split=args.split)
        dataset = dataset.filter(lambda e: e.get("image") is not None)
        if len(dataset) == 0:
            raise ValueError("No samples with images found in the selected dataset/split")
        sample_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[sample_idx]
        image_source = sample["image"]
        sample_id = sample.get("id", sample_idx)

        question = sample.get("question", "").strip()
        choices = sample.get("choices", [])
        if len(choices) > 0:
            options = "\n".join([f"{chr(65+i)}. {{{choice.strip()}}}" for i, choice in enumerate(choices)])
            prompt_text = f"[Question]: {{{question}}}\n[Options]:\n{options}\nAnswer:"
        else:
            prompt_text = f"[Question]: {{{question}}}\nAnswer:"

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_source,
                    "resized_height": args.resized_height,
                    "resized_width": args.resized_width,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(args.device)

    with torch.no_grad():
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        vision_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )

    input_ids = inputs["input_ids"]
    num_vision_tokens_in_prompt = int((input_ids == image_token_id).sum().item())

    hidden = outputs.hidden_states[-1]

    print("=== Qwen-VL Dimension Inspection ===")
    print(f"Model: {args.model_id}")
    if sample_id is not None:
        print(f"Random M3CoT sample id: {sample_id}")
    print(f"Image source: {'M3CoT random sample' if args.image is None else args.image}")
    print(f"Prompt vision token count (image token positions): {num_vision_tokens_in_prompt}")
    print(f"Vision embeddings shape: {tuple(vision_embeds.shape)}")
    print(f"  -> Number of vision tokens: {vision_embeds.shape[0]}")
    print(f"  -> Vision embedding dim: {vision_embeds.shape[-1]}")
    print(f"Last hidden states shape: {tuple(hidden.shape)}")
    print(f"  -> Hidden state dim: {hidden.shape[-1]}")


if __name__ == "__main__":
    main()
