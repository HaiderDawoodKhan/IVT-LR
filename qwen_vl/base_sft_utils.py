from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from qwen_vl_utils import process_vision_info


RESIZED_HEIGHT = 280
RESIZED_WIDTH = 280


def build_multimodal_sft_sample(
    *,
    processor,
    tokenizer,
    image: Any,
    user_text: str,
    assistant_text: str,
    resized_height: int = RESIZED_HEIGHT,
    resized_width: int = RESIZED_WIDTH,
) -> Dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "resized_height": resized_height,
                    "resized_width": resized_width,
                },
                {"type": "text", "text": user_text},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + assistant_text.strip() + tokenizer.eos_token

    image_inputs, video_inputs = process_vision_info(messages)
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = full_inputs["input_ids"][0]
    attention_mask = full_inputs["attention_mask"][0]
    labels = input_ids.clone()

    prompt_length = prompt_inputs["input_ids"].shape[1]
    labels[:prompt_length] = -100
    labels[attention_mask == 0] = -100
    if tokenizer.pad_token_id is not None:
        labels[input_ids == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist(),
        "pixel_values": full_inputs["pixel_values"][0].tolist(),
        "image_grid_thw": full_inputs["image_grid_thw"][0].tolist(),
        "prompt_text": prompt_text,
        "full_text": full_text,
    }


@dataclass
class BaseSFTCollator:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None):
        return_tensors = "pt"

        text_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            text_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        max_label_length = max(len(feature["labels"]) for feature in features)
        batch["labels"] = torch.tensor(
            [
                feature["labels"]
                + [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                for feature in features
            ],
            dtype=torch.int64,
        )

        batch["pixel_values"] = torch.stack(
            [torch.tensor(feature["pixel_values"], dtype=torch.float32) for feature in features],
            dim=0,
        )
        batch["image_grid_thw"] = torch.stack(
            [torch.tensor(feature["image_grid_thw"], dtype=torch.long) for feature in features],
            dim=0,
        )

        if "idx" in features[0]:
            batch["idx"] = torch.tensor([feature["idx"] for feature in features], dtype=torch.long)

        return batch
