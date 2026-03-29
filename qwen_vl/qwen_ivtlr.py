import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import logging
import os

LOG_DIR = os.getenv("QWEN_LOG_DIR", ".")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'qwenvl_32_infer_sqa_time_epoch4.log'),
    level=logging.DEBUG,         
    format='[%(asctime)s] %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'  
)
import pdb
from transformers.cache_utils import DynamicCache

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 4


class IVTLR(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        image_token_id,
        visual_start_id,
        visual_end_id,
        num_selected_patches: int = 32,
        mask_selected_patches: bool = True,
        split_pool_selection: bool = False,
        new_pool_patch_count: int = None,
        use_visual_latents: bool = True,
        use_last_hidden_state: bool = True,
        enable_reasoning: bool = True,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    ):

        super(IVTLR, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.image_token_id = image_token_id
        self.visual_start_id = visual_start_id
        self.visual_end_id = visual_end_id
        self.num_selected_patches = num_selected_patches
        self.mask_selected_patches = mask_selected_patches
        self.split_pool_selection = split_pool_selection
        self.new_pool_patch_count = new_pool_patch_count
        self.use_visual_latents = use_visual_latents
        self.use_last_hidden_state = use_last_hidden_state
        self.enable_reasoning = enable_reasoning
        self.model_id = model_id
        self.last_topk_trace = []

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
        
        # self.processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        if self.new_pool_patch_count is not None:
            if self.new_pool_patch_count < 0 or self.new_pool_patch_count > self.num_selected_patches:
                raise ValueError(
                    "new_pool_patch_count must be between 0 and num_selected_patches"
                )

        if (
            self.split_pool_selection
            and self.new_pool_patch_count is None
            and self.num_selected_patches % 2 != 0
        ):
            raise ValueError(
                "num_selected_patches must be even when split_pool_selection is enabled without an explicit new_pool_patch_count"
            )

    def clear_topk_trace(self):
        self.last_topk_trace = []

    def get_topk_trace(self):
        return self.last_topk_trace

    def _get_new_pool_patch_count(self):
        if self.new_pool_patch_count is not None:
            return self.new_pool_patch_count
        return self.num_selected_patches // 2

    def _rank_pool_candidates(self, scores, pool_mask, source_ids, pool_name):
        candidate_mask = pool_mask & torch.isfinite(scores) & (source_ids >= 0)
        candidate_positions = candidate_mask.nonzero(as_tuple=False).flatten()
        if candidate_positions.numel() == 0:
            return []

        candidate_scores = scores[candidate_positions]
        sort_order = torch.argsort(candidate_scores, descending=True)
        ranked_positions = candidate_positions[sort_order]

        ranked_candidates = []
        for position in ranked_positions.tolist():
            ranked_candidates.append(
                {
                    "abs_idx": position,
                    "source_id": int(source_ids[position].item()),
                    "score": float(scores[position].item()),
                    "pool": pool_name,
                }
            )
        return ranked_candidates

    def _take_unique_candidates(self, candidates, quota, excluded_source_ids=None):
        if excluded_source_ids is None:
            excluded_source_ids = set()

        selected = []
        selected_source_ids = set(excluded_source_ids)
        for candidate in candidates:
            if len(selected) >= quota:
                break
            source_id = candidate["source_id"]
            if source_id in selected_source_ids:
                continue
            selected.append(candidate)
            selected_source_ids.add(source_id)

        return selected, selected_source_ids

    def _pad_1d_tensor(self, tensor, target_len, pad_value):
        if tensor.size(0) == target_len:
            return tensor
        pad_len = target_len - tensor.size(0)
        pad_tensor = torch.full(
            (pad_len,),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, pad_tensor], dim=0)

    def _pad_2d_tensor(self, tensor, target_len):
        if tensor.size(0) == target_len:
            return tensor
        pad_len = target_len - tensor.size(0)
        pad_tensor = torch.zeros(
            (pad_len, tensor.size(1)),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, pad_tensor], dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor,        # shape = (B, S)
        attention_mask: torch.LongTensor,    # shape = (B, S)
        labels: torch.LongTensor,            # shape = (B, S)
        position_ids: torch.LongTensor,      # shape = (B, S)
        pixel_values: torch.FloatTensor,     # shape = (B, 3, H, W)
        image_grid_thw: torch.Tensor = None,
        **kwargs
    ):

        B, S = input_ids.size()
        sample_keys = kwargs.get("sample_keys", None)
        if sample_keys is None:
            sample_keys = [None for _ in range(B)]
        self.last_topk_trace = [
            {
                "batch_index": b,
                "sample_key": sample_keys[b],
                "steps": [],
            }
            for b in range(B)
        ]

        # decode
        _ = self.processor.tokenizer.batch_decode(
            input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )

        inputs_embeds = self.embedding(input_ids)  # (B, S, D)

        original_mask = torch.ones((B, S), dtype=torch.bool, device=input_ids.device)
        initial_pool_mask = torch.zeros((B, S), dtype=torch.bool, device=input_ids.device)
        new_pool_mask = torch.zeros((B, S), dtype=torch.bool, device=input_ids.device)
        source_index_map = torch.full((B, S), -1, dtype=torch.long, device=input_ids.device)

        vs_indices = (input_ids == self.visual_start_id).nonzero(as_tuple=True)
        ve_indices = (input_ids == self.visual_end_id).nonzero(as_tuple=True)
        vs_pos_per_batch = {b.item(): vs_indices[1][i].item() for i, b in enumerate(vs_indices[0])}
        ve_pos_per_batch = {b.item(): ve_indices[1][i].item() for i, b in enumerate(ve_indices[0])}

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.base_causallm.visual.get_dtype())
            image_embeds = self.base_causallm.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            if n_image_tokens != image_embeds.shape[0]:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_embeds.shape[0]}"
                )
            image_mask_init = (input_ids == self.image_token_id)  # (B, orig_S)
            expand_mask = image_mask_init.unsqueeze(-1).expand(-1, -1, inputs_embeds.size(-1))
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(expand_mask, image_embeds)
        else:
            image_mask_init = torch.zeros((B, S), dtype=torch.bool, device=input_ids.device)
        

        for b in range(B):
            vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
            image_positions = torch.arange(ve - vs - 1, device=input_ids.device, dtype=torch.long)
            initial_pool_mask[b, vs + 1:ve] = True
            source_index_map[b, vs + 1:ve] = image_positions

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == b]
            for b in range(B)
        ]
        max_n_latents = max((len(lst) for lst in latent_lists), default=0)
        if not self.enable_reasoning:
            max_n_latents = 0

        if max_n_latents > 0:
            first_latent_pos = min(lst[0] for lst in latent_lists if len(lst) > 0)
            end = first_latent_pos
        else:
            end = S
        
        kv_cache = None
        all_logits = []

        if max_n_latents > 0:
            for pass_idx in range(max_n_latents):
                start = 0
                hidden_states_offset = 0
                if kv_cache is None:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, start:end, :],  # (B, end, D)
                        attention_mask=attention_mask[:, start:end],
                        position_ids=position_ids[:, start:end],
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )
                else:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, start:end, :],
                        attention_mask=attention_mask[:, :end],
                        position_ids=position_ids[:, start:end],
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )

                logits_this = outputs.logits                   
                hidden_states = outputs.hidden_states[-1]      
                attentions    = outputs.attentions              # list of (B, heads, seq_len, seq_len)
                kv_cache      = outputs.past_key_values

                all_logits.append(logits_this)

                #   Top-K
                avg_attn = torch.cat(attentions, dim=1).mean(dim=1)  # (B, seq_len)
                current_seq_len = avg_attn.size(1)
                select_image_embeds = []
                selected_source_ids_per_batch = []
                inserted_counts = []

                for b in range(B):
                    last_attn = avg_attn[b, end - 1]  # shape=(seq_len,)
                    vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
                    if self.use_visual_latents:
                        scores = last_attn.clone()
                        allowed_positions = (
                            initial_pool_mask[b, :current_seq_len] | new_pool_mask[b, :current_seq_len]
                        )
                        invalid = ~allowed_positions
                        scores[invalid] = float("-inf")

                        current_initial_mask = initial_pool_mask[b, :current_seq_len]
                        current_new_mask = new_pool_mask[b, :current_seq_len]
                        current_source_ids = source_index_map[b, :current_seq_len]

                        initial_candidates = self._rank_pool_candidates(
                            scores,
                            current_initial_mask,
                            current_source_ids,
                            "initial",
                        )
                        new_candidates = self._rank_pool_candidates(
                            scores,
                            current_new_mask,
                            current_source_ids,
                            "new",
                        )

                        if not self.split_pool_selection:
                            selected_candidates, _ = self._take_unique_candidates(
                                initial_candidates,
                                self.num_selected_patches,
                            )
                        elif pass_idx == 0:
                            selected_candidates, _ = self._take_unique_candidates(
                                initial_candidates,
                                self.num_selected_patches,
                            )
                        else:
                            new_pool_quota = self._get_new_pool_patch_count()
                            initial_pool_quota = self.num_selected_patches - new_pool_quota
                            initial_selected, selected_source_ids = self._take_unique_candidates(
                                initial_candidates,
                                initial_pool_quota,
                            )
                            new_selected, selected_source_ids = self._take_unique_candidates(
                                new_candidates,
                                new_pool_quota,
                                excluded_source_ids=selected_source_ids,
                            )
                            selected_candidates = initial_selected + new_selected

                            if len(selected_candidates) < self.num_selected_patches:
                                combined_candidates = sorted(
                                    initial_candidates + new_candidates,
                                    key=lambda candidate: candidate["score"],
                                    reverse=True,
                                )
                                filler_candidates, _ = self._take_unique_candidates(
                                    combined_candidates,
                                    self.num_selected_patches - len(selected_candidates),
                                    excluded_source_ids={
                                        candidate["source_id"] for candidate in selected_candidates
                                    },
                                )
                                selected_candidates.extend(filler_candidates)

                        if len(selected_candidates) != self.num_selected_patches:
                            raise ValueError(
                                f"Unable to select {self.num_selected_patches} unique visual embeddings at pass {pass_idx} for batch {b}"
                            )

                        abs_idxs = torch.tensor(
                            [candidate["abs_idx"] for candidate in selected_candidates],
                            device=input_ids.device,
                            dtype=torch.long,
                        )
                        selected_source_ids = torch.tensor(
                            [candidate["source_id"] for candidate in selected_candidates],
                            device=input_ids.device,
                            dtype=torch.long,
                        )
                        selected_pools = [candidate["pool"] for candidate in selected_candidates]
                        topk_rel = selected_source_ids

                        logging.debug(f"topk_rel: {topk_rel}")
                        logging.debug(f"abs idx: {abs_idxs}")

                        if self.mask_selected_patches:
                            initial_pool_mask[b, abs_idxs] = False
                            new_pool_mask[b, abs_idxs] = False

                        picked = inputs_embeds[b, abs_idxs, :]  # (K, D)
                    else:
                        selected_candidates = []
                        abs_idxs = torch.empty((0,), device=input_ids.device, dtype=torch.long)
                        selected_source_ids = torch.empty((0,), device=input_ids.device, dtype=torch.long)
                        selected_pools = []
                        topk_rel = selected_source_ids
                        picked = inputs_embeds[b, abs_idxs, :]  # (0, D)

                    select_image_embeds.append(picked)
                    selected_source_ids_per_batch.append(selected_source_ids)
                    inserted_counts.append(abs_idxs.numel())

                    initial_abs_idxs = [
                        candidate["abs_idx"] for candidate in selected_candidates if candidate["pool"] == "initial"
                    ]
                    new_abs_idxs = [
                        candidate["abs_idx"] for candidate in selected_candidates if candidate["pool"] == "new"
                    ]
                    self.last_topk_trace[b]["steps"].append(
                        {
                            "pass_idx": pass_idx,
                            "mask_selected_patches": self.mask_selected_patches,
                            "split_pool_selection": self.split_pool_selection,
                            "new_pool_patch_count": self._get_new_pool_patch_count() if self.split_pool_selection else 0,
                            "topk_rel": topk_rel.detach().cpu().tolist(),
                            "abs_idxs": abs_idxs.detach().cpu().tolist(),
                            "initial_abs_idxs": initial_abs_idxs,
                            "new_abs_idxs": new_abs_idxs,
                            "selected_pools": selected_pools,
                            "source_ids": selected_source_ids.detach().cpu().tolist(),
                            "embeddings": picked.detach().to(torch.float16).cpu(),
                        }
                    )

                select_image_embeds = torch.stack(select_image_embeds, dim=0)  # (B, K, D)
                inputs_embeds_detached = inputs_embeds.detach().clone()
                for b in range(B):
                    if len(latent_lists[b]) > pass_idx and self.use_last_hidden_state:
                        t_idx = latent_lists[b][pass_idx]
                        rel_pos = t_idx - 1 - hidden_states_offset
                        rel_pos = max(0, min(rel_pos, hidden_states.size(1) - 1))
                        inputs_embeds_detached[b, t_idx, :] = hidden_states[b, rel_pos, :]

                inputs_embeds.data = inputs_embeds_detached
                new_inputs_embeds = []
                new_attention_mask = []
                new_position_ids = []
                new_original_mask = []
                new_initial_pool_mask = []
                new_new_pool_mask = []
                new_source_index_map = []
                batch_max_len = 0
                insert_count = inserted_counts[0]
                if any(count != insert_count for count in inserted_counts):
                    raise ValueError("All batch items must insert the same number of embeddings per pass")

                for b in range(B):
                    end_b = end
                    prefix_b = inputs_embeds[b, :end_b, :]    # (end_b, D)
                    suffix_b = inputs_embeds[b, end_b:, :]    # (old_len - end_b, D)
                    v_embed_b = select_image_embeds[b]       # (K, D)
                    merged_b = torch.cat([prefix_b, v_embed_b, suffix_b], dim=0)  # (old_len+K, D)
                    new_inputs_embeds.append(merged_b)

                    # attention_mask
                    att_pref = attention_mask[b, :end_b]      # (end_b,)
                    att_suf  = attention_mask[b, end_b:]      # (old_len-end_b,)
                    att_v    = torch.ones(insert_count, device=attention_mask.device, dtype=attention_mask.dtype)
                    merged_att = torch.cat([att_pref, att_v, att_suf], dim=0)  # (new_len,)
                    new_attention_mask.append(merged_att)

                    # position_ids 
                    new_pos = torch.arange(merged_b.size(0), device=position_ids.device)
                    new_position_ids.append(new_pos)

                    # original_mask
                    orig_pref = original_mask[b, :end_b]       # (end_b,)
                    orig_suf  = original_mask[b, end_b:]       # (old_len-end_b,)
                    orig_v    = torch.zeros(insert_count, device=input_ids.device, dtype=torch.bool)
                    merged_orig = torch.cat([orig_pref, orig_v, orig_suf], dim=0)
                    new_original_mask.append(merged_orig)

                    # initial_pool_mask
                    init_pref = initial_pool_mask[b, :end_b]
                    init_suf  = initial_pool_mask[b, end_b:]
                    init_v    = torch.zeros(insert_count, device=input_ids.device, dtype=torch.bool)
                    merged_init = torch.cat([init_pref, init_v, init_suf], dim=0)
                    new_initial_pool_mask.append(merged_init)

                    # new_pool_mask
                    new_pref = new_pool_mask[b, :end_b]
                    new_suf  = new_pool_mask[b, end_b:]
                    new_v    = torch.ones(insert_count, device=input_ids.device, dtype=torch.bool)
                    merged_new = torch.cat([new_pref, new_v, new_suf], dim=0)
                    new_new_pool_mask.append(merged_new)

                    # source_index_map
                    src_pref = source_index_map[b, :end_b]
                    src_suf  = source_index_map[b, end_b:]
                    src_v    = selected_source_ids_per_batch[b]
                    merged_src = torch.cat([src_pref, src_v, src_suf], dim=0)
                    new_source_index_map.append(merged_src)

                    batch_max_len = max(batch_max_len, merged_b.size(0))

                padded_embeds = []
                padded_att   = []
                padded_pos   = []
                padded_orig  = []
                padded_init  = []
                padded_new   = []
                padded_src   = []

                for b in range(B):
                    emb_b = self._pad_2d_tensor(new_inputs_embeds[b], batch_max_len)
                    att_b = self._pad_1d_tensor(new_attention_mask[b], batch_max_len, 0)
                    pos_b = self._pad_1d_tensor(new_position_ids[b], batch_max_len, 0)
                    orig_b = self._pad_1d_tensor(new_original_mask[b], batch_max_len, False)
                    init_b = self._pad_1d_tensor(new_initial_pool_mask[b], batch_max_len, False)
                    new_b = self._pad_1d_tensor(new_new_pool_mask[b], batch_max_len, False)
                    src_b = self._pad_1d_tensor(new_source_index_map[b], batch_max_len, -1)

                    padded_embeds.append(emb_b.unsqueeze(0))
                    padded_att.append(att_b.unsqueeze(0))
                    padded_pos.append(pos_b.unsqueeze(0))
                    padded_orig.append(orig_b.unsqueeze(0))
                    padded_init.append(init_b.unsqueeze(0))
                    padded_new.append(new_b.unsqueeze(0))
                    padded_src.append(src_b.unsqueeze(0))

                inputs_embeds = torch.cat(padded_embeds, dim=0)    
                attention_mask = torch.cat(padded_att, dim=0)      
                position_ids    = torch.cat(padded_pos, dim=0)     
                original_mask  = torch.cat(padded_orig, dim=0)
                initial_pool_mask = torch.cat(padded_init, dim=0)
                new_pool_mask = torch.cat(padded_new, dim=0)
                source_index_map = torch.cat(padded_src, dim=0)
                K = insert_count
                for b in range(B):
                    for i, pos in enumerate(latent_lists[b]):
                        if pos > end:
                            latent_lists[b][i] = pos + K
                            logging.debug(f"latent pos: {latent_lists[b][i]}")

                if pass_idx + 1 >= max_n_latents:
                    end = inputs_embeds.size(1)
                else:
                    end = end + 1 + K

            if kv_cache:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    output_attentions=False,
                )
            else:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    output_attentions=False,
                )
            all_logits.append(outputs.logits)

        else:
            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                output_attentions=False,
            )
            all_logits.append(outputs.logits)

        logits = torch.cat(all_logits, dim=-2)  # (B, total_len, V)
        B, final_S, V = logits.size()


        new_labels = torch.full((B, final_S), -100, device=input_ids.device, dtype=labels.dtype)
        for b in range(B):
            num_labels = labels.size(1)
            new_labels[:, -num_labels:] = labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = new_labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)


    def train(self, mode=True):
        self.base_causallm.train(mode)

    def eval(self):
        self.base_causallm.eval()
    
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            image_grid_thw: torch.Tensor = None,
            past_key_values: tuple = None,
            attention_mask: torch.Tensor = None,
            inputs_embeds: torch.FloatTensor = None,
            position_ids: torch.LongTensor = None,
            use_cache: bool = True,
            **kwargs
        ):
        
        self.base_causallm.prepare_inputs_for_generation(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs
        )

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        pixel_values,
        image_grid_thw,
        max_new_tokens=16,
        output_embedding=False,
        **kwargs
    ):
        self.gen_forward_cnt = 0
        eos_pos = None

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()
        
        current_ids = input_ids.clone()

        position_ids = torch.arange(
            0, current_ids.shape[1], 
            dtype=torch.long, 
            device=current_ids.device
        ).reshape(1, -1)

        outputs = self.forward(
            input_ids=current_ids,
            attention_mask=torch.ones_like(current_ids),
            labels=current_ids.clone(),  
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs
        )


        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
            

        current_inputs_embeds = outputs.inputs_embeds  # shape: (1, seq_len_after_insertion, hidden_dim)
        current_seq_len = current_inputs_embeds.shape[1]
        

        current_attention_mask = torch.ones((1, current_seq_len), device=current_inputs_embeds.device)
        

        next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1)

        self.gen_forward_cnt += 1
        

        past_key_values = None
        

        for _ in range(max_new_tokens - 1):
            if past_key_values is None:
                logging.debug(f"no kv_cache, using full embedding sequence")
                inputs_embeds_for_forward = current_inputs_embeds
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.arange(
                        0, current_inputs_embeds.shape[1], 
                    dtype=torch.long, 
                        device=current_inputs_embeds.device
                ).reshape(1, -1)
            else:
                logging.debug(f"using kv_cache, input_shape: {next_token_embedding.shape}")
                inputs_embeds_for_forward = next_token_embedding
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.tensor([[current_inputs_embeds.shape[1] - 1]], device=current_inputs_embeds.device)

            outputs = self.base_causallm.forward(
                inputs_embeds=inputs_embeds_for_forward,
                attention_mask=attention_mask_for_forward,
                position_ids=position_ids,
                pixel_values=pixel_values if past_key_values is None else None, 
                image_grid_thw=image_grid_thw if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values

            next_token = torch.argmax(outputs.logits[0, -1]).item()
            tokens.append(next_token)
            
            next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1)

            self.gen_forward_cnt += 1

            if self.gen_forward_cnt % 10 == 0 and self.gen_forward_cnt >= 10:
                logging.debug(f"gen_forward_cnt: {self.gen_forward_cnt}")

            if next_token == self.eos_token_id:
                logging.debug(f"EOS token encountered at position {len(tokens)}, stopping generation")
                break

        print("generate 315")
        
        
        if output_embedding:
            return torch.tensor(tokens).view(1, -1), current_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)

    def generate_with_selected_embeddings(
        self,
        input_ids,
        selected_step_embeddings,
        attention_mask=None,
        max_new_tokens=16,
        num_steps=None,
    ):
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        inputs_embeds = self.embedding(input_ids.clone())
        image_positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=False).flatten()

        if num_steps is None:
            num_steps = len(selected_step_embeddings)

        selected_list = []
        for step in selected_step_embeddings[:num_steps]:
            if isinstance(step, torch.Tensor):
                selected_list.append(step)
            else:
                selected_list.append(torch.tensor(step))

        if len(selected_list) > 0:
            flattened = torch.cat(selected_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            n_replace = min(image_positions.numel(), flattened.size(0))
            if n_replace > 0:
                inputs_embeds[0, image_positions[:n_replace], :] = flattened[:n_replace]

        tokens = input_ids[0].detach().tolist()
        current_inputs_embeds = inputs_embeds
        current_attention_mask = attention_mask.to(inputs_embeds.device)

        position_ids = torch.arange(
            0,
            current_inputs_embeds.shape[1],
            dtype=torch.long,
            device=current_inputs_embeds.device,
        ).reshape(1, -1)

        outputs = self.base_causallm.forward(
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            position_ids=position_ids,
            pixel_values=None,
            image_grid_thw=None,
            use_cache=True,
        )

        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        next_token_embedding = self.embedding(
            torch.tensor([[next_token]], device=current_inputs_embeds.device)
        )
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1
        )

        past_key_values = outputs.past_key_values

        for _ in range(max_new_tokens - 1):
            position_ids = torch.tensor(
                [[current_inputs_embeds.shape[1] - 1]], device=current_inputs_embeds.device
            )
            outputs = self.base_causallm.forward(
                inputs_embeds=next_token_embedding,
                attention_mask=current_attention_mask,
                position_ids=position_ids,
                pixel_values=None,
                image_grid_thw=None,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            next_token = torch.argmax(outputs.logits[0, -1]).item()
            tokens.append(next_token)

            next_token_embedding = self.embedding(
                torch.tensor([[next_token]], device=current_inputs_embeds.device)
            )
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1
            )

            if next_token == self.eos_token_id:
                break

        return torch.tensor(tokens).view(1, -1)


