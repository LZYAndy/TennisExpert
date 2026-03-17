# coding=utf-8
"""
Custom trainer with chunk-wise backward for long multi-turn conversations.

Key idea:
- Do NOT accumulate loss across sliding windows inside model.forward (that chains graphs and increases VRAM).
- Instead, do window slicing in the Trainer.training_step and call backward per-chunk.
- This keeps VRAM ~ O(window_len) rather than O(full_conversation_len).

Assumptions (same as your current model utilities):
- per-GPU batch_size == 1 (DDP is fine; global batch can be >1).
- `inputs` contains: input_ids, labels, attention_mask (optional),
  and optional multimodal fields: pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw.
- Data collator injects:
    inputs["sliding_config"] = {"size": <turns_per_window>, "stride": <stride_turns>}
    inputs["use_global_context"] = True/False
"""

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union
import torch.distributed as dist
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override


from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


def _get_turn_boundaries(input_ids):
        """
        Splits based on <|im_start|>user (151644, 872).
        This identifies the starting position of each user turn.
        """
        im_start_id = 151644
        user_id = 872  # Based on the provided tensor, the user token is 872
        
        seq = input_ids[0] # Assumes batch_size=1
        
        # Logic: Find all indices 'i' where seq[i] == im_start AND seq[i+1] == user
        
        # 1. Create offset slices for vectorized comparison
        curr_tokens = seq[:-1] # From index 0 to N-1
        next_tokens = seq[1:]  # From index 1 to N
        
        # 2. Find matching pairs
        # Condition: (curr == 151644) AND (next == 872)
        match_mask = (curr_tokens == im_start_id) & (next_tokens == user_id)
        
        # 3. Extract indices
        start_indices = match_mask.nonzero(as_tuple=True)[0].tolist()
        
        # 4. Ensure the total sequence length is included as the final endpoint
        # If start_indices is empty (e.g., no user header found), add 0 to prevent errors
        if not start_indices:
            start_indices = [0]
            
        # Append the sequence end index
        boundaries = start_indices + [input_ids.shape[1]]
        
        return boundaries

def _slice_cached_visual_by_grid(
    *,
    flat_embeds: torch.Tensor,
    deepstack_embeds: Optional[list[torch.Tensor]],
    grid_thw: torch.LongTensor,
    cum_sizes: torch.LongTensor,
    grid_range: tuple[int, int],
) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], torch.LongTensor]:
    """
    Slice cached embeddings by selecting grid entries in [start_idx, end_idx).
    IMPORTANT: We slice by embedding segment boundaries (cum_sizes) rather than grid_thw.prod(),
    because each grid entry contributes `prod(thw)//merge^2` placeholder tokens in the LLM.
    """
    start_idx, end_idx = grid_range
    if start_idx < 0:
        start_idx = 0
    if end_idx > int(grid_thw.shape[0]):
        end_idx = int(grid_thw.shape[0])
    if end_idx <= start_idx:
        return flat_embeds.new_empty((0,) + flat_embeds.shape[1:]), None, grid_thw.new_empty((0, 3))

    embed_start = 0 if start_idx == 0 else int(cum_sizes[start_idx - 1].item())
    embed_end = int(cum_sizes[end_idx - 1].item())

    flat_slice = flat_embeds[embed_start:embed_end]
    deep_slice = None
    if deepstack_embeds is not None:
        deep_slice = [d[embed_start:embed_end] for d in deepstack_embeds]

    return flat_slice, deep_slice, grid_thw[start_idx:end_idx]

def _sync_min_and_resample_chunks(chunk_list, device):
    """
    1) k = min(len(chunk_list)) across ranks
    2) all ranks participate in the same collectives
    3) if local has more than k, randomly sample k chunks (without replacement)
    """
    if (not dist.is_available()) or (not dist.is_initialized()):
        return chunk_list, len(chunk_list)

    rank = dist.get_rank()

    # --- 1) global min k ---
    local_n = torch.tensor([len(chunk_list)], device=device, dtype=torch.long)
    dist.all_reduce(local_n, op=dist.ReduceOp.MIN)
    k = int(local_n.item())

    # --- 2) broadcast a shared seed: IMPORTANT all ranks must call this ---
    seed = torch.empty((), device=device, dtype=torch.long)
    if rank == 0:
        seed.fill_(torch.randint(0, 2**31 - 1, (1,), device=device, dtype=torch.long).item())
    dist.broadcast(seed, src=0)

    # --- 3) handle k == 0 (safe now because broadcast already happened) ---
    if k <= 0:
        return [], 0

    # --- 4) resample if needed ---
    n = len(chunk_list)
    if n <= k:
        return chunk_list, n

    # Sample k indices without replacement
    g = torch.Generator(device=device)
    g.manual_seed(int(seed.item()) + rank)

    perm = torch.randperm(n, generator=g, device=device)[:k].tolist()
    new_chunk_list = [chunk_list[i] for i in perm]
    return new_chunk_list, k

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        training_args: TrainingArguments = kwargs.get("args")

        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )

        if training_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, training_args)
        
        if hasattr(self.processing_class, "tokenizer"):
            self.tokenizer = self.processing_class.tokenizer
        else:
            self.tokenizer = self.processing_class

        self._mem_prefix_ids = self.tokenizer.encode(
            "\nVideo and metadata of several past rallies: ",
            add_special_tokens=False
        )

        self._cur_prefix_ids = self.tokenizer.encode(
            "\nVideo and metadata of current rally: ",
            add_special_tokens=False
)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    # -------------------------------
    # Chunk-wise backward
    # -------------------------------

    @override
    def training_step(self, model: torch.nn.Module, inputs: dict[str, Any], num_items_in_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Chunk-wise backward for long dialogs:
        - Split a single sample into multiple sliding windows.
        - For each window, run forward once and backward immediately.
        This avoids chaining graphs across windows (which would increase VRAM).
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        sliding_config = inputs.pop("sliding_config", {'size': 5, 'stride': 1})
        #use_global_context = inputs.pop("use_global_context", True)

        # Fallback: default Trainer behavior
        if sliding_config is None or inputs.get("labels", None) is None:
            return super().training_step(model, inputs, num_items_in_batch)

        # This implementation assumes per-GPU batch_size == 1
        if inputs["input_ids"].shape[0] != 1:
            return super().training_step(model, inputs, num_items_in_batch)

        # Unwrap DDP/FSDP
        raw_model = model.module if hasattr(model, "module") else model

        # Prepare cached vision embeddings ONCE per full conversation to avoid repeated vision encoding.
        with torch.no_grad():
            vision_cache = self._preencode_full_visual(raw_model, inputs)
        

        # Build chunk inputs list (pure dict slicing; no forward yet)
        chunk_list = list(self._iter_sliding_chunk_inputs(inputs, sliding_config, vision_cache))

        chunk_list, k = _sync_min_and_resample_chunks(
            chunk_list,
            device=inputs["input_ids"].device
        )

        if not chunk_list:
            return torch.tensor(0.0, device=inputs["input_ids"].device)

        total_contrib = sum(contrib for _, contrib in chunk_list)

        if total_contrib <= 0:
            return torch.tensor(0.0, device=inputs["input_ids"].device)

        grad_accum = max(1, int(self.args.gradient_accumulation_steps))

        # For logging only (do NOT build graphs)
        loss_sum = 0.0
        token_sum = 0
        chunk = 0
        for chunk_inputs, contrib in chunk_list:
            if contrib <= 0:
                continue

            # Token-weighted mean across chunks + gradient accumulation scaling
            weight = float(contrib) / float(total_contrib)

            with self.compute_loss_context_manager():
                outputs = raw_model(**chunk_inputs)
                chunk_loss = outputs.loss

            # Important: scale for grad accumulation
            chunk_loss = (chunk_loss * weight) / grad_accum

            #  backward per chunk (graph frees per-chunk)
            self.accelerator.backward(chunk_loss)

            # logging
            loss_sum += float(outputs.loss.detach().item()) * contrib
            token_sum += contrib
            
            if dist.get_rank() == 0 and chunk % 10 == 0:
                print(f" Chunk: {chunk} loss: {outputs.loss.detach().item():.4f}")

            chunk += 1

        if token_sum == 0:
            return torch.tensor(0.0, device=inputs["input_ids"].device)
        
        avg_loss = loss_sum / token_sum
        scaled_avg_loss = avg_loss / grad_accum
        return torch.tensor(scaled_avg_loss, device=inputs["input_ids"].device)

    def _preencode_full_visual(self, raw_model: torch.nn.Module, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Pre-encode all images/videos for the full conversation once.
        Returns a cache dict used for slicing per-chunk.
        """
        cache: dict[str, Any] = {}

        base = getattr(raw_model, "model", raw_model)  # Qwen3VLForConditionalGeneration.model -> Qwen3VLModel
        vision_start_id = base.config.vision_start_token_id

        # Images
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        if pixel_values is not None and image_grid_thw is not None:
            # base.get_image_features returns (list_of_embeds, deepstack_list)
            image_embeds_list, deepstack_image_embeds = base.get_image_features(pixel_values, image_grid_thw)
            flat_image_embeds = torch.cat(image_embeds_list, dim=0)
            merge = int(base.visual.spatial_merge_size)
            split_sizes = (image_grid_thw.prod(-1) // (merge ** 2)).tolist()
            cum_sizes = torch.tensor(split_sizes, device=image_grid_thw.device).cumsum(0)
            cache["image"] = dict(
                flat=flat_image_embeds,
                deep=deepstack_image_embeds,
                grid=image_grid_thw,
                split_sizes=split_sizes,
                cum_sizes=cum_sizes,
                token_id=base.config.image_token_id,
                vision_start_id=vision_start_id,
            )

        # Videos
        pixel_values_videos = inputs.get("pixel_values_videos", None)
        video_grid_thw = inputs.get("video_grid_thw", None)
        if pixel_values_videos is not None and video_grid_thw is not None:
            video_embeds_list, deepstack_video_embeds = base.get_video_features(pixel_values_videos, video_grid_thw)
            flat_video_embeds = torch.cat(video_embeds_list, dim=0)
            merge = int(base.visual.spatial_merge_size)
            split_sizes = (video_grid_thw.prod(-1) // (merge ** 2)).tolist()
            cum_sizes = torch.tensor(split_sizes, device=video_grid_thw.device).cumsum(0)
            cache["video"] = dict(
                flat=flat_video_embeds,
                deep=deepstack_video_embeds,
                grid=video_grid_thw,
                split_sizes=split_sizes,
                cum_sizes=cum_sizes,
                token_id=base.config.video_token_id,
                vision_start_id=vision_start_id,
            )

        return cache

    def _iter_sliding_chunk_inputs(
        self,
        inputs: dict[str, Any],
        sliding_config: dict[str, int],
        vision_cache: dict[str, Any],
    ):
        """
        Sliding window chunk iterator (NO summary inference).
        Summary is already included in the dataset as a "summary" turn,
        so it will naturally be inside prev_ids/curr_ids token slices.
        """

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        boundaries = _get_turn_boundaries(input_ids)

        turns_per_window = int(sliding_config.get("size", 5))
        stride_turns = int(sliding_config.get("stride", 1))
        total_turns = len(boundaries) - 1

        # -------------------------
        # Prefix helper (optional)
        # -------------------------
        device = input_ids.device
        bsz = input_ids.shape[0]

        def _make_prefix(prefix_ids_list):
            if not prefix_ids_list:
                return None, None, None, 0

            p_ids = torch.tensor(prefix_ids_list, device=device).unsqueeze(0).expand(bsz, -1)
            p_labels = torch.full_like(p_ids, IGNORE_INDEX)
            p_mask = torch.ones_like(p_ids) if attention_mask is not None else None

            return p_ids, p_labels, p_mask, p_ids.shape[1]

        mem_p_ids, mem_p_labels, mem_p_mask, mem_p_len = _make_prefix(self._mem_prefix_ids)
        cur_p_ids, cur_p_labels, cur_p_mask, cur_p_len = _make_prefix(self._cur_prefix_ids)

        # -------------------------
        # System slice detection
        # -------------------------
        if boundaries[0] > 0:
            system_end = boundaries[0]
            has_system = True
        else:
            system_end = 0
            has_system = False

        img_cache = vision_cache.get("image", None)
        vid_cache = vision_cache.get("video", None)

        # -------------------------
        # Sliding window loop
        # -------------------------
        for i in range(1, total_turns + 1, stride_turns):
            start_turn = max(0, i - turns_per_window)
            end_turn = i
            cur_turn = end_turn - 1

            prev_start = boundaries[start_turn]
            prev_end = boundaries[cur_turn]
            curr_start = boundaries[cur_turn]
            curr_end = boundaries[end_turn]

            prev_ids = input_ids[:, prev_start:prev_end]
            prev_labels = labels[:, prev_start:prev_end]
            prev_mask = attention_mask[:, prev_start:prev_end] if attention_mask is not None else None

            curr_ids = input_ids[:, curr_start:curr_end]
            curr_labels = labels[:, curr_start:curr_end]
            curr_mask = attention_mask[:, curr_start:curr_end] if attention_mask is not None else None

            has_prev = (prev_ids.shape[1] > 0)

            if has_system:
                g_ids = input_ids[:, :system_end]
                g_labels = labels[:, :system_end]
                g_mask = attention_mask[:, :system_end] if attention_mask is not None else None
                system_len = g_ids.shape[1]
            else:
                g_ids = g_labels = g_mask = None
                system_len = 0

            # -------------------------
            # Assemble chunk
            # -------------------------
            cat_ids, cat_labels = [], []
            cat_masks = [] if attention_mask is not None else None

            def _append(x_ids, x_labels, x_mask):
                if x_ids is None or x_ids.shape[1] == 0:
                    return
                cat_ids.append(x_ids)
                cat_labels.append(x_labels)
                if attention_mask is not None:
                    cat_masks.append(x_mask)

            _append(g_ids, g_labels, g_mask)

            if has_prev:
                _append(mem_p_ids, mem_p_labels, mem_p_mask)
                _append(prev_ids, prev_labels, prev_mask)

            _append(cur_p_ids, cur_p_labels, cur_p_mask)
            _append(curr_ids, curr_labels, curr_mask)

            chunk_input_ids = torch.cat(cat_ids, dim=1)
            chunk_labels = torch.cat(cat_labels, dim=1)
            chunk_attention_mask = torch.cat(cat_masks, dim=1) if attention_mask is not None else None

            # -------------------------
            # Overlap masking
            # -------------------------
            cutoff_turn = max(i - stride_turns, start_turn)
            new_start_full = boundaries[cutoff_turn]

            prev_len = prev_ids.shape[1]
            eff_mem_len = mem_p_len if has_prev else 0
            eff_cur_len = cur_p_len

            if new_start_full < curr_start:
                rel_prev = new_start_full - prev_start
                offset_non_system = eff_mem_len + rel_prev
            else:
                rel_curr = new_start_full - curr_start
                offset_non_system = eff_mem_len + prev_len + eff_cur_len + rel_curr

            chunk_new_start = system_len + offset_non_system

            chunk_labels = chunk_labels.clone()
            chunk_labels[:, :chunk_new_start] = IGNORE_INDEX

            with torch.no_grad():
                contrib = int((chunk_labels[:, 1:] != IGNORE_INDEX).sum().item())

            # -------------------------
            # Slice cached visuals (unchanged)
            # -------------------------
            chunk_kwargs = {}
            if img_cache is not None:
                flat, deep, grid = _slice_cached_visual_by_grid(
                    flat_embeds=img_cache["flat"],
                    deepstack_embeds=img_cache["deep"],
                    grid_thw=img_cache["grid"],
                    cum_sizes=img_cache["cum_sizes"],
                    grid_range=(start_turn, end_turn),
                )
                chunk_kwargs["image_embeds"] = flat
                chunk_kwargs["deepstack_image_embeds"] = deep
                chunk_kwargs["image_grid_thw"] = grid

            if vid_cache is not None:
                flat, deep, grid = _slice_cached_visual_by_grid(
                    flat_embeds=vid_cache["flat"],
                    deepstack_embeds=vid_cache["deep"],
                    grid_thw=vid_cache["grid"],
                    cum_sizes=vid_cache["cum_sizes"],
                    grid_range=(start_turn, end_turn),
                )
                chunk_kwargs["video_embeds"] = flat
                chunk_kwargs["deepstack_video_embeds"] = deep
                chunk_kwargs["video_grid_thw"] = grid

            chunk_inputs = {
                "input_ids": chunk_input_ids,
                "attention_mask": chunk_attention_mask,
                "labels": chunk_labels,
                "pixel_values": None if "image_embeds" in chunk_kwargs else inputs.get("pixel_values", None),
                "pixel_values_videos": None if "video_embeds" in chunk_kwargs else inputs.get("pixel_values_videos", None),
                "image_grid_thw": chunk_kwargs.get("image_grid_thw", inputs.get("image_grid_thw", None)),
                "video_grid_thw": chunk_kwargs.get("video_grid_thw", inputs.get("video_grid_thw", None)),
                **chunk_kwargs,
            }

            yield chunk_inputs, contrib

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens."""
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`."""
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
