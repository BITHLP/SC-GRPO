# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _forward_micro_batch_logits(
        self,
        micro_batch,
        temperature,
        distill_topk: int = 0,
        topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
        """Return logits or top-K log-probs for response tokens."""
        if self.use_fused_kernels:
            raise NotImplementedError("SC-GRPO top-k KL weighting does not support fused actor kernels.")

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids,
                        attention_mask,
                        position_ids,
                        cu_seqlens,
                        multi_modal_inputs,
                    )

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)
                del output
                logits_rmpad.div_(temperature)

                if self.use_ulysses_sp:
                    logits_rmpad = gather_outputs_and_unpad(
                        logits_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

                if distill_topk > 0:
                    logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                    if topk_indices is not None:
                        full_topk_idx = logits_rmpad.new_zeros(
                            batch_size, seqlen, topk_indices.size(-1), dtype=topk_indices.dtype
                        )
                        full_topk_idx[:, -response_length - 1 : -1, :] = topk_indices
                        topk_idx_rmpad = index_first_axis(
                            rearrange(full_topk_idx, "b s k -> (b s) k"), indices
                        )
                        gathered_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_idx_rmpad)
                        topk_logps_rmpad = gathered_rmpad - logsumexp_rmpad
                        del logits_rmpad, gathered_rmpad, logsumexp_rmpad, full_topk_idx, topk_idx_rmpad
                        full_topk_logps = pad_input(
                            hidden_states=topk_logps_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        del topk_logps_rmpad
                        topk_logps = full_topk_logps[:, -response_length - 1 : -1, :].clone()
                        del full_topk_logps
                        return topk_logps, None

                    topk_vals_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, distill_topk, dim=-1)
                    topk_logps_rmpad = topk_vals_rmpad - logsumexp_rmpad
                    del logits_rmpad, topk_vals_rmpad, logsumexp_rmpad
                    topk_size = topk_logps_rmpad.size(-1)
                    full_topk_logps = topk_logps_rmpad.new_zeros(batch_size * seqlen, topk_size)
                    full_topk_logps = full_topk_logps.index_put((indices,), topk_logps_rmpad)
                    full_topk_logps = full_topk_logps.view(batch_size, seqlen, topk_size)
                    full_topk_indices = pad_input(
                        hidden_states=topk_indices_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    del topk_logps_rmpad, topk_indices_rmpad
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :].clone()
                    response_topk_indices = full_topk_indices[:, -response_length - 1 : -1, :].clone()
                    del full_topk_logps, full_topk_indices
                    return topk_logps, response_topk_indices

                full_logits = pad_input(
                    hidden_states=logits_rmpad,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                logits = full_logits[:, -response_length - 1 : -1, :].clone()
                del full_logits
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :].clone()
                del output

                if distill_topk > 0:
                    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                    if topk_indices is not None:
                        gathered = torch.gather(logits, dim=-1, index=topk_indices)
                        topk_logps = gathered - logsumexp
                        del logits, gathered, logsumexp
                        return topk_logps, None

                    topk_vals, response_topk_indices = torch.topk(logits, distill_topk, dim=-1)
                    topk_logps = topk_vals - logsumexp
                    del logits, topk_vals, logsumexp
                    return topk_logps, response_topk_indices

            return logits

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        kw_required_keys = [
            "kw_apply_mask",
            "kw_teacher_input_ids",
            "kw_teacher_attention_mask",
            "kw_teacher_position_ids",
        ]
        kw_optional_keys = ["kw_teacher_temperature", "kw_diversity_mask", "kw_diversity_ref_len"]
        use_sc_grpo = bool(data.meta_info.get("sc_grpo_enable", False))
        if use_sc_grpo:
            available_batch_keys = set(data.batch.keys())
            kw_missing_keys = [key for key in kw_required_keys if key not in available_batch_keys]
            if len(kw_missing_keys) == 0:
                select_keys.extend(kw_required_keys)
                for opt_key in kw_optional_keys:
                    if opt_key in available_batch_keys:
                        select_keys.append(opt_key)
            else:
                logger.warning(
                    "SC-GRPO disabled for this actor update because required keys are missing: %s",
                    kw_missing_keys,
                )
                use_sc_grpo = False

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    policy_response_mask = response_mask.clone().float()

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    if use_sc_grpo:
                        kw_topk = int(data.meta_info.get("sc_grpo_topk", 64))
                        kl_c_quantile = float(data.meta_info.get("sc_grpo_kl_c_quantile", 0.75))
                        kl_c_floor = float(data.meta_info.get("sc_grpo_kl_c_floor", 1e-4))
                        kl_weight_mode = str(
                            data.meta_info.get("sc_grpo_kl_weight_mode", "advantage")
                        ).lower()
                        if kl_weight_mode not in {"mask", "advantage"}:
                            raise ValueError(
                                "sc_grpo_kl_weight_mode must be 'mask' or 'advantage', "
                                f"got {kl_weight_mode!r}."
                            )
                        kw_apply_mask = model_inputs["kw_apply_mask"].clone()
                        # Truncate diversity rows' kw_apply_mask before KL computation
                        # so tail tokens beyond ref_len don't pollute median or f_kl
                        _div_mask = model_inputs.get("kw_diversity_mask", None)
                        _div_ref_len = model_inputs.get("kw_diversity_ref_len", None)
                        if _div_mask is not None and _div_ref_len is not None:
                            for _ri in range(_div_mask.shape[0]):
                                if _div_mask[_ri]:
                                    _rl = int(_div_ref_len[_ri].item())
                                    if 0 < _rl < kw_apply_mask.shape[1]:
                                        kw_apply_mask[_ri, _rl:] = 0
                        teacher_micro = {
                            "input_ids": model_inputs["kw_teacher_input_ids"],
                            "attention_mask": model_inputs["kw_teacher_attention_mask"],
                            "position_ids": model_inputs["kw_teacher_position_ids"],
                            "responses": model_inputs["responses"],
                        }
                        if "multi_modal_inputs" in model_inputs:
                            teacher_micro["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]

                        per_row_teacher_temp = model_inputs.get("kw_teacher_temperature", None)

                        with torch.no_grad():
                            student_topk_logps, student_topk_indices = self._forward_micro_batch_logits(
                                model_inputs,
                                temperature=temperature,
                                distill_topk=kw_topk,
                            )
                            teacher_topk_logps, _ = self._forward_micro_batch_logits(
                                teacher_micro,
                                temperature=temperature,
                                distill_topk=kw_topk,
                                topk_indices=student_topk_indices,
                            )
                            del student_topk_indices

                            def _add_tail(logps: torch.Tensor) -> torch.Tensor:
                                log_sum = torch.logsumexp(logps, dim=-1, keepdim=True).clamp(max=-1e-7)
                                tail = torch.log(-torch.expm1(log_sum))
                                return torch.cat([logps, tail], dim=-1)

                            student_tail = _add_tail(student_topk_logps)
                            teacher_tail = _add_tail(teacher_topk_logps)
                            del student_topk_logps, teacher_topk_logps

                            kl_per_token = F.kl_div(
                                student_tail,
                                teacher_tail,
                                reduction="none",
                                log_target=True,
                            ).sum(dim=-1)
                            del student_tail, teacher_tail

                            if per_row_teacher_temp is not None:
                                temp_scale = per_row_teacher_temp.to(kl_per_token.device).unsqueeze(-1)
                                kl_per_token = kl_per_token / temp_scale.clamp(min=1.0)
                                scaled_rows = (temp_scale.squeeze(-1) > 1.0).sum().item()
                                micro_batch_metrics["actor/teacher_temp_mean"] = temp_scale.mean().item()
                                micro_batch_metrics["actor/n_temp_scaled_rows"] = scaled_rows

                            active_kl = kl_per_token[kw_apply_mask.bool()]
                            if active_kl.numel() > 0:
                                median_kl = active_kl.median().item()
                                mean_kl = active_kl.mean().item()
                                kl_c_raw = torch.quantile(active_kl.float(), kl_c_quantile).item()
                            else:
                                median_kl = 1.0
                                mean_kl = 1.0
                                kl_c_raw = 1.0

                            # Use configurable quantile (default p75) as threshold c to avoid
                            # median collapse: when 90% of tokens have KL≈0, median→0 makes
                            # f(KL)≈1 everywhere. p75 keeps the bottom 75% suppressed and
                            # preserves top-25% signal. kl_c_floor prevents collapse when the
                            # entire distribution is near zero.
                            c = max(kl_c_raw, kl_c_floor)
                            f_kl = kl_per_token / (kl_per_token + c)
                            kl_weight = torch.where(
                                kw_apply_mask.bool(),
                                f_kl,
                                torch.ones_like(f_kl),
                            )

                            if kl_weight_mode == "mask":
                                policy_response_mask = policy_response_mask * kl_weight

                            if _div_mask is not None and _div_ref_len is not None:
                                for _ri in range(_div_mask.shape[0]):
                                    if _div_mask[_ri]:
                                        _rl = int(_div_ref_len[_ri].item())
                                        if 0 < _rl < policy_response_mask.shape[1]:
                                            policy_response_mask[_ri, _rl:] = 0

                            micro_batch_metrics["actor/kl_weight_mean"] = (
                                f_kl[kw_apply_mask.bool()].mean().item() if active_kl.numel() > 0 else 0.0
                            )
                            micro_batch_metrics["actor/kl_weight_c"] = c
                            micro_batch_metrics["actor/kl_per_token_mean"] = mean_kl
                            micro_batch_metrics["actor/kl_per_token_median"] = median_kl
                            micro_batch_metrics["actor/kl_weight_c_raw"] = kl_c_raw
                            micro_batch_metrics["actor/kw_active_ratio"] = kw_apply_mask.sum().item() / max(response_mask.sum().item(), 1)

                            if active_kl.numel() > 0:
                                percentiles = torch.quantile(
                                    active_kl.float(),
                                    torch.tensor([0.25, 0.75, 0.90, 0.95, 0.99], device=active_kl.device),
                                )
                                micro_batch_metrics["actor/kl_p25"] = percentiles[0].item()
                                micro_batch_metrics["actor/kl_p75"] = percentiles[1].item()
                                micro_batch_metrics["actor/kl_p90"] = percentiles[2].item()
                                micro_batch_metrics["actor/kl_p95"] = percentiles[3].item()
                                micro_batch_metrics["actor/kl_p99"] = percentiles[4].item()
                                micro_batch_metrics["actor/kl_nonzero_ratio"] = (active_kl > 1e-6).float().mean().item()

                            del kl_per_token

                        advantages = advantages.clone()
                        diversity_mask = model_inputs.get("kw_diversity_mask", None)
                        diversity_ref_len = model_inputs.get("kw_diversity_ref_len", None)
                        explore_coeff = float(data.meta_info.get("sc_grpo_explore_coeff", 0.0))
                        if diversity_mask is not None and explore_coeff > 0:
                            div_mask_bool = diversity_mask.bool()
                            n_div_rows = div_mask_bool.sum().item()
                            if n_div_rows >= 2:
                                row_scores = []
                                row_ref_lens = []
                                for row_i in range(div_mask_bool.shape[0]):
                                    if not div_mask_bool[row_i]:
                                        row_scores.append(0.0)
                                        row_ref_lens.append(0)
                                        continue
                                    resp_len = int(response_mask[row_i].sum().item())
                                    rl = int(diversity_ref_len[row_i].item()) if diversity_ref_len is not None else 0
                                    eff_len = min(resp_len, rl) if rl > 0 else resp_len
                                    row_ref_lens.append(rl)
                                    if eff_len > 0:
                                        if kl_weight_mode == "mask":
                                            score = policy_response_mask[row_i, :eff_len].sum().item() / eff_len
                                        else:
                                            active = kw_apply_mask[row_i, :eff_len].bool()
                                            if active.any():
                                                score = f_kl[row_i, :eff_len][active].mean().item()
                                            else:
                                                score = 0.0
                                    else:
                                        score = 0.0
                                    row_scores.append(score)

                                div_scores = torch.tensor(
                                    [row_scores[i] for i in range(len(row_scores)) if div_mask_bool[i]],
                                    device=advantages.device,
                                    dtype=advantages.dtype,
                                )
                                if div_scores.std() > 1e-8:
                                    div_normed = (div_scores - div_scores.mean()) / (div_scores.std() + 1e-8)
                                else:
                                    div_normed = torch.zeros_like(div_scores)

                                div_idx = 0
                                for row_i in range(advantages.shape[0]):
                                    if div_mask_bool[row_i]:
                                        rl = row_ref_lens[row_i]
                                        end = rl if rl > 0 else advantages.shape[1]
                                        advantages[row_i, :end] += div_normed[div_idx] * explore_coeff
                                        div_idx += 1

                                micro_batch_metrics["actor/diversity_score_mean"] = div_scores.mean().item()
                                micro_batch_metrics["actor/diversity_score_std"] = div_scores.std().item()
                                micro_batch_metrics["actor/diversity_pseudo_adv_mean"] = div_normed.mean().item()
                                micro_batch_metrics["actor/n_diversity_rows"] = n_div_rows

                        if kl_weight_mode == "advantage":
                            advantages = advantages * kl_weight
                        del f_kl, kl_weight

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if torch.any(policy_response_mask.bool()) and self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=policy_response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    elif torch.any(policy_response_mask.bool()):
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=policy_response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )
                    else:
                        zero = log_prob.new_zeros(())
                        pg_loss = zero
                        pg_clipfrac = zero
                        ppo_kl = zero
                        pg_clipfrac_lower = zero

                    if entropy_coeff != 0:
                        if torch.any(policy_response_mask.bool()):
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=policy_response_mask, loss_agg_mode=loss_agg_mode)
                        else:
                            entropy_loss = log_prob.new_zeros(())

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        if torch.any(policy_response_mask.bool()):
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=policy_response_mask, loss_agg_mode=loss_agg_mode)
                        else:
                            kl_loss = log_prob.new_zeros(())

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
