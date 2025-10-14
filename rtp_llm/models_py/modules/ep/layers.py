# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py
# but make some modifications for RTP-LLM
# Licensed under the Apache License, Version 2.0

import logging
import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import Module

import rtp_llm.models_py.modules.utils as utils

if utils.is_cuda():
    from librtp_compute_ops.rtp_llm_ops import trt_fp8_quantize_128
else:
    rtp_llm_ops = None
    trt_fp8_quantize_128 = None

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules.utils import ceil_div, dispose_tensor
from rtp_llm.utils.model_weight import W

if utils.is_cuda():
    from librtp_compute_ops.rtp_llm_ops import FusedMoEOp, SelectTopkOp

    from rtp_llm.models_py.modules.ep.kernels import (
        ep_gather,
        ep_scatter,
        gelu_and_mul_triton_kernel,
        grouped_gemm_triton,
        post_reorder_triton_kernel,
        pre_reorder_triton_kernel,
        run_moe_ep_preproess,
        silu_and_mul_masked_post_quant_fwd,
        silu_and_mul_triton_kernel,
        tma_align_input_scale,
    )

else:
    logging.info("can't import from rtp_llm_ops and ep.kernels, only support cuda!")
    FusedMoEOp = None
    SelectTopkOp = None

from rtp_llm.models_py.modules.ep.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
)
from rtp_llm.models_py.modules.ep.topk import select_experts

logger = logging.getLogger(__name__)


class GroupedGemmRunner(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device

    # c = a * b
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
        use_fp8_w8a8: bool = False,
        scale_a: torch.Tensor = None,
        scale_b: torch.Tensor = None,
        block_shape: Optional[List[int]] = None,
        c_dtype=None,
    ):
        assert weight_column_major == True
        c = grouped_gemm_triton(
            a,
            b,
            c,
            batch_size,
            weight_column_major,
            seg_indptr,
            weight_indices,
            use_fp8_w8a8,
            scale_a,
            scale_b,
            block_shape=block_shape,
            c_dtype=c_dtype,
        )
        return c


class FusedMoE(torch.nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_id: int,
    ):
        super().__init__()

        self.config = config
        self.layer_id = layer_id
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_inter_padding_size
        self.num_experts = config.expert_num
        self.top_k = config.moe_k
        self.up_proj = weights.get(W.moe_w1, None)
        self.down_proj = weights.get(W.moe_w2, None)
        self.select_topk_op = SelectTopkOp(config)
        self.fused_moe_op = FusedMoEOp(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length, hidden_dim = hidden_states.shape
        router_logits_fp32 = router_logits.float()
        routing_weights = torch.zeros(
            (sequence_length, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        selected_experts = torch.zeros(
            (sequence_length, self.top_k),
            dtype=torch.int32,
            device=hidden_states.device,
        )

        self.select_topk_op.forward(
            router_logits_fp32, selected_experts, routing_weights
        )
        final_hidden_states = torch.zeros(
            (sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self.fused_moe_op.forward(
            hidden_states,
            self.up_proj,
            self.down_proj,
            routing_weights,
            selected_experts,
            final_hidden_states,
        )
        return final_hidden_states


class LegacyEPMoE(torch.nn.Module):
    """
    MoE Expert Parallel Impl
    """

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_id: int,
    ):
        super().__init__()

        self.config = config
        self.layer_id = layer_id

        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank

        self.num_experts = config.expert_num
        assert self.num_experts % self.tp_size == 0
        self.num_experts_per_partition = self.num_experts // self.tp_size
        self.start_expert_id = self.tp_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = config.moe_k
        self.intermediate_size = config.moe_inter_padding_size
        self.activation = config.activation_type.lower()
        self.renormalize = True

        # Check if FP8 quantization should be used
        if self.config.quant_config:
            self.use_fp8_w8a8 = True
            self.use_block_quant = self.config.quant_config.group_size() > 0
            group_size = self.config.quant_config.group_size()
            self.block_shape = [group_size, group_size] if group_size > 0 else None
            self.fp8_dtype = torch.float8_e4m3fn
            self.activation_scheme = "dynamic"  # Default to dynamic activation scheme
        else:
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.block_shape = None
            self.fp8_dtype = None
            self.activation_scheme = None

        # Initialize MoE expert weights
        self.w13_weight = weights.get(W.moe_w1, None)
        self.w2_weight = weights.get(W.moe_w2, None)

        if self.use_fp8_w8a8:
            # FP8 quantization: weights and scales are loaded directly from checkpoint
            if self.use_block_quant:
                # Block quantization: using scale_inv
                self.w13_weight_scale_inv = weights.get(W.moe_s1, None)
                self.w2_weight_scale_inv = weights.get(W.moe_s2, None)
                self.w13_weight_scale = None
                self.w2_weight_scale = None
            else:
                # Per-tensor quantization: using scale
                self.w13_weight_scale = weights.get(W.moe_s1, None)
                self.w2_weight_scale = weights.get(W.moe_s2, None)
                self.w13_weight_scale_inv = None
                self.w2_weight_scale_inv = None

            # Input scale factors initialized to None, will be dynamically computed in forward
            self.w13_input_scale = None
            self.w2_input_scale = None

        else:
            # Non-FP8 quantization case: original weight initialization logic
            self.w13_weight_scale = None
            self.w2_weight_scale = None
            self.w13_weight_scale_inv = None
            self.w2_weight_scale_inv = None

            device = (
                self.w2_weight.device
                if self.w2_weight is not None
                else torch.device("cpu")
            )
            ones_tensor = torch.ones(
                self.num_experts_per_partition, dtype=torch.float32, device=device
            )
            self.w13_input_scale = torch.nn.Parameter(ones_tensor, requires_grad=False)
            self.w2_input_scale = torch.nn.Parameter(ones_tensor, requires_grad=False)
            self.w2_weight_scale = torch.nn.Parameter(ones_tensor, requires_grad=False)

        self.grouped_gemm_runner = None

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        hidden_states_shape = hidden_states.shape
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=False,
            renormalize=self.renormalize,
            expert_location_dispatch_info=None,
        )
        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, self.num_experts
        )

        # If using FP8 quantization and dynamic activation scheme, compute input scale factors
        if (
            self.use_fp8_w8a8
            and self.activation_scheme == "dynamic"
            and not self.use_block_quant
        ):
            max_value = (
                torch.max(hidden_states)
                .repeat(self.num_experts_per_partition)
                .to(torch.float32)
            )
            self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max

        # Prepare gateup_input, choose dtype based on whether FP8 quantization is used
        gateup_input_dtype = (
            self.fp8_dtype
            if (self.use_fp8_w8a8 and not self.use_block_quant)
            else hidden_states_dtype
        )
        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * self.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=gateup_input_dtype,
        )

        # PreReorder
        pre_reorder_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            self.w13_input_scale,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )

        seg_indptr_cur_rank = seg_indptr[self.start_expert_id : self.end_expert_id + 2]
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states_device,
            dtype=torch.int64,
        )

        # Initialize GroupedGemmRunner
        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(hidden_states_device)

        # GroupGemm-0: using Triton implementation
        weight_scales = (
            self.w13_weight_scale_inv if self.use_block_quant else self.w13_weight_scale
        )
        gateup_output = self.grouped_gemm_runner(
            a=gateup_input,
            b=self.w13_weight,
            c=None,
            c_dtype=hidden_states_dtype,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w13_input_scale,
            scale_b=weight_scales,
            block_shape=self.block_shape,
        )
        del gateup_input

        # Act
        down_input_dtype = (
            self.fp8_dtype
            if (self.use_fp8_w8a8 and not self.use_block_quant)
            else hidden_states_dtype
        )
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=down_input_dtype,
        )

        # Prepare input scaling factor for second GroupGemm
        if self.w2_input_scale is None and not self.use_block_quant:
            self.w2_input_scale = torch.ones(
                self.num_experts_per_partition,
                dtype=torch.float32,
                device=hidden_states_device,
            )

        if self.activation == "silu" or self.activation == "siglu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        elif self.activation == "gelu":
            gelu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")
        del gateup_output

        # GroupGemm-1: using Triton implementation
        weight_scales = (
            self.w2_weight_scale_inv if self.use_block_quant else self.w2_weight_scale
        )
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states_device,
            dtype=hidden_states_dtype,
        )
        down_output = self.grouped_gemm_runner(
            a=down_input,
            b=self.w2_weight,
            c=down_output,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w2_input_scale,
            scale_b=weight_scales,
            block_shape=self.block_shape,
        )
        del down_input

        # PostReorder
        output = torch.empty(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
        post_reorder_triton_kernel[(hidden_states_shape[0],)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states_shape[1],
            BLOCK_SIZE=512,
        )
        return output
