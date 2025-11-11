from typing import Any, Optional

import torch
import triton.language as tl

import rtp_llm.models_py.modules.moe.fused_moe as mm
from rtp_llm.models_py.modules.moe import TopKWeightAndReduceDelegate
from rtp_llm.models_py.modules.moe.utils import FusedMoEQuantConfig, resize_cache
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.moe.grouped_gemm import (
    invoke_moe_batched_triton_kernel,
)


class NaiveBatchedExperts(mm.FusedMoeExpertExecutor):
    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        w1: torch.Tensor,
        w2: torch.Tensor,
        block_shape: Optional[list[int]] = None,
        per_act_token_quant: bool = False,
        **kwargs,
    ):
        super().__init__(
            quant_config=FusedMoEQuantConfig(
                quant_dtype=None,
                per_act_token_quant=per_act_token_quant,
                block_shape=block_shape,
            )
        )
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers
        self.w1 = w1
        self.w2 = w2

    @property
    def local_num_experts(self) -> int:
        return self.w1.size(0)

    def finalize_weight_and_reduce_impl(self) -> mm.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    def execute(
        self,
        payload: mm.ExpertForwardPayload,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        assert payload.expert_x.dim() == 3
        assert payload.expert_tokens_meta is not None
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens

        num_local_experts = self.w1.size(0)  # Use class member

        N = self.w1.size(1) // 2  # Use class member

        num_dp = self.num_dispatchers
        output_shape = (
            num_local_experts,
            self.max_num_tokens * num_dp,
            self.w2.size(1),
        )

        output = torch.empty(
            output_shape, device=payload.expert_x.device, dtype=self.w2.dtype
        )

        workspace2 = torch.empty(
            (self.max_num_tokens * num_dp, N),
            device=payload.expert_x.device,
            dtype=self.w2.dtype,
        )

        for expert in range(num_local_experts):
            # Indexing expert_num_tokens doesn't work w/cudagraphs or inductor
            if (
                torch.compiler.is_compiling()
                or torch.cuda.is_current_stream_capturing()
            ):
                num = payload.expert_x.shape[1]
            else:
                num = int(expert_num_tokens[expert].item())

            if num == 0:
                continue

            tmp = resize_cache(workspace2, (num, N))

            if self.quant_config.is_quantized:
                # assert a1q_scale is not None and w1_scale is not None
                # input = self.dequant(payload.expert_x[expert, :, :],
                #                      a1q_scale[expert])
                # w1_dq = self.dequant(self.w1[expert], w1_scale[expert])  # Use class member
                # input = input[:num] @ w1_dq.transpose(0, 1)
                raise NotImplementedError("quantization not supported yet")
            else:
                input = payload.expert_x[expert, :num, :] @ self.w1[expert].transpose(
                    0, 1
                )  # Use class member

            silu_and_mul(tmp, input.to(tmp.dtype))
            # value, gate = torch.split(input, N, dim=-1)
            # import torch.nn.functional as F

            # tmp = F.silu(gate) * value

            if self.quant_config.is_quantized:
                # assert w2_scale is not None
                # w2_dq = self.dequant(self.w2[expert], w2_scale[expert])  # Use class member
                raise NotImplementedError("quantization not supported yet")
            else:
                w2_dq = self.w2[expert]  # Use class member

            output[expert, :num, :] = tmp @ w2_dq.transpose(0, 1).to(tmp.dtype)
        return output


class BatchedTritonExperts(mm.FusedMoeExpertExecutor):
    """
    A Triton based MoE expert class that operates on expert batched format,
    i.e. E x max_num_tokens x K.  This is the format that the pplx
    dispatch/combine kernels use.
    """

    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        w1: torch.Tensor,
        w2: torch.Tensor,
        block_shape: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__(
            quant_config=FusedMoEQuantConfig(
                quant_dtype=None,
                per_act_token_quant=False,
                block_shape=block_shape,
            )
        )
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers
        self.w1 = w1
        self.w2 = w2

    @property
    def local_num_experts(self) -> int:
        return self.w1.size(0)

    def finalize_weight_and_reduce_impl(self) -> mm.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    def execute(
        self,
        payload: mm.ExpertForwardPayload,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        # Check constraints.
        assert payload.expert_x.size(-1) == self.w1.size(
            2
        ), f"Hidden size mismatch {payload.expert_x.size(-1)} != {self.w1.size(2)}"

        assert payload.expert_x.is_contiguous(), "Hidden_states must be contiguous"
        assert self.w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert self.w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert payload.expert_x.dtype in [torch.float16, torch.bfloat16]
        assert payload.expert_tokens_meta is not None

        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens

        E = self.local_num_experts
        N = self.w1.size(1)
        assert payload.expert_topk_ids is not None
        top_k_num = payload.expert_topk_ids.size(1)

        assert self.w1.size(0) == E
        assert self.w2.size(0) == E

        if payload.expert_x.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif payload.expert_x.dtype == torch.float16:
            compute_type = tl.float16
        else:
            raise ValueError(f"Unsupported compute_type: {payload.expert_x.dtype}")

        intermediate_cache1 = torch.empty(
            (E, self.max_num_tokens, N),
            device=payload.expert_x.device,
            dtype=payload.expert_x.dtype,
        )
        intermediate_cache2 = torch.empty(
            (E, self.max_num_tokens, N // 2),
            device=payload.expert_x.device,
            dtype=payload.expert_x.dtype,
        )
        output_shape = (
            self.local_num_experts,
            self.max_num_tokens,
            self.w2.size(1),
        )
        output = torch.empty(
            output_shape, device=payload.expert_x.device, dtype=self.w2.dtype
        )

        # MM1
        invoke_moe_batched_triton_kernel(
            A=payload.expert_x,
            B=self.w1,
            C=intermediate_cache1,
            expert_num_tokens=expert_num_tokens,
            compute_type=compute_type,
        )

        intermediate_cache2.fill_(0)

        silu_and_mul(
            intermediate_cache2.view(-1, N // 2),
            intermediate_cache1.view(-1, N),
        )

        invoke_moe_batched_triton_kernel(
            A=intermediate_cache2,
            B=self.w2,
            C=output,
            expert_num_tokens=expert_num_tokens,
            compute_type=compute_type,
        )

        return output
