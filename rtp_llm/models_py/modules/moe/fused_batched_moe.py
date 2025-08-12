from dataclasses import dataclass
from typing import Any, Optional

import torch

import rtp_llm.models_py.modules.moe.fused_moe as mm
from rtp_llm.models_py.modules import FusedMoEQuantConfig, resize_cache
from rtp_llm.models_py.modules.moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNaiveBatched,
)
from rtp_llm.models_py.modules.moe.utils import (
    moe_kernel_quantize_input,
    normalize_scales_shape,
)


class BatchedDataRouter(mm.FusedMoeDataRouter):
    def __init__(
        self,
        max_num_tokens: int,
        num_local_experts: int,
        num_dispatchers: int,
        rank: int,
    ):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts
        self.rank = rank
        self.num_dispatchers_ = num_dispatchers

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_ids: torch.Tensor,
        num_experts: int,
        quant_config: FusedMoEQuantConfig,
    ) -> mm.ExpertForwardPayload:
        assert a1.dim() == 2
        assert topk_ids.dim() == 2
        assert a1.size(0) == topk_ids.size(0)

        _, hidden_dim = a1.size()
        topk = topk_ids.size(1)

        tokens_per_expert = torch.zeros(num_experts, dtype=torch.int, device=a1.device)

        num_local_experts = self.num_local_experts

        if quant_config.quant_dtype is None:
            b_type = a1.dtype
        else:
            b_type = quant_config.quant_dtype

        assert isinstance(b_type, torch.dtype)

        b_a1 = torch.zeros(
            (num_local_experts, self.max_num_tokens, hidden_dim),
            dtype=b_type,
            device=a1.device,
        )

        if quant_config.is_quantized:
            raise NotImplementedError("quantization not supported yet")
        else:
            assert a1_scale is None
            b_a1_scale = None

        first_expert = num_local_experts * self.rank
        last_expert = first_expert + num_local_experts

        a1_scale = normalize_scales_shape(a1_scale)
        a2_scale = normalize_scales_shape(a2_scale)

        for expert_id in range(first_expert, last_expert):
            topks = torch.any(topk_ids == expert_id, dim=1).flatten()
            rows = torch.count_nonzero(topks.flatten())
            if rows == 0:
                continue
            idx = expert_id - first_expert
            tokens_per_expert[idx] = rows
            rhs = a1[: topks.numel()][topks]
            if quant_config.is_quantized:
                raise NotImplementedError("quantization not supported yet")
            else:
                b_a1[idx, :rows, :] = rhs

        assert b_a1_scale is None or b_a1_scale.ndim == 3

        expert_tokens_meta = mm.ExpertTokensMetadata(
            expert_num_tokens=tokens_per_expert, expert_num_tokens_cpu=None
        )

        return mm.ExpertForwardPayload(
            expert_x=b_a1,
            expert_x_scale=b_a1_scale,
            expert_tokens_meta=expert_tokens_meta,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mm.TopKWeightAndReduce,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceNaiveBatched(self.rank)
        weight_and_reduce_impl.apply(
            output=output,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


class NaiveBatchedExperts(mm.FusedMoeExpertExecutor):
    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        block_shape: Optional[list[int]] = None,
        per_act_token_quant: bool = False,
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

    def finalize_weight_and_reduce_impl(self) -> mm.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Optional[mm.ExpertTokensMetadata],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        assert a.dim() == 2
        num_dp = self.num_dispatchers
        num_experts = local_num_experts
        workspace13 = (num_experts, self.max_num_tokens * num_dp, K)
        workspace2 = (self.max_num_tokens * num_dp, N)
        output = workspace13
        return (workspace13, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        payload: mm.ExpertForwardPayload,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ):
        assert payload.expert_x.dim() == 3
        assert payload.expert_tokens_meta is not None
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens

        num_local_experts = w1.size(0)

        N = w1.size(1) // 2

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
                # input = self.dequant(hidden_states[expert, :, :],
                #                      a1q_scale[expert])
                # w1_dq = self.dequant(w1[expert], w1_scale[expert])
                # input = input[:num] @ w1_dq.transpose(0, 1)
                raise NotImplementedError("quantization not supported yet")
            else:
                input = payload.expert_x[expert, :num, :] @ w1[expert].transpose(0, 1)

            self.activation(activation, tmp, input.to(tmp.dtype))

            if self.quant_config.is_quantized:
                # assert w2_scale is not None
                # w2_dq = self.dequant(w2[expert], w2_scale[expert])
                raise NotImplementedError("quantization not supported yet")
            else:
                w2_dq = w2[expert]

            output[expert, :num, :] = tmp @ w2_dq.transpose(0, 1).to(tmp.dtype)
