from typing import Any, Dict, Optional

import torch
from libth_transformer.rtp_llm_ops import trt_fp8_quantize_128

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.distributed.deepep_wrapper import get_deepep_wrapper
from rtp_llm.models_py.modules.fp8_kernel import scaled_fp8_per_token_quant
from rtp_llm.models_py.modules.moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
    FusedMoEQuantConfig,
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)


class DeepepNormalRouter(FusedMoeDataRouter):
    def __init__(
        self,
        config: GptInitModelParameters,
        use_fp8: bool = True,
        async_mode: bool = False,
        expert_alignment: int = 128,
    ):
        self.config = config
        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank
        self.dp_size = config.dp_size
        self.dp_rank = config.dp_rank
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.expert_num = config.expert_num
        self.expert_num_per_rank = self.expert_num // self.ep_size
        self.num_dispatchers = config.world_size // config.tp_size
        self.rank_expert_offset = self.ep_rank * self.expert_num_per_rank
        self.top_k = config.moe_topk_group
        self.deepep_buffer_wrapper = get_deepep_wrapper()
        self.use_fp8 = use_fp8
        self.async_mode = async_mode
        self.expert_alignment = expert_alignment
        if self.async_mode:
            raise ValueError("DeepEPNormal not supports async mode now")
        self.handle: Any = None

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:
        if a1_scale is not None or a2_scale is not None:
            raise ValueError("DeepEPNormal a1_scale or a2_scale should be None")
        act_dtype = a1.dtype
        if self.use_fp8:
            if quant_config.is_per_act_token:
                a1, a1_scale = scaled_fp8_per_token_quant(a1, None)
                assert a1.shape[1] % 128 == 0
                a1_scale = a1_scale.repeat(1, a1.shape[1] // 128)
            else:
                a1, a1_scale = trt_fp8_quantize_128(a1, False)

            input = (a1, a1_scale)
        else:
            input = a1
        # pre dispatch
        # topk_ids = topk_ids.long()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event1,
        ) = self.deepep_buffer_wrapper.buffer.get_dispatch_layout(topk_ids, num_experts)
        # dispatch
        (
            output,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            self.handle,
            event2,
        ) = self.deepep_buffer_wrapper.buffer.dispatch(
            input,
            None,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            topk_ids,
            topk_weights,
            expert_alignment=self.expert_alignment,
        )
        if self.use_fp8:
            if quant_config.is_per_act_token:
                expert_x, expert_x_scale = output
                expert_x_scale = expert_x_scale[:, 0].unsqueeze(1)
            else:
                expert_x, expert_x_scale = output
        else:
            expert_x = output
            expert_x_scale = None

        expert_num_tokens = torch.tensor(
            num_recv_tokens_per_expert_list, device=expert_x.device, dtype=torch.int32
        )

        if recv_topk_idx.numel() != 0 and (
            not self.use_fp8 or quant_config.is_per_act_token
        ):
            expert_topk_ids = torch.where(
                recv_topk_idx == -1,
                num_experts - 1 if self.rank_expert_offset == 0 else 0,
                recv_topk_idx + self.rank_expert_offset,
            )
        else:
            expert_topk_ids = recv_topk_idx

        return ExpertForwardPayload(
            expert_x,
            act_dtype,
            expert_x_scale,
            ExpertTokensMetadata(expert_num_tokens, num_recv_tokens_per_expert_list),
            expert_topk_ids,
            recv_topk_weights,
        )

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: Any,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            if weight_and_reduce_impl is not None:
                fused_expert_output = weight_and_reduce_impl.apply(
                    fused_expert_output=fused_expert_output,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                )

        assert self.handle is not None, "handler is None"
        out_token, _, event = self.deepep_buffer_wrapper.buffer.combine(
            fused_expert_output, self.handle
        )
        self.handle = None
        # out_token should be a tensor with shape and dtype like a1
        return out_token
