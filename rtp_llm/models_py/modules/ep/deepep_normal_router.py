import logging
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from libth_transformer.rtp_llm_ops import trt_fp8_quantize_128

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group
from rtp_llm.distribute.deep_ep_wrapper import DeepBufferWrapper
from rtp_llm.models_py.modules.ep.fuesd_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
    FusedMoEQuantConfig,
)


class DeepepNormalRouter(FusedMoeDataRouter):
    def __init__(
        self,
        config: GptInitModelParameters,
        use_fp8: bool = True,
        async_mode: bool = False,
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
        self.top_k = config.moe_topk_group
        self.deepep_buffer_wrapper = DeepBufferWrapper(config)
        self.use_fp8 = use_fp8
        self.async_mode = async_mode
        if self.async_mode:
            raise ValueError("DeepEPNormal not supports async mode now")

    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group()

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
        if self.use_fp8:
            a1, a1_scale = trt_fp8_quantize_128(a1, False)
            input = (a1, a1_scale)
        else:
            input = a1
        # pre dispatch
        topk_ids = topk_ids.long()
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
            handle,
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
            expert_alignment=128,
        )
        if self.use_fp8:
            expert_x, expert_x_scale = output
        else:
            expert_x = output
            expert_x_scale = None
        return ExpertForwardPayload(
            expert_x,
            expert_x_scale,
            ExpertTokensMetadata(num_recv_tokens_per_expert_list, None),
            recv_topk_idx,
            recv_topk_weights,
            extra_finalize_args={"handle": handle},
        )

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: Any,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> None:
        assert extra_finalize_args is not None, "extra_finalize_args is None"
        handle = extra_finalize_args["handle"]
        recv_x, _, event = self.deepep_buffer_wrapper.buffer.combine(
            fused_expert_output, handle
        )
        return recv_x
