import logging
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from libth_transformer.rtp_llm_ops import trt_fp8_quantize_128

import rtp_llm.models_py.modules.moe.fused_moe as mm
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group
from rtp_llm.distribute.deep_ep import get_deepep_wrapper
from rtp_llm.models_py.modules.moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)


class DeepEPLowLatencyRouter(FusedMoeDataRouter):
    def __init__(
        self,
        config: GptInitModelParameters,
        use_fp8: bool = True,
        async_mode: bool = False,
    ):
        super().__init__()
        self.handle = None
        self.num_dispatchers = config.world_size // config.tp_size
        self.rank = config.ep_rank
        self.num_global_experts = config.expert_num
        self.num_local_experts = self.num_global_experts // config.ep_size
        self.max_num_tokens = (
            config.max_generate_batch_size + config.tp_size - 1
        ) // config.tp_size
        self.use_fp8 = use_fp8
        self.async_mode = async_mode
        self.deepep_buffer_wrapper = get_deepep_wrapper()

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        quant_config: FusedMoEQuantConfig,
    ) -> mm.ExpertForwardPayload:
        assert topk_ids.shape[0] <= self.max_num_tokens
        assert a1.dim() == 2
        assert a1.is_contiguous()
        # TODO: impl fp8 dispatch
        assert self.use_fp8 is False

        expert_x, expert_num_tokens, self.handle, _, _ = (
            self.deepep_buffer_wrapper.buffer.low_latency_dispatch(
                a1,
                topk_ids,
                self.max_num_tokens,
                num_experts,
                use_fp8=self.use_fp8,
                async_finish=False,
                return_recv_hook=False,
            )
        )
        return mm.ExpertForwardPayload(
            expert_x_origin_dtype=a1.dtype,
            expert_x=expert_x,
            expert_x_scale=None,
            expert_topk_weights=None,
            expert_topk_ids=None,
            expert_tokens_meta=mm.ExpertTokensMetadata(expert_num_tokens, None),
        )

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mm.TopKWeightAndReduce,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        assert self.handle is not None

        out, _, _ = self.deepep_buffer_wrapper.buffer.low_latency_combine(
            fused_expert_output,
            topk_ids,
            topk_weights,
            self.handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=False,
        )

        return out
