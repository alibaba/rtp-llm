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
from rtp_llm.models_py.modules.moe.utils import moe_kernel_quantize_input


class DeepEPLowLatencyRouter(FusedMoeDataRouter):
    def __init__(
        self,
        config: GptInitModelParameters,
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
        if quant_config.is_quantized and quant_config.is_block_quantized:
            use_fp8_dispatch = True
        else:
            use_fp8_dispatch = False
        # dispatch
        assert a1.dim() == 2
        assert a1.is_contiguous()

        expert_x, expert_num_tokens, self.handle, _, _ = (
            self.deepep_buffer_wrapper.buffer.low_latency_dispatch(
                a1,
                topk_ids,
                self.max_num_tokens,
                num_experts,
                use_fp8=use_fp8_dispatch,
                async_finish=False,
                return_recv_hook=False,
            )
        )
        # Note: deepep low latency dispatch not always pad 0, this will cause an incorrect scale to be calculated here.
        if quant_config.is_per_tensor:
            E, M, H = expert_x.shape
            x = expert_x.view(-1, H)

            if torch.sum(expert_num_tokens) > 0:
                # TODO(serina.wzq): use high performance kernel impl
                index = torch.arange(
                    M, dtype=expert_num_tokens.dtype, device=expert_num_tokens.device
                ).repeat(E, 1)
                input_mask = (index < (expert_num_tokens.view(-1, 1))).view(-1)
                scale_inv = (
                    x[input_mask].abs().max() / torch.finfo(torch.float8_e4m3fn).max
                )
                scale = torch.tensor([scale_inv], dtype=torch.float32, device=x.device)
            else:
                scale = torch.tensor([1], dtype=torch.float32, device=x.device)
            q_x, expert_x_scale = moe_kernel_quantize_input(
                x, scale, torch.float8_e4m3fn, False, None
            )
            expert_x = q_x.view(E, -1, H)
        else:
            raise NotImplementedError

        return mm.ExpertForwardPayload(
            expert_x_origin_dtype=a1.dtype,
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
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
