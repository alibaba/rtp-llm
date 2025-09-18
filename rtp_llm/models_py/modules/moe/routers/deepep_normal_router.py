import logging
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from libth_transformer.rtp_llm_ops import trt_fp8_quantize_128

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group
from rtp_llm.distribute.deep_ep import get_deepep_wrapper
from rtp_llm.models_py.modules.moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
    FusedMoEQuantConfig,
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from rtp_llm.models_py.modules.moe.utils import moe_kernel_quantize_input


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
        if quant_config.is_block_quantized:
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
            ) = self.deepep_buffer_wrapper.buffer.get_dispatch_layout(
                topk_ids, num_experts
            )
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
                expert_x, expert_x_scale = output
            else:
                expert_x = output
                expert_x_scale = None

            return ExpertForwardPayload(
                expert_x,
                None,
                expert_x_scale,
                ExpertTokensMetadata(None, num_recv_tokens_per_expert_list),
                recv_topk_idx,
                recv_topk_weights,
            )
        else:  # dispatch and maybe quant
            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                dispatch_expert_num_tokens,
                is_token_in_rank,
                _,
            ) = self.deepep_buffer_wrapper.buffer.get_dispatch_layout(
                topk_idx=topk_ids,
                num_experts=num_experts,
                previous_event=None,
                async_finish=False,
                allocate_on_comm_stream=False,
            )

            # dispatch
            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert,
                self.handle,
                _,
            ) = self.deepep_buffer_wrapper.buffer.dispatch(
                x=a1,
                handle=None,
                topk_idx=topk_ids,
                topk_weights=topk_weights,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=dispatch_expert_num_tokens,
                previous_event=None,
                async_finish=False,
                allocate_on_comm_stream=False,
                expert_alignment=1,
                config=None,
            )

            # maybe quant
            if quant_config.is_per_tensor and recv_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    recv_x,
                    None,
                    quant_dtype=torch.float8_e4m3fn,
                    per_act_token_quant=False,
                    block_shape=None,
                )
            else:
                expert_x, expert_x_scale = recv_x, None

            if recv_topk_idx.numel() != 0:
                expert_topk_ids = torch.where(
                    recv_topk_idx == -1,
                    num_experts - 1 if self.rank_expert_offset == 0 else 0,
                    recv_topk_idx + self.rank_expert_offset,
                )
            else:
                expert_topk_ids = recv_topk_idx

            expert_num_tokens_cpu = torch.tensor(
                num_recv_tokens_per_expert, device="cpu", dtype=torch.int32
            )

            return ExpertForwardPayload(
                expert_x_origin_dtype=a1.dtype,
                expert_x=expert_x,
                expert_x_scale=expert_x_scale,
                expert_topk_weights=recv_topk_weights,
                expert_topk_ids=expert_topk_ids,
                expert_tokens_meta=ExpertTokensMetadata(
                    num_recv_tokens_per_expert, expert_num_tokens_cpu
                ),
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
