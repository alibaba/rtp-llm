from typing import Any, Dict, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, all_gather
from rtp_llm.models_py.distributed.deepep_initializer import DeepEpInitializer
from rtp_llm.models_py.modules.common.moe.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.type import RouterType
from rtp_llm.models_py.modules.fp8_kernel import scaled_fp8_per_token_quant
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128


class DeepepNormalRouter(FusedMoeDataRouter):
    @classmethod
    def router_type(cls):
        return RouterType.DEEPEP_NORMAL

    @classmethod
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        """Check if DeepepNormalRouter can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.use_low_latency(config))
        checker.check(DeepEpInitializer.supported())

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
        self.deepep_buffer_wrapper = DeepEpInitializer.get_deepep_wrapper(self.config)
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
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:
        if a1_scale is not None or a2_scale is not None:
            raise ValueError("DeepEPNormal a1_scale or a2_scale should be None")
        act_dtype = a1.dtype

        # scatter
        tp_size = self.config.tp_size
        tp_rank = self.config.tp_rank
        token_num = a1.size(0)
        tp_token_size = (token_num + tp_size - 1) // tp_size

        slice_begin = min(tp_token_size * tp_rank, token_num)
        slice_size = min(token_num - slice_begin, tp_token_size)

        if self.use_fp8:
            if quant_config.is_per_act_token:
                a1, a1_scale = scaled_fp8_per_token_quant(a1, None)
                assert a1.shape[1] % 128 == 0
                a1_scale = a1_scale.repeat(1, a1.shape[1] // 128)
            else:
                a1, a1_scale = trt_fp8_quantize_128(a1, False)

            tp_expert_a1 = torch.narrow(a1, 0, slice_begin, slice_size)
            tp_expert_a1_scale = torch.narrow(a1_scale, 0, slice_begin, slice_size)
            tp_expert_input = (tp_expert_a1, tp_expert_a1_scale)
        else:
            tp_expert_a1 = torch.narrow(a1, 0, slice_begin, slice_size)
            tp_expert_input = tp_expert_a1
        # pre dispatch
        # topk_ids = topk_ids.long()

        tp_expert_ids = torch.narrow(topk_ids, 0, slice_begin, slice_size)
        tp_expert_scales = torch.narrow(topk_weights, 0, slice_begin, slice_size)

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event1,
        ) = self.deepep_buffer_wrapper.buffer.get_dispatch_layout(
            tp_expert_ids, self.expert_num
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
            tp_expert_input,
            None,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            tp_expert_ids,
            tp_expert_scales,
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
                self.expert_num - 1 if self.rank_expert_offset == 0 else 0,
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
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        assert self.handle is not None, "handler is None"
        out_token, _, event = self.deepep_buffer_wrapper.buffer.combine(
            fused_expert_output, self.handle
        )
        self.handle = None

        # gather
        tp_size = self.config.tp_size
        original_num_tokens = extra_finalize_args["original_num_tokens"]
        tp_token_size = (original_num_tokens + tp_size - 1) // tp_size

        if tp_size > 1:
            # combine_x.size(0) might be 0
            if out_token.size(0) < tp_token_size:
                padding_out_token = torch.empty(
                    size=(tp_token_size - out_token.size(0), out_token.size(1)),
                    device=out_token.device,
                    dtype=out_token.dtype,
                )
                out_token = torch.cat([out_token, padding_out_token], dim=0)

            gatherd_output = all_gather(out_token, group=Group.TP).reshape(
                tp_size * tp_token_size, -1
            )
            gatherd_output = gatherd_output[:original_num_tokens, :]
            return gatherd_output

        # out_token should be a tensor with shape and dtype like a1
        return out_token
