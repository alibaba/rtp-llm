from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPWrapper,
    DeepepWrapperConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType


class DeepepNormalRouter(FusedMoeDataRouter):
    @classmethod
    def router_type(cls):
        return RouterType.DEEPEP_NORMAL

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepepNormalRouter can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.use_low_latency(config))
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)

        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank
        self.dp_size = config.dp_size
        self.dp_rank = config.dp_rank
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.expert_num = config.expert_num
        self.expert_num_per_rank = self.expert_num // self.ep_size
        self.top_k = config.moe_topk_group
        deepep_config = DeepepWrapperConfig.from_config_adapter(self.config)
        self.deepep_buffer_wrapper = DeepEPWrapper.get_instance(deepep_config)
        self.use_fp8 = True
        self.async_mode = False
        self.expert_alignment = 128
        self.handle: Any = None

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        if a1_scale is not None or a2_scale is not None:
            raise ValueError("DeepEPNormal a1_scale or a2_scale should be None")
        # if self.use_fp8:
        #    a1, a1_scale = trt_fp8_quantize_128(a1, False)
        #    input = (a1, a1_scale)
        # else:
        input = a1
        # pre dispatch
        # topk_ids = topk_ids.long()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event1,
        ) = self.deepep_buffer_wrapper.buffer.get_dispatch_layout(
            topk_ids, self.expert_num
        )
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
            expert_alignment=self.expert_alignment,
        )
        # if self.use_fp8:
        #    expert_x, expert_x_scale = output
        # else:
        expert_x = output
        expert_x_scale = None
        self.handle = handle
        return ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
            expert_x_origin_dtype=None,
            expert_topk_ids=recv_topk_idx,
            expert_topk_weights=recv_topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=None,
                expert_num_tokens_cpu=num_recv_tokens_per_expert_list,
            ),
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        assert self.handle is not None, "handler is None"
        recv_x, _, event = self.deepep_buffer_wrapper.buffer.combine(
            payload.fused_expert_output, self.handle
        )
        self.handle = None
        return recv_x
