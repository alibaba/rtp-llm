from typing import Any, Dict, Optional
import torch

from rtp_llm.models_py.distributed.moriep_wrapper import MoriEPWrapper

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

class MoriEpIntranodeRouter(FusedMoeDataRouter):
    @classmethod
    def router_type(cls):
        return RouterType.MORI_EP_INTRANODE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if MoriEpIntranodeRouter can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.use_low_latency(config))
        checker.check(MoriEPWrapper.supported())

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
        self.mori_buffer_wrapper = MoriEPWrapper.get_instance()
        self.use_fp8 = True
        self.async_mode = False
        self.expert_alignment = 128
    
    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        # TODO: implement fp8 quantization
        if a1_scale is not None or a2_scale is not None:
            raise ValueError("MoriEpIntranode a1_scale or a2_scale should be None")

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_buffer_wrapper.op.dispatch(a1, topk_weights, None, topk_ids)
        return ExpertForwardPayload(
            expert_x=dispatch_a1,
            expert_x_scale=dispatch_scale,
            expert_x_origin_dtype=None,
            expert_topk_ids=dispatch_ids,
            expert_topk_weights=dispatch_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=None,
                expert_num_tokens_cpu=dispatch_recv_token_num,
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
        recv_x = self.mori_buffer_wrapper.op.combine(payload.fused_expert_output, None, topk_ids)[0]
        return recv_x