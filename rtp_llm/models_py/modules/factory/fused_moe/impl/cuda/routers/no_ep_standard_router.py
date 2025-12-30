from typing import Any, Optional

import torch

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
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.util import (
    moe_kernel_quantize_input,
)


class DataRouterNoEPStandard(FusedMoeDataRouter):
    @classmethod
    def router_type(cls):
        """Return the router type for this class"""
        return RouterType.PURE_TP

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DataRouterNoEPStandard can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_single_gpu(config))
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method in ["FP8_PER_TENSOR_COMPRESSED", "FP8_DYNAMIC_PER_TENSOR"]
        )

    def __init__(
        self,
        num_dispatchers: int,
    ):
        super().__init__()
        self.num_dispatchers = num_dispatchers
        assert self.num_dispatchers == 1

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            a1_scale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
        )

        return ExpertForwardPayload(
            expert_x=a1q,
            expert_x_scale=a1q_scale,
            expert_x_origin_dtype=a1.dtype,
            expert_topk_ids=None,
            expert_topk_weights=None,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=None, expert_num_tokens_cpu=None
            ),
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        return payload.fused_expert_output
