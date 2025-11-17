from typing import Any, Optional

import torch

import rtp_llm.models_py.modules.common.moe.fused_moe as mm
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.cuda.moe.executors.util import moe_kernel_quantize_input
from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.type import RouterType


class DataRouterNoEPStandard(mm.FusedMoeDataRouter):
    @classmethod
    def router_type(cls):
        """Return the router type for this class"""
        return RouterType.PURE_TP

    @classmethod
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        """Check if DataRouterNoEPStandard can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.config_resolver import (
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
    ) -> mm.ExpertForwardPayload:

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            a1_scale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
        )

        return mm.ExpertForwardPayload(
            expert_x_origin_dtype=a1.dtype,
            expert_x=a1q,
            expert_x_scale=a1q_scale,
            expert_tokens_meta=mm.ExpertTokensMetadata(None, None),
        )

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        return fused_expert_output
