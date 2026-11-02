"""Pure-TP CANN MoE router.

The first Ascend MoE implementation deliberately has no EP/DP dispatch path.
Every TP rank owns all experts, runs the local CANN expert pipeline, then sums
the partitioned down projections with the existing TP collective.
"""

from typing import Any, Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
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
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class NpuPureTpRouter(FusedMoeDataRouter):
    """Route BF16 tokens locally for the non-quantized CANN executor."""

    @classmethod
    def router_type(cls) -> RouterType:
        return RouterType.PURE_TP

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.is_pure_tp_mode(config))
        checker.check(resolver.use_all_gather(config))

    def __init__(
        self, config: MoEConfigAdapter, quant_config: FusedMoEQuantConfig
    ) -> None:
        super().__init__(config, quant_config)
        self.tp_size = config.tp_size

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        if a1_scale is not None or a2_scale is not None:
            raise NotImplementedError("Ascend CANN MoE does not support quantization")
        if a1.dim() != 2 or not a1.is_contiguous():
            raise ValueError("Ascend CANN MoE input must be a contiguous [tokens, hidden] tensor")
        if topk_weights.shape != topk_ids.shape:
            raise ValueError("topk_weights and topk_ids must have identical shapes")
        if topk_ids.size(0) != a1.size(0):
            raise ValueError("top-k token dimension does not match hidden states")
        if topk_ids.dtype != torch.int32:
            raise TypeError("Ascend CANN MoE requires int32 topk_ids")
        if topk_weights.dtype != torch.float32:
            raise TypeError("Ascend CANN MoE requires float32 topk_weights")

        return ExpertForwardPayload(
            expert_x=a1,
            expert_x_origin_dtype=a1.dtype,
            expert_x_scale=None,
            expert_tokens_meta=ExpertTokensMetadata(),
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "Ascend CANN MoE applies router weights in token unpermute"
            )
        output = payload.fused_expert_output
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output
