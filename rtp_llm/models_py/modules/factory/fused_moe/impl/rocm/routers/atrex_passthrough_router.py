"""Passthrough router for ATREX MoE.

Simply passes hidden_states through without any routing/dispatch,
since ATREX's fused_moe handles everything internally.
"""

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType


class AtrexPassthroughRouter(FusedMoeDataRouter):
    """Passthrough router that delegates all work to the ATREX executor."""

    @classmethod
    def router_type(cls):
        return RouterType.BATCHED_DATA

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        pass

    def __init__(self, config: MoEConfigAdapter, quant_config: FusedMoEQuantConfig):
        super().__init__(config, quant_config)

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        return ExpertForwardPayload(
            expert_x=a1,
            expert_x_scale=a1_scale,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        return payload.fused_expert_output
