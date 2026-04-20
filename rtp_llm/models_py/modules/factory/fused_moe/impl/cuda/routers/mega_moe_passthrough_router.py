"""Passthrough router for mega_moe.

mega_moe handles EP dispatch + combine internally via NVLink symmetric memory.
This router is a no-op: it passes tensors through unchanged and does not perform
any cross-GPU communication.
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


class MegaMoePassthroughRouter(FusedMoeDataRouter):
    """No-op router for mega_moe.

    mega_moe fuses EP-dispatch + GEMM + SwiGLU + EP-combine into one kernel.
    The router simply wraps the incoming tensors into the expected payload types
    without any re-ordering or communication.
    """

    @classmethod
    def router_type(cls) -> RouterType:
        return RouterType.MEGA_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        pass

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
