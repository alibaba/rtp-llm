"""Base contract for FP8-activation/FP4-weight fused-MoE executors."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertGatePayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType


class Fp8Fp4ExecutorBase(FusedMoeExpertExecutor, torch.nn.Module):
    """Adapt an FP8/FP4 backend to the common fused-MoE executor API."""

    includes_shared_expert = False

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        torch.nn.Module.__init__(self)
        super().__init__(config, quant_config, weights)
        self.cfg = config
        self.setup_weights(weights)

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.FP8_FP4

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(getattr(config, "moe_quant_method", None) == "FP8_FP4")

    def setup_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_gate_pack(
        self,
        x: torch.Tensor,
        gate_payload: ExpertGatePayload,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return output plus the packed top-k weights and ids."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support fused gate packing"
        )

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        if activation.lower() not in ("silu", "siglu", "swiglu"):
            raise ValueError(
                f"FP8/FP4 MoE requires SiLU activation, got {activation!r}"
            )
        if expert_map is not None:
            raise ValueError("FP8/FP4 MoE does not support expert_map")
        if a2_scale is not None:
            raise ValueError("FP8/FP4 MoE does not accept an external a2_scale")
        if apply_router_weight_on_input:
            raise ValueError("FP8/FP4 MoE applies router weights during output combine")
        if payload.gate_payload is not None:
            output, topk_weights, topk_ids = self.forward_gate_pack(
                payload.expert_x,
                payload.gate_payload,
            )
            payload.expert_topk_weights = topk_weights
            payload.expert_topk_ids = topk_ids
            return CombineForwardPayload(fused_expert_output=output)
        topk_weights = payload.expert_topk_weights
        topk_ids = payload.expert_topk_ids
        if topk_weights is None or topk_ids is None:
            raise ValueError("FP8/FP4 MoE requires routed top-k weights and ids")
        return CombineForwardPayload(
            fused_expert_output=self.forward(payload.expert_x, topk_weights, topk_ids)
        )
