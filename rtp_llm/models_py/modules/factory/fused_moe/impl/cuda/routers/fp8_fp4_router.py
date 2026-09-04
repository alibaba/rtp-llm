"""Routers for fused FP8-activation/FP4-weight MoE executors."""

from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertGatePayload,
    FinalizeArgs,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType


class Fp8Fp4Router(FusedMoeDataRouter):
    """Pass routing tensors to an executor that owns local token dispatch."""

    @classmethod
    def router_type(cls) -> RouterType:
        return RouterType.FUSED_EXECUTOR

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(getattr(config, "moe_quant_method", None) == "FP8_FP4")

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        if a1_scale is not None or a2_scale is not None:
            raise ValueError("FP8/FP4 MoE quantizes activations inside its executor")
        return ExpertForwardPayload(
            expert_x=a1,
            expert_x_origin_dtype=a1.dtype,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[FinalizeArgs],
    ) -> torch.Tensor:
        return payload.fused_expert_output


class MegaMoeRouter(Fp8Fp4Router):
    """Router marker for executors whose kernel fuses EP dispatch/combine."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        super().check_conditions(checker, config)
        checker.check(config.ep_size > 1)

    @property
    def supports_gate_pack(self) -> bool:
        return True

    def prepare_gate_pack(
        self,
        a1: torch.Tensor,
        gate_payload: ExpertGatePayload,
    ) -> ExpertForwardPayload:
        return ExpertForwardPayload(
            expert_x=a1,
            expert_x_origin_dtype=a1.dtype,
            gate_payload=gate_payload,
        )
