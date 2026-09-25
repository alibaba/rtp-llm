"""DCU fused MoE executor using the DAS aiter triton fused_experts kernel."""

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.dcu.bolt_ops_compat import (
    ensure_bolt_ops_compat,
)

# DAS aiter 0.1.5+das185 imports activation helpers from a `bolt_ops` package
# that is not shipped in the DTK images, which makes every aiter MoE module
# (triton/asm/moe_c) fail at import time. Install the vendored shim first;
# once a real bolt_ops wheel is available it takes precedence automatically.
ensure_bolt_ops_compat()

from aiter.ops.triton.fused_moe import fused_experts_impl  # noqa: E402

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.impl.dcu.marlin_w16a16_pack import (
    shapes_match_marlin_packed,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.utils.model_weight import W

_ACTIVATION_MAP = {
    # rtp-llm gated activation names -> aiter moe activation names
    "SiGLU": "silu",
    "siglu": "silu",
    "swiglu": "silu",
    "silu": "silu",
    "GeGLU": "gelu",
    "geglu": "gelu",
    "gelu": "gelu",
    "relu2": "relu2",
}


def _build_expert_map(
    num_experts: int,
    ep_rank: int,
    ep_size: int,
    local_num_experts: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build expert_map for EP>1: shape (num_experts,), value is local index or -1."""
    if ep_size <= 1:
        return None
    start = ep_rank * local_num_experts
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    expert_map[start : start + local_num_experts] = torch.arange(
        local_num_experts, dtype=torch.int32, device=device
    )
    return expert_map


class DcuExpertsBf16(FusedMoeExpertExecutor):
    """DCU BF16 (no quantization) MoE expert executor.

    Runs the two-GEMM triton fused MoE kernel from the DAS aiter wheel
    (``aiter.ops.triton.fused_moe.fused_experts_impl``): tokens are sorted per
    expert (moe_sorting), GEMM1 + gated activation + GEMM2 are fused per
    block, and router weights are applied in GEMM2 before ``moe_sum``.
    """

    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        # aiter moe_sorting kernels require int32 topk_ids
        return torch.int32

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        self.num_experts = config.expert_num
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.w1 = weights[W.moe_w1]
        self.w2 = weights[W.moe_w2]
        self._marlin_packed = shapes_match_marlin_packed(self.w1, self.w2)
        # Hidden size K: packed layout stores w1 as [E, K/16, 2N*16].
        self._hidden_size = (
            self.w1.size(1) * 16 if self._marlin_packed else self.w1.size(2)
        )

        self._expert_map = _build_expert_map(
            self.num_experts,
            self.ep_rank,
            self.ep_size,
            self.w1.size(0),
            self.w1.device,
        )

    @property
    def local_num_experts(self) -> int:
        return self.w1.size(0)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "expert_x is None"
        assert payload.expert_x.size(-1) == self._hidden_size, (
            f"Hidden size mismatch {payload.expert_x.size(-1)} != {self._hidden_size}"
        )
        assert payload.expert_x.is_contiguous(), "Hidden_states must be contiguous"
        assert self.w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert self.w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert payload.expert_tokens_meta is not None

        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        assert topk_ids is not None
        assert topk_weights is not None

        assert self.w1.size(0) == self.local_num_experts
        assert self.w2.size(0) == self.local_num_experts

        hidden_states = payload.expert_x

        if apply_router_weight_on_input:
            assert (
                topk_weights.dim() == 2
            ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
            topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

        # When the router has already remapped IDs to local indices, an
        # expert_map (based on global IDs) must not be applied.
        effective_expert_map = (
            None
            if payload.expert_ids_are_local
            else (expert_map if expert_map is not None else self._expert_map)
        )

        output = fused_experts_impl(
            hidden_states=hidden_states,
            w1=self.w1,
            w2=self.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            output_dtype=hidden_states.dtype,
            activation=_ACTIVATION_MAP.get(activation, "silu"),
            global_num_experts=self.num_experts,
            expert_map=effective_expert_map,
        )
        return CombineForwardPayload(fused_expert_output=output)
