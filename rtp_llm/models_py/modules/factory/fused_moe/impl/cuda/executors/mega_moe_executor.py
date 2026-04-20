"""MegaMoe executor: wraps the DeepGEMM fp8_fp4_mega_moe kernel.

The kernel fuses EP-dispatch + L1 GEMM + SwiGLU + L2 GEMM + EP-combine into a
single operation using NVLink symmetric memory, eliminating separate deep_ep
dispatch/combine overhead.

Weight format:  FP4 (e2m1) with UE8M0 per-32-element scale factors.
Activation:     FP8 (e4m3fn) with packed UE8M0 per-32-element scale factors.

Alignment requirements (TMA):
  hidden_size % 128 == 0
  moe_intermediate_size % 512 == 0
  SM100 (GB200 / Blackwell) only

Activation via env:  USE_MEGA_MOE=1
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

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
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)

_BLOCK_M_ALIGN = 128


def _cast_grouped_weights_to_fp4(
    bf16_weights: torch.Tensor,  # [G, N, K] BF16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise each row of grouped weights from BF16 to FP4 (e2m1).

    Returns:
        packed – int8 [G, N, K//2] (two FP4 values packed per byte)
        sf     – UE8M0 scale factors in the layout required by mega_moe
    """
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp4

    num_groups, n, k = bf16_weights.shape
    packed_list: List[torch.Tensor] = []
    sf_list: List[torch.Tensor] = []
    for i in range(num_groups):
        p, s = per_token_cast_to_fp4(
            bf16_weights[i].float(),
            use_ue8m0=True,
            gran_k=32,
            use_packed_ue8m0=False,
        )
        packed_list.append(p)
        sf_list.append(s)

    packed = torch.stack(packed_list)   # [G, N, K//2]
    sf = torch.stack(sf_list)           # [G, N, K//32]
    sf = deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), num_groups)
    return packed, sf


def _prepare_mega_moe_weights(
    w1_bf16: torch.Tensor,  # [E, 2*I, H]  gate+up stacked
    w2_bf16: torch.Tensor,  # [E, H, I]    down projection
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Convert BF16 weights to FP4 and apply mega_moe TMA layout."""
    import deep_gemm

    l1_packed, l1_sf = _cast_grouped_weights_to_fp4(w1_bf16)
    l2_packed, l2_sf = _cast_grouped_weights_to_fp4(w2_bf16)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
        (l1_packed, l1_sf), (l2_packed, l2_sf)
    )
    return transformed_l1, transformed_l2


def _get_ep_group() -> Optional[dist.ProcessGroup]:
    if not dist.is_available() or not dist.is_initialized():
        return None
    return dist.group.WORLD


class MegaMoeExecutor(FusedMoeExpertExecutor):
    """Executor that calls the DeepGEMM fp8_fp4_mega_moe fused kernel.

    One instance per MoE layer; weights are converted to FP4 at construction
    and the NVLink symmetric buffer is pre-allocated.
    """

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.MEGA_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.utils.arch import get_sm

        checker.check(os.environ.get("USE_MEGA_MOE", "0") == "1")

        try:
            import deep_gemm
            checker.check(hasattr(deep_gemm, "fp8_fp4_mega_moe"))
        except ImportError:
            checker.check(False)
            return

        sm = get_sm()
        checker.check(sm[0] >= 10)  # SM100 = GB200/Blackwell

        moe_inter = config.model_config.moe_inter_size
        checker.check(moe_inter % 512 == 0,)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        import deep_gemm

        self._hidden: int = config.hidden_size
        self._moe_inter: int = config.model_config.moe_inter_size
        self._num_experts: int = config.expert_num
        self._top_k: int = config.moe_k
        self._ep_size: int = config.ep_size
        self._ep_rank: int = config.ep_rank
        self._activation_clamp: float = float(
            os.environ.get("MEGA_MOE_ACTIVATION_CLAMP", "10.0")
        )
        self._fast_math: bool = os.environ.get("MEGA_MOE_FAST_MATH", "1") != "0"

        # Load weights and convert to FP4
        w1 = weights[W.moe_w1]   # [E, 2*I, H]
        w2 = weights[W.moe_w2]   # [E, H, I]
        w1_bf16 = w1.to(torch.bfloat16)
        w2_bf16 = w2.to(torch.bfloat16)

        logger.info(
            f"[MegaMoeExecutor] Converting weights: "
            f"w1={tuple(w1_bf16.shape)}, w2={tuple(w2_bf16.shape)}"
        )
        self._l1_weights, self._l2_weights = _prepare_mega_moe_weights(
            w1_bf16, w2_bf16
        )

        # Allocate NVLink symmetric buffer
        ep_group = _get_ep_group()
        if ep_group is None:
            raise RuntimeError(
                "MegaMoeExecutor requires torch.distributed to be initialized"
            )

        max_tokens_raw = int(os.environ.get(
            "MEGA_MOE_MAX_TOKENS",
            str(max(
                getattr(config, "ll_num_max_token", 0) or 65536,
                _BLOCK_M_ALIGN,
            ))
        ))
        max_tokens = math.ceil(max_tokens_raw / _BLOCK_M_ALIGN) * _BLOCK_M_ALIGN

        logger.info(
            f"[MegaMoeExecutor] Allocating symm buffer: "
            f"ep_size={self._ep_size}, experts={self._num_experts}, "
            f"max_tokens={max_tokens}, top_k={self._top_k}, "
            f"hidden={self._hidden}, moe_inter={self._moe_inter}"
        )
        self._symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            ep_group,
            num_experts=self._num_experts,
            num_max_tokens_per_rank=max_tokens,
            num_topk=self._top_k,
            hidden=self._hidden,
            intermediate_hidden=self._moe_inter,
            use_fp8_dispatch=True,
            activation="swiglu",
        )

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict],
    ) -> CombineForwardPayload:
        import deep_gemm
        from deep_gemm.utils.math import per_token_cast_to_fp8

        hidden_states = payload.expert_x           # [T, H] BF16
        topk_ids = payload.expert_topk_ids         # [T, top_k] int32
        topk_weights = payload.expert_topk_weights # [T, top_k] float32

        num_tokens = hidden_states.size(0)
        buf = self._symm_buffer

        assert num_tokens <= buf.x.size(0), (
            f"num_tokens={num_tokens} exceeds symm buffer capacity {buf.x.size(0)}. "
            f"Set MEGA_MOE_MAX_TOKENS >= {num_tokens}."
        )

        # Quantise activations to FP8 with packed UE8M0 scaling
        x_fp8, x_sf = per_token_cast_to_fp8(
            hidden_states.float(),
            use_ue8m0=True,
            gran_k=32,
            use_packed_ue8m0=True,
        )

        # Fill symmetric buffer input slots
        buf.x[:num_tokens].copy_(x_fp8)
        buf.x_sf[:num_tokens].copy_(x_sf)
        buf.topk_idx[:num_tokens].copy_(topk_ids)
        buf.topk_weights[:num_tokens].copy_(topk_weights)

        output = torch.empty(
            (num_tokens, self._hidden),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

        deep_gemm.fp8_fp4_mega_moe(
            output,
            self._l1_weights,
            self._l2_weights,
            buf,
            activation_clamp=self._activation_clamp,
            fast_math=self._fast_math,
        )

        return CombineForwardPayload(fused_expert_output=output)
