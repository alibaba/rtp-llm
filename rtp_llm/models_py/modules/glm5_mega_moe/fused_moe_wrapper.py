"""Drop-in replacement for FusedMoe that uses GLM5MegaMoE internally.

This wrapper implements the same interface as FusedMoe so it can be used
directly in GenericMoeLayer without changing the forward() call site.

The mega kernel fuses dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine
into one kernel, so there's no separate router/executor.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .mega_moe import GLM5MegaMoE, GLM5MegaMoeCfg

logger = logging.getLogger(__name__)


class MegaMoeFusedWrapper(nn.Module):
    """FusedMoe-compatible wrapper around GLM5MegaMoE.

    Usage in GenericMoeLayer:
        self.fused_moe = MegaMoeFusedWrapper(config, parallelism_config, weights)
        # then forward as usual:
        output = self.fused_moe(hidden_states, topk_weights, topk_ids, activation="SiGLU")
    """

    def __init__(
        self,
        config,
        parallelism_config,
        weights: Dict[str, torch.Tensor],
        moe_config=None,
        layer_idx: int = 0,
    ):
        super().__init__()
        from rtp_llm.utils.model_weight import W

        ep_size = parallelism_config.ep_size
        ep_rank = parallelism_config.ep_rank

        dim = config.hidden_size
        n_routed_experts = config.expert_num
        n_activated_experts = config.moe_k

        moe_inter_dim_raw = getattr(config, "moe_inter_size", None) or None
        if moe_inter_dim_raw is None or moe_inter_dim_raw == 0:
            w1 = weights.get(W.moe_w1)
            if w1 is not None:
                moe_inter_dim_raw = w1.shape[1] // 2
            else:
                raise ValueError("Cannot determine moe_intermediate_size")

        max_tokens_per_rank = 8192
        if moe_config is not None:
            ll = getattr(moe_config, "ll_num_max_token", 0)
            if ll and ll > 0:
                max_tokens_per_rank = ll

        self.mega_moe = GLM5MegaMoE.from_params(
            layer_id=layer_idx,
            dim=dim,
            moe_inter_dim=moe_inter_dim_raw,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
            max_tokens_per_rank=max_tokens_per_rank,
        )

        w1 = weights.pop(W.moe_w1, None)
        w2 = weights.pop(W.moe_w2, None)
        s1 = weights.pop(W.moe_s1, None)
        s2 = weights.pop(W.moe_s2, None)

        if w1 is not None and w1.dtype == torch.float8_e4m3fn and s1 is not None:
            E = w1.shape[0]
            half_n = w1.shape[1] // 2
            if s1.shape[1] < w1.shape[1]:
                s1 = s1.repeat_interleave(128, dim=1)[:, : w1.shape[1], :]
            if s2.shape[1] < w2.shape[1]:
                s2 = s2.repeat_interleave(128, dim=1)[:, : w2.shape[1], :]
            w1_gate = w1[:, :half_n, :].contiguous()
            w1_up = w1[:, half_n:, :].contiguous()
            s1_gate = s1[:, :half_n, :].contiguous()
            s1_up = s1[:, half_n:, :].contiguous()
            del w1, s1
            self.mega_moe.setup_weights_from_fp8(
                w1_fp8=w1_gate,
                w1_scale=s1_gate,
                w2_fp8=w2,
                w2_scale=s2,
                w3_fp8=w1_up,
                w3_scale=s1_up,
            )
            del w1_gate, w1_up, s1_gate, s1_up, w2, s2
            torch.cuda.empty_cache()
        elif w1 is not None and w1.dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            E = w1.shape[0]
            half_n = w1.shape[1] // 2
            w1_gate = w1[:, :half_n, :].to(torch.bfloat16).contiguous()
            w1_up = w1[:, half_n:, :].to(torch.bfloat16).contiguous()
            w2_bf16 = w2.to(torch.bfloat16)
            del w1, w2, s1, s2
            torch.cuda.empty_cache()
            self.mega_moe.setup_weights_from_bf16(
                w1_bf16=w1_gate,
                w2_bf16=w2_bf16,
                w3_bf16=w1_up,
            )
            del w1_gate, w1_up, w2_bf16
            torch.cuda.empty_cache()
        else:
            raise ValueError(
                "MegaMoeFusedWrapper: unsupported weight dtype %s"
                % (w1.dtype if w1 is not None else "None")
            )

        self.expert_num = n_routed_experts

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int64

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        expert_map: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        extra_expert_args: Optional[Dict[str, Any]] = None,
        extra_finalize_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return self.mega_moe(hidden_states, topk_weights, topk_ids)
