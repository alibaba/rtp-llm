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

from rtp_llm.models_py.modules.dsv4.moe.moe_layer import resolve_moe_max_tokens_per_rank

from .mega_moe import GLM5MegaMoE

logger = logging.getLogger(__name__)


def _split_stacked_moe_w1_up_gate(
    w1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split RTP stacked MoE W1 layout into up/value and gate halves."""
    if w1.dim() != 3:
        raise ValueError(
            f"Expected stacked moe_w1 to be 3D, got shape={tuple(w1.shape)}"
        )
    if w1.shape[1] % 2 != 0:
        raise ValueError(
            f"Expected even stacked moe_w1 N dimension, got shape={tuple(w1.shape)}"
        )
    half_n = w1.shape[1] // 2
    # RTP's generic MoE kernels use [up/value | gate].  DeepGEMM Mega MoE
    # follows the DSV4 convention [gate | up], so callers must reorder.
    w1_up = w1[:, :half_n, :].contiguous()
    w1_gate = w1[:, half_n:, :].contiguous()
    return w1_up, w1_gate


def _restack_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.cat([gate, up], dim=1).contiguous()


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
        max_generate_batch_size: int = 0,
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

        max_seq_len = getattr(config, "max_seq_len", 0)
        max_tokens_per_rank = max(8192, max_seq_len) if max_seq_len > 0 else 8192
        if moe_config is not None:
            ll = getattr(moe_config, "ll_num_max_token", 0)
            if ll and ll > 0:
                max_tokens_per_rank = max(max_tokens_per_rank, ll)

        # Detect decode role from parallelism_config.role_type
        is_decode_role = False
        try:
            from rtp_llm.ops import RoleType

            role_type = getattr(parallelism_config, "role_type", None)
            if role_type is not None:
                is_decode_role = role_type == RoleType.DECODE
        except Exception:
            pass

        # Resolve CP: when prefill_cp_config is enabled, per-rank tokens
        # are bounded by max_seq_len / cp_size (same logic as DSv4).
        cp_size = 1
        if (
            not is_decode_role
            and parallelism_config is not None
            and getattr(parallelism_config, "prefill_cp_config", None) is not None
        ):
            try:
                if parallelism_config.prefill_cp_config.is_enabled():
                    cp_size = int(getattr(parallelism_config, "tp_size", 1) or 1)
            except Exception:
                pass

        # Resolve max_tokens_per_rank based on role
        max_generate_batch_size = (
            int(max_generate_batch_size) if max_generate_batch_size > 0 else 0
        )
        gen_num_per_cycle = int(getattr(config, "gen_num_per_cycle", 0) or 0)
        resolved = resolve_moe_max_tokens_per_rank(
            max_seq_len=max_seq_len,
            current_max_tokens_per_rank=max_tokens_per_rank,
            cp_size=cp_size,
            max_generate_batch_size=max(max_generate_batch_size, 1),
            is_decode_role=is_decode_role,
            is_speculative=gen_num_per_cycle > 0,
            gen_num_per_cycle=gen_num_per_cycle,
        )
        if resolved != max_tokens_per_rank:
            logger.info(
                "[GLM5 MegaMoE] max_tokens_per_rank %d -> %d "
                "(role=%s, cp=%d, max_batch=%d, gen_num_per_cycle=%d)",
                max_tokens_per_rank,
                resolved,
                "DECODE" if is_decode_role else "PREFILL",
                cp_size,
                max_generate_batch_size,
                gen_num_per_cycle,
            )
            max_tokens_per_rank = resolved

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

        if w1 is None or w2 is None or s1 is None or s2 is None:
            raise ValueError(
                "MegaMoeFusedWrapper requires load-time FP4 MoE weights "
                "(moe_w1, moe_w2, moe_s1, moe_s2). Runtime/module-init "
                "FP4 quantization is not supported."
            )
        if w1.dtype != torch.int8 or w2.dtype != torch.int8:
            raise ValueError(
                "MegaMoeFusedWrapper only accepts load-time FP4 int8 weights. "
                "Runtime/module-init FP4 quantization is not supported. "
                f"Got moe_w1 dtype={w1.dtype}, moe_w2 dtype={w2.dtype}."
            )

        w1_up, w1_gate = _split_stacked_moe_w1_up_gate(w1)
        s1_up, s1_gate = _split_stacked_moe_w1_up_gate(s1)
        del w1, s1
        self.mega_moe.setup_weights_from_fp4(
            w1_w=_restack_gate_up(w1_gate, w1_up),
            w1_s=_restack_gate_up(s1_gate, s1_up),
            w2_w=w2,
            w2_s=s2,
        )
        del w1_up, w1_gate, s1_up, s1_gate, w2, s2
        torch.cuda.empty_cache()

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
