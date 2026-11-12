"""Drop-in replacement for routed FusedMoe that uses GLM5MegaMoE internally.

This wrapper implements the same interface as FusedMoe so it can be used
directly in GenericMoeLayer without changing the forward() call site.

The mega kernel fuses dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine
into one kernel, so there's no separate router/executor.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.moe.moe_layer import resolve_moe_max_tokens_per_rank

from .mega_moe import GLM5MegaMoE

logger = logging.getLogger(__name__)
_CHUNKED_GLM5_MOE_LOGGED = False


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


class MegaMoeWrapper(nn.Module):
    """FusedMoe-compatible wrapper around GLM5MegaMoE.

    Usage in GenericMoeLayer:
        self.fused_moe = MegaMoeWrapper(config, parallelism_config, weights)
        # then forward as usual:
        output = self.fused_moe(hidden_states, topk_weights, topk_ids, activation="SiGLU")
    """

    def _get_mega_moe_cls(self):
        return GLM5MegaMoE

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
        eplb_config = getattr(config, "eplb_config", None)
        n_routed_experts = (
            config.expert_num
            if eplb_config is None
            else eplb_config.phy_exp_num(config.expert_num)
        )
        if n_routed_experts % ep_size != 0:
            raise ValueError("MegaMoE physical expert count must be divisible by EP")
        n_activated_experts = config.moe_k
        swiglu_limit = float(getattr(config, "swiglu_limit", 10.0) or 10.0)
        self._activation_clamp = (
            swiglu_limit
            if getattr(config, "model_type", "") in ("hy_v4", "hy_v4_mtp")
            else None
        )

        moe_inter_dim_raw = getattr(config, "moe_inter_size", None) or None
        if moe_inter_dim_raw is None or moe_inter_dim_raw == 0:
            w1 = weights.get(W.moe_w1)
            if w1 is not None:
                moe_inter_dim_raw = w1.shape[1] // 2
            else:
                raise ValueError("Cannot determine moe_intermediate_size")

        max_seq_len = int(getattr(config, "max_seq_len", 0) or 0)
        max_tokens_per_rank = max(max_seq_len, 128) if max_seq_len > 0 else 8192
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

        self.mega_moe = self._get_mega_moe_cls().from_params(
            layer_id=layer_idx,
            dim=dim,
            moe_inter_dim=moe_inter_dim_raw,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            swiglu_limit=swiglu_limit,
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
                "MegaMoeWrapper requires load-time FP4 or FP8 MoE weights "
                "(moe_w1, moe_w2, moe_s1, moe_s2). Runtime/module-init "
                "quantization without checkpoint scales is not supported."
            )

        if w1.dtype == torch.int8 and w2.dtype == torch.int8:
            self.mega_moe.setup_weights_from_fp4(
                w1_w=w1,
                w1_s=s1,
                w2_w=w2,
                w2_s=s2,
                w1_layout="up_gate",
            )
            del w1, s1, w2, s2
        elif w1.dtype == torch.float8_e4m3fn and w2.dtype == torch.float8_e4m3fn:
            w1_up, w1_gate = _split_stacked_moe_w1_up_gate(w1)
            s1_up, s1_gate = _split_stacked_moe_w1_up_gate(s1)
            del w1, s1
            self.mega_moe.setup_weights_from_fp8(
                w1_fp8=w1_gate,
                w1_scale=s1_gate,
                w2_fp8=w2,
                w2_scale=s2,
                w3_fp8=w1_up,
                w3_scale=s1_up,
            )
            del w1_up, w1_gate, s1_up, s1_gate, w2, s2
        else:
            raise ValueError(
                "MegaMoeWrapper only accepts load-time FP4 int8 or FP8 e4m3 "
                "weights with checkpoint scales. "
                f"Got moe_w1 dtype={w1.dtype}, moe_w2 dtype={w2.dtype}."
            )
        torch.cuda.empty_cache()

        self.expert_num = n_routed_experts

    def _forward_chunked(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        forward_fn,
        eplb=None,
    ) -> torch.Tensor:
        capacity = int(
            getattr(
                getattr(self.mega_moe, "_mega_buf", None), "num_max_tokens_per_rank", 0
            )
            or 0
        )
        tokens = int(hidden_states.size(0))
        if capacity <= 0 or tokens <= capacity:
            return forward_fn(
                hidden_states,
                topk_weights,
                topk_ids,
                **({"eplb": eplb, "token_offset": 0} if eplb is not None else {}),
            )

        global _CHUNKED_GLM5_MOE_LOGGED
        if not _CHUNKED_GLM5_MOE_LOGGED:
            _CHUNKED_GLM5_MOE_LOGGED = True
            logger.info(
                "[GLM5 MegaMoE] chunked forward enabled: tokens=%d "
                "chunk_tokens=%d chunks=%d hidden=%s device=%s",
                tokens,
                capacity,
                (tokens + capacity - 1) // capacity,
                tuple(hidden_states.shape),
                hidden_states.device,
            )

        output = torch.empty_like(hidden_states)
        for token_start in range(0, tokens, capacity):
            token_end = min(token_start + capacity, tokens)
            output[token_start:token_end].copy_(
                forward_fn(
                    hidden_states[token_start:token_end],
                    topk_weights[token_start:token_end],
                    topk_ids[token_start:token_end],
                    **(
                        {"eplb": eplb, "token_offset": token_start}
                        if eplb is not None
                        else {}
                    ),
                )
            )
        return output

    supports_shared_mega_buf = True

    def supports_gate_pack(self, x, bias, config, eplb) -> bool:
        from .eplb_routing import maps_supported

        mode = os.environ.get("RTP_LLM_GLM5_GATE_PACK", "auto")
        return (
            (mode == "1" or (mode == "auto" and eplb is not None))
            and getattr(config, "model_type", "") == "glm_5"
            and getattr(self.mega_moe._input_packer, "name", None) == "fused"
            and x.is_cuda
            and x.ndim == 2
            and x.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.shape[1] % 128 == 0
            and bias is not None
            and bias.ndim == 1
            and bias.device == x.device
            and bias.dtype == torch.float32
            and bias.is_contiguous()
            and 2 <= int(config.moe_k) <= min(32, bias.numel())
            and bias.numel() <= 512
            and int(config.moe_n_group) == int(config.moe_topk_group) == 1
            and bool(config.has_moe_norm)
            and (eplb is None or maps_supported(eplb, x.device))
        )

    def forward_gate_pack(self, x, logits, bias, route_scale, eplb=None):
        capacity = int(self.mega_moe._mega_buf.num_max_tokens_per_rank)

        def run(start, end):
            return self.mega_moe._forward_impl(
                x[start:end],
                None,
                None,
                inputs_prepacked=False,
                activation_clamp=self._activation_clamp,
                gate_logits=logits[start:end],
                gate_bias=bias,
                route_scale=route_scale,
                eplb=eplb,
                token_offset=start,
            )

        if x.shape[0] <= capacity:
            return run(0, x.shape[0])
        output = torch.empty_like(x)
        for start in range(0, x.shape[0], capacity):
            end = min(start + capacity, x.shape[0])
            output[start:end].copy_(run(start, end))
        return output

    def supports_eplb_input_pack(self, layer, hidden_states) -> bool:
        from .eplb_routing import maps_supported

        return (
            maps_supported(layer, hidden_states.device)
            and getattr(self.mega_moe._input_packer, "name", None) == "fused"
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[1] % 128 == 0
        )

    def get_eplb_weights(self) -> tuple[torch.Tensor, ...]:
        """Return converted routed weights and scales, excluding shared experts."""
        moe = self.mega_moe
        return (moe._mega_l1_w, moe._mega_l1_sf, moe._mega_l2_w, moe._mega_l2_sf)

    def clone_for_cuda_graph(self, *, share_mega_buf: bool = False) -> "MegaMoeWrapper":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)
        clone.mega_moe = self.mega_moe.clone_for_cuda_graph(
            share_mega_buf=share_mega_buf
        )
        clone.expert_num = self.expert_num
        clone._activation_clamp = self._activation_clamp
        return clone

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int64

    def prepacked_input_views(
        self, tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mega_moe.prepacked_input_views(tokens)

    def forward_prepacked(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The subclasses own different fused-shared/FP8 clamp contracts. This
        # routed-only override is intentionally limited to plain mega_moe.
        if type(self) is MegaMoeWrapper:
            return self.mega_moe.forward_prepacked(
                hidden_states, activation_clamp=self._activation_clamp
            )
        return self.mega_moe.forward_prepacked(hidden_states)

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
        eplb=None,
    ) -> torch.Tensor:
        swiglu_limit = float(
            (extra_expert_args or {}).get("swiglu_limit", self._activation_clamp or 0.0)
        )
        activation_clamp = swiglu_limit if swiglu_limit > 0 else None
        return self._forward_chunked(
            hidden_states,
            topk_weights,
            topk_ids,
            lambda x, weights, indices, **kwargs: self.mega_moe(
                x,
                weights,
                indices,
                activation_clamp=activation_clamp,
                **kwargs,
            ),
            eplb=eplb,
        )
