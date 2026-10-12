"""Gemma4 geometry, normalization, RoPE, dense and routed expert math."""

import os
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.modules import FusedMoeFactory
from rtp_llm.models_py.modules.base.common.activation import gelu_and_mul
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.gemma4.geometry import (
    GEMMA4_TAG_FULL,
    GEMMA4_TAG_SWA,
    Gemma4LayerGeometry,
    _layer_type_at,
    build_gemma4_layer_geometry,
)
from rtp_llm.ops import HybridAttentionType, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import rtp_llm_ops
from rtp_llm.utils.model_weight import W

# ---------------------------------------------------------------------------
# RMSNorm (HF Gemma4RMSNorm): fp32 computation, plain weight multiplication.
# ---------------------------------------------------------------------------


def gemma4_rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    weight_float: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """HF Gemma4RMSNorm: ``x * rsqrt(mean(x^2)+eps) * weight`` (no +1)."""
    if (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() in (2, 3)
        and x.size(-1) in (256, 512, 2816)
        and x.stride(-1) == 1
        and torch.version.hip is None
    ):
        # Single-pass fused kernel: bit-equal to the square->inv->apply
        # chain (identical lane/tree reduction order and apply math) while
        # eliminating the fp32 square copy round-trip (~14B/elem -> ~4B).
        weight_arg = None
        if weight is not None:
            weight_arg = weight.float() if weight_float is None else weight_float
        return rtp_llm_ops.gemma4_rms_norm_fused_bf16(x, weight_arg, eps)
    dtype = x.dtype
    xf = x.float()
    mean_squared = xf.pow(2).mean(-1, keepdim=True) + eps
    out = xf * torch.pow(mean_squared, -0.5)
    if weight is not None:
        out = out * (weight.float() if weight_float is None else weight_float)
    return out.to(dtype)


class Gemma4RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("weight_float", weight.float(), persistent=False)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return gemma4_rms_norm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.weight_float,
        )


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def _proportional_inv_freq(
    head_dim: int, base: float, partial_rotary_factor: float
) -> torch.Tensor:
    """HF ``_compute_proportional_rope_parameters`` (factor=1.0).

    Returns ``inv_freq`` of length ``head_dim // 2``: the first
    ``int(partial * head_dim // 2)`` entries follow ``base ** (-2j/head_dim)``
    and the rest are zeros (NoPE slots). With ``partial == 1.0`` this reduces
    to the default rope table.
    """
    half = head_dim // 2
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    rope_angles = max(0, min(rope_angles, half))
    j = torch.arange(rope_angles, dtype=torch.float32)
    inv_rotated = 1.0 / (float(base) ** (2.0 * j / float(head_dim)))
    nope = half - rope_angles
    if nope > 0:
        return torch.cat([inv_rotated, torch.zeros(nope, dtype=torch.float32)], dim=0)
    return inv_rotated


class Gemma4RopeTable:
    """Per-tag rope table: cos/sin computed on the fly from ``inv_freq``.

    HF computes ``emb = cat(freqs, freqs)`` and applies a full-width
    ``rotate_half`` over ``head_dim``; zero-frequency slots make the NoPE
    dims the identity, which is exactly HF's partial rotation.
    """

    def __init__(
        self,
        head_dim: int,
        base: float,
        partial_rotary_factor: float = 1.0,
    ):
        self.head_dim = head_dim
        self.inv_freq = _proportional_inv_freq(head_dim, base, partial_rotary_factor)
        self._device_inv_freq: Dict[torch.device, torch.Tensor] = {}

    def _inv_freq_on(self, device: torch.device) -> torch.Tensor:
        cached = self._device_inv_freq.get(device)
        if cached is None:
            cached = self.inv_freq.to(device)
            self._device_inv_freq[device] = cached
        return cached

    def cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions [T] (int) -> (cos, sin) [T, head_dim] float32."""
        inv_freq = self._inv_freq_on(positions.device)
        freqs = positions.float().unsqueeze(-1) * inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def cos_sin_bf16(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._inv_freq_on(positions.device)
        if (
            positions.is_cuda
            and positions.dtype == torch.int32
            and positions.is_contiguous()
            and inv_freq.dtype == torch.float32
            and inv_freq.is_contiguous()
            and torch.version.hip is None
        ):
            return rtp_llm_ops.gemma4_rope_cos_sin_bf16(positions, inv_freq)
        cos, sin = self.cos_sin(positions)
        return cos.to(torch.bfloat16), sin.to(torch.bfloat16)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_gemma4_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """x [T, heads, head_dim]; cos/sin [T, head_dim] (any float dtype)."""
    cos = cos.to(x.dtype)[:, None, :]
    sin = sin.to(x.dtype)[:, None, :]
    return (x * cos) + (_rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Router / experts / dense MLP
# ---------------------------------------------------------------------------


class Gemma4Router(nn.Module):
    """HF Gemma4TextRouter (modeling_gemma4.py:1283-1316), fp32 math."""

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        hidden_size: int,
        top_k: int,
        eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.variance_epsilon = eps
        self.scale = weights[W.moe_router_scale]  # [H]
        self.per_expert_scale = weights[W.moe_router_expert_scale]  # [E]
        # [H, E]: checkpoint router proj already transposed by the loader.
        self.proj_weight = weights[W.moe_gate]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = gemma4_rms_norm(x, None, self.variance_epsilon)
        if (
            h.is_cuda
            and h.dtype == torch.bfloat16
            and self.scale.dtype == torch.bfloat16
            and h.is_contiguous()
            and self.scale.is_contiguous()
            and h.size(-1) % 8 == 0
            and torch.version.hip is None
        ):
            h = rtp_llm_ops.gemma4_router_scale_bf16(
                h, self.scale, self.hidden_size**-0.5
            )
        else:
            h = h * self.scale * (self.hidden_size**-0.5)
        scores = h @ self.proj_weight  # [T, E]
        probs = gemma4_router_softmax(scores)
        if (
            probs.is_cuda
            and probs.dtype == torch.bfloat16
            and probs.is_contiguous()
            and probs.size(-1) == 128
            and self.top_k == 8
            and torch.version.hip is None
        ):
            top_w, top_idx = rtp_llm_ops.gemma4_topk_8_bf16(probs)
        else:
            top_w, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        if (
            top_w.is_cuda
            and top_w.dtype == torch.bfloat16
            and top_idx.dtype == torch.int64
            and self.per_expert_scale.dtype == torch.bfloat16
            and top_w.is_contiguous()
            and top_idx.is_contiguous()
            and self.per_expert_scale.is_contiguous()
            and self.top_k == 8
            and torch.version.hip is None
        ):
            top_w = rtp_llm_ops.gemma4_finalize_router_weights_bf16(
                top_w, top_idx, self.per_expert_scale
            )
        else:
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
            top_w = top_w * self.per_expert_scale[top_idx]
        return top_w, top_idx


def gemma4_add_bf16(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if (
        lhs.is_cuda
        and lhs.dtype == torch.bfloat16
        and rhs.dtype == torch.bfloat16
        and lhs.is_contiguous()
        and rhs.is_contiguous()
        and lhs.numel() % 8 == 0
        and torch.version.hip is None
    ):
        return rtp_llm_ops.gemma4_add_bf16(lhs, rhs)
    return lhs + rhs


class Gemma4Experts(nn.Module):
    """HF Gemma4TextExperts in the frozen runtime weight layout.

    ``moe_w1``: [E, 2N, H] with checkpoint order ``[gate; up]``.
    ``moe_w2``: [E, H, N].
    Activation: ``gelu_pytorch_tanh``.
    """

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        parallelism_config: Optional[ParallelismConfig] = None,
        model_config: Optional[ModelConfig] = None,
        moe_config: Optional[MoeConfig] = None,
        enable_cuda_graph: bool = False,
    ):
        super().__init__()
        self.w1 = weights[W.moe_w1]  # [E, 2N, H]
        self.w2 = weights[W.moe_w2]  # [E, H, N]
        self.num_experts = self.w1.shape[0]
        self.inter_size = self.w1.shape[1] // 2
        self.fused_moe = None
        self.collective_ep = False
        if parallelism_config is not None and parallelism_config.ep_size > 1:
            if model_config is None:
                raise RuntimeError("Gemma4 EP requires the model config")
            low_latency = bool(
                moe_config is not None
                and (
                    moe_config.use_deepep_low_latency
                    or "low_latency" in moe_config.moe_strategy
                )
            )
            if enable_cuda_graph or low_latency:
                raise RuntimeError(
                    "Gemma4 EP CUDA graph/low-latency execution is not implemented"
                )
            backend = os.environ.get("RTP_LLM_GEMMA4_EP_BACKEND", "deepep")
            if backend not in ("deepep", "collective"):
                raise ValueError(f"unsupported Gemma4 EP backend: {backend}")
            if backend == "collective":
                if parallelism_config.ep_size != parallelism_config.world_size:
                    raise ValueError(
                        "Gemma4 collective EP requires one EP group covering the world"
                    )
                self.collective_ep = True
                self.ep_rank = int(parallelism_config.ep_rank)
                self.world_rank = int(parallelism_config.world_rank)
                self.world_size = int(parallelism_config.world_size)
                self.global_expert_count = int(model_config.expert_num)
                if self.num_experts * self.world_size != self.global_expert_count:
                    raise ValueError(
                        "Gemma4 EP checkpoint shards do not cover all experts"
                    )
            else:
                adapter = MoEConfigAdapter(
                    model_config=model_config,
                    parallelism_config=parallelism_config,
                    moe_config=moe_config,
                    quant_config=model_config.quant_config,
                    enable_cuda_graph=False,
                )
                self.fused_moe = FusedMoeFactory().create_fused_moe(adapter, weights)
        self.ffn_tp_size = (
            parallelism_config.get_ffn_tp_size()
            if parallelism_config is not None
            else 1
        )
        # graph-capture state: set by the decoder layer when the model runs
        # under cuda-graph capture; the grouped path is exact (device-side
        # segment offsets), no capacity state is kept
        self._graph_mode = False
        self._decode_mode = False
        self._path_counts = {"eager": 0, "grouped": 0, "batched": 0}
        self._grouped_mm = None
        if os.environ.get("RTP_LLM_GEMMA4_GROUPED_GEMM_BACKEND", "torch") == "deepgemm":
            from rtp_llm.models_py.modules.gemma4.grouped_moe import create_grouped_mm

            self._grouped_mm = create_grouped_mm(self.w1, self.w2)

    def forward(
        self,
        x: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
    ) -> torch.Tensor:
        if self.collective_ep:
            return self._forward_collective_ep(x, top_idx, top_w)
        if self.fused_moe is not None:
            return self.fused_moe(
                hidden_states=x,
                topk_weights=top_w.float(),
                topk_ids=top_idx.to(self.fused_moe.topk_ids_dtype),
                activation="GeGLU",
            )
        grouped_stride_aligned = (self.w2.shape[-1] * self.w2.element_size()) % 16 == 0
        grouped_supported = (
            self.w1.dtype == torch.bfloat16
            and hasattr(torch, "_grouped_mm")
            and grouped_stride_aligned
        )
        if self._graph_mode and not grouped_supported:
            raise RuntimeError(
                "Gemma4 CUDA graph MoE requires BF16 aligned weights and torch._grouped_mm"
            )
        if grouped_supported:
            self._path_counts["grouped"] += 1
            output = self._forward_grouped_device(x, top_idx, top_w)
        elif self._decode_mode:
            self._path_counts["batched"] += 1
            output = self._forward_batched_device(x, top_idx, top_w)
        else:
            self._path_counts["eager"] += 1
            output = self._forward_eager(x, top_idx, top_w)
        if self.ffn_tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output

    def _forward_collective_ep(self, x, top_idx, top_w):
        # Shape exchange precedes variable-length data gathers. Every rank,
        # including an empty rank, participates in the same collectives.
        counts = (
            all_gather(
                torch.tensor([x.size(0)], dtype=torch.int32, device=x.device),
                group=Group.DP_AND_TP,
            )
            .cpu()
            .tolist()
        )
        capacity = max(1, max(counts))
        padded_x = F.pad(x, (0, 0, 0, capacity - x.size(0)))
        padded_idx = F.pad(top_idx, (0, 0, 0, capacity - x.size(0)))
        padded_w = F.pad(top_w, (0, 0, 0, capacity - x.size(0)))
        global_x = all_gather(padded_x, group=Group.DP_AND_TP)
        global_idx = all_gather(padded_idx, group=Group.DP_AND_TP)
        global_w = all_gather(padded_w, group=Group.DP_AND_TP)
        first_expert = self.ep_rank * self.num_experts
        local_idx = global_idx - first_expert
        owned = (local_idx >= 0) & (local_idx < self.num_experts)
        local_w = torch.where(owned, global_w, 0)
        local_idx = local_idx.clamp(0, self.num_experts - 1)
        local_sum = self._forward_grouped_device(
            global_x, local_idx, local_w, output_dtype=torch.float32
        )
        combined = all_reduce(local_sum, group=Group.DP_AND_TP)
        begin = self.world_rank * capacity
        return combined[begin : begin + x.size(0)].to(x.dtype)

    def _forward_batched_device(
        self,
        x: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden = x.shape
        top_k = top_idx.shape[-1]
        token_idx = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(-1, top_k)
            .reshape(-1)
        )
        expert_ids = top_idx.reshape(-1)
        selected_hidden = x[token_idx]
        selected_w1 = self.w1[expert_ids]
        gate_up = torch.bmm(selected_w1, selected_hidden.unsqueeze(-1)).squeeze(-1)
        expert_output = gelu_and_mul(gate_up)
        selected_w2 = self.w2[expert_ids]
        expert_output = torch.bmm(selected_w2, expert_output.unsqueeze(-1)).squeeze(-1)
        weighted_output = expert_output * top_w.reshape(-1, 1)
        return weighted_output.view(num_tokens, top_k, hidden).sum(dim=1).to(x.dtype)

    def _forward_eager(
        self,
        x: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden = x.shape
        out = torch.zeros(num_tokens, hidden, dtype=x.dtype, device=x.device)
        flat_idx = top_idx.reshape(-1)
        flat_w = top_w.reshape(-1)
        top_k = top_idx.shape[-1]
        sorted_idx, perm = torch.sort(flat_idx, stable=True)
        sorted_w = flat_w[perm]
        sorted_tok = perm // top_k
        counts_host = torch.bincount(sorted_idx, minlength=self.num_experts).tolist()
        if len(counts_host) > self.num_experts:
            raise IndexError(
                f"router selected expert {int(sorted_idx.max().item())} out of range"
            )
        start = 0
        for expert, count in enumerate(counts_host):
            if count == 0:
                continue
            end = start + count
            token_idx = sorted_tok[start:end]
            xe = x[token_idx]
            gate_up = xe @ self.w1[expert].t()
            y = gelu_and_mul(gate_up)
            y = y @ self.w2[expert].t()
            contribution = y * sorted_w[start:end, None]
            out.index_add_(0, token_idx, contribution.to(out.dtype))
            start = end
        return out

    def _forward_grouped_device(
        self,
        x: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        num_tokens, hidden = x.shape
        top_k = top_idx.shape[-1]
        flat_idx = top_idx.reshape(-1)
        flat_w = top_w.reshape(-1)
        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and flat_idx.dtype == torch.int64
            and flat_w.dtype == torch.bfloat16
            and x.is_contiguous()
            and flat_idx.is_contiguous()
            and flat_w.is_contiguous()
            and hidden % 8 == 0
            and torch.version.hip is None
        ):
            perm, inv_perm, sorted_w, offsets = rtp_llm_ops.gemma4_prepare_grouped_moe(
                flat_idx, flat_w, self.num_experts
            )
            if (
                self._grouped_mm is not None
                and top_k == 8
                and output_dtype != torch.float32
            ):
                return self._grouped_mm.forward_routed(
                    x,
                    self.w1,
                    self.w2,
                    flat_idx,
                    perm,
                    inv_perm,
                    sorted_w,
                    offsets,
                    gelu_and_mul,
                )
            sorted_x = rtp_llm_ops.gemma4_gather_sorted_expert_input_bf16(
                x, perm, top_k
            )
        else:
            perm = torch.argsort(flat_idx)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(perm.numel(), device=x.device)
            sorted_idx = flat_idx[perm]
            sorted_w = flat_w[perm]
            token_idx = (
                torch.arange(num_tokens, device=x.device)
                .unsqueeze(1)
                .expand(-1, top_k)
                .reshape(-1)
            )
            sorted_x = x[token_idx[perm]]
            histc_input = (
                sorted_idx.float() if x.device.type == "cpu" else sorted_idx.int()
            )
            counts = torch.histc(
                histc_input,
                bins=self.num_experts,
                min=0,
                max=self.num_experts - 1,
            )
            offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        grouped_mm = getattr(F, "grouped_mm", torch._grouped_mm)
        use_grouped_candidate = (
            self._grouped_mm is not None
            and top_k == 8
            and sorted_x.dtype == torch.bfloat16
            and offsets.dtype == torch.int32
            and sorted_x.is_contiguous()
            and offsets.is_contiguous()
        )
        if use_grouped_candidate:
            expert_output = self._grouped_mm.forward_moe(
                sorted_x, self.w1, self.w2, offsets, gelu_and_mul
            )
        else:
            gate_up = grouped_mm(
                sorted_x.to(self.w1.dtype), self.w1.transpose(1, 2), offs=offsets
            )
            expert_output = gelu_and_mul(gate_up)
            expert_output = grouped_mm(
                expert_output.to(self.w2.dtype),
                self.w2.transpose(1, 2),
                offs=offsets,
            )
        if output_dtype == torch.float32:
            weighted = (expert_output * sorted_w.unsqueeze(-1)).to(expert_output.dtype)
            weighted = weighted[inv_perm].view(num_tokens, top_k, hidden)
            return weighted.float().sum(dim=1)
        if (
            expert_output.is_cuda
            and top_k == 8
            and expert_output.dtype == torch.bfloat16
            and sorted_w.dtype == torch.bfloat16
            and expert_output.is_contiguous()
            and sorted_w.is_contiguous()
            and inv_perm.is_contiguous()
            and expert_output.size(1) % 8 == 0
            and torch.version.hip is None
        ):
            weighted_output = rtp_llm_ops.gemma4_weighted_reorder_bf16(
                expert_output, sorted_w, inv_perm
            )
            return rtp_llm_ops.gemma4_top8_sum_bf16(
                weighted_output.view(num_tokens, top_k, hidden)
            )
        weighted_output = expert_output * sorted_w.unsqueeze(-1)
        weighted_output = weighted_output[inv_perm]
        return weighted_output.view(num_tokens, top_k, hidden).sum(dim=1).to(x.dtype)


# ---------------------------------------------------------------------------
# Torch reference FMHA implementation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Residual norm fusions preserve the original rounding boundaries. An unset
# flag uses validated shapes; explicit 0 disables them and 1 opts in elsewhere.
# ---------------------------------------------------------------------------


def _fused_residual_enabled() -> bool:
    import os

    return os.environ.get("GEMMA4_FUSED_RESIDUAL", "0") == "1"


def gemma4_norm_add(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    residual: torch.Tensor,
    eps: float = 1e-6,
    weight_float: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused RMSNorm(x) + residual with reference fallback."""
    if _fused_residual_enabled() or (
        os.environ.get("GEMMA4_FUSED_RESIDUAL") is None
        and x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[1] == 2816
        and 0 < x.shape[0] <= 8192
        and weight is not None
    ):
        from rtp_llm.models_py.modules.gemma4.norm_fusions import norm_add

        output = norm_add(
            x, weight if weight_float is None else weight_float, residual, eps
        )
        if output is not None:
            return output
    return gemma4_add_bf16(residual, gemma4_rms_norm(x, weight, eps, weight_float))


def gemma4_add_norm(
    x: torch.Tensor,
    other: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float = 1e-6,
    weight_float: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused RMSNorm(x + other) with reference fallback."""
    if _fused_residual_enabled() or (
        os.environ.get("GEMMA4_FUSED_RESIDUAL") is None
        and x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[1] == 2816
        and 0 < x.shape[0] <= 1024
        and weight is not None
    ):
        from rtp_llm.models_py.modules.gemma4.norm_fusions import add_norm

        output = add_norm(
            x, other, weight if weight_float is None else weight_float, eps
        )
        if output is not None:
            return output
    return gemma4_rms_norm(gemma4_add_bf16(x, other), weight, eps, weight_float)


def gemma4_residual_scale(
    x: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Fused (x + residual) * scale with reference fallback."""
    if (
        x.is_cuda
        and x.dtype == residual.dtype == scale.dtype == torch.bfloat16
        and x.device == residual.device == scale.device
        and x.shape == residual.shape
        and x.is_contiguous()
        and residual.is_contiguous()
        and scale.is_contiguous()
        and scale.numel() == 1
        and x.numel() % 8 == 0
        and torch.version.hip is None
    ):
        return rtp_llm_ops.gemma4_add_scale_bf16(residual, x, scale)
    if _fused_residual_enabled():
        from rtp_llm.models_py.modules.gemma4.norm_fusions import residual_scale

        output = residual_scale(x, residual, scale)
        if output is not None:
            return output
    return gemma4_add_bf16(x, residual) * scale


def gemma4_router_softmax(scores: torch.Tensor) -> torch.Tensor:
    """BF16 router softmax retaining the native 128-column reduction order."""
    if (
        scores.is_cuda
        and scores.dtype == torch.bfloat16
        and scores.dim() == 2
        and torch.version.hip is None
    ):
        from rtp_llm.models_py.modules.gemma4.router_softmax import router_probabilities

        output = router_probabilities(
            scores, variant="warp1" if scores.shape[0] <= 1536 else "warp4"
        )
        if output is not None:
            return output
    return torch.softmax(scores, dim=-1)
