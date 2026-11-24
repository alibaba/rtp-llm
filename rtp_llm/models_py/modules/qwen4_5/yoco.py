"""YoCo cross-layer KV / GDN sharing plan and decode readout for Qwen 4.5-Flash.

SSM State Convention (shared across ``yoco.py``, ``gdn.py``, and ``contracts.py``):
    All Gated DeltaNet recurrent SSM states use layout:
        ``[B, H_v, D_k, D_v]`` in ``torch.float32``
    where for Qwen 4.5-Flash: ``H_v = 128``, ``D_k = 64``, ``D_v = 64``.

    With state ``S[b, h_v, k, v]``:
      - State prediction: ``v_pred[b, h_v, v] = sum_k S[b, h_v, k, v] * k_t[b, h_k, k]``
      - State update:     ``S[b, h_v, k, v] += k_t[b, h_k, k] * delta[b, h_v, v]``
      - Readout:          ``o[b, t, h_v, v] = (sum_k S[b, h_v, k, v] * q_t[b, t, h_k, k]) * (D_k ** -0.5)``
        which is ``einsum("bhkv,bthk->bthv", S, q) * (D_k ** -0.5)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_5.contracts import (
    Qwen4_5ArchConfig,
    SharedGDNInputs,
)

SSM_STATE_LAYOUT = "(batch, num_v_heads, head_k_dim, head_v_dim)"


@dataclass(frozen=True)
class YoCoSharingPlan:
    """Resolved 0-based YoCo cross-layer attention sharing topology.

    For Qwen 4.5-Flash (40 layers, 1-based ``share_map``):
      - Full Attention source: layer 19 -> consumers (23, 27, 31, 35, 39) [5 layers]
      - GDN Linear Attention source: layer 20 -> consumers
        (21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38) [14 layers]
    """

    consumer_to_source: Dict[int, int]
    source_to_consumers: Dict[int, Tuple[int, ...]]
    full_attn_sources: Tuple[int, ...]
    full_attn_consumers: Tuple[int, ...]
    gdn_sources: Tuple[int, ...]
    gdn_consumers: Tuple[int, ...]

    @classmethod
    def from_config(cls, config: Qwen4_5ArchConfig) -> "YoCoSharingPlan":
        c2s = config.get_attention_sharing()
        s2c_lists: Dict[int, list[int]] = {}
        for consumer, source in sorted(c2s.items()):
            s2c_lists.setdefault(source, []).append(consumer)
        s2c = {src: tuple(dsts) for src, dsts in sorted(s2c_lists.items())}

        full_sources = tuple(
            src for src in sorted(s2c.keys()) if config.layer_types[src] == "full_attention"
        )
        full_consumers = tuple(
            dst for dst in sorted(c2s.keys()) if config.layer_types[dst] == "full_attention"
        )
        gdn_sources = tuple(
            src for src in sorted(s2c.keys()) if config.layer_types[src] == "linear_attention"
        )
        gdn_consumers = tuple(
            dst for dst in sorted(c2s.keys()) if config.layer_types[dst] == "linear_attention"
        )
        return cls(
            consumer_to_source=dict(c2s),
            source_to_consumers=s2c,
            full_attn_sources=full_sources,
            full_attn_consumers=full_consumers,
            gdn_sources=gdn_sources,
            gdn_consumers=gdn_consumers,
        )

    def is_consumer(self, layer_idx: int) -> bool:
        return layer_idx in self.consumer_to_source

    def is_source(self, layer_idx: int) -> bool:
        return layer_idx in self.source_to_consumers

    def is_full_attn_consumer(self, layer_idx: int) -> bool:
        return layer_idx in self.full_attn_consumers

    def is_full_attn_source(self, layer_idx: int) -> bool:
        return layer_idx in self.full_attn_sources

    def is_gdn_consumer(self, layer_idx: int) -> bool:
        return layer_idx in self.gdn_consumers

    def is_gdn_source(self, layer_idx: int) -> bool:
        return layer_idx in self.gdn_sources

    def source_for(self, layer_idx: int) -> Optional[int]:
        return self.consumer_to_source.get(layer_idx)

    def consumers_of(self, layer_idx: int) -> Tuple[int, ...]:
        return self.source_to_consumers.get(layer_idx, ())

    def kv_cache_owner(self, layer_idx: int) -> int:
        """Return the layer index owning the KV cache used by ``layer_idx``."""
        return self.consumer_to_source.get(layer_idx, layer_idx)


@triton.jit
def _yoco_gdn_decode_readout_kernel(
    Q_ptr,
    S_ptr,
    Out_ptr,
    stride_qb,
    stride_qt,
    stride_qhk,
    stride_qd,
    stride_sb,
    stride_shv,
    stride_sk,
    stride_sv,
    stride_ob,
    stride_ot,
    stride_ohv,
    stride_od,
    T,
    scale,
    eps,
    GQA_RATIO: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
):
    """Fused L2-norm(q) + GQA broadcast + FP32 matvec against shared SSM state [B, H_v, D_k, D_v]."""
    bt = tl.program_id(0)
    h_v = tl.program_id(1)
    b = bt // T
    t = bt % T
    h_k = h_v // GQA_RATIO

    offs_k = tl.arange(0, D_K)
    offs_v = tl.arange(0, D_V)

    q_ptrs = Q_ptr + b * stride_qb + t * stride_qt + h_k * stride_qhk + offs_k * stride_qd
    q = tl.load(q_ptrs).to(tl.float32)
    q_inv_norm = tl.rsqrt(tl.sum(q * q, axis=0) + eps)
    q_scaled = q * (q_inv_norm * scale)

    s_ptrs = (
        S_ptr
        + b * stride_sb
        + h_v * stride_shv
        + offs_k[:, None] * stride_sk
        + offs_v[None, :] * stride_sv
    )
    s = tl.load(s_ptrs).to(tl.float32)

    out = tl.sum(s * q_scaled[:, None], axis=0)
    out_ptrs = Out_ptr + b * stride_ob + t * stride_ot + h_v * stride_ohv + offs_v * stride_od
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty))


def yoco_gdn_decode_readout_ref(
    q_pre_l2: torch.Tensor,
    shared_ssm_state: torch.Tensor,
    head_k_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference PyTorch FP32 implementation of YoCo GDN decode readout.

    Args:
        q_pre_l2: Query tensor post-conv, pre-L2-norm, shape ``[B, T, H_k, D_k]``.
        shared_ssm_state: Source layer post-update SSM state, shape ``[B, H_v, D_k, D_v]`` (FP32).
        head_k_dim: Key head dimension ``D_k`` (64 for Qwen 4.5-Flash).
        eps: L2-normalization epsilon (1e-6).

    Returns:
        Core attention readout tensor of shape ``[B, T, H_v, D_v]`` in ``q_pre_l2.dtype``.
    """
    if q_pre_l2.ndim != 4:
        raise ValueError(f"Expected q_pre_l2 [B, T, H_k, D_k], got {tuple(q_pre_l2.shape)}")
    if shared_ssm_state.ndim != 4:
        raise ValueError(
            f"Expected shared_ssm_state [B, H_v, D_k, D_v], got {tuple(shared_ssm_state.shape)}"
        )
    bsz, _, num_k_heads, d_k = q_pre_l2.shape
    s_bsz, num_v_heads, s_dk, _ = shared_ssm_state.shape
    if bsz != s_bsz or d_k != head_k_dim or s_dk != head_k_dim:
        raise ValueError(
            f"Shape mismatch: q_pre_l2={tuple(q_pre_l2.shape)}, "
            f"shared_ssm_state={tuple(shared_ssm_state.shape)}, head_k_dim={head_k_dim}"
        )
    if num_v_heads % num_k_heads != 0:
        raise ValueError(f"H_v ({num_v_heads}) must be divisible by H_k ({num_k_heads})")

    q_f = q_pre_l2.float()
    q_norm = q_f * torch.rsqrt(q_f.square().sum(dim=-1, keepdim=True) + eps)
    gqa_ratio = num_v_heads // num_k_heads
    if gqa_ratio > 1:
        q_norm = q_norm.repeat_interleave(gqa_ratio, dim=-2)
    core = torch.einsum("bhkv,bthk->bthv", shared_ssm_state.float(), q_norm) * (head_k_dim ** -0.5)
    return core.to(q_pre_l2.dtype)


def yoco_gdn_decode_readout(
    q_pre_l2: torch.Tensor,
    shared_ssm_state: torch.Tensor,
    head_k_dim: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fast YoCo GDN decode readout against source layer FP32 SSM state.

    Computes:
        ``q = L2Norm(q_pre_l2, eps=1e-6)`` (broadcast 4:1 across ``H_v // H_k``)
        ``core = einsum("bhkv,bthk->bthv", shared_ssm_state.float(), q) * (head_k_dim ** -0.5)``

    Uses a fused single-pass Triton kernel on CUDA when ``D_k`` and ``D_v`` are powers of 2 <= 128,
    avoiding intermediate ``repeat_interleave`` allocations.
    """
    if (
        not q_pre_l2.is_cuda
        or not shared_ssm_state.is_cuda
        or q_pre_l2.ndim != 4
        or shared_ssm_state.ndim != 4
    ):
        return yoco_gdn_decode_readout_ref(q_pre_l2, shared_ssm_state, head_k_dim, eps=eps)

    bsz, seq_len, num_k_heads, d_k = q_pre_l2.shape
    s_bsz, num_v_heads, s_dk, d_v = shared_ssm_state.shape
    if (
        bsz != s_bsz
        or d_k != head_k_dim
        or s_dk != head_k_dim
        or num_v_heads % num_k_heads != 0
        or (d_k & (d_k - 1)) != 0
        or (d_v & (d_v - 1)) != 0
        or d_k > 128
        or d_v > 128
    ):
        return yoco_gdn_decode_readout_ref(q_pre_l2, shared_ssm_state, head_k_dim, eps=eps)

    gqa_ratio = num_v_heads // num_k_heads
    out = torch.empty(
        (bsz, seq_len, num_v_heads, d_v),
        dtype=q_pre_l2.dtype,
        device=q_pre_l2.device,
    )
    grid = (bsz * seq_len, num_v_heads)
    _yoco_gdn_decode_readout_kernel[grid](
        q_pre_l2,
        shared_ssm_state,
        out,
        q_pre_l2.stride(0),
        q_pre_l2.stride(1),
        q_pre_l2.stride(2),
        q_pre_l2.stride(3),
        shared_ssm_state.stride(0),
        shared_ssm_state.stride(1),
        shared_ssm_state.stride(2),
        shared_ssm_state.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        seq_len,
        float(head_k_dim ** -0.5),
        float(eps),
        GQA_RATIO=gqa_ratio,
        D_K=d_k,
        D_V=d_v,
        num_warps=4,
    )
    return out


__all__ = [
    "SSM_STATE_LAYOUT",
    "SharedGDNInputs",
    "YoCoSharingPlan",
    "yoco_gdn_decode_readout",
    "yoco_gdn_decode_readout_ref",
]
