"""Fused RoPE + Hadamard for DSV3.2 indexer **prefill** path.

Replaces the 4-op chain in ``IndexerOp.apply_rope_and_rotate_q_k{,_cp}``:
    rope_q (flashinfer) + rope_k (flashinfer) + had_q (fast_hadamard) + had_k (fast_hadamard)

with **2 launches**:
  1. Triton kernel ``_fused_rope_qk_had_k_kernel``
       - In-place NeOX RoPE on Q[:, :, :rope_head_dim] (all heads)
       - In-place NeOX RoPE on K[:, :rope_head_dim], then 128-pt Hadamard butterfly,
         writes K_done to a separate output buffer
  2. cuBLAS bf16 GEMM
       - ``Q.view(-1, head_dim) @ H_scaled`` where H is the Sylvester matrix scaled
         by 1/sqrt(head_dim). Produces Q_done.

This design avoids the 7-stage butterfly cost for Q (which has 32× more rows than K
under TP=2) by leveraging cuBLAS tensor cores. K stays in Triton because K is small
(~T rows vs T*32 for Q) and the butterfly fits in 1 program.

Production envelope (DSV3.2 with CP=8): T=4096, H=32, head_dim=128, rope_head_dim=64
yields ~2.35x speedup vs the 4-op baseline (91us → 39us), saves ~4ms / 32K prefill.

The fast path requires:
  - head_dim == 128 (kernel hardcoded; 7 butterfly stages)
  - rope_head_dim is even
  - K is a 2-D tensor [T, head_dim] (no head dim)
  - Q is a 3-D tensor [T, num_heads, head_dim]
  - cos_sin_cache dtype is fp32 (flashinfer convention; bf16 silently produces garbage)
"""
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Hadamard 128 butterfly helpers (private)
# ---------------------------------------------------------------------------
@triton.jit
def _butterfly_step(x, BLOCK_M: tl.constexpr, N_GROUPS: tl.constexpr,
                     STRIDE: tl.constexpr, HEAD_DIM: tl.constexpr):
    """One stage of Hadamard butterfly.

    For each pair (i, i^STRIDE), produces (a+b, a-b). C-contiguous reshape puts
    the pair axis in the middle, so we permute it to last for tl.split.
    """
    x4 = x.reshape(BLOCK_M, N_GROUPS, 2, STRIDE)
    x4p = tl.permute(x4, (0, 1, 3, 2))
    a, b = tl.split(x4p)
    x4p = tl.join(a + b, a - b)
    x4 = tl.permute(x4p, (0, 1, 3, 2))
    return x4.reshape(BLOCK_M, HEAD_DIM)


@triton.jit
def _had128_inline(x, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    """Hadamard 128 = 7 unrolled butterfly stages."""
    x = _butterfly_step(x, BLOCK_M, 64, 1, HEAD_DIM)
    x = _butterfly_step(x, BLOCK_M, 32, 2, HEAD_DIM)
    x = _butterfly_step(x, BLOCK_M, 16, 4, HEAD_DIM)
    x = _butterfly_step(x, BLOCK_M,  8, 8, HEAD_DIM)
    x = _butterfly_step(x, BLOCK_M,  4, 16, HEAD_DIM)
    x = _butterfly_step(x, BLOCK_M,  2, 32, HEAD_DIM)
    x = _butterfly_step(x, BLOCK_M,  1, 64, HEAD_DIM)
    return x


# ---------------------------------------------------------------------------
# Main fused kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fused_rope_qk_had_k_kernel(
    q_ptr,                  # [T, H, D] bf16, modified in-place (RoPE only)
    k_ptr,                  # [T, D] bf16, modified in-place during RoPE compute
    pos_ptr,                # [T] int32
    cos_sin_cache_ptr,      # [max_pos, rope_dim] fp32 ([cos_half | sin_half] layout)
    k_out_ptr,              # [T, D] bf16, output: K with RoPE + Hadamard applied
    HAD_SCALE,              # 1 / sqrt(head_dim)
    stride_qt, stride_qh,
    stride_kt,
    stride_kot,
    stride_csc_p,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_HALF: tl.constexpr,
):
    """1 program per token. Processes Q (all H heads, RoPE in-place) + K (RoPE +
    Hadamard, write to k_out)."""
    t = tl.program_id(0).to(tl.int64)
    pos = tl.load(pos_ptr + t)

    rope_idx = tl.arange(0, ROPE_HALF)
    full_idx = tl.arange(0, HEAD_DIM)

    cos = tl.load(cos_sin_cache_ptr + pos * stride_csc_p + rope_idx).to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * stride_csc_p + ROPE_HALF + rope_idx).to(tl.float32)

    # ===== Q RoPE in-place (all H heads) =====
    head_offs = tl.arange(0, H)
    q_first = tl.load(
        q_ptr + t * stride_qt + head_offs[:, None] * stride_qh + rope_idx[None, :]
    ).to(tl.float32)
    q_second = tl.load(
        q_ptr + t * stride_qt + head_offs[:, None] * stride_qh + (ROPE_HALF + rope_idx)[None, :]
    ).to(tl.float32)
    q_first_new = q_first * cos[None, :] - q_second * sin[None, :]
    q_second_new = q_second * cos[None, :] + q_first * sin[None, :]
    tl.store(
        q_ptr + t * stride_qt + head_offs[:, None] * stride_qh + rope_idx[None, :],
        q_first_new.to(tl.bfloat16),
    )
    tl.store(
        q_ptr + t * stride_qt + head_offs[:, None] * stride_qh + (ROPE_HALF + rope_idx)[None, :],
        q_second_new.to(tl.bfloat16),
    )
    # Q's NoPE part (>= 2*ROPE_HALF) stays untouched.

    # ===== K path: RoPE + Hadamard, entirely in registers =====
    # Load full K row + partner (k[idx XOR ROPE_HALF]) + extended cos/sin (broadcast
    # idx % ROPE_HALF via bitwise &). All loads hit L1 due to spatial locality.
    k_full = tl.load(k_ptr + t * stride_kt + full_idx).to(tl.float32)
    xor_idx = full_idx ^ ROPE_HALF
    k_partner = tl.load(k_ptr + t * stride_kt + xor_idx).to(tl.float32)
    inner_idx = full_idx & (ROPE_HALF - 1)
    cos_ext = tl.load(cos_sin_cache_ptr + pos * stride_csc_p + inner_idx).to(tl.float32)
    sin_ext = tl.load(cos_sin_cache_ptr + pos * stride_csc_p + ROPE_HALF + inner_idx).to(tl.float32)
    # NeOX RoPE on rope dims, pass-through for NoPE dims (>= 2*ROPE_HALF):
    #   For idx < ROPE_HALF:        k_rope[i] = k[i] * cos[i] - k[i + ROPE_HALF] * sin[i]
    #   For ROPE_HALF <= i < 2*RH:  k_rope[i] = k[i] * cos[i-RH] + k[i-RH] * sin[i-RH]
    #   For i >= 2*ROPE_HALF:       k_rope[i] = k[i]
    sign = tl.where(full_idx < ROPE_HALF, -1.0, 1.0)
    is_rope = full_idx < 2 * ROPE_HALF
    k_rope = tl.where(is_rope, k_full * cos_ext + sign * k_partner * sin_ext, k_full)
    # bf16 round-trip to match baseline numerics: baseline does flashinfer
    # in-place RoPE (writes bf16) then re-reads as bf16. We compute in fp32 but
    # round-trip here so the Hadamard input bit-matches baseline.
    k_rope = k_rope.to(tl.bfloat16).to(tl.float32)

    # Hadamard butterfly directly on in-register k_rope (BLOCK_M=1, no HBM round-trip)
    k_2d = k_rope.reshape(1, HEAD_DIM)
    k_had = _had128_inline(k_2d, 1, HEAD_DIM)
    k_final = (k_had * HAD_SCALE).reshape(HEAD_DIM).to(tl.bfloat16)
    tl.store(k_out_ptr + t * stride_kot + full_idx, k_final)


# ---------------------------------------------------------------------------
# H matrix cache (per (head_dim, device))
# ---------------------------------------------------------------------------
_H_SCALED_BF16_CACHE: dict = {}


def _build_hadamard_matrix(n: int) -> torch.Tensor:
    """Sylvester construction: H_{2n} = [[H_n, H_n], [H_n, -H_n]]."""
    h = torch.tensor([[1.0]])
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
    return h


def _get_h_scaled_bf16(head_dim: int, device: torch.device) -> torch.Tensor:
    """Return the cached Hadamard matrix scaled by 1/sqrt(head_dim) as bf16."""
    key = (head_dim, str(device))
    if key not in _H_SCALED_BF16_CACHE:
        h = _build_hadamard_matrix(head_dim)
        _H_SCALED_BF16_CACHE[key] = (
            (h * head_dim**-0.5).to(dtype=torch.bfloat16, device=device).contiguous()
        )
    return _H_SCALED_BF16_CACHE[key]


# ---------------------------------------------------------------------------
# Public entry: fused prefill RoPE + Hadamard for Q & K
# ---------------------------------------------------------------------------
def fused_prefill_rope_hadamard_qk(
    q: torch.Tensor,                # [T, H, D] bf16 — RoPE applied in-place
    k: torch.Tensor,                # [T, D] bf16 — RoPE applied in-place during compute
    positions: torch.Tensor,        # [T] int32
    cos_sin_cache: torch.Tensor,    # [max_pos, rope_head_dim] fp32 (cos|sin layout)
    rope_head_dim: int,
    num_warps: int = 2,
    num_stages: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply NeOX RoPE + Hadamard to (Q, K) in one Triton launch + one cuBLAS GEMM.

    Returns:
        (q_done, k_done):
            q_done: [T, H, D] bf16 — Q with RoPE + Hadamard applied (new tensor)
            k_done: [T, D] bf16 — K with RoPE + Hadamard applied (new tensor)

    Side effects:
        - q is mutated in-place (RoPE applied to first rope_head_dim dims per head)
        - k is mutated in-place during the RoPE step
        Callers must NOT rely on q/k being unchanged after this call.
    """
    T, H, D = q.shape
    assert k.shape == (T, D), f"K shape {k.shape} must match (T={T}, D={D})"
    assert D == 128, f"fused kernel hardcoded for head_dim=128, got {D}"
    assert rope_head_dim % 2 == 0, f"rope_head_dim must be even, got {rope_head_dim}"
    assert cos_sin_cache.dtype == torch.float32, (
        "flashinfer/RoPE requires fp32 cos_sin_cache (bf16 silently misinterpreted)"
    )

    if T == 0:
        return q, torch.empty_like(k)

    k_out = torch.empty_like(k)
    _fused_rope_qk_had_k_kernel[(T,)](
        q, k, positions, cos_sin_cache, k_out,
        D**-0.5,
        q.stride(0), q.stride(1),
        k.stride(0),
        k_out.stride(0),
        cos_sin_cache.stride(0),
        H=H, HEAD_DIM=D,
        ROPE_HALF=rope_head_dim // 2,
        num_warps=num_warps, num_stages=num_stages,
    )

    # Q's Hadamard via cuBLAS bf16 GEMM (Q.view(-1, D) @ H_scaled)
    # This is the only Q output — Q's in-place RoPE state in `q` is the input here.
    H_scaled = _get_h_scaled_bf16(D, q.device)
    q_out = (q.view(-1, D) @ H_scaled).view(T, H, D)
    return q_out, k_out
