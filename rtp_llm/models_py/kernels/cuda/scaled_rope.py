"""Llama scaled RoPE for FlashInfer prefill, using its replayable metadata.

The coefficient definitions match rotary_position_embedding.h's Llama3Rope
and DynamicNTK. No process-global Base cache or capture-time sequence length
is used. Q/K may be strided views of fused QKV; V and non-rotary lanes stay intact.
"""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl


@triton.jit
def _scaled_rope(
    Q,
    K,
    Positions,
    BatchIndices,
    KvLengths,
    Q_TOKEN: tl.constexpr,
    Q_HEAD: tl.constexpr,
    K_TOKEN: tl.constexpr,
    K_HEAD: tl.constexpr,
    Q_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BASE: tl.constexpr,
    SCALE: tl.constexpr,
    ORIGINAL: tl.constexpr,
    LLAMA3: tl.constexpr,
    LOW: tl.constexpr,
    HIGH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    mask = lane < DIM // 2
    pos = tl.load(Positions + token).to(tl.float32)
    base = tl.full((), BASE, tl.float32)
    if not LLAMA3:
        batch = tl.load(BatchIndices + token)
        length = tl.load(KvLengths + batch).to(tl.float32)
        factor = tl.maximum(SCALE * length / ORIGINAL - (SCALE - 1.0), 1.0)
        base = base * tl.exp(tl.log(factor) * (DIM / (DIM - 2.0)))
    inv_freq = tl.exp(-2.0 * lane / DIM * tl.log(base))
    if LLAMA3:
        wavelength = 6.283185307179586 / inv_freq
        smooth = (ORIGINAL / wavelength - LOW) / (HIGH - LOW)
        interpolated = (1.0 - smooth) * inv_freq / SCALE + smooth * inv_freq
        inv_freq = tl.where(
            wavelength > ORIGINAL / LOW,
            inv_freq / SCALE,
            tl.where(wavelength < ORIGINAL / HIGH, inv_freq, interpolated),
        )
    angle = pos * inv_freq
    cosine, sine = tl.cos(angle), tl.sin(angle)
    if head < Q_HEADS:
        ptr = Q + token * Q_TOKEN + head * Q_HEAD
        x = tl.load(ptr + lane, mask=mask, other=0).to(tl.float32)
        y = tl.load(ptr + lane + DIM // 2, mask=mask, other=0).to(tl.float32)
        tl.store(ptr + lane, x * cosine - y * sine, mask=mask)
        tl.store(ptr + lane + DIM // 2, y * cosine + x * sine, mask=mask)
    else:
        ptr = K + token * K_TOKEN + (head - Q_HEADS) * K_HEAD
        x = tl.load(ptr + lane, mask=mask, other=0).to(tl.float32)
        y = tl.load(ptr + lane + DIM // 2, mask=mask, other=0).to(tl.float32)
        tl.store(ptr + lane, x * cosine - y * sine, mask=mask)
        tl.store(ptr + lane + DIM // 2, y * cosine + x * sine, mask=mask)


def scaled_rope_qk_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    batch_indices: torch.Tensor,
    kv_lengths: torch.Tensor,
    *,
    rotary_dim: int,
    base: float,
    scale: float,
    original_max_position: int,
    llama3: bool,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
) -> None:
    if query.ndim != 3 or key.ndim != 3 or query.shape[::2] != key.shape[::2]:
        raise ValueError("scaled RoPE requires matching [tokens, heads, head_dim] Q/K")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > query.shape[-1]:
        raise ValueError(
            "scaled RoPE requires a positive even rotary_dim within the head"
        )
    if not llama3 and rotary_dim <= 2:
        raise ValueError("dynamic NTK requires rotary_dim > 2")
    if base <= 0 or scale <= 0 or original_max_position <= 0:
        raise ValueError(
            "scaled RoPE requires positive base, scale and original length"
        )
    if llama3 and not 0 < low_freq_factor < high_freq_factor:
        raise ValueError("Llama3 RoPE requires 0 < low_freq_factor < high_freq_factor")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("scaled RoPE requires contiguous head dimensions")
    tokens = query.shape[0]
    if positions.numel() < tokens or batch_indices.numel() < tokens:
        raise ValueError("scaled RoPE token metadata is too short")
    if tokens == 0:
        return
    # Select the input device, not whichever GPU the caller left current.
    # The CPU branch is only useful for Triton's numerical interpreter tests.
    with torch.cuda.device(query.device) if query.is_cuda else nullcontext():
        _scaled_rope[(tokens, query.shape[1] + key.shape[1])](
            query,
            key,
            positions,
            batch_indices,
            kv_lengths,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            query.shape[1],
            rotary_dim,
            float(base),
            float(scale),
            original_max_position,
            llama3,
            float(low_freq_factor),
            float(high_freq_factor),
            triton.next_power_of_2(rotary_dim // 2),
            num_warps=4,
            enable_fp_fusion=False,
        )
