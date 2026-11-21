"""Llama scaled RoPE shared by prefill and decode, using replayable metadata.

The coefficient definitions match rotary_position_embedding.h's Llama3Rope
and DynamicNTK. No process-global Base cache or capture-time sequence length
is used. Q/K may be strided views of fused QKV; V and non-rotary lanes stay intact.
"""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_offset(token, head, token_stride: tl.constexpr, head_stride: tl.constexpr):
    # Cast BEFORE multiplying: fused QKV views retain the entire row stride,
    # and long/batched prefill can cross INT32_MAX even with small head sizes.
    return tl.cast(token, tl.int64) * token_stride + tl.cast(head, tl.int64) * head_stride


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
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    mask = lane < DIM // 2
    pos = tl.load(Positions + token).to(tl.float32)
    base = tl.full((), BASE, tl.float32)
    if not LLAMA3 and ORIGINAL > 0:
        batch = tl.load(BatchIndices + token)
        length = tl.load(KvLengths + batch).to(tl.float32)
        factor = tl.maximum(SCALE * length / ORIGINAL - (SCALE - 1.0), 1.0)
        base = base * tl.exp(tl.log(factor) * (DIM / (DIM - 2.0)))
    inv_freq = tl.exp(-2.0 * lane / DIM * tl.log(base))
    if LLAMA3 and SCALE != 1.0:
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
        ptr = Q + _qk_offset(token, head, Q_TOKEN, Q_HEAD)
        x = tl.load(ptr + lane, mask=mask, other=0).to(tl.float32)
        y = tl.load(ptr + lane + DIM // 2, mask=mask, other=0).to(tl.float32)
        tl.store(ptr + lane, x * cosine - y * sine, mask=mask)
        tl.store(ptr + lane + DIM // 2, y * cosine + x * sine, mask=mask)
    else:
        ptr = K + _qk_offset(token, head - Q_HEADS, K_TOKEN, K_HEAD)
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
    if not llama3 and original_max_position > 0 and rotary_dim <= 2:
        raise ValueError("dynamic NTK requires rotary_dim > 2")
    # A style-only override has scale=1 and no original length. It is ordinary
    # RoPE. With a valid original length, DynamicNTK still extrapolates past
    # that threshold even when scale=1 (the native kernel has the same rule).
    if base <= 0 or scale <= 0 or (scale != 1.0 and original_max_position <= 0):
        raise ValueError(
            "scaled RoPE requires positive base, scale and original length"
        )
    if llama3 and scale != 1.0 and not 0 < low_freq_factor < high_freq_factor:
        raise ValueError("Llama3 RoPE requires 0 < low_freq_factor < high_freq_factor")
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        raise ValueError("scaled RoPE requires contiguous head dimensions")
    tokens = query.shape[0]
    if positions.numel() < tokens or batch_indices.numel() < tokens:
        raise ValueError("scaled RoPE token metadata is too short")
    if tokens == 0:
        return
    if not llama3 and original_max_position > 0 and kv_lengths.numel() == 0:
        raise ValueError("dynamic NTK requires per-request KV lengths")
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


def scaled_rope_qkv(
    qkv: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_config,
    llama3: bool,
    positions: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    prefix_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate a copy before a native layout/KV-write op with RoPE disabled.

    Decode lengths are history lengths, so the current token contributes +1.
    Prefill lengths come from each request, never the padded batch maximum.
    Explicit positions change angles, not physical KV slots or NTK lengths.
    All data-dependent operations stay on device and are replayed in a graph;
    no tensor values become capture-time Python scalars.
    """
    tokens = qkv.shape[0]
    if qkv.ndim != 2 or qkv.shape[1] != (num_heads + 2 * num_kv_heads) * head_size:
        raise ValueError("scaled RoPE requires packed [tokens, QKV hidden] input")
    if (sequence_lengths is None) == (cu_seqlens is None):
        raise ValueError("provide either decode lengths or prefill cu_seqlens")

    def on_device(tensor):
        # Engine metadata is CUDA or pinned CPU. Keep captured host copies
        # tied to their stable source; never silently capture a pageable copy.
        if qkv.is_cuda and not tensor.is_cuda:
            with torch.cuda.device(qkv.device):
                if torch.cuda.is_current_stream_capturing() and not tensor.is_pinned():
                    raise ValueError("graph RoPE metadata must be CUDA or pinned CPU")
        return tensor.to(device=qkv.device, non_blocking=True)

    token = torch.arange(tokens, device=qkv.device, dtype=torch.int64)
    if sequence_lengths is not None:
        if sequence_lengths.numel() != tokens:
            raise ValueError("decode requires one history length per token")
        history = on_device(sequence_lengths)
        batch_indices, kv_lengths = token, history + 1
        default_positions = history
    else:
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("prefill requires a non-empty cumulative-length table")
        cu = on_device(cu_seqlens)
        batch_indices = torch.searchsorted(cu[1:], token, right=True)
        if prefix_lengths is None or prefix_lengths.numel() == 0:
            prefixes = torch.zeros_like(cu[:-1])
        else:
            if prefix_lengths.numel() != cu.numel() - 1:
                raise ValueError("prefill requires one prefix length per request")
            prefixes = on_device(prefix_lengths)
        kv_lengths = cu[1:] - cu[:-1] + prefixes
        default_positions = token - cu[batch_indices] + prefixes[batch_indices]
    if positions is None or positions.numel() == 0:
        position_ids = default_positions
    else:
        stride = rope_config.index_factor
        if stride <= 0 or positions.numel() < tokens * stride:
            raise ValueError("scaled RoPE position IDs do not match token count/stride")
        # Graph metadata can reserve more capacity than this invocation uses.
        position_ids = (
            on_device(positions)
            .flatten()[: tokens * stride]
            .reshape(tokens, stride)[:, 0]
            .contiguous()
        )

    rotated = qkv.clone()
    packed = rotated.view(tokens, num_heads + 2 * num_kv_heads, head_size)
    scaled_rope_qk_inplace(
        packed[:, :num_heads],
        packed[:, num_heads : num_heads + num_kv_heads],
        position_ids,
        batch_indices,
        kv_lengths,
        rotary_dim=rope_config.dim,
        base=rope_config.base,
        scale=rope_config.scale,
        original_max_position=rope_config.max_pos,
        llama3=llama3,
        low_freq_factor=rope_config.factor1,
        high_freq_factor=rope_config.factor2,
    )
    return rotated
