"""In-place BF16 inverse RoPE without full-size FP32 intermediates."""

import torch
import triton
import triton.language as tl


@triton.jit
def _inverse_rope(
    x,
    freqs,
    PAIRS: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    PER_BATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    half: tl.constexpr = R // 2
    token = pair // (H * half)
    head = (pair // half) % H
    channel = pair % half
    offset = (token * H + head) * D + D - R + 2 * channel
    freq_token = token // S if PER_BATCH else token
    freq_offset = freq_token * R + 2 * channel
    a = tl.load(x + offset, pair < PAIRS, 0).to(tl.float32)
    b = tl.load(x + offset + 1, pair < PAIRS, 0).to(tl.float32)
    c = tl.load(freqs + freq_offset, pair < PAIRS, 0)
    s = tl.load(freqs + freq_offset + 1, pair < PAIRS, 0)
    tl.store(x + offset, a * c + b * s, pair < PAIRS)
    tl.store(x + offset + 1, b * c - a * s, pair < PAIRS)


def is_supported(x, freqs, rope_dim):
    if (
        x.ndim != 4
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or freqs.dtype != torch.complex64
        or freqs.device != x.device
        or not freqs.is_contiguous()
        or freqs.is_conj()
        or freqs.ndim < 2
        or rope_dim <= 0
        or rope_dim % 2
        or rope_dim > x.shape[-1]
        or freqs.shape[-1] != rope_dim // 2
    ):
        return False
    batch, seq = x.shape[:2]
    per_batch = freqs.ndim == 2 and freqs.shape[0] == batch
    return (
        per_batch or freqs.numel() == batch * seq * (rope_dim // 2)
    ) and torch.cuda.get_device_name(x.device) == "ZW-M890P"


def inverse_rope_inplace(x, freqs, rope_dim):
    if not is_supported(x, freqs, rope_dim):
        raise ValueError(
            "PPU inverse RoPE requires contiguous BF16 [B,S,H,D] and complex64 frequencies"
        )
    if not x.numel() or not freqs.numel():
        return x
    batch, seq, heads, dim = x.shape
    per_batch = freqs.ndim == 2 and freqs.shape[0] == batch
    pairs = batch * seq * heads * (rope_dim // 2)
    _inverse_rope[(triton.cdiv(pairs, 256),)](
        x,
        torch.view_as_real(freqs),
        pairs,
        seq,
        heads,
        dim,
        rope_dim,
        per_batch,
        256,
        num_warps=4,
        # Match the existing GPU complex multiplication's fused arithmetic;
        # splitting the products changes BF16 rounding near cancellation.
        enable_fp_fusion=True,
    )
    return x
