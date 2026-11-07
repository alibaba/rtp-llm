"""V4.1 SWA KV numerical boundaries shared by fresh and cached attention.

RMSNorm and RoPE each materialize BF16 before group-32 MXFP8. Fresh
attention consumes the same decoded values as the SWA cache, including
rows that do not own a cache slot. The operation is independent of request
batching, prefix reuse, CP partitioning, and DSpark query/commit roles.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def norm_rope_bf16(values, weight, cos, sin, EPS: tl.constexpr):
    """One 512-channel KV row; preserve both reference BF16 boundaries."""
    values = values.to(tl.float32)
    inv = tl.rsqrt(tl.sum(values * values, 0) / 512 + EPS)
    normalized = (values * inv * weight.to(tl.float32)).to(tl.bfloat16)
    pair = tl.arange(0, 32)
    real = tl.gather(normalized, 448 + 2 * pair, 0).to(tl.float32)
    imag = tl.gather(normalized, 449 + 2 * pair, 0).to(tl.float32)
    # DI uses an explicit IEEE FMA after materializing the norm to BF16.
    real_out = tl.fma(real, cos, -(imag * sin))
    imag_out = tl.fma(real, sin, imag * cos)
    tail = tl.interleave(real_out, imag_out).to(tl.bfloat16)
    columns = tl.arange(0, 512)
    return tl.where(
        columns < 448,
        normalized,
        tl.gather(tail, tl.maximum(columns - 448, 0), 0),
    )


@triton.jit
def quantize_swa_row(values):
    """Return E4M3 data, UE8M0 exponent, and its BF16 decoded value."""
    groups = values.to(tl.float32).reshape((16, 32))
    maximum = tl.maximum(tl.max(tl.abs(groups), 1), 1e-4)
    # ceil(log2(maximum / 448)), without approximate log2 at power-of-two
    # boundaries. This is the reference's fast_round_scale operation.
    raw_scale = maximum * (1.0 / 448.0)
    bits = raw_scale.to(tl.uint32, bitcast=True)
    encoded = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    scale = (encoded << 23).to(tl.float32, bitcast=True)
    scaled = tl.clamp(groups / scale[:, None], -448.0, 448.0)
    payload = scaled.to(tl.float8e4nv)
    decoded = (payload.to(tl.float32) * scale[:, None]).to(tl.bfloat16)
    return payload.reshape((512,)), encoded.to(tl.uint8), decoded.reshape((512,))


@triton.jit
def _swa_kv_kernel(
    raw, weight, freqs, output, input_stride, freq_stride, EPS: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, 512)
    pair = tl.arange(0, 32)
    values = tl.load(raw + row * input_stride + columns)
    gamma = tl.load(weight + columns)
    cos = tl.load(freqs + row * freq_stride + 2 * pair)
    sin = tl.load(freqs + row * freq_stride + 2 * pair + 1)
    rounded = norm_rope_bf16(values, gamma, cos, sin, EPS)
    _, _, decoded = quantize_swa_row(rounded)
    tl.store(output + row * 512 + columns, decoded)


def swa_kv_norm_rope(
    raw: torch.Tensor,
    weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize, rotate, and MXFP8-roundtrip all fresh V4.1 SWA rows.

    Frequencies are already selected by actual absolute position. Strided
    views from merged Q/KV projection are accepted without copying the input.
    """
    assert raw.is_cuda and raw.dtype == torch.bfloat16
    assert raw.shape[-1] == 512 and raw.stride(-1) == 1 and rope_head_dim == 64
    assert weight.shape == (512,) and weight.device == raw.device
    flat = raw.view(-1, 512)
    freqs = torch.view_as_real(freqs_cis.reshape(-1, 32))
    assert freqs_cis.dtype == torch.complex64 and freqs_cis.device == raw.device
    assert freqs.shape[0] == flat.shape[0] and freqs.stride(1) == 2
    output = torch.empty(raw.shape, device=raw.device, dtype=raw.dtype)
    if flat.shape[0]:
        _swa_kv_kernel[(flat.shape[0],)](
            flat,
            weight,
            freqs,
            output,
            flat.stride(0),
            freqs.stride(0),
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
