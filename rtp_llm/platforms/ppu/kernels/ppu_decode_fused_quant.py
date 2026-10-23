"""PPU decode producers that retain the BF16 boundary before block-FP8 quant.

These kernels emit the existing E4M3/FP32 scale ABI. They do not use UE8M0
scales or change GEMM accumulation. Callers opt in and retain their old path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_block(value, AXIS: tl.constexpr = 0):
    # The separate producer writes BF16 before the v2 quantizer reads it.
    value = value.to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(value), AXIS), 1.0e-4)
    scale = amax / 448.0
    # The v2 CUDA kernel multiplies by a correctly rounded 448/amax.
    # Reciprocal(scale) and approximate division change FP8 tie rounding.
    inv_scale = tl.div_rn(448.0, amax)
    if AXIS == 1:
        inv_scale = inv_scale[:, None]
    quant = tl.minimum(tl.maximum(value * inv_scale, -448.0), 448.0)
    return quant, scale


@triton.jit
def _swiglu_fp8(
    gateup,
    payload,
    scales,
    M: tl.constexpr,
    H: tl.constexpr,
    LIMIT: tl.constexpr,
    COLUMN: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    col = group[:, None] * 128 + tl.arange(0, 128)[None, :]
    gate = tl.load(gateup + row * (2 * H) + col, col < H, 0).to(tl.float32)
    up = tl.load(gateup + row * (2 * H) + H + col, col < H, 0).to(tl.float32)
    if LIMIT > 0:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    value = (gate * tl.sigmoid(gate)) * up
    quant, scale = _quant_block(value, 1)
    tl.store(payload + row * H + col, quant, col < H)
    offset = group * M + row if COLUMN else row * (H // 128) + group
    tl.store(scales + offset, scale, group < H // 128)


def supports_swiglu_fp8(gate_up, quantization):
    return (
        gate_up.is_cuda
        and gate_up.dtype == torch.bfloat16
        and gate_up.ndim >= 2
        and gate_up.is_contiguous()
        and gate_up.shape[-1] > 0
        and gate_up.shape[-1] % 256 == 0
        and quantization in ("v2_row", "v2_column")
    )


def swiglu_fp8(gate_up, limit=0.0, *, quantization="v2_column"):
    """Return 2D FP8 payload and exact consumer-selected FP32 scale layout."""
    if not supports_swiglu_fp8(gate_up, quantization):
        raise ValueError("Fused SwiGLU quant requires contiguous BF16 and v2 scales")
    h = gate_up.shape[-1] // 2
    m = gate_up.numel() // (2 * h)
    payload = torch.empty((m, h), device=gate_up.device, dtype=torch.float8_e4m3fn)
    if quantization == "v2_column":
        scale = torch.empty(
            (h // 128, m), device=gate_up.device, dtype=torch.float32
        ).t()
    else:
        scale = torch.empty((m, h // 128), device=gate_up.device, dtype=torch.float32)
    if m:
        groups = 1 if m <= 8 else 4
        _swiglu_fp8[(m, triton.cdiv(h // 128, groups))](
            gate_up,
            payload,
            scale,
            m,
            h,
            float(limit),
            quantization == "v2_column",
            groups,
            num_warps=4,
        )
    return payload, scale


@triton.jit
def _inverse_rope_fp8_permute(
    x,
    freqs,
    payload,
    scales,
    M: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    G: tl.constexpr,
    PER_BATCH: tl.constexpr,
    GROUPS: tl.constexpr,
):
    token = tl.program_id(0)
    head_group = tl.program_id(1)
    quant_group = tl.program_id(2) * GROUPS + tl.arange(0, GROUPS)
    K: tl.constexpr = H // G * D
    col = quant_group[:, None] * 128 + tl.arange(0, 128)[None, :]
    head_col = col % D
    base = token * H * D + head_group * K
    value = tl.load(x + base + col, col < K, 0).to(tl.float32)
    is_rope = (head_col >= D - R) & (col < K)
    pair_col = col // 2 * 2
    a = tl.load(x + base + pair_col, is_rope, 0).to(tl.float32)
    b = tl.load(x + base + pair_col + 1, is_rope, 0).to(tl.float32)
    freq_token = token // S if PER_BATCH else token
    freq_col = (head_col - (D - R)) // 2 * 2
    c = tl.load(freqs + freq_token * R + freq_col, is_rope, 0)
    s = tl.load(freqs + freq_token * R + freq_col + 1, is_rope, 0)
    # Keep both rounded outputs of the old in-place inverse RoPE.
    rotated = tl.where(head_col % 2 == 0, a * c + b * s, b * c - a * s)
    value = tl.where(is_rope, rotated, value)
    quant, scale = _quant_block(value, 1)
    tl.store(payload + (head_group * M + token) * K + col, quant, col < K)
    tl.store(
        scales + (head_group * M + token) * (K // 128) + quant_group,
        scale,
        quant_group < K // 128,
    )


def supports_inverse_rope_fp8_permute(x, freqs, rope_dim, groups):
    from .ppu_inverse_rope import is_supported

    return (
        is_supported(x, freqs, rope_dim)
        and groups in (1, 2, 4, 8)
        and x.shape[2] % groups == 0
        and x.shape[-1] % 128 == 0
    )


def inverse_rope_fp8_permute(x, freqs, rope_dim, groups):
    """Produce contiguous [G,M,K] payload/scales for PPU fp8_bmm.

    The input is read-only. The old inverse RoPE mutation is not observable
    outside the output-projection producer/consumer boundary.
    """
    if not supports_inverse_rope_fp8_permute(x, freqs, rope_dim, groups):
        raise ValueError("Unsupported PPU inverse RoPE / block-FP8 layout")
    batch, seq, heads, dim = x.shape
    m = batch * seq
    k = heads // groups * dim
    payload = torch.empty((groups, m, k), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((groups, m, k // 128), device=x.device, dtype=torch.float32)
    if m:
        groups_per_cta = 4
        _inverse_rope_fp8_permute[(m, groups, triton.cdiv(k // 128, groups_per_cta))](
            x,
            torch.view_as_real(freqs),
            payload,
            scale,
            m,
            seq,
            heads,
            dim,
            rope_dim,
            groups,
            freqs.ndim == 2 and freqs.shape[0] == batch,
            groups_per_cta,
            num_warps=2,
            enable_fp_fusion=True,
        )
    return payload, scale
