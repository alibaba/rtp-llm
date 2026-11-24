"""MXFP4 quantization, dequantization, and B300 (SM103) Triton ``tl.dot_scaled``
Grouped MoE GEMM engine for Qwen 4.5-Flash (1109-bf16-re-mxfp4-ple-fp8).

Checkpoint format (OCP MXFP4 / PyTorch ``float4_e2m1fn_x2``):
  - Weight tensor: ``[..., N, K // 2]`` ``uint8``
      Each byte packs two FP4 E2M1 nibbles along K:
        - low nibble  (``byte & 0x0F``)      -> even K index ``2 * j``
        - high nibble (``(byte >> 4) & 0x0F``) -> odd  K index ``2 * j + 1``
      Unsigned magnitude LUT (bits 0..2): ``[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]``
      Sign bit (bit 3): ``1`` -> negate.
  - Scale tensor: ``[..., N, K // 32]`` ``uint8`` (UE8M0)
      Exponent-only scale: ``2^(scale_u8 - 127)`` in FP32 for ``0..254``; ``255`` is NaN.

Hardware path on B300 (SM103, Triton 3.6+):
  - Uses native ``tl.dot_scaled(a, None, "bf16", b, b_scale, "e2m1")`` (W4A16) by
    default — empirically both faster (1.5x-3.0x on MoE token-per-expert counts
    m_e=1..32) and far more accurate (zero activation quantization loss, exact
    0.0 error on integer-BF16 x E2M1) than W4A8 ``e4m3 x e2m1``.
  - Optional ``compute_mode="e4m3"`` path is also provided for benchmarking.
  - All inter-expert pointer offsets use ``tl.int64`` scalar arithmetic outside
    the K-loop (mandatory because ``640 * 5120 * 1280 = 4.19 GB > INT32_MAX``),
    while keeping intra-tile offsets in ``tl.int32`` for 128-bit coalesced loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_5.activation import (
    situ_and_mul_ref,
    situ_and_mul_triton,
)

MXFP4_BLOCK_SIZE = 32

_E2M1_LUT_CPU = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
_E2M1_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_e2m1_lut(device: torch.device) -> torch.Tensor:
    lut = _E2M1_LUT_CACHE.get(device)
    if lut is None:
        lut = _E2M1_LUT_CPU.to(device=device)
        _E2M1_LUT_CACHE[device] = lut
    return lut


def supports_triton_mxfp4(device: torch.device) -> bool:
    """Return True if ``tl.dot_scaled`` MXFP4 hardware/Triton execution is supported."""
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 9 and hasattr(tl, "dot_scaled")


# ---------------------------------------------------------------------------
# MXFP4 Quantize & Dequantize (Reference + Triton)
# ---------------------------------------------------------------------------


def quantize_mxfp4(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize floating-point tensor ``[..., N, K]`` into MXFP4 ``(w_u8, scale_u8)``.

    Args:
        w: Input tensor of shape ``[..., N, K]`` with ``K % 32 == 0``.

    Returns:
        w_u8: Packed E2M1 ``uint8`` tensor of shape ``[..., N, K // 2]``
              (low nibble = even K, high nibble = odd K).
        scale_u8: UE8M0 ``uint8`` tensor of shape ``[..., N, K // 32]``.
    """
    if w.ndim < 2:
        raise ValueError(f"Expected weight tensor with ndim >= 2, got {w.shape}")
    K = w.shape[-1]
    if K % MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"Last dimension K={K} must be divisible by {MXFP4_BLOCK_SIZE}"
        )

    prefix_shape = w.shape[:-1]
    num_groups = K // MXFP4_BLOCK_SIZE

    w_f32 = w.float().reshape(*prefix_shape, num_groups, MXFP4_BLOCK_SIZE)
    amax = w_f32.abs().amax(dim=-1)  # [..., N, K // 32]

    is_zero = amax == 0.0
    # Shared exponent: smallest power of two >= amax / 6.0 so max value maps <= 6.0.
    safe_amax = torch.where(is_zero, torch.ones_like(amax), amax)
    exp = torch.ceil(torch.log2(safe_amax / 6.0)).clamp(-127.0, 127.0).to(torch.int32)
    exp = torch.where(is_zero, torch.zeros_like(exp), exp)
    scale_u8 = (exp + 127).to(torch.uint8)

    inv_scale = torch.ldexp(torch.ones_like(safe_amax), -exp)
    inv_scale = torch.where(is_zero, torch.zeros_like(inv_scale), inv_scale)
    scaled = w_f32 * inv_scale.unsqueeze(-1)

    # IEEE-754 round-to-nearest-even to E2M1 grid {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    abs_v = scaled.abs()
    mag = (
        (abs_v > 0.25).to(torch.uint8)
        + (abs_v >= 0.75).to(torch.uint8)
        + (abs_v > 1.25).to(torch.uint8)
        + (abs_v >= 1.75).to(torch.uint8)
        + (abs_v > 2.5).to(torch.uint8)
        + (abs_v >= 3.5).to(torch.uint8)
        + (abs_v > 5.0).to(torch.uint8)
    )
    sign = (scaled < 0.0).to(torch.uint8) << 3
    nibbles = (sign | mag).reshape(*prefix_shape, K // 2, 2)
    w_u8 = nibbles[..., 0] | (nibbles[..., 1] << 4)
    return w_u8.contiguous(), scale_u8.contiguous()


def dequantize_mxfp4_ref(
    w_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Exact vectorized PyTorch reference MXFP4 dequantizer.

    Args:
        w_u8: ``[..., N, K // 2]`` ``uint8`` packed E2M1 weights.
        scale_u8: ``[..., N, K // 32]`` ``uint8`` UE8M0 scales.
        dtype: Target floating-point dtype.
    """
    if w_u8.dtype != torch.uint8 or scale_u8.dtype != torch.uint8:
        raise TypeError("w_u8 and scale_u8 must be torch.uint8")
    prefix_shape = w_u8.shape[:-1]
    K_half = w_u8.shape[-1]
    K = K_half * 2
    if K % MXFP4_BLOCK_SIZE != 0:
        raise ValueError(f"Unpacked K={K} must be divisible by {MXFP4_BLOCK_SIZE}")
    expected_scale_shape = (*prefix_shape, K // MXFP4_BLOCK_SIZE)
    if tuple(scale_u8.shape) != expected_scale_shape:
        raise ValueError(
            f"Expected scale_u8 shape {expected_scale_shape}, got {tuple(scale_u8.shape)}"
        )

    lut = _get_e2m1_lut(w_u8.device)
    lo = lut[(w_u8 & 0x0F).to(torch.int64)]
    hi = lut[((w_u8 >> 4) & 0x0F).to(torch.int64)]
    vals = torch.stack((lo, hi), dim=-1).reshape(*prefix_shape, K)

    exp_i32 = scale_u8.to(torch.int32) - 127
    scale_f32 = torch.ldexp(
        torch.ones(scale_u8.shape, dtype=torch.float32, device=w_u8.device), exp_i32
    )
    if (scale_u8 == 255).any():
        scale_f32 = torch.where(
            scale_u8 == 255, torch.full_like(scale_f32, float("nan")), scale_f32
        )
    scale_f32 = scale_f32.repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)
    return (vals * scale_f32).to(dtype)


@triton.jit
def _dequantize_mxfp4_kernel(
    w_ptr,
    sc_ptr,
    out_ptr,
    num_rows,
    K_half,
    stride_w_row: tl.int64,
    stride_sc_row: tl.int64,
    stride_out_row: tl.int64,
    BLOCK_K2: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    pid_col = tl.program_id(1)
    offs_k2 = pid_col * BLOCK_K2 + tl.arange(0, BLOCK_K2)
    mask_k2 = offs_k2 < K_half
    offs_sc = offs_k2 // 16

    b = tl.load(w_ptr + pid_row * stride_w_row + offs_k2, mask=mask_k2, other=0)
    sc_u8 = tl.load(
        sc_ptr + pid_row * stride_sc_row + offs_sc, mask=mask_k2, other=127
    )

    sc_i32 = sc_u8.to(tl.int32)
    sc_normal = (sc_i32 << 23).to(tl.float32, bitcast=True)
    sc_subnormal = tl.full(sc_i32.shape, 0x00400000, tl.int32).to(
        tl.float32, bitcast=True
    )
    sc_nan = tl.full(sc_i32.shape, 0x7FC00000, tl.int32).to(tl.float32, bitcast=True)
    sc_f32 = tl.where(sc_i32 == 0, sc_subnormal, tl.where(sc_i32 == 255, sc_nan, sc_normal))

    lo_n = b & 0x0F
    hi_n = (b >> 4) & 0x0F
    lo_mag = (lo_n & 7).to(tl.int32)
    hi_mag = (hi_n & 7).to(tl.int32)
    lo_sign = tl.where((lo_n & 8) != 0, -1.0, 1.0)
    hi_sign = tl.where((hi_n & 8) != 0, -1.0, 1.0)

    # For mag >= 2: value = (1 + 0.5 * (mag & 1)) * 2^((mag >> 1) - 1)
    # In IEEE 754 FP32 bits: exp = (mag >> 1) + 126, mantissa MSB = (mag & 1) << 22.
    lo_norm = (((lo_mag >> 1) + 126) << 23 | ((lo_mag & 1) << 22)).to(
        tl.float32, bitcast=True
    )
    hi_norm = (((hi_mag >> 1) + 126) << 23 | ((hi_mag & 1) << 22)).to(
        tl.float32, bitcast=True
    )
    lo_val = tl.where(lo_mag < 2, lo_mag.to(tl.float32) * 0.5, lo_norm) * lo_sign * sc_f32
    hi_val = tl.where(hi_mag < 2, hi_mag.to(tl.float32) * 0.5, hi_norm) * hi_sign * sc_f32

    out_base = out_ptr + pid_row * stride_out_row
    out_pair = tl.interleave(lo_val, hi_val).to(out_ptr.dtype.element_ty)
    offs_k = pid_col * (2 * BLOCK_K2) + tl.arange(0, 2 * BLOCK_K2)
    mask_k = offs_k < (2 * K_half)
    tl.store(out_base + offs_k, out_pair, mask=mask_k)


def dequantize_mxfp4(
    w_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fast Triton MXFP4 dequantizer with automatic CPU fallback to reference."""
    if not w_u8.is_cuda or not scale_u8.is_cuda:
        return dequantize_mxfp4_ref(w_u8, scale_u8, dtype=dtype)
    if w_u8.dtype != torch.uint8 or scale_u8.dtype != torch.uint8:
        raise TypeError("w_u8 and scale_u8 must be torch.uint8")

    prefix_shape = w_u8.shape[:-1]
    K_half = w_u8.shape[-1]
    K = K_half * 2
    if K % MXFP4_BLOCK_SIZE != 0:
        raise ValueError(f"Unpacked K={K} must be divisible by {MXFP4_BLOCK_SIZE}")
    expected_scale_shape = (*prefix_shape, K // MXFP4_BLOCK_SIZE)
    if tuple(scale_u8.shape) != expected_scale_shape:
        raise ValueError(
            f"Expected scale_u8 shape {expected_scale_shape}, got {tuple(scale_u8.shape)}"
        )

    if w_u8.numel() == 0:
        return torch.empty((*prefix_shape, K), dtype=dtype, device=w_u8.device)

    w_2d = w_u8.reshape(-1, K_half)
    sc_2d = scale_u8.reshape(-1, K // MXFP4_BLOCK_SIZE)
    if w_2d.stride(-1) != 1:
        w_2d = w_2d.contiguous()
    if sc_2d.stride(-1) != 1:
        sc_2d = sc_2d.contiguous()

    num_rows = w_2d.shape[0]
    out = torch.empty((num_rows, K), dtype=dtype, device=w_u8.device)
    BLOCK_K2 = 512 if K_half >= 512 else triton.next_power_of_2(max(K_half, 16))
    grid = (num_rows, triton.cdiv(K_half, BLOCK_K2))
    _dequantize_mxfp4_kernel[grid](
        w_2d,
        sc_2d,
        out,
        num_rows,
        K_half,
        w_2d.stride(0),
        sc_2d.stride(0),
        out.stride(0),
        BLOCK_K2=BLOCK_K2,
        num_warps=4,
    )
    return out.reshape(*prefix_shape, K)


# ---------------------------------------------------------------------------
# MXFP8 (E4M3 + UE8M0) Dynamic Activation Quantizer (for optional W4A8 path)
# ---------------------------------------------------------------------------


@triton.jit
def _quant_mxfp8_e4m3_kernel(
    x_ptr,
    q_ptr,
    sc_ptr,
    num_rows,
    num_groups,
    stride_x_row: tl.int64,
    stride_q_row: tl.int64,
    stride_sc_row: tl.int64,
    GROUPS_PER_CTA: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    pid_g = tl.program_id(1)
    g_offs = pid_g * GROUPS_PER_CTA + tl.arange(0, GROUPS_PER_CTA)
    g_mask = g_offs < num_groups
    elem_offs = g_offs[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(
        x_ptr + pid_row * stride_x_row + elem_offs,
        mask=g_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=1)
    is_zero = amax == 0.0
    safe_amax = tl.where(is_zero, 1.0, amax)
    # E4M3FN max finite value is 448.0
    ratio = safe_amax * (1.0 / 448.0)
    # ceil(log2(ratio)) via FP32 exponent bits:
    bits = ratio.to(tl.int32, bitcast=True)
    raw_exp = ((bits >> 23) & 0xFF) - 127
    has_mant = (bits & 0x007FFFFF) != 0
    exp = raw_exp + has_mant.to(tl.int32)
    exp = tl.maximum(tl.minimum(exp, 127), -126)
    exp = tl.where(is_zero, 0, exp)
    sc_u8 = (exp + 127).to(tl.uint8)
    inv_sc = ((127 - exp) << 23).to(tl.float32, bitcast=True)
    inv_sc = tl.where(is_zero, 0.0, inv_sc)

    q = (x * inv_sc[:, None]).to(tl.float8e4nv)
    tl.store(
        q_ptr + pid_row * stride_q_row + elem_offs,
        q,
        mask=g_mask[:, None],
    )
    tl.store(sc_ptr + pid_row * stride_sc_row + g_offs, sc_u8, mask=g_mask)


def quantize_mxfp8_e4m3(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-32-element MXFP8 (E4M3FN + UE8M0) activation quantizer."""
    if x.shape[-1] % MXFP4_BLOCK_SIZE != 0:
        raise ValueError(f"K={x.shape[-1]} must be divisible by 32")
    orig_shape = x.shape[:-1]
    K = x.shape[-1]
    num_groups = K // MXFP4_BLOCK_SIZE
    x_2d = x.reshape(-1, K)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    num_rows = x_2d.shape[0]
    q = torch.empty((num_rows, K), dtype=torch.float8_e4m3fn, device=x.device)
    sc = torch.empty((num_rows, num_groups), dtype=torch.uint8, device=x.device)
    if num_rows == 0:
        return q.reshape(*orig_shape, K), sc.reshape(*orig_shape, num_groups)
    GROUPS_PER_CTA = 16
    grid = (num_rows, triton.cdiv(num_groups, GROUPS_PER_CTA))
    _quant_mxfp8_e4m3_kernel[grid](
        x_2d,
        q,
        sc,
        num_rows,
        num_groups,
        x_2d.stride(0),
        q.stride(0),
        sc.stride(0),
        GROUPS_PER_CTA=GROUPS_PER_CTA,
        num_warps=4,
    )
    return q.reshape(*orig_shape, K), sc.reshape(*orig_shape, num_groups)


# ---------------------------------------------------------------------------
# Single-Matrix MXFP4 Linear (BF16 x E2M1 via tl.dot_scaled)
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_linear_bf16_kernel(
    a_ptr,
    b_ptr,
    b_sc_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am: tl.int64,
    stride_bn,
    stride_bsn,
    stride_cm: tl.int64,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    K_DIVISIBLE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_k2 = tl.arange(0, BK // 2)
    offs_ks = tl.arange(0, BK // 32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k2[None, :]
    bs_ptrs = b_sc_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    K_half = K // 2
    K_sc = K // 32
    for k0 in range(0, K, BK):
        if K_DIVISIBLE:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None], other=0)
            b_sc = tl.load(bs_ptrs, mask=mask_n[:, None], other=127)
        else:
            mask_k = (k0 + offs_k) < K
            mask_k2 = (k0 // 2 + offs_k2) < K_half
            mask_ks = (k0 // 32 + offs_ks) < K_sc
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k2[None, :], other=0)
            b_sc = tl.load(bs_ptrs, mask=mask_n[:, None] & mask_ks[None, :], other=127)
        b = tl.trans(b)
        acc = tl.dot_scaled(a, None, "bf16", b, b_sc, "e2m1", acc=acc)
        a_ptrs += BK
        b_ptrs += BK // 2
        bs_ptrs += BK // 32

    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def mxfp4_linear(
    x: torch.Tensor,
    w_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    use_triton: bool = True,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Single-matrix MXFP4 GEMM: ``x @ dequant(w_u8, scale_u8).T``.

    Uses native SM103 ``tl.dot_scaled(bf16, e2m1)`` when available.
    """
    target_dtype = out_dtype if out_dtype is not None else x.dtype
    orig_shape = x.shape[:-1]
    K = x.shape[-1]
    N = w_u8.shape[0]
    if w_u8.shape != (N, K // 2) or scale_u8.shape != (N, K // 32):
        raise ValueError(
            f"Weight/scale mismatch for K={K}: w_u8={tuple(w_u8.shape)}, scale_u8={tuple(scale_u8.shape)}"
        )

    if (
        not use_triton
        or not x.is_cuda
        or not supports_triton_mxfp4(x.device)
        or x.dtype != torch.bfloat16
    ):
        w_f = dequantize_mxfp4_ref(w_u8, scale_u8, dtype=torch.float32)
        out = F.linear(x.float(), w_f)
        return out.to(target_dtype)

    x_2d = x.reshape(-1, K)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    if w_u8.stride(-1) != 1:
        w_u8 = w_u8.contiguous()
    if scale_u8.stride(-1) != 1:
        scale_u8 = scale_u8.contiguous()

    M = x_2d.shape[0]
    out = torch.empty((M, N), dtype=target_dtype, device=x.device)
    if M == 0:
        return out.reshape(*orig_shape, N)

    BM = 16 if M <= 16 else (32 if M <= 64 else 64)
    BN = 64 if N < 128 else 128
    BK = 64 if (K < 128 or K % 128 != 0) else 128
    grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))
    _mxfp4_linear_bf16_kernel[grid](
        x_2d,
        w_u8,
        scale_u8,
        out,
        M,
        N,
        K,
        x_2d.stride(0),
        w_u8.stride(0),
        scale_u8.stride(0),
        out.stride(0),
        BM=BM,
        BN=BN,
        BK=BK,
        K_DIVISIBLE=(K % BK == 0),
        num_warps=4,
        num_stages=3,
    )
    return out.reshape(*orig_shape, N)


# ---------------------------------------------------------------------------
# Weight Layout Preparation
# ---------------------------------------------------------------------------


@dataclass
class MXFP4MoEWeights:
    """Container for packed MXFP4 routed expert weights ready for SM103 GEMM.

    Stores native checkpoint K-major layout ``[E, N, K // 2]`` by default
    (zero extra copy/transpose on B300 SM103, since ``tl.trans`` inside
    ``tl.dot_scaled`` is faster on MoE tile sizes and avoids doubling HBM).
    """

    w13_u8: torch.Tensor  # [E, 2*I, L//2] uint8 (or [E, L//2, 2*I] if transposed)
    w13_scale: torch.Tensor  # [E, 2*I, L//32] uint8
    w2_u8: torch.Tensor  # [E, L, I//2] uint8 (or [E, I//2, L] if transposed)
    w2_scale: torch.Tensor  # [E, L, I//32] uint8
    weights_transposed: bool = False

    @property
    def num_experts(self) -> int:
        return int(self.w13_u8.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.w2_scale.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.w13_scale.shape[1] // 2)

    def nbytes(self) -> int:
        return (
            self.w13_u8.numel() * self.w13_u8.element_size()
            + self.w13_scale.numel() * self.w13_scale.element_size()
            + self.w2_u8.numel() * self.w2_u8.element_size()
            + self.w2_scale.numel() * self.w2_scale.element_size()
        )


def prepare_mxfp4_moe_weights(
    *,
    w13_u8: Optional[torch.Tensor] = None,
    w13_scale: Optional[torch.Tensor] = None,
    w_gate_u8: Optional[torch.Tensor] = None,
    w_gate_scale: Optional[torch.Tensor] = None,
    w_up_u8: Optional[torch.Tensor] = None,
    w_up_scale: Optional[torch.Tensor] = None,
    w2_u8: torch.Tensor,
    w2_scale: torch.Tensor,
    transpose_weights: bool = False,
) -> MXFP4MoEWeights:
    """Prepare MXFP4 routed-expert weights from checkpoint tensors.

    Accepts either fused ``(w13_u8, w13_scale)`` or separate
    ``(w_gate_u8, w_gate_scale, w_up_u8, w_up_scale)`` alongside ``(w2_u8, w2_scale)``.
    All input tensors use checkpoint layout ``[E, N, K // 2]`` and ``[E, N, K // 32]``.
    """
    if w13_u8 is None or w13_scale is None:
        if (
            w_gate_u8 is None
            or w_gate_scale is None
            or w_up_u8 is None
            or w_up_scale is None
        ):
            raise ValueError(
                "Provide either (w13_u8, w13_scale) or all of (w_gate_u8, w_gate_scale, w_up_u8, w_up_scale)"
            )
        w13_u8 = torch.cat([w_gate_u8, w_up_u8], dim=1)
        w13_scale = torch.cat([w_gate_scale, w_up_scale], dim=1)

    w13_u8 = w13_u8.contiguous()
    w13_scale = w13_scale.contiguous()
    w2_u8 = w2_u8.contiguous()
    w2_scale = w2_scale.contiguous()

    if transpose_weights:
        w13_u8 = w13_u8.transpose(1, 2).contiguous()
        w2_u8 = w2_u8.transpose(1, 2).contiguous()

    return MXFP4MoEWeights(
        w13_u8=w13_u8,
        w13_scale=w13_scale,
        w2_u8=w2_u8,
        w2_scale=w2_scale,
        weights_transposed=transpose_weights,
    )


# ---------------------------------------------------------------------------
# Zero-Sync GPU MoE Tile Schedule Builder
# ---------------------------------------------------------------------------


@dataclass
class MoETileSchedule:
    """Block-aligned per-expert M-tile schedule skipping all 0-token experts."""

    perm: torch.Tensor  # [M_total] int32: sorted slot -> flat (tok*top_k + k)
    sorted_tok_ids: torch.Tensor  # [M_total] int32: sorted slot -> orig token id
    tile_expert_id: torch.Tensor  # [max_num_tiles] int32 (-1 for inactive tail tiles)
    tile_m_start: torch.Tensor  # [max_num_tiles] int32
    tile_m_count: torch.Tensor  # [max_num_tiles] int32
    max_num_tiles: int
    top_k: int
    num_tokens: int
    block_m: int


@triton.jit
def _scatter_moe_tile_schedule_kernel(
    counts_ptr,
    m_starts_ptr,
    tile_starts_ptr,
    tile_expert_id_ptr,
    tile_m_start_ptr,
    tile_m_count_ptr,
    BM: tl.constexpr,
):
    e = tl.program_id(0)
    c = tl.load(counts_ptr + e).to(tl.int32)
    if c == 0:
        return
    m0 = tl.load(m_starts_ptr + e).to(tl.int32)
    t0 = tl.load(tile_starts_ptr + e).to(tl.int32)
    nt = (c + BM - 1) // BM
    for i in range(nt):
        t = t0 + i
        rem = c - i * BM
        cnt = tl.minimum(rem, BM)
        tl.store(tile_expert_id_ptr + t, e)
        tl.store(tile_m_start_ptr + t, m0 + i * BM)
        tl.store(tile_m_count_ptr + t, cnt)


_M1_SCHED_CACHE: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_m1_static_tensors(
    device: torch.device, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device, top_k)
    cached = _M1_SCHED_CACHE.get(key)
    if cached is None:
        perm = torch.arange(top_k, dtype=torch.int32, device=device)
        sorted_tok_ids = torch.zeros((top_k,), dtype=torch.int32, device=device)
        tile_m_start = perm
        tile_m_count = torch.ones((top_k,), dtype=torch.int32, device=device)
        cached = (perm, sorted_tok_ids, tile_m_start, tile_m_count)
        _M1_SCHED_CACHE[key] = cached
    return cached



@triton.jit
def _moe_expert_bincount_kernel(
    ids_ptr,
    counts_ptr,
    m_total,
    num_experts,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < m_total
    e = tl.load(ids_ptr + offs, mask=mask, other=-1)
    valid = mask & (e >= 0) & (e < num_experts)
    tl.atomic_add(counts_ptr + e, 1, mask=valid)


def _gpu_expert_bincount(flat_local: torch.Tensor, num_experts: int) -> torch.Tensor:
    """CUDA-Graph-safe, zero-sync expert histogram (replaces torch.bincount)."""
    if not flat_local.is_cuda:
        mask = (flat_local >= 0) & (flat_local < num_experts)
        return torch.bincount(flat_local[mask].to(torch.int64), minlength=num_experts)
    counts = torch.zeros((num_experts,), dtype=torch.int32, device=flat_local.device)
    m_total = flat_local.numel()
    if m_total == 0:
        return counts
    flat_c = flat_local.to(torch.int32).contiguous()
    BLOCK = 256
    _moe_expert_bincount_kernel[(triton.cdiv(m_total, BLOCK),)](
        flat_c,
        counts,
        m_total,
        num_experts,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return counts


def build_moe_tile_schedule(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_m: int,
    *,
    unique_per_token: bool = False,
    ep_start: int = 0,
    ep_size: int = 1,
) -> MoETileSchedule:
    """Sort token-expert pairs and build a block-aligned M-tile table on GPU.

    Performs ZERO CPU-GPU synchronization and skips every expert with 0 tokens
    (and every remote expert outside ``[ep_start, ep_start + num_experts)``
    when ``ep_size > 1``). When ``num_tokens == 1`` and ``unique_per_token=True``
    (guaranteed by ``torch.topk``), bypasses sort/bincount/cumsum/scatter completely.
    """
    if topk_ids.ndim != 2:
        raise ValueError(f"Expected topk_ids of shape [M, top_k], got {topk_ids.shape}")
    num_tokens, top_k = topk_ids.shape
    m_total = num_tokens * top_k
    device = topk_ids.device

    if num_tokens == 1 and unique_per_token and top_k > 0:
        perm, sorted_tok_ids, tile_m_start, tile_m_count = _get_m1_static_tensors(
            device, top_k
        )
        if ep_size > 1:
            local_ids = topk_ids[0].to(torch.int32) - int(ep_start)
            valid = (local_ids >= 0) & (local_ids < num_experts)
            tile_expert_id = torch.where(
                valid, local_ids, torch.full_like(local_ids, -1)
            ).contiguous()
        else:
            tile_expert_id = topk_ids[0].to(torch.int32).contiguous()
        return MoETileSchedule(
            perm=perm,
            sorted_tok_ids=sorted_tok_ids,
            tile_expert_id=tile_expert_id,
            tile_m_start=tile_m_start,
            tile_m_count=tile_m_count,
            max_num_tiles=top_k,
            top_k=top_k,
            num_tokens=1,
            block_m=block_m,
        )

    if m_total == 0:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        return MoETileSchedule(
            perm=empty_i32,
            sorted_tok_ids=empty_i32,
            tile_expert_id=empty_i32,
            tile_m_start=empty_i32,
            tile_m_count=empty_i32,
            max_num_tiles=0,
            top_k=top_k,
            num_tokens=num_tokens,
            block_m=block_m,
        )

    flat_ids = topk_ids.reshape(-1)
    if ep_size > 1:
        ep_end = int(ep_start) + num_experts
        in_range = (flat_ids >= ep_start) & (flat_ids < ep_end)
        flat_local = torch.where(
            in_range,
            flat_ids - int(ep_start),
            torch.full_like(flat_ids, num_experts),
        )
        _, perm_i64 = torch.sort(flat_local)
        perm = perm_i64.to(torch.int32)
        sorted_tok_ids = (perm_i64 // top_k).to(torch.int32)
        counts = _gpu_expert_bincount(flat_local, num_experts)
    else:
        _, perm_i64 = torch.sort(flat_ids)
        perm = perm_i64.to(torch.int32)
        sorted_tok_ids = (perm_i64 // top_k).to(torch.int32)
        counts = _gpu_expert_bincount(flat_ids, num_experts)
    m_starts = torch.cumsum(counts, dim=0) - counts
    num_tiles_per_e = (counts + block_m - 1) // block_m
    tile_starts = torch.cumsum(num_tiles_per_e, dim=0) - num_tiles_per_e

    # Tight mathematical upper bound on total active tiles:
    # A_max = min(E, M_total); sum_e ceil(c_e / BM) <= A_max + (M_total - A_max) // BM.
    a_max = min(num_experts, m_total)
    max_num_tiles = a_max + (m_total - a_max) // block_m

    tile_expert_id = torch.full(
        (max_num_tiles,), -1, dtype=torch.int32, device=device
    )
    tile_m_start = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)
    tile_m_count = torch.empty((max_num_tiles,), dtype=torch.int32, device=device)

    _scatter_moe_tile_schedule_kernel[(num_experts,)](
        counts,
        m_starts,
        tile_starts,
        tile_expert_id,
        tile_m_start,
        tile_m_count,
        BM=block_m,
        num_warps=1,
    )

    return MoETileSchedule(
        perm=perm,
        sorted_tok_ids=sorted_tok_ids,
        tile_expert_id=tile_expert_id,
        tile_m_start=tile_m_start,
        tile_m_count=tile_m_count,
        max_num_tiles=max_num_tiles,
        top_k=top_k,
        num_tokens=num_tokens,
        block_m=block_m,
    )


# ---------------------------------------------------------------------------
# B300 SM103 Triton tl.dot_scaled Grouped MXFP4 GEMM Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_grouped_gemm_bf16_kernel(
    a_ptr,
    b_ptr,
    b_sc_ptr,
    c_ptr,
    sorted_tok_ids_ptr,
    perm_ptr,
    tile_expert_id_ptr,
    tile_m_start_ptr,
    tile_m_count_ptr,
    N,
    K,
    stride_am,
    stride_be: tl.int64,
    stride_b0,
    stride_bse: tl.int64,
    stride_bsn,
    stride_cm,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TRANS_B: tl.constexpr,
    GATHER_INPUT: tl.constexpr,
    SCATTER_OUTPUT: tl.constexpr,
    K_DIVISIBLE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_tile = tl.program_id(1)

    expert_id = tl.load(tile_expert_id_ptr + pid_tile)
    if expert_id < 0:
        return

    m_start = tl.load(tile_m_start_ptr + pid_tile)
    m_count = tl.load(tile_m_count_ptr + pid_tile)

    offs_local = tl.arange(0, BM)
    mask_m = offs_local < m_count
    slot_ids = m_start + offs_local

    if GATHER_INPUT:
        row_ids_a = tl.load(sorted_tok_ids_ptr + slot_ids, mask=mask_m, other=0)
    else:
        row_ids_a = slot_ids

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BK)
    offs_k2 = tl.arange(0, BK // 2)
    offs_ks = tl.arange(0, BK // 32)

    # Scalar 64-bit expert base offset computed ONCE outside the K-loop
    # (avoids int32 overflow for E=640 weights > 2 GiB while keeping inner offsets int32).
    expert_id_64 = expert_id.to(tl.int64)
    b_base = b_ptr + expert_id_64 * stride_be
    bs_base = b_sc_ptr + expert_id_64 * stride_bse

    a_ptrs = a_ptr + row_ids_a[:, None] * stride_am + offs_k[None, :]
    if TRANS_B:
        b_ptrs = b_base + offs_n[:, None] * stride_b0 + offs_k2[None, :]
        b_step = BK // 2
    else:
        b_ptrs = b_base + offs_k2[:, None] * stride_b0 + offs_n[None, :]
        b_step = (BK // 2) * stride_b0
    bs_ptrs = bs_base + offs_n[:, None] * stride_bsn + offs_ks[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    K_half = K // 2
    K_sc = K // 32
    for k0 in range(0, K, BK):
        if K_DIVISIBLE:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if TRANS_B:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[:, None],
                    other=0,
                    eviction_policy="evict_first",
                )
                b = tl.trans(b)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
            b_sc = tl.load(
                bs_ptrs,
                mask=mask_n[:, None],
                other=127,
                eviction_policy="evict_first",
            )
        else:
            mask_k = (k0 + offs_k) < K
            mask_k2 = (k0 // 2 + offs_k2) < K_half
            mask_ks = (k0 // 32 + offs_ks) < K_sc
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            if TRANS_B:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[:, None] & mask_k2[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
                b = tl.trans(b)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=mask_k2[:, None] & mask_n[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
            b_sc = tl.load(
                bs_ptrs,
                mask=mask_n[:, None] & mask_ks[None, :],
                other=127,
                eviction_policy="evict_first",
            )
        acc = tl.dot_scaled(a, None, "bf16", b, b_sc, "e2m1", acc=acc)
        a_ptrs += BK
        b_ptrs += b_step
        bs_ptrs += BK // 32

    if SCATTER_OUTPUT:
        row_ids_c = tl.load(perm_ptr + slot_ids, mask=mask_m, other=0)
    else:
        row_ids_c = slot_ids

    c_ptrs = c_ptr + row_ids_c[:, None] * stride_cm + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _mxfp4_grouped_gemm_e4m3_kernel(
    a_ptr,
    a_sc_ptr,
    b_ptr,
    b_sc_ptr,
    c_ptr,
    sorted_tok_ids_ptr,
    perm_ptr,
    tile_expert_id_ptr,
    tile_m_start_ptr,
    tile_m_count_ptr,
    N,
    K,
    stride_am,
    stride_asm,
    stride_be: tl.int64,
    stride_b0,
    stride_bse: tl.int64,
    stride_bsn,
    stride_cm,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TRANS_B: tl.constexpr,
    GATHER_INPUT: tl.constexpr,
    SCATTER_OUTPUT: tl.constexpr,
    K_DIVISIBLE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_tile = tl.program_id(1)

    expert_id = tl.load(tile_expert_id_ptr + pid_tile)
    if expert_id < 0:
        return

    m_start = tl.load(tile_m_start_ptr + pid_tile)
    m_count = tl.load(tile_m_count_ptr + pid_tile)

    offs_local = tl.arange(0, BM)
    mask_m = offs_local < m_count
    slot_ids = m_start + offs_local

    if GATHER_INPUT:
        row_ids_a = tl.load(sorted_tok_ids_ptr + slot_ids, mask=mask_m, other=0)
    else:
        row_ids_a = slot_ids

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < N
    offs_k = tl.arange(0, BK)
    offs_k2 = tl.arange(0, BK // 2)
    offs_ks = tl.arange(0, BK // 32)

    expert_id_64 = expert_id.to(tl.int64)
    b_base = b_ptr + expert_id_64 * stride_be
    bs_base = b_sc_ptr + expert_id_64 * stride_bse

    a_ptrs = a_ptr + row_ids_a[:, None] * stride_am + offs_k[None, :]
    as_ptrs = a_sc_ptr + row_ids_a[:, None] * stride_asm + offs_ks[None, :]
    if TRANS_B:
        b_ptrs = b_base + offs_n[:, None] * stride_b0 + offs_k2[None, :]
        b_step = BK // 2
    else:
        b_ptrs = b_base + offs_k2[:, None] * stride_b0 + offs_n[None, :]
        b_step = (BK // 2) * stride_b0
    bs_ptrs = bs_base + offs_n[:, None] * stride_bsn + offs_ks[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    K_half = K // 2
    K_sc = K // 32
    for k0 in range(0, K, BK):
        if K_DIVISIBLE:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            a_sc = tl.load(as_ptrs, mask=mask_m[:, None], other=127)
            if TRANS_B:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[:, None],
                    other=0,
                    eviction_policy="evict_first",
                )
                b = tl.trans(b)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
            b_sc = tl.load(
                bs_ptrs,
                mask=mask_n[:, None],
                other=127,
                eviction_policy="evict_first",
            )
        else:
            mask_k = (k0 + offs_k) < K
            mask_k2 = (k0 // 2 + offs_k2) < K_half
            mask_ks = (k0 // 32 + offs_ks) < K_sc
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            a_sc = tl.load(as_ptrs, mask=mask_m[:, None] & mask_ks[None, :], other=127)
            if TRANS_B:
                b = tl.load(
                    b_ptrs,
                    mask=mask_n[:, None] & mask_k2[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
                b = tl.trans(b)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=mask_k2[:, None] & mask_n[None, :],
                    other=0,
                    eviction_policy="evict_first",
                )
            b_sc = tl.load(
                bs_ptrs,
                mask=mask_n[:, None] & mask_ks[None, :],
                other=127,
                eviction_policy="evict_first",
            )
        acc = tl.dot_scaled(a, a_sc, "e4m3", b, b_sc, "e2m1", acc=acc)
        a_ptrs += BK
        as_ptrs += BK // 32
        b_ptrs += b_step
        bs_ptrs += BK // 32

    if SCATTER_OUTPUT:
        row_ids_c = tl.load(perm_ptr + slot_ids, mask=mask_m, other=0)
    else:
        row_ids_c = slot_ids

    c_ptrs = c_ptr + row_ids_c[:, None] * stride_cm + offs_n[None, :]
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _moe_weighted_reduce_kernel(
    in_ptr,
    weights_ptr,
    out_ptr,
    M,
    K,
    stride_im: tl.int64,
    stride_ik,
    stride_wm,
    stride_om: tl.int64,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    in_row_base = in_ptr + pid_m * stride_im
    w_row_base = weights_ptr + pid_m * stride_wm
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k_idx in tl.static_range(TOP_K):
        w = tl.load(w_row_base + k_idx).to(tl.float32)
        if w != 0.0:
            v = tl.load(
                in_row_base + k_idx * stride_ik + offs_k, mask=mask_k, other=0.0
            ).to(tl.float32)
            acc += v * w

    tl.store(
        out_ptr + pid_m * stride_om + offs_k,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_k,
    )


def moe_weighted_reduce_triton(
    expert_outputs: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """Deterministic FP32 weighted sum over ``top_k`` expert outputs: ``[M, top_k, K] -> [M, K]``."""
    M, top_k, K = expert_outputs.shape
    if not expert_outputs.is_cuda:
        return (
            (expert_outputs.float() * topk_weights.float().unsqueeze(-1))
            .sum(dim=1)
            .to(expert_outputs.dtype)
        )
    out = torch.empty((M, K), dtype=expert_outputs.dtype, device=expert_outputs.device)
    if M == 0:
        return out
    w = topk_weights.float().contiguous()
    if expert_outputs.stride(-1) != 1:
        expert_outputs = expert_outputs.contiguous()
    BLOCK_K = 512 if K >= 512 else triton.next_power_of_2(max(K, 64))
    grid = (M, triton.cdiv(K, BLOCK_K))
    _moe_weighted_reduce_kernel[grid](
        expert_outputs,
        w,
        out,
        M,
        K,
        expert_outputs.stride(0),
        expert_outputs.stride(1),
        w.stride(0),
        out.stride(0),
        TOP_K=top_k,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out


def _select_grouped_gemm_config(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    N: int,
    K: int,
    block_m: Optional[int] = None,
) -> Tuple[int, int, int, int, int]:
    """Select tuned B300 SM103 ``(BM, BN, BK, num_warps, num_stages)``."""
    m_total = num_tokens * top_k
    est_active = max(1, min(num_experts, m_total))
    avg_m_per_e = m_total / est_active

    if block_m is None:
        if avg_m_per_e <= 12.0:
            BM = 16
        elif avg_m_per_e <= 24.0:
            BM = 32
        else:
            BM = 64
    else:
        BM = block_m

    if N < 128:
        BN = 64 if N >= 64 else max(16, triton.next_power_of_2(N))
    elif num_tokens <= 4 and N <= 2560:
        # Increase CTA wave utilization on 148-SM B300 during tiny-batch decode Stage 2.
        BN = 64
    else:
        BN = 128

    if K < 128 or (K % 128 != 0):
        BK = 64
    elif BM >= 32:
        # Empirically optimal at Prefill M=2048 (BM=32, BN=128, BK=64 -> 1748 GB/s).
        BK = 64
    else:
        # Empirically optimal at Decode M=1..64 & Prefill M=512 (BM=16, BN=128, BK=128 -> 1800 GB/s).
        BK = 128

    num_warps = 4
    num_stages = 4 if (BM == 16 and BK == 128) else 3
    return BM, BN, BK, num_warps, num_stages


def mxfp4_grouped_moe_gemm(
    x: torch.Tensor,
    w_u8: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    schedule: Optional[MoETileSchedule] = None,
    topk_ids: Optional[torch.Tensor] = None,
    gather_input: bool = True,
    scatter_output: bool = False,
    weights_transposed: bool = False,
    compute_mode: str = "bf16",
    use_triton: bool = True,
) -> torch.Tensor:
    """Execute one grouped MXFP4 GEMM over active MoE experts.

    Args:
        x: Activations of shape ``[M, K]`` (when ``gather_input=True``) or
           ``[M_total, K]`` (when ``gather_input=False``), BF16.
        w_u8: Packed MXFP4 weights ``[E, N, K // 2]`` (or ``[E, K // 2, N]`` if
              ``weights_transposed=True``).
        scale_u8: UE8M0 scales ``[E, N, K // 32]``.
        schedule: Prebuilt :class:`MoETileSchedule` (optional; built from
                  ``topk_ids`` if omitted).
        topk_ids: ``[M, top_k]`` routing indices (required if ``schedule is None``).
        gather_input: Gather rows from ``[M, K]`` using ``schedule.sorted_tok_ids``.
        scatter_output: Scatter output rows to ``schedule.perm`` order.
        weights_transposed: False for native checkpoint layout ``[E, N, K // 2]``;
                            True for pre-transposed ``[E, K // 2, N]``.
        compute_mode: ``"bf16"`` (default, W4A16 exact) or ``"e4m3"`` (W4A8).
        use_triton: Use Triton SM103 ``tl.dot_scaled`` kernel when True.

    Returns:
        Output tensor of shape ``[M_total, N]`` (BF16).
    """
    num_experts = int(scale_u8.shape[0])
    N = int(scale_u8.shape[1])
    K = int(x.shape[-1])
    if schedule is None:
        if topk_ids is None:
            raise ValueError("Provide either schedule or topk_ids")
        BM_init, _, _, _, _ = _select_grouped_gemm_config(
            topk_ids.shape[0], topk_ids.shape[1], num_experts, N, K
        )
        schedule = build_moe_tile_schedule(topk_ids, num_experts, BM_init)

    m_total = schedule.num_tokens * schedule.top_k
    out = torch.empty((m_total, N), dtype=x.dtype, device=x.device)
    if m_total == 0:
        return out

    if (
        not use_triton
        or not x.is_cuda
        or not supports_triton_mxfp4(x.device)
        or x.dtype != torch.bfloat16
    ):
        # Clean dequant-BF16 fallback (only dequantizes experts that appear in schedule)
        w_nk = w_u8.transpose(1, 2).contiguous() if weights_transposed else w_u8
        tile_e = schedule.tile_expert_id.tolist()
        tile_s = schedule.tile_m_start.tolist()
        tile_c = schedule.tile_m_count.tolist()
        deq_cache: dict[int, torch.Tensor] = {}
        for e, m0, cnt in zip(tile_e, tile_s, tile_c):
            if e < 0:
                continue
            w_f = deq_cache.get(e)
            if w_f is None:
                w_f = dequantize_mxfp4_ref(w_nk[e], scale_u8[e], dtype=x.dtype)
                deq_cache[e] = w_f
            slots = slice(m0, m0 + cnt)
            if gather_input:
                a_sub = x.index_select(0, schedule.sorted_tok_ids[slots].to(torch.int64))
            else:
                a_sub = x[slots]
            y_sub = F.linear(a_sub, w_f)
            if scatter_output:
                out.index_copy_(0, schedule.perm[slots].to(torch.int64), y_sub)
            else:
                out[slots] = y_sub
        return out

    if x.stride(-1) != 1:
        x = x.contiguous()
    if w_u8.stride(-1) != 1:
        w_u8 = w_u8.contiguous()
    if scale_u8.stride(-1) != 1:
        scale_u8 = scale_u8.contiguous()

    BM, BN, BK, num_warps, num_stages = _select_grouped_gemm_config(
        schedule.num_tokens,
        schedule.top_k,
        num_experts,
        N,
        K,
        block_m=schedule.block_m,
    )
    grid = (triton.cdiv(N, BN), schedule.max_num_tiles)
    k_divisible = (K % BK) == 0

    if compute_mode == "bf16":
        _mxfp4_grouped_gemm_bf16_kernel[grid](
            x,
            w_u8,
            scale_u8,
            out,
            schedule.sorted_tok_ids,
            schedule.perm,
            schedule.tile_expert_id,
            schedule.tile_m_start,
            schedule.tile_m_count,
            N,
            K,
            x.stride(0),
            w_u8.stride(0),
            w_u8.stride(1),
            scale_u8.stride(0),
            scale_u8.stride(1),
            out.stride(0),
            BM=BM,
            BN=BN,
            BK=BK,
            TRANS_B=not weights_transposed,
            GATHER_INPUT=gather_input,
            SCATTER_OUTPUT=scatter_output,
            K_DIVISIBLE=k_divisible,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif compute_mode == "e4m3":
        x_q, x_sc = quantize_mxfp8_e4m3(x)
        _mxfp4_grouped_gemm_e4m3_kernel[grid](
            x_q,
            x_sc,
            w_u8,
            scale_u8,
            out,
            schedule.sorted_tok_ids,
            schedule.perm,
            schedule.tile_expert_id,
            schedule.tile_m_start,
            schedule.tile_m_count,
            N,
            K,
            x_q.stride(0),
            x_sc.stride(0),
            w_u8.stride(0),
            w_u8.stride(1),
            scale_u8.stride(0),
            scale_u8.stride(1),
            out.stride(0),
            BM=BM,
            BN=BN,
            BK=BK,
            TRANS_B=not weights_transposed,
            GATHER_INPUT=gather_input,
            SCATTER_OUTPUT=scatter_output,
            K_DIVISIBLE=k_divisible,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        raise ValueError(f"Unsupported compute_mode={compute_mode!r}; expected bf16 or e4m3")

    return out


# ---------------------------------------------------------------------------
# Full 2-Stage Routed MXFP4 Experts Forward (Triton SM103 + Reference)
# ---------------------------------------------------------------------------


def mxfp4_moe_forward(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MXFP4MoEWeights,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    compute_mode: str = "bf16",
    use_triton: bool = True,
    ep_start: int = 0,
    ep_size: int = 1,
) -> torch.Tensor:
    """Full 2-stage MXFP4 Routed Experts execution:
      Stage 1: fused gate+up MXFP4 grouped GEMM -> SiTU-GLU
      Stage 2: down_proj MXFP4 grouped GEMM -> deterministic FP32 weighted reduce

    When ``ep_size > 1``, ``weights`` holds only the ``local_E`` experts covering
    global ids ``[ep_start, ep_start + local_E)``. Remote slots are skipped with
    zero CPU-GPU synchronization and receive weight ``0.0`` in the weighted sum.
    """
    if not use_triton or not x.is_cuda or not supports_triton_mxfp4(x.device):
        return mxfp4_moe_forward_ref(
            x,
            topk_ids,
            topk_weights,
            weights,
            gate_beta=gate_beta,
            up_beta=up_beta,
            ep_start=ep_start,
            ep_size=ep_size,
        )

    M, L = x.shape
    top_k = topk_ids.shape[1]
    if M == 0:
        return torch.empty_like(x)

    num_experts = weights.num_experts
    two_I = weights.w13_scale.shape[1]
    BM, _, _, _, _ = _select_grouped_gemm_config(M, top_k, num_experts, two_I, L)
    schedule = build_moe_tile_schedule(
        topk_ids,
        num_experts,
        BM,
        unique_per_token=True,
        ep_start=ep_start,
        ep_size=ep_size,
    )

    # Stage 1: fused gate_proj + up_proj ([M, L] -> [M_total, 2*I])
    gate_up = mxfp4_grouped_moe_gemm(
        x,
        weights.w13_u8,
        weights.w13_scale,
        schedule=schedule,
        gather_input=True,
        scatter_output=False,
        weights_transposed=weights.weights_transposed,
        compute_mode=compute_mode,
        use_triton=True,
    )

    # Fused SiTU-GLU activation ([M_total, 2*I] -> [M_total, I])
    act = situ_and_mul_triton(gate_up, gate_beta=gate_beta, up_beta=up_beta)

    # Stage 2: down_proj ([M_total, I] -> [M * top_k, L] scattered to original token/k slots)
    down_out = mxfp4_grouped_moe_gemm(
        act,
        weights.w2_u8,
        weights.w2_scale,
        schedule=schedule,
        gather_input=False,
        scatter_output=True,
        weights_transposed=weights.weights_transposed,
        compute_mode=compute_mode,
        use_triton=True,
    )

    if ep_size > 1:
        ep_end = int(ep_start) + num_experts
        in_range = (topk_ids >= ep_start) & (topk_ids < ep_end)
        reduce_weights = torch.where(
            in_range, topk_weights, torch.zeros_like(topk_weights)
        )
    else:
        reduce_weights = topk_weights

    # Deterministic FP32 weighted sum over top_k -> [M, L] BF16
    return moe_weighted_reduce_triton(down_out.view(M, top_k, L), reduce_weights)


def mxfp4_moe_forward_ref(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MXFP4MoEWeights,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    ep_start: int = 0,
    ep_size: int = 1,
) -> torch.Tensor:
    """Exact dequant-BF16 reference for MXFP4 Routed Experts.

    Only dequantizes active local experts present in ``topk_ids`` so large
    ``E=640`` tests remain fast and memory-light.
    """
    M, L = x.shape
    if M == 0:
        return torch.empty_like(x)

    out_fp32 = torch.zeros((M, L), dtype=torch.float32, device=x.device)
    w13_nk = (
        weights.w13_u8.transpose(1, 2).contiguous()
        if weights.weights_transposed
        else weights.w13_u8
    )
    w2_nk = (
        weights.w2_u8.transpose(1, 2).contiguous()
        if weights.weights_transposed
        else weights.w2_u8
    )
    local_e_count = weights.num_experts
    ep_end = int(ep_start) + local_e_count if ep_size > 1 else local_e_count
    e_lo = int(ep_start) if ep_size > 1 else 0

    unique_experts = torch.unique(topk_ids).tolist()
    for e in unique_experts:
        if e < e_lo or e >= ep_end:
            continue
        local_e = e - e_lo
        tok_idx, k_idx = torch.where(topk_ids == e)
        if tok_idx.numel() == 0:
            continue
        w13_bf16 = dequantize_mxfp4_ref(
            w13_nk[local_e], weights.w13_scale[local_e], dtype=x.dtype
        )
        w2_bf16 = dequantize_mxfp4_ref(
            w2_nk[local_e], weights.w2_scale[local_e], dtype=x.dtype
        )

        x_e = x.index_select(0, tok_idx)
        gate_up = F.linear(x_e, w13_bf16)
        act = situ_and_mul_ref(gate_up, gate_beta=gate_beta, up_beta=up_beta)
        down = F.linear(act, w2_bf16)

        w_e = topk_weights[tok_idx, k_idx].float().unsqueeze(-1)
        out_fp32.index_add_(0, tok_idx, down.float() * w_e)

    return out_fp32.to(x.dtype)
