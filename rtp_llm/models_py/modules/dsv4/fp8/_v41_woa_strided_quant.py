"""E5a (Batch-4b part 2): strided-read group32 UE8M0 FP8 quantizer for the
wo_a einsum output.

Production path today (``round_kv_norm_to_input_dtype`` layers):
``torch.einsum("tgd,grd->tgr")`` materializes [g,t,r] and returns the
permuted view [t,g,r]; ``projected.flatten(1)`` then copies the whole
bf16 tensor (t*g*r*2 bytes) purely to make rows contiguous before
``wo_b`` re-quantizes it with ``sgl_per_token_group_quant_fp8``
(group 32, UE8M0, int32 column-major TMA scales).

Under ``DSV41_E5A_WOA_STRIDED_QUANT`` the quantizer reads the permuted
einsum result directly (each 32-element group lies inside one contiguous
r-slice when r % 32 == 0) and writes the exact sgl outputs: contiguous
E4M3 [t, K] plus int32 column-major TMA-packed scales. The bf16 copy
disappears; quantized bytes must be identical (GPU byte-compare gated).

Recipe replicated from the LEGACY CUDA kernel
``per_token_group_quant_8bit_kernel`` (the auto-selected variant while
t*K < 4M, which covers production t <= 4096, K = 512):

    amax  = max(eps, max|x|)                 # eps = fp32 tiny
    y_s   = amax / 448.0f                    # fp32 division
    y_s   = exp2f(ceilf(log2f(max(y_s, 1e-10f))))
    byte  = (uint8)((int)log2f(y_s) + 127)
    q     = cvt_rn_satfinite_e4m3(clamp(val / y_s, -448, 448))

The v2 kernel differs at degenerate magnitudes (eps floor placement and
reciprocal-multiply); the byte-compare gate covers both regimes.
"""

import os
import sys

import torch
import triton
import triton.language as tl

_E5A_FLAG = "DSV41_E5A_WOA_STRIDED_QUANT"
_E5A_ANNOUNCED = False

# torch.finfo(torch.float32).tiny — the eps utils._quantize_input passes.
_FP32_TINY = 1.1754943508222875e-38


def woa_strided_quant_enabled() -> bool:
    """E5a: strided-read wo_a quantizer. Default ON since Batch-5 (R16):
    GPU byte-gate 0/112.7M q + 0 s over 10 shapes x 7 distributions in
    both sgl regimes, GPU parity byte-exact with engagement
    proven (x8), A/B median −12.6 ms with 3/3 sign.
    ``=0`` (any non-1 value) restores the flatten-copy + sgl path."""
    return os.environ.get(_E5A_FLAG, "1") == "1"


def _announce_e5a() -> None:
    global _E5A_ANNOUNCED
    if not _E5A_ANNOUNCED:
        _E5A_ANNOUNCED = True
        print(
            "[E5AWOA] strided wo_a quantizer engaged",
            file=sys.stderr,
            flush=True,
        )


@triton.jit
def _woa_strided_quant_kernel(
    X,
    Q,
    S,
    stride_t,
    stride_g,
    R: tl.constexpr,
    K: tl.constexpr,
    K32: tl.constexpr,
    ALIGNED_MN,
    EPS: tl.constexpr,
    V2: tl.constexpr,
):
    """One program per token row; K must be a power of two >= 32 and
    K32 = K/32 a multiple of 4 (wrapper-gated).

    V2=False replicates the LEGACY CUDA recipe (per_token_group_quant_8bit
    _kernel): y_s = exp2f(ceilf(log2f(max(amax/448, 1e-10)))), q = val/y_s.
    V2=True replicates per_token_group_quant_8bit_v2 (DeepEP-style bit
    tricks): exp = fast_log2_ceil(amax * (1/448)) on the raw fp32 bits,
    scale = fast_pow2(-exp), q = val * scale. The two differ only at
    degenerate magnitudes (tiny/zero groups); sgl's auto dispatch selects
    v2 at numel >= 4M, and the wrapper mirrors that rule."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, K)
    g = cols // R
    r = cols % R
    x = tl.load(X + row * stride_t + g.to(tl.int64) * stride_g + r).to(tl.float32)
    if V2:
        # CUDA v2 runs FTZ: subnormal fp32 multiplicands are flushed to
        # SIGN-PRESERVING zero before the scale multiply (matters when the
        # scale is 2^127, i.e. zero/tiny groups — the -0.0 products become
        # fp8 0x80). Triton keeps subnormals; emulate the signed flush.
        # Harmless for amax (|subnormal| < eps floor).
        xb = x.to(tl.uint32, bitcast=True)
        signed_zero = (xb & 0x80000000).to(tl.float32, bitcast=True)
        x = tl.where((xb & 0x7F800000) == 0, signed_zero, x)
    groups = x.reshape((K32, 32))
    amax = tl.maximum(tl.max(tl.abs(groups), 1), EPS)
    if V2:
        scaled = amax * 0.0022321428571428575  # fp32(1.0f/448.0f)
        bits = scaled.to(tl.uint32, bitcast=True)
        # The CUDA v2 kernel runs FTZ: when the true product is a nonzero
        # fp32 subnormal (zero/tiny groups; amax >= eps is the init floor)
        # it is flushed to all-zero bits and fast_log2_ceil yields -127
        # (byte 0) — NOT the subnormal-preserving -126. Triton keeps the
        # subnormal here, so force the flushed semantics on exp field 0.
        exp_field = ((bits >> 23) & 0xFF).to(tl.int32)
        exp_i = tl.where(
            exp_field == 0,
            -127,
            (exp_field - 127) + ((bits & 0x7FFFFF) != 0).to(tl.int32),
        )
        y_s = (((127 - exp_i).to(tl.uint32)) << 23).to(tl.float32, bitcast=True)
        q = tl.clamp(groups * y_s[:, None], -448.0, 448.0).to(
            tl.float8e4nv, fp_downcast_rounding="rtne"
        )
        byte = exp_i + 127
    else:
        y_s = tl.div_rn(amax, 448.0)
        y_s = tl.maximum(y_s, 1e-10)
        y_s = tl.exp2(tl.ceil(tl.log2(y_s)))
        q = tl.clamp(tl.div_rn(groups, y_s[:, None]), -448.0, 448.0).to(
            tl.float8e4nv, fp_downcast_rounding="rtne"
        )
        byte = tl.log2(y_s).to(tl.int32) + 127
    tl.store(Q + row * K + cols, q.to(tl.uint8, bitcast=True).reshape((K,)))
    # pack 4 little-endian bytes per int32 (disjoint lanes: sum == or)
    cols32 = tl.arange(0, K32)
    words = (byte << ((cols32 % 4) * 8)).reshape((K32 // 4, 4))
    packed = tl.sum(words, 1)
    tl.store(
        S + tl.arange(0, K32 // 4).to(tl.int64) * ALIGNED_MN + row,
        packed,
    )


def quantize_strided_gtr(projected: torch.Tensor):
    """Quantize the permuted einsum result [t, g, r] without flattening.

    Returns ``(x_q, x_s)`` byte-identical to
    ``sgl_per_token_group_quant_fp8(projected.flatten(1).contiguous(),
    group_size=32, eps=fp32_tiny, column_major_scales=True,
    scale_tma_aligned=True, scale_ue8m0=True)``. Returns ``None`` when the
    layout is not the supported permuted one (caller falls back).
    """
    if (
        projected.dim() != 3
        or projected.dtype != torch.bfloat16
        or not projected.is_cuda
        or projected.shape[0] == 0
    ):
        return None
    t, g, r = projected.shape
    k = g * r
    if (
        r % 32
        or k & (k - 1)
        or k < 32
        or (k // 32) % 4
        or projected.stride(2) != 1
        or projected.stride(0) != r
        or projected.stride(1) != t * r
    ):
        return None
    k32 = k // 32
    aligned_mn = (t + 3) // 4 * 4
    x_q = torch.empty((t, k), dtype=torch.float8_e4m3fn, device=projected.device)
    x_s = torch.empty(
        (k32 // 4, aligned_mn), dtype=torch.int32, device=projected.device
    ).transpose(-1, -2)[:t]
    _announce_e5a()
    _woa_strided_quant_kernel[(t,)](
        projected,
        x_q.view(torch.uint8),
        x_s,
        projected.stride(0),
        projected.stride(1),
        R=r,
        K=k,
        K32=k32,
        ALIGNED_MN=aligned_mn,
        EPS=_FP32_TINY,
        V2=t * k >= 4 * 1024 * 1024,  # sgl auto dispatch rule
        num_warps=4,
        enable_fp_fusion=False,
    )
    return x_q, x_s
