"""e4m3fn conversion helpers for pre-sm89 NVIDIA GPUs (A100 / sm80).

Triton's ``float8e4nv`` element type only lowers on sm89+ (native ``cvt``
instructions); kernels that use it fail to *compile* on sm80. The dsv41 fused
kernels use these conversions in ~30 sites, so sm80 needs a bit-op emulation
with identical semantics.

The hardware path Triton emits on sm89+/sm100 (verified from generated PTX)
is a two-step conversion::

    fp32 --(cvt.rn)--> fp16 --(cvt.rn.satfinite.e4m3x2.f16x2)--> e4m3fn

i.e. values that overflow the e4m3fn range saturate to +/-448 (``satfinite``,
NOT NaN like the torch/ml_dtypes CPU casts), and NaN maps to 0x7F. The
emulation below reproduces exactly that: a real hardware ``fp32 -> fp16`` cast
(legal on sm80) followed by integer RTNE-with-saturation rounding on the fp16
bits.

Validation (CPU, numpy mirror of the exact expressions in ``_emu_downcast``):

* exhaustive over all 48,770 finite fp16 patterns with |x| <= 464 against a
  two-step ``ml_dtypes`` reference: bit-exact;
* saturation edges (|x| > 464, +/-inf -> 0x7E/0xFE) and NaN (-> 0x7F): match
  the PTX ``satfinite`` semantics;
* 1M random fp32 samples in range: bit-exact;
* upcast: exhaustive over all 256 e4m3fn byte patterns: bit-exact.

The helpers are compile-time switched: on sm89+ they forward to the native
Triton conversion (zero change to frozen behavior); on sm80 they expand to the
emulation. Kernel call sites replace expressions like::

    val.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.uint8, bitcast=True)
    raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

with ``to_e4m3_bits(val)`` / ``e4m3_bits_to_fp32(raw)``.
"""

from __future__ import annotations

import logging

import triton
import triton.language as tl


def _detect_hw_e4m3() -> bool:
    """True when the local GPU lowers tl.float8e4nv natively (sm89+)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return True  # kernels never launch without CUDA; keep frozen path
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (8, 9)
    except Exception as e:  # pragma: no cover - defensive
        logging.warning("[sm80_fp8_compat] capability probe failed (%s); assuming sm89+", e)
        return True


# Compile-time constant consumed by the @triton.jit helpers below. Callers may
# override per-kernel via the HW constexpr argument (used by tests).
HW_E4M3: tl.constexpr = tl.constexpr(_detect_hw_e4m3())


@triton.jit
def e4m3_bits_to_fp32(raw):
    """Upcast packed e4m3fn bytes (uint8) to fp32. Exact, arch-neutral.

    Layout: S EEEE MMMM, bias 7, no infinities, 0x7F/0xFF = NaN.
    """
    u = raw.to(tl.uint32)
    sign = (u >> 7) & 1
    em = u & 0x7F
    exp = em >> 3
    man = em & 0x7
    normal = (((exp + 120) << 23) | (man << 20)).to(tl.float32, bitcast=True)
    sub = man.to(tl.float32) * 0.001953125  # man * 2^-9
    val = tl.where(exp == 0, sub, normal)
    val = tl.where(em == 0x7F, float("nan"), val)
    return tl.where(sign == 1, -val, val).to(tl.float32)


@triton.jit
def _emu_downcast(x):
    """fp32 -> e4m3fn bits, two-step via fp16 with satfinite RTNE (see header)."""
    h = x.to(tl.float32).to(tl.float16)
    b16 = h.to(tl.uint16, bitcast=True)
    b = b16.to(tl.uint32)
    sign = (b >> 8) & 0x80
    exp16 = (b >> 10) & 0x1F
    man16 = b & 0x3FF
    absf = (b16 & 0x7FFF).to(tl.float16, bitcast=True).to(tl.float32)
    isnan = (exp16 == 31) & (man16 != 0)
    # Normal branch (|x| >= 2^-6): target exponent = exp16 - 8, keep 3 mantissa
    # bits with RTNE; carry into the exponent; saturate at the finite max.
    t_exp = exp16.to(tl.int32) - 8
    man3 = (man16 >> 7).to(tl.int32)
    rb = (man16 >> 6) & 1
    sticky = (man16 & 0x3F) != 0
    roundup = (rb == 1) & (sticky | ((man3 & 1) == 1))
    man3r = man3 + roundup.to(tl.int32)
    carry = man3r == 8
    man3r = tl.where(carry, 0, man3r)
    t_expr = t_exp + carry.to(tl.int32)
    sat = (t_expr >= 16) | ((t_expr == 15) & (man3r >= 7))
    normal_res = tl.where(sat, 0x7E, (t_expr << 3) | man3r).to(tl.uint32)
    # Subnormal/underflow branch (|x| < 2^-6): encodings are contiguous with the
    # normals, so RTNE rounding of |x| * 2^9 covers subnormals and the
    # subnormal->normal transition in one expression.
    q = absf * 512.0
    f = tl.math.floor(q)
    r = q - f
    fi = f.to(tl.int32)
    up = (r > 0.5) | ((r == 0.5) & ((fi & 1) == 1))
    sub_res = (f + up.to(tl.float32)).to(tl.int32).to(tl.uint32) & 0xF
    res = tl.where(absf < 0.015625, sub_res, normal_res)
    res = tl.where(isnan, 0x7F, res)
    return (res | sign).to(tl.uint8)


@triton.jit
def fp32_to_e4m3_bits(x, HW: tl.constexpr = HW_E4M3):
    """Downcast fp32/bf16/fp16 values to packed e4m3fn bytes (uint8).

    HW=True (sm89+): native Triton conversion (frozen behavior).
    HW=False (sm80): bit-op emulation matching the native PTX semantics.
    """
    if HW:
        # Untouched frozen path; no dtype pre-cast so bf16/fp16 inputs keep
        # their native direct conversion.
        return x.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.uint8, bitcast=True)
    return _emu_downcast(x.to(tl.float32))


@triton.jit
def e4m3_to_fp32(raw, HW: tl.constexpr = HW_E4M3):
    """Upcast e4m3fn bytes (uint8) to fp32. Emulation is exact and arch-neutral,
    so both branches compute the same thing; the HW flag is kept for symmetry
    and future specialization."""
    return e4m3_bits_to_fp32(raw)
