"""Architecture gates for the dsv41 SM100-native fast path.

The frozen dsv41 execution stack (DeepGEMM ``fp8_fp4_gemm_nt`` /
``fp8_fp4_mega_moe``, FlashMLA, mega symmetric-memory MoE) targets SM100+
(``capability[0] >= 10``, matching ``docs/dsv4/v41_runtime_defaults.md``'s
"major == 10 admission" contract). On pre-Blackwell GPUs (A100/sm80, L20/sm89)
those kernels cannot launch, so weight-GEMM and attention paths must take the
reference implementations instead:

* FP4/FP8 weights stay in their packed checkpoint form in HBM (no new
  quantization, no format substitution); the reference path dequantizes
  exactly (e2m1/e4m3/ue8m0 all embed losslessly into fp32/BF16) and computes
  with BF16 cuBLAS GEMMs.
* Triton e4m3 conversion sites go through ``sm80_fp8_compat`` (hardware cvt on
  sm89+, bit-op emulation with identical PTX semantics on sm80).

Single source of truth so every dsv41 call site gates identically.
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def has_sm100_native_kernels() -> bool:
    """True when DeepGEMM FP4/FP8 + FlashMLA fast paths can run locally."""
    try:
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        return major >= 10
    except Exception:
        return False


def use_reference_gemm_paths() -> bool:
    """True when quantized-weight GEMMs must use exact-dequant references."""
    return not has_sm100_native_kernels()


# ---------------------------------------------------------------------------
# Exact dequantization references (torch-level, arch-neutral).
#
# Checkpoint formats embed losslessly into fp32/BF16:
#   * e2m1 (FP4): 1+2+1 bits, values {0,±0.5,±1,±1.5,±2,±3,±4,±6}
#   * e4m3fn (FP8): 3-bit mantissa ⊂ BF16 7-bit mantissa
#   * ue8m0 scales: powers of two, exact in fp32/BF16 range
# So the reference GEMM paths below change no numerics of the stored weights;
# they only compute the same products through BF16 cuBLAS instead of SM100
# FP4/FP8 tensor-core kernels.

def fp4_packed_to_bf16(weight_int8: torch.Tensor, scale_ue8m0: torch.Tensor) -> torch.Tensor:
    """Dequantize packed FP4 ``[N, K/2]`` int8 + UE8M0 ``[N, K/32]`` -> BF16 ``[N, K]``.

    Nibble order matches the checkpoint loader (low nibble = even column),
    identical to ``qlinear._fp4_unpack_to_fp32`` but without the int64 LUT
    temporaries (bit-arithmetic decode, ~4x less transient memory).
    """
    w = weight_int8.view(torch.uint8).to(torch.int32)
    lo = w & 0xF
    hi = (w >> 4) & 0xF

    def _dec(n: torch.Tensor) -> torch.Tensor:
        s = (n >> 3) & 1
        e = (n >> 1) & 3
        m = n & 1
        # OCP e2m1: e == 0 -> m * 0.5 (subnormal); e > 0 -> (1 + m/2) * 2^(e-1)
        mag = torch.where(
            e == 0,
            m.to(torch.float32) * 0.5,
            (2 + m).to(torch.float32) * torch.exp2((e - 2).to(torch.float32)),
        )
        return torch.where(s == 1, -mag, mag)

    n, half_k = w.shape
    out = torch.empty((n, half_k * 2), dtype=torch.float32, device=w.device)
    out[:, 0::2] = _dec(lo)
    out[:, 1::2] = _dec(hi)
    scale = scale_ue8m0.view(torch.uint8).to(torch.int32)
    scale_f = torch.exp2((scale - 127).to(torch.float32)).repeat_interleave(32, dim=1)[:, : half_k * 2]
    return (out * scale_f).to(torch.bfloat16)


def fp8_block32_to_bf16(weight_e4m3: torch.Tensor, scales_e8m0: torch.Tensor) -> torch.Tensor:
    """Dequantize e4m3fn ``[N, K]`` with 32x32 UE8M0 block scales -> BF16.

    Exact: e4m3 significands (4 bits) and power-of-two scales both represent
    losslessly in BF16.
    """
    n, k = weight_e4m3.shape
    rows = torch.arange(n, device=scales_e8m0.device) // 32
    cols = torch.arange(k, device=scales_e8m0.device) // 32
    scale_f = scales_e8m0.to(torch.float32)[rows][:, cols]
    return (weight_e4m3.to(torch.float32) * scale_f).to(torch.bfloat16)


def mxfp8_packed_input_to_bf16(x_q: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
    """Dequantize the MXFP8 activation pair produced by unpack_packed_rows.

    ``x_q``: ``[T, W]`` e4m3fn values. ``x_s``: ``[T, ceil(W/32/4)]`` int32
    words (MN-major as_strided view); byte j of word c is the UE8M0 exponent
    of group 4*c+j (little endian), one exponent per 32-value group.

    Exact: e4m3 significands embed in BF16 and UE8M0 scales are powers of
    two, so this loses nothing relative to the SM100 fp8-activation GEMM
    inputs; only the GEMM itself changes (BF16 cuBLAS, fp32 accumulate).
    """
    t, w = x_q.shape
    groups = w // 32
    words = x_s.to(torch.int64)
    scale_bytes = torch.stack(
        [
            words & 0xFF,
            (words >> 8) & 0xFF,
            (words >> 16) & 0xFF,
            (words >> 24) & 0xFF,
        ],
        dim=-1,
    ).reshape(t, -1)[:, :groups]
    scales = torch.exp2((scale_bytes - 127).to(torch.float32))
    vals = x_q.float().reshape(t, groups, 32) * scales.unsqueeze(-1)
    return vals.reshape(t, w).to(torch.bfloat16)
