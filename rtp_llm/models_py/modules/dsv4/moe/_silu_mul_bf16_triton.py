"""Fused SiLU + (optional clamp) + mul over a merged BF16 ``[M, 2N]`` buffer.

Replaces the V4.1 MXFP8 ``W13SharedExpert`` mid-chain::

    gate_up = w13(x).float()                # BF16 -> FP32 cast (kernel + alloc)
    gate, up = gate_up.chunk(2, dim=-1)     # two non-contiguous views
    hidden = silu_mul_split(                # two .contiguous() copies (kernels)
        gate.contiguous(),                  # + FP32 silu kernel + FP32 alloc
        up.contiguous(),
        clamp_limit=L,
    )
    hidden.to(x.dtype)                      # FP32 -> BF16 cast (kernel + alloc)

with a single Triton launch that reads both halves of the contiguous BF16
``gate_up`` in place (no chunk views, no contiguous copies, no FP32
intermediate) and writes the BF16 activation consumed by the following
MXFP8 linear.

``silu_mul_fp8_g32_quant`` additionally fuses the following group32 quantizer,
keeping the BF16 rounding in registers and returning only E4M3/UE8M0 outputs.
The BF16-only entry retains its original kernel and allocation contract.

Numerical contract (bit-identical to the chain above):
  - BF16 -> FP32 conversion is exact, so loading the BF16 halves and
    converting in-kernel yields the same FP32 ``gate``/``up`` values the old
    ``.float()`` produced.
  - The clamp/silu/mul order and FP32 accumulation match
    ``_silu_mul_split_triton.silu_mul_split`` exactly.
  - The old chain rounded once (FP32 ``hidden`` -> BF16 via ``.to(dtype)``);
    this kernel applies the same single round-to-nearest-even on store.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover — keep the module importable without Triton
    triton = None
    tl = None


if triton is not None:

    @triton.jit(do_not_specialize=["M"])
    def _silu_mul_split_bf16_kernel(
        gate_up_ptr,  # [M, 2N] bf16 row-major (row stride 2N)
        out_ptr,  # [M, N] bf16 row-major
        M,
        N: tl.constexpr,
        APPLY_CLAMP: tl.constexpr,  # bool
        CLAMP_LIMIT: tl.constexpr,  # float; ignored when APPLY_CLAMP is False
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0).to(tl.int64)
        pid_n = tl.program_id(axis=1).to(tl.int64)
        if pid_m >= M:
            return

        n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask = n_off < N
        row = pid_m * (2 * N)

        g = tl.load(gate_up_ptr + row + n_off, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(gate_up_ptr + row + N + n_off, mask=mask, other=0.0).to(tl.float32)

        if APPLY_CLAMP:
            # clamp(up, -L, L); clamp(gate, max=L) — same order as the split kernel.
            u = tl.where(u > CLAMP_LIMIT, CLAMP_LIMIT, u)
            u = tl.where(u < -CLAMP_LIMIT, -CLAMP_LIMIT, u)
            g = tl.where(g > CLAMP_LIMIT, CLAMP_LIMIT, g)

        # F.silu(g) = g * sigmoid(g), computed in fp32 (inputs converted above).
        s = g * tl.sigmoid(g)
        out = s * u

        tl.store(
            out_ptr + pid_m * N + n_off, out.to(out_ptr.dtype.element_ty), mask=mask
        )

    @triton.jit
    def _mul_ftz(a, b):
        return tl.inline_asm_elementwise(
            "mul.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit(do_not_specialize=["scale_stride", "legacy"])
    def _silu_mul_fp8_g32_quant_kernel(
        gate_up_ptr,
        q_ptr,
        scales_ptr,
        scale_stride,
        legacy,
        N: tl.constexpr,
        APPLY_CLAMP: tl.constexpr,
        CLAMP_LIMIT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        GROUPS: tl.constexpr = BLOCK_N // 32
        PACKS: tl.constexpr = GROUPS // 4
        pid_m = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1).to(tl.int64)
        n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask = n_off < N
        row = pid_m * (2 * N)
        g = tl.load(gate_up_ptr + row + n_off, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(gate_up_ptr + row + N + n_off, mask=mask, other=0.0).to(tl.float32)
        if APPLY_CLAMP:
            u = tl.where(u > CLAMP_LIMIT, CLAMP_LIMIT, u)
            u = tl.where(u < -CLAMP_LIMIT, -CLAMP_LIMIT, u)
            g = tl.where(g > CLAMP_LIMIT, CLAMP_LIMIT, g)
        s = g * tl.sigmoid(g)
        # Quantize the same BF16-rounded value as the two-launch chain.
        y = (s * u).to(tl.bfloat16).to(tl.float32).reshape((GROUPS, 32))
        # Native fmaxf ignores NaNs and seeds each lane with FP32 tiny.
        amax = tl.max(
            tl.maximum(tl.abs(y), 2.0**-126, propagate_nan=tl.PropagateNan.NONE), 1
        )
        if legacy:
            raw = tl.maximum(amax / 448.0, 1.0e-10)
            scale = tl.exp2(tl.ceil(tl.log2(raw)))
            scaled = y / scale[:, None]
            biased = tl.log2(scale).to(tl.int32) + 127
        else:
            # CUDA13 native v2 uses FTZ and fast_log2_ceil/fast_pow2.
            raw = _mul_ftz(amax, 1.0 / 448.0)
            bits = raw.to(tl.int32, bitcast=True)
            biased = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
            reciprocal = ((254 - biased) << 23).to(tl.float32, bitcast=True)
            scaled = _mul_ftz(y, reciprocal[:, None])
        q = tl.minimum(
            tl.maximum(scaled, -448.0, propagate_nan=tl.PropagateNan.NONE), 448.0
        ).reshape((BLOCK_N,))
        tl.store(q_ptr + pid_m * N + n_off, q.to(tl.float8e4nv), mask=mask)
        group = pid_n * GROUPS + tl.arange(0, GROUPS)
        biased = tl.where(group < N // 32, biased, 0).to(tl.uint8).to(tl.uint32)
        packed = tl.sum(biased.reshape((PACKS, 4)) << (tl.arange(0, 4)[None, :] * 8), 1)
        pack = pid_n * PACKS + tl.arange(0, PACKS)
        tl.store(
            scales_ptr + pack * scale_stride + pid_m, packed, pack < tl.cdiv(N, 128)
        )


def silu_mul_split_bf16(
    gate_up: torch.Tensor,  # [M, 2N] bf16, contiguous
    clamp_limit: float = 0.0,
    out: Optional[torch.Tensor] = None,  # [M, N] bf16, contiguous
) -> torch.Tensor:
    """Fused SiLU + optional SwiGLU clamp + mul over a merged BF16 buffer.

    Equivalent to the split-input chain documented in the module docstring;
    see :func:`rtp_llm.models_py.modules.dsv4._silu_mul_split_triton.silu_mul_split`
    for the clamp semantics (``clamp(up, ±L)`` + ``clamp(gate, max=L)``).
    """
    M, two_n = gate_up.shape
    N = two_n // 2
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=gate_up.device)
    if M * N == 0:
        return out
    block_n = 1024
    grid = (M, triton.cdiv(N, block_n))
    apply_clamp = clamp_limit is not None and clamp_limit > 0.0
    _silu_mul_split_bf16_kernel[grid](
        gate_up,
        out,
        M,
        N=N,
        APPLY_CLAMP=apply_clamp,
        CLAMP_LIMIT=float(clamp_limit) if apply_clamp else 0.0,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out


def silu_mul_fp8_g32_quant(
    gate_up: torch.Tensor, clamp_limit: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU followed by native group32 E4M3/UE8M0 quantization.

    The MXFP8 caller supplies contiguous BF16 [M,2N], with N divisible by
    32. Preserve the BF16 rounding boundary without materializing hidden.
    Quantization honors the existing per-call DSV4_FP8_QUANT_KERNEL policy;
    auto selects v2 at M*N >= 4Mi elements. Legacy's unused scale-pack bytes
    are unspecified; this entry initializes them to zero, as v2 does.
    """
    m, two_n = gate_up.shape
    n = two_n // 2
    quant = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=gate_up.device)
    scales = torch.empty(
        (triton.cdiv(n, 128), triton.cdiv(m, 4) * 4),
        dtype=torch.int32,
        device=gate_up.device,
    ).T[:m]
    if m * n == 0:
        return quant, scales
    mode = os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
    legacy = mode == "legacy" or (mode == "auto" and m * n < 4 * 1024 * 1024)
    apply_clamp = clamp_limit is not None and clamp_limit > 0.0
    _silu_mul_fp8_g32_quant_kernel[(m, triton.cdiv(n, 1024))](
        gate_up,
        quant,
        scales,
        scales.stride(1),
        legacy,
        N=n,
        APPLY_CLAMP=apply_clamp,
        CLAMP_LIMIT=float(clamp_limit) if apply_clamp else 0.0,
        BLOCK_N=1024,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return quant, scales
