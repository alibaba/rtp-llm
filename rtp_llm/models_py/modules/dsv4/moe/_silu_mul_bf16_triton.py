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

from typing import Optional

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
    if triton is None:
        raise RuntimeError("DSV4 fused BF16 SiLU path requires Triton")
    if not gate_up.is_cuda:
        raise RuntimeError("DSV4 fused BF16 SiLU path requires CUDA tensors")
    if gate_up.dim() != 2 or gate_up.dtype != torch.bfloat16:
        raise ValueError(
            f"gate_up must be 2D bf16, got dim={gate_up.dim()} dtype={gate_up.dtype}"
        )
    if not gate_up.is_contiguous():
        raise ValueError("gate_up must be contiguous")
    M, two_n = gate_up.shape
    if two_n % 2 != 0:
        raise ValueError(f"gate_up last dim must be even, got {two_n}")
    N = two_n // 2
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=gate_up.device)
    elif (
        tuple(out.shape) != (M, N)
        or out.dtype != torch.bfloat16
        or out.device != gate_up.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous bf16 [M, N] on the gate_up device; got "
            f"shape={tuple(out.shape)}, dtype={out.dtype}"
        )
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
