"""Fused SiLU + (optional clamp) + element-wise mul for V4 Expert.forward.

Replaces the per-expert chain in ``moe.py:Expert.forward``::

    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)   # 1 launch
        gate = torch.clamp(gate, max=swiglu_limit)                  # 1 launch
    x = F.silu(gate) * up                                            # 2 launches

with one Triton launch.  Inputs/outputs are FP32 (Expert.forward casts
the FP8 GEMM output via ``.float()`` upstream).

Why a *split* kernel (vs the existing ``silu_and_mul`` that takes a
concatenated ``[B, 2N]`` tensor): w1 and w3 are separate FP8 GEMMs
producing separate ``gate`` / ``up`` tensors, so feeding the existing
kernel would require a ``torch.cat`` (one extra launch) and a buffer
allocation.  The split-input kernel avoids both.

Shape contract (V4-Flash):
  gate, up, out: [..., D] FP32, contiguous along the last dim.
  D is moe_intermediate_size for shared-expert (e.g. 2048).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_split_kernel(
    gate_ptr,    # [N, D] fp32 contiguous
    up_ptr,      # [N, D] fp32 contiguous
    out_ptr,     # [N, D] fp32 contiguous
    N: tl.int32,
    D: tl.int32,
    row_stride: tl.int32,    # stride between rows for gate/up/out (= D for contiguous)
    CLAMP_LIMIT: tl.constexpr,   # float; <= 0 disables the clamp branch
    APPLY_CLAMP: tl.constexpr,   # bool
    BLOCK_D: tl.constexpr,
):
    """One program per (row, D-block).  Streams D in BLOCK_D-wide tiles."""
    pid_n = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    if pid_n >= N:
        return

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D
    base = pid_n * row_stride

    g = tl.load(gate_ptr + base + d_off, mask=mask, other=0.0)
    u = tl.load(up_ptr + base + d_off, mask=mask, other=0.0)

    if APPLY_CLAMP:
        # clamp(up, -L, L); clamp(gate, max=L)
        u = tl.where(u > CLAMP_LIMIT, CLAMP_LIMIT, u)
        u = tl.where(u < -CLAMP_LIMIT, -CLAMP_LIMIT, u)
        g = tl.where(g > CLAMP_LIMIT, CLAMP_LIMIT, g)

    # F.silu(g) = g * sigmoid(g) (compute in fp32; inputs already fp32).
    s = g * tl.sigmoid(g)
    out = s * u

    tl.store(out_ptr + base + d_off, out, mask=mask)


def silu_mul_split(
    gate: torch.Tensor,    # [..., D] fp32 contiguous
    up: torch.Tensor,      # [..., D] fp32 contiguous, same shape as gate
    clamp_limit: float = 0.0,    # > 0 enables clamp(up,±L) + clamp(gate, max=L)
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused SiLU + optional SwiGLU clamp + multiply.

    Equivalent to (matching ``moe.py:Expert.forward``)::
        if clamp_limit > 0:
            up = torch.clamp(up, -clamp_limit, clamp_limit)
            gate = torch.clamp(gate, max=clamp_limit)
        return F.silu(gate) * up

    Returns ``out`` (allocated when None).
    """
    assert gate.shape == up.shape, f"gate {gate.shape} vs up {up.shape}"
    assert gate.dtype == torch.float32 and up.dtype == torch.float32, (
        f"gate {gate.dtype} / up {up.dtype}; expect fp32"
    )
    assert gate.is_contiguous() and up.is_contiguous(), "gate/up must be contiguous"

    # Flatten leading dims; kernel works on [N, D].
    orig_shape = gate.shape
    D = orig_shape[-1]
    N = gate.numel() // D

    if out is None:
        out = torch.empty_like(gate)
    else:
        assert out.shape == gate.shape and out.dtype == torch.float32
        assert out.is_contiguous()

    if N == 0 or D == 0:
        return out

    g_flat = gate.reshape(N, D)
    u_flat = up.reshape(N, D)
    o_flat = out.reshape(N, D)

    BLOCK_D = 1024 if D >= 1024 else triton.next_power_of_2(D)
    grid = (N, triton.cdiv(D, BLOCK_D))

    _silu_mul_split_kernel[grid](
        g_flat, u_flat, o_flat,
        N=N, D=D,
        row_stride=g_flat.stride(0),
        CLAMP_LIMIT=float(clamp_limit) if clamp_limit > 0 else 0.0,
        APPLY_CLAMP=clamp_limit > 0,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out
