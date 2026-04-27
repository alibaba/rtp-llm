"""DeepSeek-V4 fused RMSNorm Triton kernel — replaces the per-tensor
6-launch chain ``x.float() → square() → mean(-1) → +eps → rsqrt() →
mul → cast`` in attention.py.

Replaces three call sites in attention.py (final_plan.md P4, minimal C):

  1. ``qr = self._rmsnorm_weighted(self._lin(self.wq_a, x), self.q_norm.weight)``
     — Q-LoRA RMSNorm with learned weight, D=q_lora_rank=1024.
  2. ``q = q * torch.rsqrt(q.float().square().mean(-1) + eps).to(q.dtype)``
     — per-head Q-RMSNorm, NO weight, D=head_dim=512.
  3. ``kv = self._rmsnorm_weighted(self._lin(self.wkv, x), self.kv_norm.weight)``
     — KV RMSNorm with learned weight, D=head_dim=512.

All three normalize the LAST dim of the input.  The kernel takes the
input flattened to [N, D] and an optional weight [D]; one program per
row.  D ≤ 4096 fits in registers; we cap BLOCK_D at next_power_of_2(D)
which is 512 / 1024 in practice.

Numerical fidelity matches REF: fp32 reduction (square + mean), fp32
rsqrt, fp32 weight multiply, cast back to input dtype at the end.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _v4_rmsnorm_fwd(
    x_ptr,                  # [N, D] input
    w_ptr,                  # [D] fp32 weight (or 0 if HAS_WEIGHT==False)
    out_ptr,                # [N, D] output, same dtype as x
    x_n: tl.constexpr,      # x stride along N; D-stride==1 assumed
    out_n: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    x_ptrs = x_ptr + pid * x_n + d_off
    x = tl.load(x_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / D
    inv = tl.rsqrt(var + EPS)
    y = x * inv

    if HAS_WEIGHT:
        w = tl.load(w_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
        y = y * w

    out_ptrs = out_ptr + pid * out_n + d_off
    # Triton infers the store dtype from out_ptr's element dtype; the
    # `.to(...)` cast happens implicitly via the store's dtype.
    tl.store(out_ptrs, y, mask=d_mask)


def v4_rmsnorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMSNorm over the last dim of ``x``.

    ``x`` may be any shape; the last dim is the feature dim.  Returns
    output in ``x``'s dtype with the same shape.  ``weight`` (if given)
    must match the last dim and may be any float dtype (cast to fp32
    inside the kernel).

    Math is bit-equivalent (within fp32 noise) to:

        x32 = x.float()
        rms = (x32.square().mean(-1, keepdim=True) + eps).rsqrt()
        y = (x32 * rms * (weight if weight is not None else 1.0)).to(x.dtype)
    """
    assert x.is_cuda
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D).contiguous()
    N = x_2d.shape[0]
    out = torch.empty_like(x_2d)

    if N == 0:
        return out.view(*orig_shape)

    if weight is not None:
        assert weight.shape == (D,), f"weight shape {weight.shape} != ({D},)"
        weight = weight.contiguous()
        if weight.dtype != torch.float32:
            weight = weight.float()
        has_weight = True
    else:
        # Pass a dummy 1-element fp32 buffer; the kernel guards on
        # HAS_WEIGHT before dereferencing.
        weight = torch.empty(1, dtype=torch.float32, device=x.device)
        has_weight = False

    BLOCK_D = triton.next_power_of_2(D)
    assert BLOCK_D <= 4096, f"D={D} too large for single-row kernel"

    _v4_rmsnorm_fwd[(N,)](
        x_2d, weight, out,
        x_2d.stride(0), out.stride(0),
        D=D, BLOCK_D=BLOCK_D, EPS=eps,
        HAS_WEIGHT=has_weight,
        num_warps=4 if BLOCK_D <= 512 else 8,
    )
    return out.view(*orig_shape)
