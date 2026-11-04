"""Fused residual-add + per-layer scalar Triton kernel (one pass, one launch).

Contract 6 (opt r8, BinaryFunctor 138ms/39300x elimination): the
DiffusionGemma layer tail used to run, per layer forward,

    hidden_states = residual + hidden_states    # CUDAFunctor_add launch
    return hidden_states * scalar               # BinaryFunctor Mul launch

— two launches and three full [M, H] bf16 passes (read a, read b, write
sum, read sum, write out) on tensors that are pure launch-overhead at the
canvas M=256 x H=2816 shape. The elementwise chain is a single
``out = (a + b) * scalar`` expression, so one Triton launch reads a and b
once, applies the fp32 add and scale in registers, and writes the bf16 out
once (the "residual+mul+add 单核多元素融合" contract item).

Value contract (A/B referee ``[dgm-addscale-ab]``, DGM_ADDSCALE_AB=1): the
torch chain rounds the bf16 sum before the fp32 scale-mul, the kernel keeps
fp32 through both and rounds once at the store — <= 1 bf16 ulp, the same
ruling class as the fused add-norm (107ba336c1). ``scalar`` may be a 0-dim /
1-element device tensor (the layer_scalar weights) or None (pure add).

The kernel compiles through the normal Triton JIT (TRITON_CACHE_DIR, the
categorical_* kernel / jit cache manager pattern).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_NUM_WARPS = 8


@triton.jit
def _add_scale_kernel(
    a_ptr,
    b_ptr,
    s_ptr,
    out_ptr,
    n_elements,
    HAS_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 offsets: M * H reaches 4096 * 2816 > 2^24 comfortably but stays
    # < 2^31; keep int64 anyway — the same kernel serves any canvas/prefill.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.float32)
    acc = a + b
    if HAS_SCALE:
        s = tl.load(s_ptr).to(tl.float32)
        acc = acc * s
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


def add_scale_usable(
    a: torch.Tensor, b: torch.Tensor, scalar: Optional[torch.Tensor]
) -> bool:
    """Gate: CUDA bf16 contiguous flat-compatible pair, same shape/stride."""
    if not (a.is_cuda and a.dim() == 2 and a.dtype == torch.bfloat16):
        return False
    if b.dtype != torch.bfloat16 or b.shape != a.shape:
        return False
    if a.stride() != b.stride() or a.stride(1) != 1:
        return False
    if a.data_ptr() == b.data_ptr():
        return False
    if scalar is not None:
        if scalar.numel() != 1:
            return False
        if scalar.dtype not in (torch.bfloat16, torch.float32, torch.float16):
            return False
        if scalar.device != a.device:
            return False
    return True


def add_scale(
    a: torch.Tensor, b: torch.Tensor, scalar: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``(a + b) * scalar`` in one launch: bf16 [M, N] contiguous pair in,
    bf16 out. ``scalar=None`` degenerates to the pure add (same fusion)."""
    n = a.numel()
    out = torch.empty_like(a)
    if n == 0:
        return out
    grid = (triton.cdiv(n, _BLOCK),)
    _add_scale_kernel[grid](
        a,
        b,
        scalar if scalar is not None else a,
        out,
        n,
        HAS_SCALE=scalar is not None,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out
