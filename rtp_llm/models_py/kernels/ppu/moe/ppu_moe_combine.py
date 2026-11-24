"""Scale routed FP32 sums, round to BF16, then add the shared TP partial."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _combine(
    routed, shared, output, N: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    r = tl.load(routed + offsets, offsets < N, 0)
    s = tl.load(shared + offsets, offsets < N, 0).to(tl.float32)
    # SG ep_gather stores the scaled accumulator in BF16 before shared add.
    # Preserve that rounding boundary, even though both operations are fused.
    scaled = (r * SCALE).to(tl.bfloat16).to(tl.float32)
    tl.store(output + offsets, scaled + s, offsets < N)


def is_supported(routed, shared):
    return (
        routed.is_cuda
        and shared.device == routed.device
        and routed.dtype == torch.float32
        and shared.dtype == torch.bfloat16
        and routed.ndim == 2
        and routed.shape == shared.shape
        and routed.is_contiguous()
        and shared.is_contiguous()
    )


def combine_tp_partials(routed, shared, route_scale):
    if not is_supported(routed, shared):
        raise ValueError(
            "PPU MoE combine requires same-shape contiguous FP32/BF16 matrices"
        )
    if not math.isfinite(route_scale):
        raise ValueError("MoE route scale must be finite")
    out = torch.empty_like(shared)
    if out.numel():
        _combine[(triton.cdiv(out.numel(), 1024),)](
            routed,
            shared,
            out,
            out.numel(),
            float(route_scale),
            1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _add_cast_bf16(a, b, output, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(a + offsets, mask, 0)
    y = tl.load(b + offsets, mask, 0)
    tl.store(output + offsets, (x + y).to(tl.bfloat16), mask)


def add_pair_to_bf16(a, b):
    """Fused FP32 add + RNE BF16 cast matching (a + b).bfloat16()."""
    if not a.is_cuda:
        # CPU oracle path (contract tests); GPU always takes the fused kernel.
        return (a + b).bfloat16()
    if (
        a.dtype != torch.float32
        or b.dtype != torch.float32
        or a.shape != b.shape
        or not a.is_contiguous()
        or not b.is_contiguous()
    ):
        raise ValueError("fused add/cast requires same-shape contiguous FP32 inputs")
    out = torch.empty(a.shape, dtype=torch.bfloat16, device=a.device)
    n = a.numel()
    BLOCK = 2048
    _add_cast_bf16[(triton.cdiv(n, BLOCK),)](a, b, out, n, BLOCK=BLOCK)
    return out
