"""Exact GLM53 single-group routing, keeping native FTZ and tie semantics."""

from functools import lru_cache

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from triton.experimental.gluon.language.extra import libdevice

_NUM_WARPS = 4  # One warp per row; four independent rows share a CTA.


@gluon.jit(do_not_specialize=["M"])
def _route_single_group(
    logits,
    bias,
    weights,
    ids,
    E: tl.constexpr,
    STRIDE: tl.constexpr,
    BS: tl.constexpr,
    K: tl.constexpr,
    NORM: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    M,
    BM: tl.constexpr,
):
    layout: tl.constexpr = BlockedLayout([1, 1], [1, 32], [BM, 1], [1, 0])
    row = (
        tl.program_id(0).to(tl.int64) * BM
        + tl.arange(0, BM, layout=SliceLayout(1, layout))
    )[:, None]
    col = tl.arange(0, BLOCK, layout=SliceLayout(0, layout))[None, :]
    x = tl.load(logits + row * STRIDE + col, (row < M) & (col < E), 0)
    b = tl.load(bias + col * BS, col < E, 0)
    score = libdevice.div_rn(1.0, libdevice.add_rn(1.0, libdevice.exp(-x)))
    biased = libdevice.add_rn(score, b)
    # Torch sigmoid/add retain denormals; the original native TopK is built
    # with --ftz=true. Match only the native comparison/reduction stage here.
    biased = tl.inline_asm_elementwise(
        "add.rn.ftz.f32 $0, $1, 0f00000000;",
        constraints="=f,f",
        args=[biased],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    # Native top-2 ignores NaNs, retains infinities, and defaults the whole row
    # if its group score is nonfinite or the lowest finite FP32 value.
    group_values = tl.where((col < E) & (biased == biased), biased, -float("inf"))
    largest = tl.max(group_values, 1)[:, None]
    first = tl.min(tl.where(group_values == largest, col, 2147483647), 1)[:, None]
    second = tl.max(tl.where(col == first, -float("inf"), group_values), 1)[:, None]
    group = libdevice.add_rn(largest, second)
    lowest: tl.constexpr = -3.4028234663852886e38
    proceed = (tl.abs(group) < float("inf")) & (group != lowest)
    candidates = tl.where((col < E) & (tl.abs(biased) < float("inf")), biased, lowest)
    candidates = tl.where(col < E, candidates, -float("inf"))
    slots = tl.arange(0, 32, layout=SliceLayout(0, layout))[None, :]
    selected_ids = tl.full((BM, 32), 0, tl.int32, layout=layout)
    for i in tl.static_range(K):
        best = tl.max(candidates, 1)[:, None]
        index = tl.min(tl.where(candidates == best, col, 2147483647), 1)[:, None]
        selected_ids = tl.where(slots == i, index, selected_ids)
        candidates = tl.where(col == index, -float("inf"), candidates)
    # Recompute just K sigmoids after selection. This avoids K extra full-row
    # reductions and holding all E original scores live across the TopK loop.
    selected_x = tl.load(
        logits + row * STRIDE + selected_ids, (row < M) & (slots < K), 0
    )
    selected_scores = libdevice.div_rn(
        1.0, libdevice.add_rn(1.0, libdevice.exp(-selected_x))
    )
    if NORM:
        norm_scores = tl.inline_asm_elementwise(
            "add.rn.ftz.f32 $0, $1, 0f00000000;",
            constraints="=f,f",
            args=[tl.where(slots < K, selected_scores, 0.0)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        denominator = libdevice.add_rn(tl.sum(norm_scores, 1)[:, None], 1e-20)
    else:
        denominator = 1.0
    # Native CUDA divides FP32, then promotes to double for its scale argument.
    result = tl.inline_asm_elementwise(
        """{
          .reg .f32 ratio;
          .reg .f64 wide;
          div.rn.ftz.f32 ratio, $1, $2;
          cvt.ftz.f64.f32 wide, ratio;
          mul.f64 wide, wide, $3;
          cvt.rn.ftz.f32.f64 $0, wide;
        }""",
        constraints="=f,f,f,d",
        args=[selected_scores, denominator, tl.full((), SCALE, tl.float64)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    result = tl.where(proceed, result, 1.0 / K)
    selected_ids = tl.where(proceed, selected_ids, slots)
    tl.store(weights + row * K + slots, result, (row < M) & (slots < K))
    tl.store(ids + row * K + slots, selected_ids, (row < M) & (slots < K))


@lru_cache(maxsize=128)
def _runner(device, index_dtype, e, stride, bs, k, norm, scale, warps, m):
    # M controls only the grid and tail mask. Keep it runtime so variable
    # request lengths reuse the compiled code (including M=1).
    grid = (triton.cdiv(m, warps), 1, 1)
    constants = (e, stride, bs, k, norm, scale, triton.next_power_of_2(e), m, warps)
    kernel = _route_single_group.warmup(
        torch.float32,
        torch.float32,
        torch.float32,
        index_dtype,
        *constants,
        grid=grid,
        num_warps=warps,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return kernel[grid]


def fused_router_topk(weights, ids, scores, bias, groups, selected, k, norm, scale):
    """Write caller-owned outputs or return False for the ordinary native path."""
    if groups != 1 or selected != 1 or k != 8 or torch.version.hip is not None:
        return False
    if (
        scores.ndim != 2
        or scores.shape[1] != 288
        or not 0 < scores.shape[0] <= 16384
        or bias.shape != (288,)
        or weights.shape != (scores.shape[0], 8)
        or ids.shape != weights.shape
        or scores.dtype != torch.float32
        or bias.dtype != torch.float32
        or weights.dtype != torch.float32
        or ids.dtype not in (torch.int32, torch.int64)
        or not scores.is_cuda
    ):
        return False
    tensors = (scores, bias, weights, ids)
    device = scores.device
    if any(
        t.device != device or not t.is_contiguous() or t.requires_grad for t in tensors
    ):
        return False
    if device.index != torch.cuda.current_device():
        return False
    m = scores.shape[0]
    pointers = tuple(t.data_ptr() for t in tensors)
    sizes = (
        m * 288 * 4,
        288 * 4,
        m * 8 * 4,
        m * 8 * (8 if ids.dtype == torch.int64 else 4),
    )
    # Inputs may alias each other. Outputs must not overlap inputs or each other:
    # the fused kernel writes before other rows have necessarily read their logits.
    for output in (2, 3):
        for other in range(output):
            if (
                pointers[output] < pointers[other] + sizes[other]
                and pointers[other] < pointers[output] + sizes[output]
            ):
                return False
    warps = _NUM_WARPS or 4
    constants = (288, 288, 1, 8, norm, scale, 512, m, warps)
    if (
        all(ptr % 16 == 0 for ptr in pointers)
        and not torch.cuda.is_current_stream_capturing()
    ):
        _runner(device.index, ids.dtype, 288, 288, 1, 8, norm, scale, warps, m)(
            scores, bias, weights, ids, *constants
        )
    else:
        _route_single_group[(triton.cdiv(m, warps),)](
            scores,
            bias,
            weights,
            ids,
            *constants,
            num_warps=warps,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
    return True
