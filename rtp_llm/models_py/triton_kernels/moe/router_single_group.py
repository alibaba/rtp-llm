"""Single-group sigmoid router; preserves stable expert ordering and FP32 scores."""

import torch
import triton as tr
import triton.language as tl
from rtp_llm.models_py.triton_kernels.common import fast_launch as _cpu_launch
from triton.language.extra.cuda import libdevice

_kernels = {}


@tr.jit
def _div(x, y):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@tr.jit
def _add(x, y):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@tr.jit
def _cuda_route_div(x, y):
    # The CUDA routing extension uses FTZ for its final division; Torch's
    # preceding sigmoid/bias do not. Keep those two numerical contracts apart.
    return tl.inline_asm_elementwise(
        "div.rn.ftz.f32 $0, $1, $2;",
        "=f,f,f",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@tr.jit
def _route(
    X,
    B,
    W,
    IDS,
    E: tl.constexpr,
    K: tl.constexpr,
    NORM: tl.constexpr,
    FACTOR: tl.constexpr,
):
    row = tl.program_id(0)
    e = tl.arange(0, E)
    x = tl.load(X + row * E + e).to(tl.float32)
    s = _div(1.0, 1.0 + libdevice.exp(-x))
    biased = _add(s, tl.load(B + e))
    # CUDA's comparisons/reductions flush subnormal operands, unlike the
    # Torch bias addition. This also preserves lower-ID ties near zero.
    biased = tl.inline_asm_elementwise(
        "add.ftz.f32 $0, $1, 0f00000000;",
        "=f,f",
        [biased],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    # The CUDA top-two group reduction ignores NaNs for E > 32. Infinite
    # group sums trigger its uniform fallback, even in a single group.
    group_values = tl.where(biased == biased, biased, -float("inf"))
    first = tl.max(group_values, 0)
    first_id = tl.min(tl.where(group_values == first, e, 2147483647), 0)
    second = tl.max(tl.where(e == first_id, -float("inf"), group_values), 0)
    group_sum = first + second
    valid = (tl.abs(group_sum) < float("inf")) & (group_sum != -3.4028234663852886e38)
    candidates = tl.where(tl.abs(biased) < float("inf"), biased, -3.4028234663852886e38)
    lane = tl.arange(0, 32)
    selected_s = tl.full((32,), 0, tl.float32)
    selected_i = tl.full((32,), 0, tl.int32)
    for i in tl.static_range(K):
        best = tl.max(candidates, 0)
        idx = tl.min(tl.where(candidates == best, e, 2147483647), 0)
        value = tl.sum(tl.where(e == idx, s, 0.0), 0)
        selected_s = tl.where(lane == i, value, selected_s)
        selected_i = tl.where(lane == i, idx, selected_i)
        candidates = tl.where(e == idx, -float("inf"), candidates)
    denom = _add(tl.sum(selected_s, 0), 1.0e-20) if NORM else 1.0
    weight = (
        _cuda_route_div(selected_s, denom).to(tl.float64)
        * tl.full((), FACTOR, tl.float64)
    ).to(tl.float32)
    weight = tl.where(valid, weight, 1.0 / K)
    index = tl.where(valid, selected_i, lane)
    tl.store(W + row * K + lane, weight, lane < K)
    tl.store(IDS + row * K + lane, index, lane < K)


def try_single_group_router(
    x, bias, weights, ids, n_group, topk_group, topk, renormalize, factor
):
    if not (
        n_group == topk_group == 1
        and topk == 8
        and x.is_cuda
        and x.ndim == 2
        and x.shape[1] == 256
        and x.is_contiguous()
        and x.dtype in (torch.float32, torch.bfloat16)
        and not x.requires_grad
        and bias.device == x.device
        and bias.dtype == torch.float32
        and bias.shape == (256,)
        and bias.is_contiguous()
        and not bias.requires_grad
        and weights.device == ids.device == x.device
        and weights.dtype == torch.float32
        and ids.dtype in (torch.int32, torch.int64)
        and weights.shape == ids.shape == (x.shape[0], topk)
        and weights.is_contiguous()
        and ids.is_contiguous()
    ):
        return False
    if x.shape[0]:
        args = (x, bias, weights, ids, 256, topk, renormalize, factor)
        key = (
            x.device.index,
            x.dtype,
            ids.dtype,
            x.data_ptr() % 16,
            bias.data_ptr() % 16,
            weights.data_ptr() % 16,
            ids.data_ptr() % 16,
            renormalize,
            factor,
        )
        kernel = _kernels.get(key)
        if kernel is None:
            kernel = _route[(x.shape[0], 1, 1)](
                *args, num_warps=1, enable_fp_fusion=False
            )
            if len(_kernels) >= 128:
                _kernels.pop(next(iter(_kernels)))
            _kernels[key] = kernel
        else:
            _cpu_launch.launch_compiled(kernel, (x.shape[0], 1, 1), args)
    return True
