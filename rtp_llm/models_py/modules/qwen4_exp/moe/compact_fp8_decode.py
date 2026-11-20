"""Compact block-scaled MoE for replicated decode inputs on Hopper."""

from contextvars import ContextVar

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import ep_gather
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    invoke_fused_moe_kernel,
)

from .moe_align_triton import moe_align_decode, moe_align_large_decode

# The model supplies the phase at capture time; Graph replay uses that path.
h20_compact_moe_decode: ContextVar[bool] = ContextVar(
    "h20_compact_moe_decode", default=False
)

_STABLE_ALIGNMENT_MAX_ROUTES = 2560
_DENSE_EXPERT_ROWS = 128

# H20 WGMMA sees the 64 output columns as its M dimension. The token
# dimension is 16, so each expert needs only ceil(actual_rows / 16) tiles.
_GEMM_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 4,
    "num_warps": 4,
    "num_stages": 3,
}
_SCALE_BLOCK = [128, 128]


@triton.jit
def _silu_mul_quantize(up_gate, ids, quantized, scales, GROUPS: tl.constexpr):
    route = tl.program_id(0)
    group = tl.program_id(1)
    if tl.load(ids + route) >= 0:
        col = group * 128 + tl.arange(0, 128)
        row = up_gate + route * (GROUPS * 256)
        up = tl.load(row + col)
        gate = tl.load(row + GROUPS * 128 + col).to(tl.float32)
        value = up * (gate / (1.0 + tl.exp(-gate)))
        scale = tl.maximum(tl.max(tl.abs(value)), 1e-10) / 448.0
        q = tl.clamp(value / scale, -448.0, 448.0)
        tl.store(quantized + route * (GROUPS * 128) + col, q)
        tl.store(scales + route * GROUPS + group, scale)


@triton.jit
def _silu_mul_quantize_tiles(
    up_gate, ids, quantized, scales, routes, GROUPS: tl.constexpr, ROWS: tl.constexpr
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group = tl.program_id(1)
    expert = tl.load(ids + row, row < routes, -1)
    live = expert >= 0
    col = group * 128 + tl.arange(0, 128)
    offset = row[:, None] * (GROUPS * 256) + col[None, :]
    up = tl.load(up_gate + offset, live[:, None], 0)
    gate = tl.load(up_gate + offset + GROUPS * 128, live[:, None], 0).to(tl.float32)
    value = up * (gate / (1.0 + tl.exp(-gate)))
    scale = tl.maximum(tl.max(tl.abs(value), axis=1), 1e-10) / 448.0
    q = tl.clamp(value / scale[:, None], -448.0, 448.0)
    offset = row[:, None] * (GROUPS * 128) + col[None, :]
    tl.store(quantized + offset, q, live[:, None])
    tl.store(scales + row * GROUPS + group, scale, live)


def _project(x, x_scale, weights, out, ids, route_weights, top_k, plans):
    for config, swap_ab, (sorted_ids, expert_ids, padded) in plans:
        invoke_fused_moe_kernel(
            x,
            weights[0],
            out,
            route_weights,
            ids,
            sorted_ids,
            expert_ids,
            padded,
            False,
            top_k,
            config,
            tl.bfloat16,
            A_scale=x_scale,
            B_scale=weights[1],
            block_shape=_SCALE_BLOCK,
            swap_ab=swap_ab,
        )


def compact_fp8_decode(
    x, x_scale, ids, route_weights, expert_counts, up_weights, down_weights
):
    """Run local routes and keep BF16 rounding before the FP32 weighted sum."""
    tokens, hidden = x.shape
    top_k = ids.shape[1]
    experts, _, intermediate = down_weights[0].shape
    routes = tokens * top_k
    # PureTP emits only local expert IDs and -1 for routes owned by other ranks.
    ids = ids.to(dtype=torch.int32).contiguous()
    large = routes > _STABLE_ALIGNMENT_MAX_ROUTES
    if large:
        # Keep smaller tiles for sparse routes. Group heavily used experts
        # into 64-row tiles to reuse the same weights across more tokens.
        small_m, small_n = (32, 64) if tokens >= 1024 else (16, 128)
        small_config = dict(_GEMM_CONFIG, BLOCK_SIZE_M=small_m, BLOCK_SIZE_N=small_n)
        dense_config = dict(_GEMM_CONFIG, BLOCK_SIZE_M=64)
        small, dense = moe_align_large_decode(
            ids,
            expert_counts,
            small_m,
            dense_config["BLOCK_SIZE_M"],
            _DENSE_EXPERT_ROWS,
        )
        plans = ((small_config, True, small), (dense_config, False, dense))
    else:
        layout = moe_align_decode(
            ids,
            _GEMM_CONFIG["BLOCK_SIZE_M"],
            experts,
            max_routes=_STABLE_ALIGNMENT_MAX_ROUTES,
        )
        plans = ((_GEMM_CONFIG, True, layout),)
    flat_ids = ids.view(-1)
    flat_weights = route_weights.reshape(-1)

    up = x.new_empty((routes, 2 * intermediate), dtype=torch.bfloat16)
    _project(x, x_scale, up_weights, up, flat_ids, flat_weights, top_k, plans)

    groups = intermediate // _SCALE_BLOCK[0]
    down_input = x.new_empty((routes, intermediate))
    down_scale = x.new_empty((routes, groups), dtype=torch.float32)
    if large:
        _silu_mul_quantize_tiles[(triton.cdiv(routes, 4), groups)](
            up, flat_ids, down_input, down_scale, routes, groups, 4, num_warps=4
        )
    else:
        _silu_mul_quantize[(routes, groups)](
            up, flat_ids, down_input, down_scale, groups, num_warps=1
        )
    down = x.new_empty((routes, hidden), dtype=torch.bfloat16)
    _project(
        down_input, down_scale, down_weights, down, flat_ids, flat_weights, 1, plans
    )

    out = x.new_empty((tokens, hidden), dtype=torch.bfloat16)
    input_indices = torch.arange(routes, device=x.device, dtype=torch.int32).view_as(
        ids
    )
    ep_gather(down, ids, route_weights, input_indices, out)
    return out
