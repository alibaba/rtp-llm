"""Compact block-scaled MoE for replicated inputs on Hopper."""

from contextvars import ContextVar
import os
from typing import Literal

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import ep_gather_global_routes
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    invoke_fused_moe_kernel,
)
from rtp_llm.models_py.utils.device_capability import sm90_pdl_enabled

from rtp_llm.models_py.triton_kernels.moe.decode_alignment import moe_align_decode, moe_align_large_decode

# Capture records the dispatch selected for this forward. Replay keeps it.
h20_compact_moe_phase: ContextVar[
    Literal["eager", "decode", "target_verify"] | None
] = ContextVar("h20_compact_moe_phase", default=None)
compact_moe_output_buffer: ContextVar[torch.Tensor | None] = ContextVar(
    "compact_moe_output_buffer", default=None
)

_STABLE_ALIGNMENT_MAX_ROUTES = 5120
_DENSE_EXPERT_ROWS = 128
_H20_COMPACT_EAGER_MAX_ROWS = 5120

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
def _silu_mul_quantize(
    up_gate,
    ids,
    quantized,
    scales,
    GROUPS: tl.constexpr,
    GLOBAL_START: tl.constexpr = 0,
    EXPERTS: tl.constexpr = 256,
    PDL: tl.constexpr = False,
):
    route = tl.program_id(0)
    group = tl.program_id(1)
    expert = tl.load(ids + route)
    local = (expert >= GLOBAL_START) & (expert < GLOBAL_START + EXPERTS)
    if PDL:
        tl.extra.cuda.gdc_wait()
    if local:
        col = group * 128 + tl.arange(0, 128)
        row = up_gate + route * (GROUPS * 256)
        up = tl.load(row + col)
        gate = tl.load(row + GROUPS * 128 + col).to(tl.float32)
        value = up * (gate / (1.0 + tl.exp(-gate)))
        scale = tl.maximum(tl.max(tl.abs(value)), 1e-10) / 448.0
        q = tl.clamp(value / scale, -448.0, 448.0)
        tl.store(quantized + route * (GROUPS * 128) + col, q)
        tl.store(scales + route * GROUPS + group, scale)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _silu_mul_quantize_tiles(
    up_gate,
    ids,
    quantized,
    scales,
    routes,
    GROUPS: tl.constexpr,
    ROWS: tl.constexpr,
    GLOBAL_START: tl.constexpr = 0,
    EXPERTS: tl.constexpr = 256,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group = tl.program_id(1)
    expert = tl.load(ids + row, row < routes, -1)
    live = (expert >= GLOBAL_START) & (expert < GLOBAL_START + EXPERTS)
    col = group * 128 + tl.arange(0, 128)
    offset = row[:, None] * (GROUPS * 256) + col[None, :]
    if PDL:
        tl.extra.cuda.gdc_wait()
    up = tl.load(up_gate + offset, live[:, None], 0)
    gate = tl.load(up_gate + offset + GROUPS * 128, live[:, None], 0).to(tl.float32)
    value = up * (gate / (1.0 + tl.exp(-gate)))
    scale = tl.maximum(tl.max(tl.abs(value), axis=1), 1e-10) / 448.0
    q = tl.clamp(value / scale[:, None], -448.0, 448.0)
    offset = row[:, None] * (GROUPS * 128) + col[None, :]
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(quantized + offset, q, live[:, None])
    tl.store(scales + row * GROUPS + group, scale, live)


def _project(
    x,
    x_scale,
    weights,
    out,
    ids,
    route_weights,
    top_k,
    plans,
    *,
    pre_ready_routing: bool = False,
):
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
            pre_ready_routing=pre_ready_routing,
        )


def compact_fp8_decode(
    x,
    x_scale,
    ids,
    route_weights,
    expert_counts,
    up_weights,
    down_weights,
    *,
    global_start: int = 0,
):
    """Run local routes and keep BF16 rounding before the FP32 weighted sum."""
    tokens, hidden = x.shape
    top_k = ids.shape[1]
    experts, _, intermediate = down_weights[0].shape
    routes = tokens * top_k
    large = routes > _STABLE_ALIGNMENT_MAX_ROUTES
    if large:
        if global_start != 0 or expert_counts is None:
            raise ValueError("large compact MoE requires local expert IDs and counts")
        if ids.dtype != torch.int32 or not ids.is_contiguous():
            ids = ids.to(dtype=torch.int32).contiguous()
        is_graph = (
            torch.cuda.is_current_stream_capturing()
            or os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD") == "1"
        )
        if not is_graph:
            if tokens < 768:
                bm, bn, nw, ns, sw = 16, 128, 4, 3, True
            elif tokens < 1792:
                bm, bn, nw, ns, sw = 32, 64, 4, 3, True
            elif tokens <= 4096:
                bm, bn, nw, ns, sw = 64, 128, 8, 4, False
            else:
                bm, bn, nw, ns, sw = 32, 64, 4, 3, True
            single_cfg = {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "num_warps": nw,
                "num_stages": ns,
            }
            small, _ = moe_align_large_decode(
                ids, expert_counts, bm, 64, routes + 1
            )
            plans = ((single_cfg, sw, small),)
        else:
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
        if not ids.is_contiguous():
            ids = ids.contiguous()
        layout = moe_align_decode(
            ids,
            _GEMM_CONFIG["BLOCK_SIZE_M"],
            experts,
            max_routes=_STABLE_ALIGNMENT_MAX_ROUTES,
            global_start=global_start,
            allow_unordered=routes > 640,
        )
        plans = ((_GEMM_CONFIG, True, layout),)
    flat_ids = ids.view(-1)
    flat_weights = route_weights.reshape(-1)

    up = x.new_empty((routes, 2 * intermediate), dtype=torch.bfloat16)
    _project(x, x_scale, up_weights, up, flat_ids, flat_weights, top_k, plans)

    groups = intermediate // _SCALE_BLOCK[0]
    down_input = x.new_empty((routes, intermediate))
    down_scale = x.new_empty((routes, groups), dtype=torch.float32)
    pdl = sm90_pdl_enabled(x.device)
    if large or routes > 640:
        tile_rows = 8 if routes >= 5120 else 4
        _silu_mul_quantize_tiles[(triton.cdiv(routes, tile_rows), groups)](
            up,
            flat_ids,
            down_input,
            down_scale,
            routes,
            groups,
            tile_rows,
            GLOBAL_START=global_start,
            EXPERTS=experts,
            PDL=pdl,
            num_warps=4,
            launch_pdl=pdl,
        )
    else:
        _silu_mul_quantize[(routes, groups)](
            up,
            flat_ids,
            down_input,
            down_scale,
            groups,
            GLOBAL_START=global_start,
            EXPERTS=experts,
            PDL=pdl,
            num_warps=1,
            launch_pdl=pdl,
        )
    down = x.new_empty((routes, hidden), dtype=torch.bfloat16)
    _project(
        down_input,
        down_scale,
        down_weights,
        down,
        flat_ids,
        flat_weights,
        1,
        plans,
        pre_ready_routing=True,
    )

    staged_out = compact_moe_output_buffer.get()
    if (
        staged_out is not None
        and staged_out.shape == (tokens, hidden)
        and staged_out.dtype == torch.bfloat16
        and staged_out.device == x.device
        and staged_out.is_contiguous()
    ):
        out = staged_out
    else:
        out = x.new_empty((tokens, hidden), dtype=torch.bfloat16)
    ep_gather_global_routes(down, ids, route_weights, out, global_start, experts)
    return out
