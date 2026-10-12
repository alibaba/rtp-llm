"""Replay the existing fixed-row FP32 projections without Python chunk loops."""

import os
from collections import Counter, OrderedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_CACHE = OrderedDict()
_STATS = Counter()
_MAX_BYTES = 512 * 1024**2


def projection_graph_stats():
    return {
        **_STATS,
        "entries": len(_CACHE),
        "bytes": sum(e[4] for e in _CACHE.values()),
    }


@triton.jit(do_not_specialize=["REAL"])
def _pack(
    x,
    w,
    px,
    pw,
    REAL,
    CAP: tl.constexpr,
    WEIGHTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    real = REAL.to(tl.int64)
    tl.store(px + i, tl.load(x + i, i < real, 0), i < CAP)
    tl.store(pw + i, tl.load(w + i, i < WEIGHTS, 0), i < WEIGHTS)


@triton.jit(do_not_specialize=["SIZE"])
def _copy(out, src, SIZE, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    size = SIZE.to(tl.int64)
    tl.store(out + i, tl.load(src + i, i < size, 0), i < size)


def fixed_projection_graph(x, weight, rows):
    if os.environ.get("GLM53_FIXED_PROJECTION_GRAPH", "0") == "0":
        return None
    if (
        x.ndim != 2
        or weight.ndim != 2
        or not x.is_cuda
        or weight.device != x.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or weight.dtype != torch.float32
        or x.shape[1] != weight.shape[1]
        or not 0 < x.shape[0] <= 16384
        or not 0 < rows <= 8192
        or x.requires_grad
        or weight.requires_grad
        or torch.is_autocast_enabled()
        or torch.backends.cuda.matmul.allow_tf32
        or torch.cuda.is_current_stream_capturing()
    ):
        _STATS["unsupported"] += 1
        return None
    capacity = triton.next_power_of_2(triton.cdiv(x.shape[0], rows)) * rows
    estimate = 2 * (
        capacity * x.shape[1] * x.element_size()
        + weight.numel() * 4
        + capacity * weight.shape[0] * 4
        + rows * x.shape[1] * 4
    )
    if estimate > _MAX_BYTES:
        _STATS["workspace_limit"] += 1
        return None
    stream = torch.cuda.current_stream(x.device)
    key = (
        x.device.index,
        stream.cuda_stream,
        x.dtype,
        x.shape[1],
        weight.shape[0],
        rows,
        capacity,
    )
    entry = _CACHE.get(key)
    if entry is None:
        px = x.new_empty((capacity, x.shape[1]))
        pw = torch.empty_like(weight)
        output = weight.new_empty((capacity, weight.shape[0]))

        def reference():
            for begin in range(0, capacity, rows):
                output[begin : begin + rows].copy_(
                    F.linear(px[begin : begin + rows].float(), pw)
                )

        capture_stream = torch.cuda.Stream(device=x.device)
        capture_stream.wait_stream(stream)
        with torch.cuda.stream(capture_stream):
            _pack[(triton.cdiv(max(px.numel(), pw.numel()), 1024),)](
                x, weight, px, pw, x.numel(), px.numel(), pw.numel(), 1024
            )
            for _ in range(2):
                reference()
        stream.wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, stream=capture_stream, capture_error_mode="thread_local"
        ):
            reference()
        stream.wait_stream(capture_stream)
        pool = tuple(graph.pool())
        pool_bytes = sum(
            s["total_size"]
            for s in torch.cuda.memory_snapshot()
            if tuple(s.get("segment_pool_id", (0, 0))) == pool
            and s["device"] == x.device.index
        )
        used = (
            px.numel() * px.element_size()
            + pw.numel() * 4
            + output.numel() * 4
            + pool_bytes
        )
        if not pool_bytes:
            used = estimate
        entry = (px, pw, output, graph, used)
        _CACHE[key] = entry
        _STATS["captures"] += 1
        while (
            len(_CACHE) > 16
            or sum(e[4] for k, e in _CACHE.items() if k[0] == x.device.index)
            > _MAX_BYTES
        ):
            if len(_CACHE) == 1:
                break
            _CACHE.popitem(last=False)
            _STATS["evictions"] += 1
    else:
        _CACHE.move_to_end(key)
    px, pw, output, graph, _ = entry
    _pack[(triton.cdiv(max(px.numel(), pw.numel()), 1024),)](
        x, weight, px, pw, x.numel(), px.numel(), pw.numel(), 1024
    )
    graph.replay()
    _STATS["replays"] += 1
    result = output.new_empty((x.shape[0], weight.shape[0]))
    _copy[(triton.cdiv(result.numel(), 1024),)](result, output, result.numel(), 1024)
    return result
