"""Replay the existing fixed-row FP32 projections without Python chunk loops."""

import logging
import os
from collections import Counter, OrderedDict
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from rtp_llm.models_py.triton_kernels.common.host_math import ceil_div

_CACHE = OrderedDict()
_STATS = Counter()
_MAX_BYTES = 512 * 1024**2
_COMPILED_DISPATCH = os.environ.get("GLM53_COMPILED_PROJECTION_DISPATCH", "0") == "1"
_COMPACT_MIDDLE = os.environ.get("GLM53_COMPACT_PROJECTION_MIDDLE", "0") == "1"
_COMPACT_TAIL = os.environ.get("GLM53_COMPACT_PROJECTION_TAIL", "0") == "1"


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


@lru_cache(maxsize=128)
def _pack_runner(device, dtype, capacity, weights):
    # Keep code only, with device-specific CUDA handles and no tensor storage.
    # An odd runtime size avoids assumptions about its divisibility/alignment.
    grid = (ceil_div(max(capacity, weights), 1024), 1, 1)
    kernel = _pack.warmup(
        dtype,
        torch.float32,
        dtype,
        torch.float32,
        13,
        capacity,
        weights,
        1024,
        grid=grid,
    )
    return kernel[grid]


@lru_cache(maxsize=128)
def _copy_runner(device, size):
    grid = (ceil_div(size, 1024), 1, 1)
    kernel = _copy.warmup(torch.float32, torch.float32, 13, 1024, grid=grid)
    # The public runner resolves the current stream for every call.
    return kernel[grid]


def _projection_entry(x, weight, rows, capacity, stream):
    hidden, outputs = x.shape[1], weight.shape[0]
    estimate = 2 * (
        capacity * hidden * x.element_size()
        + hidden * outputs * 4
        + capacity * outputs * 4
        + rows * hidden * 4
    )
    if estimate > _MAX_BYTES:
        _STATS["workspace_limit"] += 1
        return None
    key = (
        x.device.index,
        stream.cuda_stream,
        x.dtype,
        hidden,
        outputs,
        rows,
        capacity,
    )
    entry = _CACHE.get(key)
    if entry is None:
        px = x.new_empty((capacity, hidden))
        pw = torch.empty_like(weight)
        output = weight.new_empty((capacity, outputs))

        def reference():
            for begin in range(0, capacity, rows):
                output[begin : begin + rows].copy_(
                    F.linear(px[begin : begin + rows].float(), pw)
                )

        capture_stream = torch.cuda.Stream(device=x.device)
        capture_stream.wait_stream(stream)
        with torch.cuda.stream(capture_stream):
            _pack[(ceil_div(max(px.numel(), pw.numel()), 1024),)](
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
        logging.info(
            "[GLM53 fixed projection] capture hidden=%d outputs=%d rows=%d capacity=%d input_rows=%d",
            hidden,
            outputs,
            rows,
            capacity,
            x.shape[0],
        )
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
    return entry


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
    tokens, hidden = x.shape
    outputs = weight.shape[0]
    weight_elements = hidden * outputs
    # Keep only a small-row and a bounded slab template per projection shape.
    # Larger batches replay the same slab twice; every GEMM still has `rows`
    # rows, including the independent zero rows of its final chunk.
    capacity = rows if tokens <= rows else max(1, 8192 // rows) * rows
    if (
        _COMPACT_MIDDLE
        and rows == 512
        and x.dtype == torch.bfloat16
        and (hidden, outputs) in ((4096, 32), (4096, 288), (8192, 32), (8192, 256))
        and 512 < tokens <= 2048
    ):
        # Preserve the exact 512-row GEMMs; omit twelve independent zero-row
        # chunks for these middle-sized batches. Keep the original memory cap.
        capacity = 2048
    stream = torch.cuda.current_stream(x.device.index)
    entry = _projection_entry(x, weight, rows, capacity, stream)
    if entry is None:
        return None
    px, pw, output, graph, _ = entry
    compiled_dispatch = (
        _COMPILED_DISPATCH
        # Each following slab must retain 16-byte pointer alignment for all
        # supported input dtypes and FP32 output widths, including odd widths.
        and capacity % 8 == 0
        and x.device.index == torch.cuda.current_device()
        and x.data_ptr() % 16 == weight.data_ptr() % 16 == 0
    )
    packed_elements = capacity * hidden
    pack = (
        _pack_runner(x.device.index, x.dtype, packed_elements, weight_elements)
        if compiled_dispatch
        else _pack[(ceil_div(max(packed_elements, weight_elements), 1024),)]
    )
    result = output.new_empty((tokens, outputs))
    compact_tail = (
        _COMPACT_TAIL
        and rows == 512
        and x.dtype == torch.bfloat16
        and (hidden, outputs) in ((4096, 32), (4096, 288), (8192, 32), (8192, 256))
        and 8192 < tokens <= 10240
    )
    for begin in range(0, tokens, capacity):
        end = min(begin + capacity, tokens)
        slab = x[begin:end]
        if compact_tail and begin:
            # Use the existing 512/2048-row template for the last slab, while
            # preserving every real 512-row GEMM and the owned output buffer.
            tail_capacity = 512 if end - begin <= 512 else 2048
            entry = _projection_entry(slab, weight, rows, tail_capacity, stream)
            if entry is None:
                return None
            px, pw, output, graph, _ = entry
            packed_elements = tail_capacity * hidden
            pack = (
                _pack_runner(x.device.index, x.dtype, packed_elements, weight_elements)
                if compiled_dispatch
                else _pack[(ceil_div(max(packed_elements, weight_elements), 1024),)]
            )
        pack(
            slab,
            weight,
            px,
            pw,
            (end - begin) * hidden,
            packed_elements,
            weight_elements,
            1024,
        )
        graph.replay()
        _STATS["replays"] += 1
        destination = result[begin:end]
        copy = (
            _copy_runner(x.device.index, (end - begin) * outputs)
            if compiled_dispatch
            else _copy[(ceil_div((end - begin) * outputs, 1024),)]
        )
        copy(destination, output, (end - begin) * outputs, 1024)
    return result
