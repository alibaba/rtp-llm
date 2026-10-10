"""Replay the reference chunk-state math for short cached GLM prefills."""

import logging
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_delta_h import (
    _sequence_boundaries,
    chunk_gated_delta_rule_fwd_h_cublas,
)

# Shared across layers on one stream. Returned outputs have independent storage.
# Bound shape-dependent graph memory for varying incremental request lengths.
_GRAPHS = OrderedDict()
_MAX_GRAPHS = 8


@triton.jit
def _copy_inputs(
    k, w, u, g, h, kd, wd, ud, gd, hd,
    NK: tl.constexpr, NW: tl.constexpr, NU: tl.constexpr,
    NG: tl.constexpr, NH: tl.constexpr, BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(kd + i, tl.load(k + i, i < NK, 0), i < NK)
    tl.store(wd + i, tl.load(w + i, i < NW, 0), i < NW)
    tl.store(ud + i, tl.load(u + i, i < NU, 0), i < NU)
    tl.store(gd + i, tl.load(g + i, i < NG, 0), i < NG)
    tl.store(hd + i, tl.load(h + i, i < NH, 0), i < NH)


@triton.jit
def _copy_outputs(
    h, v, f, hd, vd, fd,
    NH: tl.constexpr, NV: tl.constexpr, NF: tl.constexpr, BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(hd + i, tl.load(h + i, i < NH, 0), i < NH)
    tl.store(vd + i, tl.load(v + i, i < NV, 0), i < NV)
    tl.store(fd + i, tl.load(f + i, i < NF, 0), i < NF)


def _copy_into(inputs, buffers):
    sizes = tuple(t.numel() for t in inputs)
    _copy_inputs[(triton.cdiv(max(sizes), 256),)](*inputs, *buffers, *sizes, 256)


def _supported(k, w, u, gk, initial_state, options):
    tensors = (k, w, u, gk, initial_state)
    if any(t is None or not t.is_cuda or not t.is_contiguous() for t in tensors):
        return False
    if any(t.device != k.device for t in tensors):
        return False
    if (
        k.ndim != 4 or k.shape[0] != 1 or not 0 < k.shape[1] <= 512
        or k.shape[-1] != 128 or w.shape != k.shape or u.shape != k.shape
        or gk.shape != k.shape or initial_state.shape != (1, k.shape[2], 128, 128)
        or any(t.dtype != torch.bfloat16 for t in (k, w, u))
        or gk.dtype != torch.float32 or initial_state.dtype != torch.float32
        or options.get("g") is not None
        or options.get("chunk_size", 64) != 64
        or not options.get("use_exp2", True)
        or options.get("transpose_state_layout", False)
        or not options.get("output_final_state", False)
        or not options.get("save_new_value", True)
        or options.get("intermediate_state_dtype") != torch.float32
    ):
        return False
    return not torch.cuda.is_current_stream_capturing()


def chunk_gated_delta_rule_fwd_h_reuse_graph(k, w, u, gk, initial_state, **options):
    """Keep cuBLAS full-K accumulation and replay its existing CUDA operations.

    The former Triton fusion split K into 64-element partial products and added
    a dot directly to a nonzero FP32 state. Both differ from the reference math.
    This graph changes CPU dispatch only. Unsupported shapes retain the same
    reference backend, and each call returns fresh output tensors.
    """
    reference = dict(k=k, w=w, u=u, gk=gk, initial_state=initial_state, **options)
    if not _supported(k, w, u, gk, initial_state, options):
        return chunk_gated_delta_rule_fwd_h_cublas(**reference)
    cu = options.get("cu_seqlens")
    if cu is not None:
        if cu.numel() != 2:
            return chunk_gated_delta_rule_fwd_h_cublas(**reference)
        _sequence_boundaries(cu, k.shape[1])

    stream = torch.cuda.current_stream(k.device)
    key = (k.device.index, stream.cuda_stream, k.shape, gk.dtype)
    entry = _GRAPHS.get(key)
    inputs = (k, w, u, gk, initial_state)
    if entry is None:
        buffers = tuple(torch.empty_like(t) for t in inputs)
        captured = dict(reference)
        captured.update(zip(("k", "w", "u", "gk", "initial_state"), buffers))
        # One sequence has identical layout with or without cu_seqlens. Keeping
        # it None removes host metadata reads from the captured reference loop.
        captured["cu_seqlens"] = None
        captured["chunk_indices"] = None
        capture_stream = torch.cuda.Stream(device=k.device)
        capture_stream.wait_stream(stream)
        with torch.cuda.stream(capture_stream):
            _copy_into(inputs, buffers)
            for _ in range(2):
                chunk_gated_delta_rule_fwd_h_cublas(**captured)
        stream.wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            outputs = chunk_gated_delta_rule_fwd_h_cublas(**captured)
        stream.wait_stream(capture_stream)
        entry = (buffers, graph, outputs)
        _GRAPHS[key] = entry
        while len(_GRAPHS) > _MAX_GRAPHS:
            _GRAPHS.popitem(last=False)
        logging.info(
            "[KDA reuse fusion] reference state graph enabled tokens=%d heads=%d",
            k.shape[1], k.shape[2],
        )
    else:
        _GRAPHS.move_to_end(key)
    buffers, graph, outputs = entry
    _copy_into(inputs, buffers)
    graph.replay()
    result = tuple(torch.empty_like(t) for t in outputs)
    sizes = tuple(t.numel() for t in outputs)
    _copy_outputs[(triton.cdiv(max(sizes), 256),)](
        *outputs, *result, *sizes, 256
    )
    return result
