"""Replay the reference chunk-state math for short cached GLM prefills."""

import logging
import os
from collections import OrderedDict
from itertools import accumulate

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
    k,
    w,
    u,
    g,
    h,
    kd,
    wd,
    ud,
    gd,
    hd,
    NK: tl.constexpr,
    NW: tl.constexpr,
    NU: tl.constexpr,
    NG: tl.constexpr,
    NH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    tl.store(kd + i, tl.load(k + i, i < NK, 0), i < NK)
    tl.store(wd + i, tl.load(w + i, i < NW, 0), i < NW)
    tl.store(ud + i, tl.load(u + i, i < NU, 0), i < NU)
    tl.store(gd + i, tl.load(g + i, i < NG, 0), i < NG)
    tl.store(hd + i, tl.load(h + i, i < NH, 0), i < NH)


@triton.jit
def _copy_outputs(
    h,
    v,
    f,
    hd,
    vd,
    fd,
    NH: tl.constexpr,
    NV: tl.constexpr,
    NF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
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
        k.ndim != 4
        or k.shape[0] != 1
        or not 0 < k.shape[1] <= 16384
        or k.shape[-1] != 128
        or w.shape != k.shape
        or u.shape != k.shape
        or gk.shape != k.shape
        or initial_state.ndim != 4
        or not 1 <= initial_state.shape[0] <= 32
        or initial_state.shape[1:] != (k.shape[2], 128, 128)
        or any(t.dtype != torch.bfloat16 for t in (k, w, u))
        or gk.dtype != torch.float32
        or initial_state.dtype != torch.float32
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
    boundaries = (
        (0, k.shape[1]) if cu is None else tuple(_sequence_boundaries(cu, k.shape[1]))
    )
    lengths = tuple(stop - start for start, stop in zip(boundaries, boundaries[1:]))
    if (
        len(lengths) != initial_state.shape[0]
        or any(length <= 0 or length > 512 for length in lengths)
        or (
            len(lengths) > 1
            and os.environ.get("GLM53_KDA_BATCH_REUSE_GRAPH", "1") == "0"
        )
    ):
        return chunk_gated_delta_rule_fwd_h_cublas(**reference)

    sequence_count = len(lengths)
    graph_lengths = lengths
    if len(set(lengths)) == 1:
        # Adjacent dynamic batches reuse one graph. Add independent dummy
        # sequences, preserving the exact GEMM length of every real chunk.
        graph_lengths = lengths[:1] * triton.next_power_of_2(sequence_count)
    graph_tokens = sum(graph_lengths)
    graph_shape = torch.Size((1, graph_tokens, *k.shape[2:]))
    stream = torch.cuda.current_stream(k.device)
    # Identical total token counts can have different sequence/chunk layouts.
    key = (k.device.index, stream.cuda_stream, graph_shape, gk.dtype, graph_lengths)
    entry = _GRAPHS.get(key)
    inputs = (k, w, u, gk, initial_state)
    if entry is None:
        buffers = tuple(t.new_zeros(graph_shape) for t in inputs[:4]) + (
            initial_state.new_zeros((len(graph_lengths), *initial_state.shape[1:])),
        )
        captured = dict(reference)
        captured.update(zip(("k", "w", "u", "gk", "initial_state"), buffers))
        # The reference reads boundaries only on the host. Freeze a CPU copy
        # for capture, preserving every sequence's original GEMM and chunk order
        # without a device-to-host transfer inside the captured graph.
        captured["cu_seqlens"] = (
            None
            if len(graph_lengths) == 1
            else torch.tensor(
                (0, *accumulate(graph_lengths)),
                dtype=torch.int32,
                device="cpu",
            )
        )
        captured["chunk_indices"] = None
        capture_stream = torch.cuda.Stream(device=k.device)
        capture_stream.wait_stream(stream)
        with torch.cuda.stream(capture_stream):
            _copy_into(inputs, buffers)
            for _ in range(2):
                chunk_gated_delta_rule_fwd_h_cublas(**captured)
        stream.wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        # Cache IO can allocate/copy/synchronize on other host threads while
        # this graph is captured. Its buffers are private to this stream.
        with torch.cuda.graph(
            graph, stream=capture_stream, capture_error_mode="thread_local"
        ):
            outputs = chunk_gated_delta_rule_fwd_h_cublas(**captured)
        stream.wait_stream(capture_stream)
        entry = (buffers, graph, outputs)
        _GRAPHS[key] = entry
        while len(_GRAPHS) > _MAX_GRAPHS:
            _GRAPHS.popitem(last=False)
        logging.info(
            "[KDA reuse fusion] reference state graph enabled tokens=%d heads=%d sequences=%d capacity=%d",
            k.shape[1],
            k.shape[2],
            len(lengths),
            len(graph_lengths),
        )
    else:
        _GRAPHS.move_to_end(key)
    buffers, graph, outputs = entry
    _copy_into(inputs, buffers)
    graph.replay()
    real_chunks = sum(triton.cdiv(length, 64) for length in lengths)
    result = (
        outputs[0].new_empty((1, real_chunks, *outputs[0].shape[2:])),
        torch.empty_like(u),
        torch.empty_like(initial_state),
    )
    # Real sequences occupy a contiguous prefix of every captured output.
    sizes = tuple(t.numel() for t in result)
    _copy_outputs[(triton.cdiv(max(sizes), 256),)](*outputs, *result, *sizes, 256)
    return result
