"""Replay exact KDA math with length-grouped layouts, independent of request order."""

import logging
import os
from collections import Counter, OrderedDict
from contextlib import nullcontext
from itertools import accumulate

import torch
import triton
import triton.language as tl
from rtp_llm.models_py.modules.dsv4.forward_metadata import (
    _FORWARD_METADATA,
    metadata_cache,
    tensor_key,
)
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_delta_h import _sequence_boundaries

_GRAPHS = OrderedDict()
_STATS = Counter()
_MAX_GRAPHS = 256
_MAX_BYTES = 16 * 1024**3
_BUDGETS = {}
_BASE_LENGTHS = {}


def _workspace_budget(device):
    if device.index not in _BUDGETS:
        # Bounded by device capacity; query properties only when building a
        # new layout, never cudaMemGetInfo in the steady launch path.
        total = torch.cuda.get_device_properties(device).total_memory
        _BUDGETS[device.index] = min(_MAX_BYTES, total // 8)
    return _BUDGETS[device.index]


def graph_stats():
    from rtp_llm.models_py.modules.glm53_fixed_projection_graph import (
        projection_graph_stats,
    )

    return {
        **{
            f"projection_{key}": value
            for key, value in projection_graph_stats().items()
        },
        **_STATS,
        "entries": len(_GRAPHS),
        "estimated_bytes": sum(e[4] for e in _GRAPHS.values()),
    }


@triton.jit
def _pack_inputs(
    q,
    k,
    v,
    g,
    beta,
    state,
    a,
    bias,
    cu,
    inverse,
    pq,
    pk,
    pv,
    pg,
    pb,
    ps,
    pa,
    pd,
    pcu,
    HD: tl.constexpr,
    H: tl.constexpr,
    STATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    seq = tl.load(inverse + slot).to(tl.int64)
    start = tl.load(cu + seq, seq >= 0, 0).to(tl.int64)
    target = tl.load(pcu + slot).to(tl.int64)
    length = tl.load(pcu + slot + 1).to(tl.int64) - target
    src = start * HD + i
    dst = target * HD + i
    m = i < length * HD
    tl.store(pq + dst, tl.load(q + src, m & (seq >= 0), 0), m)
    tl.store(pk + dst, tl.load(k + src, m & (seq >= 0), 0), m)
    tl.store(pv + dst, tl.load(v + src, m & (seq >= 0), 0), m)
    tl.store(pg + dst, tl.load(g + src, m & (seq >= 0), 0), m)
    mb = i < length * H
    tl.store(pb + target * H + i, tl.load(beta + start * H + i, mb & (seq >= 0), 0), mb)
    ms = i < STATE
    tl.store(
        ps + slot * STATE + i, tl.load(state + seq * STATE + i, ms & (seq >= 0), 0), ms
    )
    if slot == 0:
        tl.store(pa + i, tl.load(a + i, i < H, 0), i < H)
        tl.store(pd + i, tl.load(bias + i, i < HD, 0), i < HD)


@triton.jit
def _unpack_outputs(
    o,
    f,
    h,
    out,
    final,
    checkpoints,
    cu,
    forward,
    real_chunks,
    pcu,
    packed_chunks,
    HD: tl.constexpr,
    STATE: tl.constexpr,
    BLOCK: tl.constexpr,
    selected=None,
    SUBSET: tl.constexpr = False,
):
    seq = tl.program_id(0).to(tl.int64)
    if SUBSET:
        seq = tl.load(selected + seq).to(tl.int64)
    i = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    slot = tl.load(forward + seq).to(tl.int64)
    start = tl.load(cu + seq).to(tl.int64)
    length = tl.load(cu + seq + 1).to(tl.int64) - start
    packed_start = tl.load(pcu + slot).to(tl.int64)
    m = i < length * HD
    tl.store(out + start * HD + i, tl.load(o + packed_start * HD + i, m, 0), m)
    mf = i < STATE
    tl.store(final + seq * STATE + i, tl.load(f + slot * STATE + i, mf, 0), mf)
    source_chunk = tl.load(packed_chunks + slot).to(tl.int64)
    target_chunk = tl.load(real_chunks + seq).to(tl.int64)
    mh = i < tl.cdiv(length, 64) * STATE
    tl.store(
        checkpoints + target_chunk * STATE + i,
        tl.load(h + source_chunk * STATE + i, mh, 0),
        mh,
    )


def _device_indices(values, device):
    host = torch.tensor(values, dtype=torch.int64, device="cpu", pin_memory=True)
    return host.to(device, non_blocking=True), host


def _layout(lengths, device):
    cache = metadata_cache()
    key = ("kda_full_graph_layout", lengths, device)
    if cache is not None and key in cache:
        return cache[key]
    groups = {}
    for seq, length in enumerate(lengths):
        groups.setdefault(length, []).append(seq)
    if len(groups) == 2:
        # Reserve both lengths for every request. A majority switch (e.g.
        # 7/32 -> 21/18 in a 39-request batch) must not capture a new graph.
        # Padding adds independent sequences, preserving real partial GEMMs.
        # Small batches use <=4x tokens; larger eight-row buckets use <=3x.
        capacity = (
            triton.next_power_of_2(len(lengths))
            if len(lengths) <= 16
            else triton.cdiv(len(lengths), 8) * 8
        )
        capacities = {length: capacity for length in groups}
    else:
        capacities = {
            length: triton.next_power_of_2(len(sequences))
            for length, sequences in groups.items()
        }
    packed_lengths, inverse = [], []
    forward = [0] * len(lengths)
    for length, sequences in sorted(groups.items()):
        capacity = capacities[length]
        first = len(packed_lengths)
        packed_lengths.extend([length] * capacity)
        inverse.extend(sequences + [-1] * (capacity - len(sequences)))
        for index, seq in enumerate(sequences):
            forward[seq] = first + index
    real_offsets = (0, *accumulate(triton.cdiv(length, 64) for length in lengths))
    gpu = [_device_indices(v, device) for v in (inverse, forward, real_offsets)]
    result = (tuple(packed_lengths), tuple(v[0] for v in gpu), tuple(v[1] for v in gpu))
    if cache is not None:
        cache[key] = result
    return result


def _reference(buffers, cu, indices, norm_rows):
    # Import lazily to keep chunk.py's public wrapper free of an import cycle.
    from rtp_llm.models_py.triton_kernels.kimi_kda.chunk_fwd import chunk_kda_fwd

    q, k, v, g, beta, state, a, bias = buffers
    q = l2norm_fwd(q, config_rows=norm_rows)
    k = l2norm_fwd(k, config_rows=norm_rows)
    result = chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        state,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=indices,
        chunk_size=64,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=a,
        dt_bias=bias,
        return_intermediate_states=True,
        fuse_state_recurrence=False,
    )
    return result[0], result[1], result[10]


@triton.jit
def _unpack_group_outputs(
    o,
    f,
    h,
    out,
    final,
    checkpoints,
    cu,
    sequences,
    real_chunks,
    LENGTH: tl.constexpr,
    HD: tl.constexpr,
    STATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local = tl.program_id(0).to(tl.int64)
    seq = tl.load(sequences + local).to(tl.int64)
    i = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    target = tl.load(cu + seq).to(tl.int64)
    m = i < LENGTH * HD
    tl.store(out + target * HD + i, tl.load(o + local * LENGTH * HD + i, m, 0), m)
    mf = i < STATE
    tl.store(final + seq * STATE + i, tl.load(f + local * STATE + i, mf, 0), mf)
    target_chunk = tl.load(real_chunks + seq).to(tl.int64)
    chunks: tl.constexpr = (LENGTH + 63) // 64
    mh = i < chunks * STATE
    tl.store(
        checkpoints + target_chunk * STATE + i,
        tl.load(h + local * chunks * STATE + i, mh, 0),
        mh,
    )


def _length_group_plans(lengths, device, skip=0):
    cache = metadata_cache()
    key = ("kda_length_group_plans", lengths, device, skip)
    if key in cache:
        return cache[key]
    real_chunks, real_host = _device_indices(
        (0, *accumulate(triton.cdiv(n, 64) for n in lengths)), device
    )
    groups = {}
    for seq, length in enumerate(lengths):
        groups.setdefault(length, []).append(seq)
    plans = []
    for length, sequences in sorted(groups.items()):
        sequences = sequences[skip:]
        if not sequences:
            continue
        ids, ids_host = _device_indices(sequences, device)
        boundaries = tuple(i * length for i in range(len(sequences) + 1))
        host = torch.tensor(boundaries, dtype=torch.int32, pin_memory=True)
        cu = host.to(device, non_blocking=True)
        cache[("kda_sequence_boundaries", tensor_key(cu), boundaries[-1])] = boundaries
        plans.append((length, len(sequences), ids, cu, ids_host, host))
    result = (plans, real_chunks, real_host)
    cache[key] = result
    return result


def _base_layout(lengths, device, heads):
    if os.environ.get("GLM53_KDA_BASE_GROUP_GRAPH", "0") == "0" or len(lengths) <= 16:
        return None
    # Reserve only 1/2 complete requests per observed length. Excess requests
    # retain their exact length graph. This never pads a real partial GEMM.
    capacity = 1 if len(lengths) <= 48 else 2
    stream = torch.cuda.current_stream(device).cuda_stream
    identity = (device.index, stream, heads, capacity)
    universe = _BASE_LENGTHS.setdefault(identity, set())
    universe.update(lengths)
    if len(universe) > 64 or capacity * sum(universe) > 3 * sum(lengths):
        return None
    cache = metadata_cache()
    key = ("kda_base_layout", identity, lengths, tuple(sorted(universe)))
    if key in cache:
        return cache[key]
    groups = {}
    for seq, length in enumerate(lengths):
        groups.setdefault(length, []).append(seq)
    packed, inverse, selected = [], [], []
    forward = [-1] * len(lengths)
    for length in sorted(universe):
        sequences = groups.get(length, [])[:capacity]
        first = len(packed)
        packed.extend([length] * capacity)
        inverse.extend(sequences + [-1] * (capacity - len(sequences)))
        for slot, seq in enumerate(sequences):
            forward[seq] = first + slot
            selected.append(seq)
    chunks = (0, *accumulate(triton.cdiv(n, 64) for n in lengths))
    gpu = [_device_indices(v, device) for v in (inverse, forward, chunks, selected)]
    result = (
        tuple(packed),
        tuple(v[0] for v in gpu),
        tuple(v[1] for v in gpu),
        capacity,
    )
    cache[key] = result
    return result


def _length_group_reuse(inputs, cu, lengths, norm_rows):
    # A new heterogeneous batch must not invalidate every existing graph.
    # Capture each repeated length independently; frequency changes use the
    # existing power-of-two capacities. All input/output storage is independent.
    token = _FORWARD_METADATA.set({}) if metadata_cache() is None else None
    try:
        q, k, v, g, beta, state, a, bias = inputs
        base = _base_layout(lengths, q.device, q.shape[2])
        result = None
        if base is not None:
            result = chunk_kda_full_reuse_graph(
                *inputs[:6],
                cu,
                inputs[6],
                inputs[7],
                _normalization_rows=norm_rows,
                _packed_layout=base,
            )
        skip = base[3] if result is not None else 0
        plans, real_chunks, _ = _length_group_plans(lengths, q.device, skip=skip)
        heads, state_size = q.shape[2], q.shape[2] * 128 * 128
        if result is None:
            result = (
                torch.empty_like(q),
                torch.empty_like(state),
                state.new_empty(
                    (1, sum(triton.cdiv(n, 64) for n in lengths), heads, 128, 128)
                ),
            )
        for length, count, sequence_ids, group_cu, *_ in plans:
            shape = (1, count * length, heads, 128)
            buffers = tuple(t.new_empty(shape) for t in (q, k, v, g)) + (
                beta.new_empty(shape[:3]),
                state.new_empty((count, heads, 128, 128)),
                torch.empty_like(a),
                torch.empty_like(bias),
            )
            _pack_inputs[
                (count, triton.cdiv(max(length * heads * 128, state_size), 1024))
            ](
                *inputs,
                cu,
                sequence_ids,
                *buffers,
                group_cu,
                heads * 128,
                heads,
                state_size,
                1024,
            )
            outputs = chunk_kda_full_reuse_graph(
                *buffers[:5],
                buffers[5],
                group_cu,
                buffers[6],
                buffers[7],
                _normalization_rows=norm_rows,
            )
            if outputs is None:
                return None
            _unpack_group_outputs[
                (
                    count,
                    triton.cdiv(
                        max(length * heads * 128, triton.cdiv(length, 64) * state_size),
                        1024,
                    ),
                )
            ](
                *outputs,
                *result,
                cu,
                sequence_ids,
                real_chunks,
                length,
                heads * 128,
                state_size,
                1024,
            )
        _STATS["grouped_calls"] += 1
        return result
    finally:
        if token is not None:
            _FORWARD_METADATA.reset(token)


def chunk_kda_full_reuse_graph(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    cu_seqlens,
    A_log,
    dt_bias,
    *,
    _normalization_rows=None,
    _packed_layout=None,
):
    """Return fresh outputs, or None so callers retain their supported fallback."""
    tensors = (q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias)
    if os.environ.get("GLM53_KDA_FULL_REUSE_GRAPH", "1") == "0":
        return None
    if (
        any(t is None or not t.is_cuda or not t.is_contiguous() for t in tensors)
        or any(t.device != q.device for t in tensors)
        or q.ndim != 4
        or q.shape[0] != 1
        or q.shape[-1] != 128
        or not 0 < q.shape[1] <= 16384
        or not 1 <= initial_state.shape[0] <= 64
        or any(t.shape != q.shape or t.dtype != torch.bfloat16 for t in (q, k, v, g))
        or beta.shape != q.shape[:3]
        or beta.dtype != torch.float32
        or initial_state.shape[1:] != (q.shape[2], 128, 128)
        or initial_state.dtype != torch.float32
        or A_log.numel() != q.shape[2]
        or A_log.dtype != torch.float32
        or dt_bias.numel() != q.shape[2] * 128
        or dt_bias.dtype != torch.float32
        or torch.cuda.is_current_stream_capturing()
    ):
        _STATS["unsupported"] += 1
        return None
    boundaries = _sequence_boundaries(cu_seqlens, q.shape[1])
    lengths = tuple(b - a for a, b in zip(boundaries, boundaries[1:]))
    if len(lengths) != initial_state.shape[0] or any(n <= 0 for n in lengths):
        _STATS["invalid_layout"] += 1
        return None
    norm_rows = (
        q.shape[1] * q.shape[2] if _normalization_rows is None else _normalization_rows
    )
    if _packed_layout is None and (
        len(set(lengths)) > 2
        or (len(set(lengths)) == 2 and max(lengths) * 10 > min(lengths) * 11)
    ):
        return _length_group_reuse(
            (q, k, v, g, beta, initial_state, A_log, dt_bias),
            cu_seqlens,
            lengths,
            norm_rows,
        )
    selected = None
    if _packed_layout is None:
        packed_lengths, (inverse, forward, real_chunks), hosts = _layout(
            lengths, q.device
        )
    else:
        packed_lengths, (inverse, forward, real_chunks, selected), hosts, _ = (
            _packed_layout
        )
    packed_tokens, heads = sum(packed_lengths), q.shape[2]
    packed_states = len(packed_lengths) * heads * 128 * 128
    packed_checkpoints = (
        sum(triton.cdiv(n, 64) for n in packed_lengths) * heads * 128 * 128
    )
    conservative_bytes = 2 * (
        packed_tokens * heads * 128 * 10
        + packed_tokens * heads * 4
        + packed_states * 8
        + packed_checkpoints * 4
    )
    budget = _workspace_budget(q.device)
    if conservative_bytes > budget:
        _STATS["workspace_limit"] += 1
        return None
    # Preserve L2Norm's sealed reduction configuration across capacity padding.
    norm_small = norm_rows <= 65536
    stream = torch.cuda.current_stream(q.device)
    key = (q.device.index, stream.cuda_stream, heads, packed_lengths, norm_small)
    entry = _GRAPHS.get(key)
    inputs = (q, k, v, g, beta, initial_state, A_log, dt_bias)
    scope = (
        torch.profiler.record_function("kda.full_graph")
        if os.environ.get("GLM53_KDA_GRAPH_DIAGNOSTICS") == "1"
        else nullcontext()
    )
    with scope:
        if entry is None:
            shape = (1, packed_tokens, heads, 128)
            buffers = tuple(t.new_empty(shape) for t in (q, k, v, g)) + (
                beta.new_empty((1, packed_tokens, heads)),
                initial_state.new_empty((len(packed_lengths), heads, 128, 128)),
                torch.empty_like(A_log),
                torch.empty_like(dt_bias),
            )
            pcu_host = torch.tensor(
                (0, *accumulate(packed_lengths)), dtype=torch.int32, pin_memory=True
            )
            pcu = pcu_host.to(q.device, non_blocking=True)
            offsets = (0, *accumulate(triton.cdiv(n, 64) for n in packed_lengths))
            packed_chunks, chunk_host = _device_indices(offsets, q.device)
            indices_host = torch.tensor(
                [
                    (i, j)
                    for i, n in enumerate(packed_lengths)
                    for j in range(triton.cdiv(n, 64))
                ],
                dtype=torch.int32,
                pin_memory=True,
            )
            indices = indices_host.to(q.device, non_blocking=True)
            capture_stream = torch.cuda.Stream(device=q.device)
            capture_stream.wait_stream(stream)
            token = _FORWARD_METADATA.set({})
            try:
                with torch.cuda.stream(capture_stream):
                    _pack_inputs[
                        (
                            len(packed_lengths),
                            triton.cdiv(
                                max(
                                    max(packed_lengths) * heads * 128, heads * 128 * 128
                                ),
                                1024,
                            ),
                        )
                    ](
                        *inputs,
                        cu_seqlens,
                        inverse,
                        *buffers,
                        pcu,
                        heads * 128,
                        heads,
                        heads * 128 * 128,
                        1024,
                    )
                    # Warm every native/Triton operation with frozen metadata.
                    for _ in range(2):
                        _reference(buffers, pcu, indices, norm_rows)
                stream.wait_stream(capture_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, stream=capture_stream, capture_error_mode="thread_local"
                ):
                    outputs = _reference(buffers, pcu, indices, norm_rows)
                stream.wait_stream(capture_stream)
            finally:
                _FORWARD_METADATA.reset(token)
            # Count private-pool reserved segments, not unrelated reusable
            # allocator blocks. A 2x tensor-size estimate evicted live hot
            # templates under concurrent admission and forced recapture.
            pool = tuple(graph.pool())
            pool_bytes = sum(
                segment["total_size"]
                for segment in torch.cuda.memory_snapshot()
                if tuple(segment.get("segment_pool_id", (0, 0))) == pool
                and segment["device"] == q.device.index
            )
            estimated_bytes = (
                sum(t.numel() * t.element_size() for t in buffers) + pool_bytes
                if pool_bytes
                else 2 * sum(t.numel() * t.element_size() for t in (*buffers, *outputs))
            )
            entry = (
                buffers,
                graph,
                outputs,
                (pcu, packed_chunks, indices, pcu_host, chunk_host, indices_host),
                estimated_bytes,
            )
            _GRAPHS[key] = entry
            _STATS["captures"] += 1
            while (
                len(_GRAPHS) > _MAX_GRAPHS
                or sum(e[4] for k, e in _GRAPHS.items() if k[0] == q.device.index)
                > budget
            ):
                if len(_GRAPHS) == 1:
                    break
                _GRAPHS.popitem(last=False)
                _STATS["evictions"] += 1
            logging.info(
                "[KDA full reuse graph] capture sequences=%d capacity=%d tokens=%d heads=%d groups=%d resident_bytes=%d",
                len(lengths),
                len(packed_lengths),
                q.shape[1],
                heads,
                len(set(lengths)),
                estimated_bytes,
            )
        else:
            _GRAPHS.move_to_end(key)
        buffers, graph, outputs, (pcu, packed_chunks, *_), _ = entry
        _pack_inputs[
            (
                len(packed_lengths),
                triton.cdiv(
                    max(max(packed_lengths) * heads * 128, heads * 128 * 128), 1024
                ),
            )
        ](
            *inputs,
            cu_seqlens,
            inverse,
            *buffers,
            pcu,
            heads * 128,
            heads,
            heads * 128 * 128,
            1024,
        )
        graph.replay()
        _STATS["replays"] += 1
        result = (
            torch.empty_like(q),
            torch.empty_like(initial_state),
            initial_state.new_empty(
                (1, sum(triton.cdiv(n, 64) for n in lengths), heads, 128, 128)
            ),
        )
        _unpack_outputs[
            (
                len(lengths) if selected is None else selected.numel(),
                triton.cdiv(
                    max(
                        max(lengths) * heads * 128,
                        triton.cdiv(max(lengths), 64) * heads * 128 * 128,
                    ),
                    1024,
                ),
            )
        ](
            *outputs,
            *result,
            cu_seqlens,
            forward,
            real_chunks,
            pcu,
            packed_chunks,
            heads * 128,
            heads * 128 * 128,
            1024,
            selected,
            selected is not None,
        )
        if selected is not None:
            _STATS["base_replays"] += 1
        return result
