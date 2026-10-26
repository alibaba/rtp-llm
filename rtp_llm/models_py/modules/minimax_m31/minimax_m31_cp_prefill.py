"""MiniMax-M3.1 CP Prefill over main's tagged, page-mapped KV groups."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.minimax_m31.kernels.minimax_m31_cache import (
    m31_cache_views,
)
from rtp_llm.models_py.modules.minimax_m31.kernels.minimax_m31_cp_wire import (
    pack_m31_cp_wire,
    write_m31_cp_wire,
)
from rtp_llm.models_py.modules.minimax_m31.kernels.minimax_m31_gemma_rope import (
    minimax_m31_gemma_norm_rope_,
)
from rtp_llm.models_py.modules.minimax_m31.minimax_m31_addressing import _decode_table
from rtp_llm.models_py.modules.minimax_m31.msa.common.minimax_m31_index_score import (
    q8kv4_cp_prefill_index_score,
)
from rtp_llm.models_py.modules.minimax_m31.msa.decode.minimax_m31_sparse_decode import (
    m31_sparse_decode,
)
from rtp_llm.models_py.modules.minimax_m31.msa.prefill.minimax_m31_cp_pages import (
    pack_prefix_pages,
    restore_prefix_pages,
)
from rtp_llm.models_py.modules.minimax_m31.msa.prefill.minimax_m31_prefill_fmha import (
    m31_sparse_prefill_fmha,
)


@dataclass(frozen=True)
class M31CPPlan:
    local_request_ids: torch.Tensor
    local_positions: torch.Tensor
    restore_indices: torch.Tensor
    full_lengths: tuple[int, ...]
    prefix_lengths: tuple[int, ...]
    input_lengths: tuple[int, ...]
    write_slots: torch.Tensor


def m31_cp_native_segments(
    input_lengths: tuple[int, ...], cp_rank: int, cp_size: int
) -> list[tuple[int, int, int]] | None:
    """Return the two contiguous Zigzag spans per request, if unpadded."""
    if any(n % (2 * cp_size) or (n // cp_size) % 32 for n in input_lengths):
        return None
    segments = []
    for request, length in enumerate(input_lengths):
        pair = length // (2 * cp_size)
        segments.extend(
            (
                (request, cp_rank * pair, pair),
                (request, length - (cp_rank + 1) * pair, pair),
            )
        )
    return segments


def plan_m31_cp_prefill(
    main_inputs, index_inputs, rows: int, cp_size: int
) -> M31CPPlan:
    """Map rank-local zigzag rows to global request positions."""
    info = main_inputs.context_parallel_info
    if info is None or index_inputs.context_parallel_info is None:
        raise ValueError("M3.1 CP requires tagged context metadata")
    if cp_size <= 1 or not main_inputs.is_prefill or not index_inputs.is_prefill:
        raise ValueError("M3.1 CP requires multi-rank Prefill")
    full_lengths = tuple(int(x) for x in info.prefill_actual_input_lengths_cpu.tolist())
    prefix_cpu = info.prefill_prefix_lengths_cpu
    prefix_lengths = (
        tuple(int(x) for x in prefix_cpu.tolist())
        if prefix_cpu is not None and prefix_cpu.numel()
        else (0,) * len(full_lengths)
    )
    if (
        len(full_lengths) != len(prefix_lengths)
        or not full_lengths
        or any(n <= 0 or p < 0 for n, p in zip(full_lengths, prefix_lengths))
    ):
        raise ValueError("M3.1 CP request lengths are invalid")
    local_lengths = tuple(
        ((n + 2 * cp_size - 1) // (2 * cp_size)) * 2 for n in full_lengths
    )
    if sum(local_lengths) != rows or info.prefill_shuffle_indices.numel() != rows:
        raise ValueError("M3.1 CP local rows disagree with zigzag metadata")
    mask = info.prefill_qkv_padding_mask_cpu
    restore = info.prefill_qkv_restore_indice_cpu
    if mask.numel() != sum(local_lengths) * cp_size or restore.shape != mask.shape:
        raise ValueError("M3.1 CP global restore geometry is invalid")
    valid = mask.to(torch.bool)
    if int(valid.sum()) != sum(full_lengths):
        raise ValueError("M3.1 CP valid token count differs from request lengths")
    restore_indices = restore[valid].to(
        device=info.prefill_shuffle_indices.device, dtype=torch.int64, non_blocking=True
    )
    device = info.prefill_shuffle_indices.device
    request_ids = torch.repeat_interleave(
        torch.arange(len(local_lengths), device=device, dtype=torch.int64),
        torch.tensor(local_lengths, device=device, dtype=torch.int64),
        output_size=rows,
    )
    prefixes = torch.tensor(prefix_lengths, device=device, dtype=torch.int64)
    lengths = torch.tensor(full_lengths, device=device, dtype=torch.int64)
    shuffle = info.prefill_shuffle_indices.to(torch.int64)
    relative = torch.minimum(
        shuffle.clamp_min(0), lengths.index_select(0, request_ids) - 1
    )
    positions = prefixes.index_select(0, request_ids) + relative
    total_lengths = tuple(p + n for p, n in zip(prefix_lengths, full_lengths))
    starts = [1]
    for length in total_lengths[:-1]:
        starts.append(starts[-1] + (length + 127) // 128)
    full_request_ids = torch.repeat_interleave(
        torch.arange(len(full_lengths), device=device, dtype=torch.int64),
        torch.tensor(full_lengths, device=device, dtype=torch.int64),
        output_size=sum(full_lengths),
    )
    row_starts = torch.tensor(
        [sum(full_lengths[:i]) for i in range(len(full_lengths))],
        device=device,
        dtype=torch.int64,
    )
    write_slots = (
        torch.arange(sum(full_lengths), device=device)
        - row_starts.index_select(0, full_request_ids)
        + prefixes.index_select(0, full_request_ids)
        + 128
        * torch.tensor(starts, device=device, dtype=torch.int64).index_select(
            0, full_request_ids
        )
    )
    return M31CPPlan(
        request_ids,
        positions,
        restore_indices,
        total_lengths,
        prefix_lengths,
        full_lengths,
        write_slots,
    )


@lru_cache(maxsize=8)
def _prefix_page_indices(
    prefix_counts: tuple[int, ...], cp_rank: int, cp_size: int, sharded: bool
) -> tuple[tuple[int, ...], ...]:
    ranks = cp_size if sharded else 1
    rank = cp_rank if sharded else 0
    send_offsets = []
    src_requests = []
    src_blocks = []
    dst_requests = []
    dst_blocks = []
    recv_offsets = []
    send_count = 0
    for request, count in enumerate(prefix_counts):
        slots = (count + ranks - 1) // ranks
        for slot in range(slots):
            logical = rank + slot * ranks
            if logical < count:
                send_offsets.append(send_count + slot)
                src_requests.append(request)
                src_blocks.append(slot if sharded else logical)
        send_count += slots
    for owner in range(ranks):
        offset = 0
        for request, count in enumerate(prefix_counts):
            slots = (count + ranks - 1) // ranks
            for slot in range(slots):
                logical = owner + slot * ranks
                if logical < count:
                    dst_requests.append(request)
                    dst_blocks.append(logical)
                    recv_offsets.append(owner * send_count + offset + slot)
            offset += slots
    return tuple(
        map(
            tuple,
            (
                send_offsets,
                src_requests,
                src_blocks,
                dst_requests,
                dst_blocks,
                recv_offsets,
            ),
        )
    )


@dataclass(frozen=True)
class _WorkingGeometry:
    compact: torch.Tensor
    starts: tuple[int, ...]
    page_count: int
    send_count: int
    src_requests: torch.Tensor
    src_blocks: torch.Tensor
    send_offsets: torch.Tensor
    dst_pages: torch.Tensor
    recv_offsets: torch.Tensor
    publish_requests: torch.Tensor
    publish_blocks: torch.Tensor
    publish_sources: torch.Tensor


@lru_cache(maxsize=8)
def _working_geometry(
    full_lengths: tuple[int, ...],
    prefix_lengths: tuple[int, ...],
    cp_rank: int,
    cp_size: int,
    sharded: bool,
    device: torch.device,
) -> _WorkingGeometry:
    page_counts = tuple((n + 127) // 128 for n in full_lengths)
    prefix_counts = tuple((n + 127) // 128 for n in prefix_lengths)
    starts = []
    next_page = 1
    compact = torch.zeros(
        (len(full_lengths), max(page_counts)), dtype=torch.int32, device=device
    )
    for request, count in enumerate(page_counts):
        starts.append(next_page)
        compact[request, :count] = torch.arange(
            next_page, next_page + count, dtype=torch.int32, device=device
        )
        next_page += count
    (send_offsets, src_requests, src_blocks, dst_requests, dst_blocks, recv_offsets) = (
        _prefix_page_indices(prefix_counts, cp_rank, cp_size, sharded)
    )
    ranks = cp_size if sharded else 1
    send_count = sum((count + ranks - 1) // ranks for count in prefix_counts)
    publish_requests = []
    publish_blocks = []
    publish_sources = []
    for request, (full, prefix) in enumerate(zip(full_lengths, prefix_lengths)):
        for logical in range(prefix // 128, (full + 127) // 128):
            if sharded and logical % cp_size != cp_rank:
                continue
            publish_requests.append(request)
            publish_blocks.append(logical // cp_size if sharded else logical)
            publish_sources.append(starts[request] + logical)

    def indices(values):
        return torch.tensor(values, dtype=torch.int64, device=device)

    return _WorkingGeometry(
        compact,
        tuple(starts),
        next_page,
        send_count,
        indices(src_requests),
        indices(src_blocks),
        indices(send_offsets),
        indices(
            [
                starts[request] + logical
                for request, logical in zip(dst_requests, dst_blocks)
            ]
        ),
        indices(recv_offsets),
        indices(publish_requests),
        indices(publish_blocks),
        indices(publish_sources),
    )


def _working_pages(
    pool: torch.Tensor,
    table: torch.Tensor,
    full_lengths: tuple[int, ...],
    prefix_lengths: tuple[int, ...],
    cp_rank: int,
    cp_size: int,
    sharded: bool,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Build per-forward compact pages; gather only active prefix pages."""
    geometry = _working_geometry(
        full_lengths, prefix_lengths, cp_rank, cp_size, sharded, table.device
    )
    use_fused_copy = (
        pool.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(pool.device) == (10, 3)
    )
    complete_pages = use_fused_copy and all(
        full % 128 == 0 and prefix % 128 == 0
        for full, prefix in zip(full_lengths, prefix_lengths)
    )
    working = (torch.empty if complete_pages else torch.zeros)(
        (geometry.page_count, pool.shape[1]), dtype=torch.uint8, device=pool.device
    )
    if complete_pages:
        working[0].zero_()
    if not geometry.send_count:
        return working, geometry.compact, geometry.starts
    send = (
        torch.empty
        if use_fused_copy and geometry.src_requests.numel() == geometry.send_count
        else torch.zeros
    )((geometry.send_count, pool.shape[1]), dtype=torch.uint8, device=pool.device)
    if use_fused_copy:
        pack_prefix_pages(
            pool,
            table,
            send,
            geometry.src_requests,
            geometry.src_blocks,
            geometry.send_offsets,
        )
    elif geometry.src_requests.numel():
        src = table[geometry.src_requests, geometry.src_blocks].to(torch.int64)
        send.index_copy_(
            0,
            geometry.send_offsets,
            pool.index_select(0, src),
        )
    received = all_gather(send, Group.TP) if sharded else send
    if use_fused_copy:
        restore_prefix_pages(
            received, working, geometry.dst_pages, geometry.recv_offsets
        )
    else:
        working.index_copy_(
            0,
            geometry.dst_pages,
            received.index_select(0, geometry.recv_offsets),
        )
    return working, geometry.compact, geometry.starts


_prefix_streams: dict[torch.device, torch.cuda.Stream] = {}


def start_m31_cp_prefix_gather(
    main_inputs,
    index_inputs,
    main_cache,
    index_cache,
    plan: M31CPPlan,
    cp_rank: int,
    cp_size: int,
    sharded: bool,
):
    """Overlap this layer's prefix transfer with its QKV projection."""
    main_table = _decode_table(main_inputs, "main")
    index_table = _decode_table(index_inputs, "indexer")
    if main_table.shape != index_table.shape:
        raise ValueError("M3.1 CP tagged cache tables must have matching geometry")
    device = main_cache.kv_cache_base.device
    main_stream = torch.cuda.current_stream(device)
    side_stream = _prefix_streams.get(device)
    if side_stream is None:
        side_stream = torch.cuda.Stream(device=device)
        _prefix_streams[device] = side_stream
    side_stream.wait_stream(main_stream)
    with torch.cuda.stream(side_stream):
        main_work, work_table, starts = _working_pages(
            main_cache.kv_cache_base,
            main_table,
            plan.full_lengths,
            plan.prefix_lengths,
            cp_rank,
            cp_size,
            sharded,
        )
        index_work, _, _ = _working_pages(
            index_cache.kv_cache_base,
            index_table,
            plan.full_lengths,
            plan.prefix_lengths,
            cp_rank,
            cp_size,
            sharded,
        )
        ready = torch.cuda.Event()
        ready.record(side_stream)
    for tensor in (main_work, index_work, work_table):
        tensor.record_stream(main_stream)
    return main_work, index_work, work_table, starts, ready


def _publish_local_pages(
    pool: torch.Tensor,
    table: torch.Tensor,
    working: torch.Tensor,
    starts: tuple[int, ...],
    full_lengths: tuple[int, ...],
    prefix_lengths: tuple[int, ...],
    cp_rank: int,
    cp_size: int,
    sharded: bool,
) -> None:
    geometry = _working_geometry(
        full_lengths, prefix_lengths, cp_rank, cp_size, sharded, table.device
    )
    if geometry.publish_sources.numel():
        destination = table[geometry.publish_requests, geometry.publish_blocks].to(
            torch.int64
        )
        pool.index_copy_(
            0, destination, working.index_select(0, geometry.publish_sources)
        )


def m31_cp_prefill_projected_attention(
    qkv,
    index_q,
    index_k,
    raw_norms,
    cos_sin_cache,
    main_inputs,
    index_inputs,
    main_cache,
    index_cache,
    *,
    cp_rank: int,
    cp_size: int,
    sharded: bool,
    norm_eps: float,
    init_blocks: int,
    local_blocks: int,
    plan: M31CPPlan | None = None,
    prefix_gather=None,
):
    rows = qkv.shape[0]
    if plan is None:
        plan = plan_m31_cp_prefill(main_inputs, index_inputs, rows, cp_size)
    segments = m31_cp_native_segments(plan.input_lengths, cp_rank, cp_size)
    main_table = _decode_table(main_inputs, "main")
    index_table = _decode_table(index_inputs, "indexer")
    if main_table.shape != index_table.shape:
        raise ValueError("M3.1 CP tagged cache tables must have matching geometry")
    index_q8 = torch.empty((rows, 4, 128), dtype=torch.float8_e4m3fn, device=qkv.device)
    query = torch.empty((rows, 64, 128), dtype=torch.float8_e4m3fn, device=qkv.device)
    minimax_m31_gemma_norm_rope_(
        qkv,
        index_q,
        index_k,
        raw_norms,
        plan.local_positions,
        cos_sin_cache,
        num_q_heads=64,
        num_kv_heads=4,
        num_index_heads=4,
        eps=norm_eps,
        query_fp8_outputs=(query, index_q8),
    )
    packed = torch.cat(
        (qkv[:, 64 * 128 : 68 * 128], qkv[:, 68 * 128 :], index_k), dim=-1
    ).contiguous()
    if prefix_gather is not None:
        torch.cuda.current_stream(qkv.device).wait_event(prefix_gather[-1])
    gathered_wire = all_gather(pack_m31_cp_wire(packed), Group.TP)
    if prefix_gather is None:
        main_work, work_table, starts = _working_pages(
            main_cache.kv_cache_base,
            main_table,
            plan.full_lengths,
            plan.prefix_lengths,
            cp_rank,
            cp_size,
            sharded,
        )
        index_work, _, _ = _working_pages(
            index_cache.kv_cache_base,
            index_table,
            plan.full_lengths,
            plan.prefix_lengths,
            cp_rank,
            cp_size,
            sharded,
        )
    else:
        main_work, index_work, work_table, starts, _ = prefix_gather
    work_views = m31_cache_views(main_work, index_work, 4, 128, 128, 128)
    write_m31_cp_wire(gathered_wire, plan.restore_indices, plan.write_slots, work_views)
    _publish_local_pages(
        main_cache.kv_cache_base,
        main_table,
        main_work,
        starts,
        plan.full_lengths,
        plan.prefix_lengths,
        cp_rank,
        cp_size,
        sharded,
    )
    _publish_local_pages(
        index_cache.kv_cache_base,
        index_table,
        index_work,
        starts,
        plan.full_lengths,
        plan.prefix_lengths,
        cp_rank,
        cp_size,
        sharded,
    )
    for inputs, cache in ((main_inputs, main_cache), (index_inputs, index_cache)):
        if inputs.cache_store_inputs is not None:
            writer = common.create_write_cache_store_impl(inputs)
            common.apply_write_cache_store(writer, inputs, cache)

    if segments is not None and torch.cuda.get_device_capability(qkv.device) == (10, 3):
        try:
            return m31_sparse_prefill_fmha(
                query,
                index_q8,
                work_views,
                work_table,
                work_table,
                plan.local_request_ids,
                plan.local_positions,
                list(plan.input_lengths),
                list(plan.prefix_lengths),
                init_blocks=init_blocks,
                local_blocks=local_blocks,
                query_segments=segments,
            )
        except ModuleNotFoundError as error:
            if error.name not in ("fmha_sm100", "fmha_sm100.cute"):
                raise

    output = torch.empty((rows, 64, 128), dtype=torch.bfloat16, device=qkv.device)
    tiled_score = torch.cuda.get_device_capability(qkv.device) == (10, 3) and all(
        ((n + 2 * cp_size - 1) // (2 * cp_size) * 2) % 32 == 0
        for n in plan.input_lengths
    )
    for start in range(0, rows, 256):
        end = min(start + 256, rows)
        request = plan.local_request_ids[start:end]
        scores = None
        if tiled_score:
            scores = torch.empty(
                (4, end - start, work_table.shape[1]),
                dtype=torch.float32,
                device=qkv.device,
            )
            q8kv4_cp_prefill_index_score(
                index_q8[start:end],
                work_views.index_k,
                work_views.index_k_scale,
                work_table,
                request,
                plan.local_positions[start:end],
                scores,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
            )
        output[start:end] = m31_sparse_decode(
            query[start:end],
            index_q8[start:end],
            work_views,
            work_table.index_select(0, request),
            work_table.index_select(0, request),
            (plan.local_positions[start:end] + 1).to(torch.int32),
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            prefill_scores=scores,
        )
    return output
