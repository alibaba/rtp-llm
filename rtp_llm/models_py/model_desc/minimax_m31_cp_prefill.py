"""MiniMax-M3.1 CP Prefill over main's tagged, page-mapped KV groups."""

from dataclasses import dataclass

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.model_desc.minimax_m31_addressing import _decode_table
from rtp_llm.models_py.model_desc.minimax_m31_cache import m31_cache_views
from rtp_llm.models_py.model_desc.minimax_m31_cp_wire import (
    pack_m31_cp_wire,
    write_m31_cp_wire,
)
from rtp_llm.models_py.model_desc.minimax_m31_gemma_rope import (
    minimax_m31_gemma_norm_rope_,
)
from rtp_llm.models_py.model_desc.minimax_m31_sparse_decode import m31_sparse_decode
from rtp_llm.models_py.modules.factory.attention import common


@dataclass(frozen=True)
class M31CPPlan:
    local_request_ids: torch.Tensor
    local_positions: torch.Tensor
    restore_indices: torch.Tensor
    full_lengths: tuple[int, ...]
    prefix_lengths: tuple[int, ...]
    input_lengths: tuple[int, ...]


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
    return M31CPPlan(
        request_ids,
        positions,
        restore_indices,
        tuple(p + n for p, n in zip(prefix_lengths, full_lengths)),
        prefix_lengths,
        full_lengths,
    )


def _working_pages(
    pool: torch.Tensor,
    table: torch.Tensor,
    full_lengths: tuple[int, ...],
    prefix_lengths: tuple[int, ...],
    cp_rank: int,
    cp_size: int,
    sharded: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build per-forward compact pages; gather only active prefix pages."""
    page_counts = [(n + 127) // 128 for n in full_lengths]
    prefix_counts = [(n + 127) // 128 for n in prefix_lengths]
    starts = []
    next_page = 1  # page zero is reserved for padded table entries
    compact = torch.zeros(
        (len(full_lengths), max(page_counts)), dtype=torch.int32, device=table.device
    )
    for request, count in enumerate(page_counts):
        starts.append(next_page)
        compact[request, :count] = torch.arange(
            next_page, next_page + count, dtype=torch.int32, device=table.device
        )
        next_page += count
    working = torch.zeros(
        (next_page, pool.shape[1]), dtype=torch.uint8, device=pool.device
    )
    if not any(prefix_counts):
        return working, compact, starts

    ranks = cp_size if sharded else 1
    rank = cp_rank if sharded else 0
    send_offsets = []
    src_requests = []
    src_blocks = []
    dst_pages = []
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
    send = torch.zeros(
        (send_count, pool.shape[1]), dtype=torch.uint8, device=pool.device
    )
    if src_requests:
        src = table[
            torch.tensor(src_requests, device=table.device),
            torch.tensor(src_blocks, device=table.device),
        ].to(torch.int64)
        send.index_copy_(
            0,
            torch.tensor(send_offsets, device=pool.device),
            pool.index_select(0, src),
        )
    received = all_gather(send, Group.TP) if sharded else send
    for owner in range(ranks):
        offset = 0
        for request, count in enumerate(prefix_counts):
            slots = (count + ranks - 1) // ranks
            for slot in range(slots):
                logical = owner + slot * ranks
                if logical < count:
                    dst_pages.append(starts[request] + logical)
                    recv_offsets.append(owner * send_count + offset + slot)
            offset += slots
    working.index_copy_(
        0,
        torch.tensor(dst_pages, device=pool.device),
        received.index_select(0, torch.tensor(recv_offsets, device=pool.device)),
    )
    return working, compact, starts


def _publish_local_pages(
    pool: torch.Tensor,
    table: torch.Tensor,
    working: torch.Tensor,
    starts: list[int],
    full_lengths: tuple[int, ...],
    prefix_lengths: tuple[int, ...],
    cp_rank: int,
    cp_size: int,
    sharded: bool,
) -> None:
    source = []
    requests = []
    blocks = []
    for request, (full, prefix) in enumerate(zip(full_lengths, prefix_lengths)):
        for logical in range(prefix // 128, (full + 127) // 128):
            if sharded and logical % cp_size != cp_rank:
                continue
            source.append(starts[request] + logical)
            requests.append(request)
            blocks.append(logical // cp_size if sharded else logical)
    if source:
        destination = table[
            torch.tensor(requests, device=table.device),
            torch.tensor(blocks, device=table.device),
        ].to(torch.int64)
        pool.index_copy_(
            0,
            destination,
            working.index_select(0, torch.tensor(source, device=pool.device)),
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
):
    rows = qkv.shape[0]
    plan = plan_m31_cp_prefill(main_inputs, index_inputs, rows, cp_size)
    main_table = _decode_table(main_inputs, "main")
    index_table = _decode_table(index_inputs, "indexer")
    if main_table.shape != index_table.shape:
        raise ValueError("M3.1 CP tagged cache tables must have matching geometry")
    q8 = torch.empty((rows, 64, 128), dtype=torch.float8_e4m3fn, device=qkv.device)
    index_q8 = torch.empty((rows, 4, 128), dtype=torch.float8_e4m3fn, device=qkv.device)
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
        query_fp8_outputs=(q8, index_q8),
    )
    packed = torch.cat(
        (qkv[:, 64 * 128 : 68 * 128], qkv[:, 68 * 128 :], index_k), dim=-1
    ).contiguous()
    gathered_wire = all_gather(pack_m31_cp_wire(packed), Group.TP)
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
    work_views = m31_cache_views(main_work, index_work, 4, 128, 128, 128)
    request_ids = torch.repeat_interleave(
        torch.arange(len(plan.input_lengths), device=qkv.device, dtype=torch.int64),
        torch.tensor(plan.input_lengths, device=qkv.device, dtype=torch.int64),
        output_size=sum(plan.input_lengths),
    )
    starts_per_request = torch.tensor(starts, device=qkv.device, dtype=torch.int64)
    prefixes = torch.tensor(plan.prefix_lengths, device=qkv.device, dtype=torch.int64)
    row_starts = torch.tensor(
        [sum(plan.input_lengths[:i]) for i in range(len(plan.input_lengths))],
        device=qkv.device,
        dtype=torch.int64,
    )
    positions = (
        torch.arange(sum(plan.input_lengths), device=qkv.device)
        - row_starts.index_select(0, request_ids)
        + prefixes.index_select(0, request_ids)
    )
    slots = starts_per_request.index_select(0, request_ids) * 128 + positions
    write_m31_cp_wire(gathered_wire, plan.restore_indices, slots, work_views)
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

    output = torch.empty((rows, 64, 128), dtype=torch.bfloat16, device=qkv.device)
    for start in range(0, rows, 256):
        end = min(start + 256, rows)
        request = plan.local_request_ids[start:end]
        output[start:end] = m31_sparse_decode(
            q8[start:end],
            index_q8[start:end],
            work_views,
            work_table.index_select(0, request),
            work_table.index_select(0, request),
            (plan.local_positions[start:end] + 1).to(torch.int32),
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )
    return output
