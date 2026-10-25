"""MiniMax-M3.1 ordinary Prefill on main's independent tagged KV groups."""

from dataclasses import dataclass

import torch

from rtp_llm.models_py.model_desc.minimax_m31_addressing import _decode_table
from rtp_llm.models_py.model_desc.minimax_m31_cache import M31CacheViews
from rtp_llm.models_py.model_desc.minimax_m31_gemma_rope import (
    minimax_m31_gemma_norm_rope_,
)
from rtp_llm.models_py.model_desc.minimax_m31_kv_write import write_m31_nvfp4_rows
from rtp_llm.models_py.model_desc.minimax_m31_prefill_fmha import (
    m31_sparse_prefill_fmha,
)
from rtp_llm.models_py.model_desc.minimax_m31_sparse_decode import m31_sparse_decode
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


@dataclass(frozen=True)
class M31PrefillAddressing:
    request_ids: torch.Tensor
    positions: torch.Tensor
    main_slots: torch.Tensor
    index_slots: torch.Tensor
    main_table: torch.Tensor
    index_table: torch.Tensor
    query_lengths: tuple[int, ...]


def m31_prefill_addressing(
    main_inputs: PyAttentionInputs,
    index_inputs: PyAttentionInputs,
    rows: int,
) -> M31PrefillAddressing:
    if not main_inputs.is_prefill or not index_inputs.is_prefill:
        raise ValueError("M3.1 Prefill requires both tagged inputs in Prefill phase")
    if (
        main_inputs.context_parallel_info is not None
        or index_inputs.context_parallel_info is not None
    ):
        raise ValueError("M3.1 ordinary Prefill does not accept CP metadata")
    main_table = _decode_table(main_inputs, "main")
    index_table = _decode_table(index_inputs, "indexer")
    if main_table.shape != index_table.shape or main_table.device != index_table.device:
        raise ValueError(
            "M3.1 Prefill cache tables must have matching request geometry"
        )
    prefixes = main_inputs.prefix_lengths.to(torch.int64).tolist()
    lengths = main_inputs.input_lengths.to(torch.int64).tolist()
    if (
        prefixes != index_inputs.prefix_lengths.to(torch.int64).tolist()
        or lengths != index_inputs.input_lengths.to(torch.int64).tolist()
        or len(lengths) != main_table.shape[0]
        or any(p < 0 for p in prefixes)
        or any(length <= 0 for length in lengths)
        or sum(lengths) != rows
    ):
        raise ValueError("M3.1 Prefill tagged request lengths must match token rows")
    if (
        max((p + n for p, n in zip(prefixes, lengths)), default=0)
        > main_table.shape[1] * 128
    ):
        raise ValueError("M3.1 Prefill cache page table is too short")
    device = main_table.device
    device_lengths = getattr(main_inputs, "input_lengths_device", None)
    counts = (
        device_lengths.to(torch.int64)
        if device_lengths is not None and device_lengths.device == device
        else torch.tensor(lengths, dtype=torch.int64, device=device)
    )
    request_ids = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int64, device=device),
        counts,
        output_size=rows,
    )
    starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    device_prefixes = getattr(main_inputs, "prefix_lengths_device", None)
    prefix = (
        device_prefixes.to(torch.int64)
        if device_prefixes is not None and device_prefixes.device == device
        else torch.tensor(prefixes, dtype=torch.int64, device=device)
    )
    positions = (
        torch.arange(rows, dtype=torch.int64, device=device)
        - starts.index_select(0, request_ids)
        + prefix.index_select(0, request_ids)
    )
    page = torch.div(positions, 128, rounding_mode="floor").unsqueeze(1)
    offset = positions.remainder(128)
    main_id = (
        main_table.index_select(0, request_ids)
        .gather(1, page)
        .squeeze(1)
        .to(torch.int64)
    )
    index_id = (
        index_table.index_select(0, request_ids)
        .gather(1, page)
        .squeeze(1)
        .to(torch.int64)
    )
    return M31PrefillAddressing(
        request_ids,
        positions,
        main_id * 128 + offset,
        index_id * 128 + offset,
        main_table,
        index_table,
        tuple(lengths),
    )


def m31_prefill_projected_attention(
    qkv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    raw_norms: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cos_sin_cache: torch.Tensor,
    views: M31CacheViews,
    main_inputs: PyAttentionInputs,
    index_inputs: PyAttentionInputs,
    main_cache: LayerKVCache,
    index_cache: LayerKVCache,
    *,
    norm_eps: float,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    rows = qkv.shape[0]
    if (
        qkv.shape != (rows, 72 * 128)
        or index_q.shape != (rows, 4 * 128)
        or index_k.shape != (rows, 128)
    ):
        raise ValueError("M3.1 Prefill projection geometry mismatch")
    if not qkv.is_cuda:
        raise ValueError("M3.1 Prefill attention requires CUDA")
    address = m31_prefill_addressing(main_inputs, index_inputs, rows)
    q8 = torch.empty((rows, 64, 128), dtype=torch.float8_e4m3fn, device=qkv.device)
    index_q8 = torch.empty((rows, 4, 128), dtype=torch.float8_e4m3fn, device=qkv.device)
    minimax_m31_gemma_norm_rope_(
        qkv,
        index_q,
        index_k,
        raw_norms,
        address.positions,
        cos_sin_cache,
        num_q_heads=64,
        num_kv_heads=4,
        num_index_heads=4,
        eps=norm_eps,
        query_fp8_outputs=(q8, index_q8),
    )
    k = qkv[:, 64 * 128 : 68 * 128].view(rows, 4, 128)
    v = qkv[:, 68 * 128 : 72 * 128].view(rows, 4, 128)
    write_m31_nvfp4_rows(k, v, index_k, address.main_slots, address.index_slots, views)
    for inputs, cache in ((main_inputs, main_cache), (index_inputs, index_cache)):
        if inputs.cache_store_inputs is not None:
            writer = common.create_write_cache_store_impl(inputs)
            common.apply_write_cache_store(writer, inputs, cache)

    # On SM103, the native Q8KV4 page reader avoids staging all active pages.
    sm103 = torch.cuda.get_device_capability(qkv.device) == (10, 3)
    if not sm103:
        try:
            return m31_sparse_prefill_fmha(
                q8,
                index_q8,
                views,
                address.main_table,
                address.index_table,
                address.request_ids,
                address.positions,
                main_inputs.input_lengths.to(torch.int64).tolist(),
                main_inputs.prefix_lengths.to(torch.int64).tolist(),
                init_blocks=init_blocks,
                local_blocks=local_blocks,
            )
        except ModuleNotFoundError as error:
            if error.name not in ("fmha_sm100", "fmha_sm100.cute"):
                raise

    max_width = max(address.query_lengths, default=0)
    cu_seqlens = main_inputs.cu_seqlens_device
    if (
        main_inputs.is_target_verify
        and sm103
        and rows <= 256
        and 2 <= max_width <= 16
        and cu_seqlens is not None
        and cu_seqlens.numel() == len(address.query_lengths) + 1
    ):
        seq_lens = (address.positions + 1).to(torch.int32)
        return m31_sparse_decode(
            q8,
            index_q8,
            views,
            address.main_table.index_select(0, address.request_ids),
            address.index_table.index_select(0, address.request_ids),
            seq_lens,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            query_cu_seqlens=cu_seqlens,
            max_query_width=max_width,
        )
    output = torch.empty((rows, 64, 128), dtype=torch.bfloat16, device=qkv.device)
    # Bound per-query TopK and selected-page scratch. Every chunk sees the
    # request's complete physical table while the kernels enforce each row's
    # causal position through sequence_lengths.
    for start in range(0, rows, 256):
        end = min(start + 256, rows)
        req = address.request_ids[start:end]
        seq_lens = (address.positions[start:end] + 1).to(torch.int32)
        output[start:end] = m31_sparse_decode(
            q8[start:end],
            index_q8[start:end],
            views,
            address.main_table.index_select(0, req),
            address.index_table.index_select(0, req),
            seq_lens,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )
    return output
