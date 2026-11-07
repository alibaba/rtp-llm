"""Bound scratch for the engine's reserved-page Prefill warmup only."""

from collections.abc import Mapping
import logging

import torch

from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelOutputs


def reserved_page_warmup_groups(v4, parallelism, attention_inputs, cu_seqlens,
                                total_tokens, prepare_hidden_fn, capture_context):
    budget = int(v4._prefill_ws_q_rows)
    if total_tokens <= budget or not v4.fp8_kv_cache:
        return None
    if (prepare_hidden_fn is not None
            or (capture_context is not None and capture_context.enabled)
            or int(getattr(v4, "_cp_size", 1)) != 1
            or parallelism is None
            or parallelism.ep_size != 1
            or parallelism.world_size != parallelism.tp_size
            or cu_seqlens.device.type != "cpu"):
        return None
    entries = list(attention_inputs.values()) if isinstance(attention_inputs, Mapping) else [attention_inputs]
    if not entries:
        return None
    batch = cu_seqlens.numel() - 1
    for attn in entries:
        table = attn.kv_cache_block_id
        # Real allocations exclude physical block 0 (IBlockPool). fakeInitKVBlock
        # alone maps every request/tag to it; absence of a PD writer is insufficient.
        if (not attn.is_prefill or attn.is_cuda_graph or attn.is_s_padded
                or attn.is_target_verify or attn.context_parallel_info is not None
                or attn.cache_store_inputs is not None or attn.cache_store_writer is not None
                or table is None or table.device.type != "cpu" or table.ndim != 2
                or table.size(0) != batch or table.numel() == 0
                or bool(torch.any(table != 0))):
            return None
        if (attn.input_lengths is None or attn.prefix_lengths is None
                or attn.input_lengths.device.type != "cpu"
                or attn.prefix_lengths.device.type != "cpu"
                or attn.prefix_lengths.numel() != batch
                or not torch.equal(attn.input_lengths, torch.diff(cu_seqlens))
                or attn.headwise_config is not None):
            return None
    boundaries = cu_seqlens.tolist()
    if boundaries[0] != 0 or boundaries[-1] != total_tokens:
        raise ValueError("PPU warmup request boundaries do not cover input tokens")
    groups, first = [], 0
    for request in range(batch):
        size = boundaries[request + 1] - boundaries[request]
        if size <= 0 or size > budget:
            raise ValueError("PPU reserved-page warmup requires whole requests within workspace capacity")
        if boundaries[request + 1] - boundaries[first] > budget:
            groups.append((first, request, boundaries[first], boundaries[request]))
            first = request
    groups.append((first, batch, boundaries[first], boundaries[-1]))
    return groups


def slice_warmup_attention(attn, r0, r1, t0, t1, cu_seqlens, positions):
    # Construct rather than shallow-copy: readonly length mirrors in a copy
    # would still contain the full batch and override the sliced host lengths.
    result = PyAttentionInputs()
    result.is_prefill = True
    result.dtype = attn.dtype
    for name in ("input_lengths", "prefix_lengths", "sequence_lengths",
                 "kv_cache_block_id", "kv_cache_block_id_device",
                 "kv_cache_kernel_block_id", "kv_cache_kernel_block_id_device",
                 "sequence_lengths_plus_1_device"):
        value = getattr(attn, name)
        if value is not None and value.numel():
            result_value = value[r0:r1]
            setattr(result, name, result_value)
    result.cu_seqlens = cu_seqlens[r0:r1 + 1] - t0
    result.cu_seqlens_device = result.cu_seqlens.to(positions.device)
    result.combo_position_ids = positions[t0:t1]
    if attn.cu_kv_seqlens_device is not None and attn.cu_kv_seqlens_device.numel():
        kv_cu = attn.cu_kv_seqlens_device
        result.cu_kv_seqlens_device = kv_cu[r0:r1 + 1] - kv_cu[r0]
    result.total_tokens = t1 - t0
    result.context_total_kv_length = int(
        (result.input_lengths + result.prefix_lengths).sum())
    result.headwise_config = attn.headwise_config
    return result


def forward_warmup_groups(v4, kv_cache, input_ids, positions, cu_seqlens,
                          block_tables, attention_inputs, groups, forward_layers):
    """Keep full outputs, reuse one scratch allocation, restore registered buffers."""
    fixed = v4._mtp_hidden_buffer
    fixed_valid = v4._mtp_hidden_valid_tokens
    last = v4._mtp_last_hidden_buffer
    last_valid = v4._mtp_last_hidden_valid_tokens
    workspace = PrefillWorkspace(input_ids.device, q_rows=v4._prefill_ws_q_rows,
                                 q_dim=v4._prefill_ws_q_dim, reserve_cp=False)
    aux = None if fixed is None else fixed.new_empty((input_ids.numel(), fixed.size(1)))
    logging.info("[PPU_PREFILL_WARMUP] reserved-page requests=%d tokens=%d groups=%d scratch_rows=%d",
                 cu_seqlens.numel() - 1, input_ids.numel(), len(groups), v4._prefill_ws_q_rows)
    hidden = None
    try:
        for r0, r1, t0, t1 in groups:
            if isinstance(attention_inputs, Mapping):
                sliced = {tag: slice_warmup_attention(a, r0, r1, t0, t1, cu_seqlens, positions)
                          for tag, a in attention_inputs.items()}
                primary = next(iter(sliced.values()))
            else:
                sliced = primary = slice_warmup_attention(
                    attention_inputs, r0, r1, t0, t1, cu_seqlens, positions)
            v4._mtp_hidden_buffer = None if aux is None else aux[t0:t1]
            v4._mtp_last_hidden_buffer = None if last is None else last[r0:r1]
            output = forward_layers(
                v4, kv_cache, input_ids[t0:t1], positions[t0:t1],
                cu_seqlens[r0:r1 + 1] - t0,
                None if block_tables is None else {tag: table[r0:r1] for tag, table in block_tables.items()},
                attn_inputs=primary, attention_inputs=sliced, workspace=workspace)
            if hidden is None:
                hidden = output.new_empty((input_ids.numel(), output.size(1)))
            hidden[t0:t1].copy_(output)
            del output
    finally:
        v4._mtp_hidden_buffer = fixed
        v4._mtp_hidden_valid_tokens = fixed_valid
        v4._mtp_last_hidden_buffer = last
        v4._mtp_last_hidden_valid_tokens = last_valid
        del workspace
    return PyModelOutputs(hidden) if aux is None else PyModelOutputs(hidden, aux)
