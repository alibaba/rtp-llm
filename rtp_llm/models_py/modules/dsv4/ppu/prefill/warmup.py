"""Bound scratch for the engine's reserved-page Prefill warmup only."""

from collections.abc import Mapping
import logging

import torch

from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelOutputs


def _warmup_reject_reason(attn, batch, cu_seqlens):
    """Return None when this attention input is a reserved-page warmup stream."""
    table = attn.kv_cache_block_id
    if not attn.is_prefill:
        return "not_prefill"
    if attn.is_cuda_graph:
        return "cuda_graph"
    # `is_s_padded` is deliberately not checked: PyWrappedModel.cc sets it for
    # `inputs.warmup || inputs.is_fake_stream`, i.e. exactly the streams this
    # helper exists for. The reserved-block-zero check below is the real signal.
    if attn.is_target_verify or attn.context_parallel_info is not None:
        return "target_verify_or_cp"
    if attn.cache_store_inputs is not None or attn.cache_store_writer is not None:
        return "cache_store"
    if table is None or table.device.type != "cpu" or table.ndim != 2:
        return "block_table_layout"
    if table.size(0) != batch or table.numel() == 0:
        return "block_table_batch"
    # Real allocations exclude physical block 0 (IBlockPool). fakeInitKVBlock
    # alone maps every request/tag to it; absence of a PD writer is insufficient.
    if bool(torch.any(table != 0)):
        return "live_blocks"
    if (attn.input_lengths is None or attn.prefix_lengths is None
            or attn.input_lengths.device.type != "cpu"
            or attn.prefix_lengths.device.type != "cpu"
            or attn.prefix_lengths.numel() != batch):
        return "lengths"
    if not torch.equal(attn.input_lengths, torch.diff(cu_seqlens)):
        return "lengths_not_matching_cu_seqlens"
    if attn.headwise_config is not None:
        return "headwise_config"
    return None


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
        reason = _warmup_reject_reason(attn, batch, cu_seqlens)
        if reason is not None:
            logging.log(
                20 if getattr(attn, "is_s_padded", False) else 10,
                "[PPU_PREFILL_WARMUP] skip reason=%s is_prefill=%s is_cuda_graph=%s "
                "is_s_padded=%s is_target_verify=%s cp=%s cache_store=%s batch=%d",
                reason, attn.is_prefill, attn.is_cuda_graph,
                getattr(attn, "is_s_padded", None), attn.is_target_verify,
                attn.context_parallel_info is not None,
                attn.cache_store_inputs is not None or attn.cache_store_writer is not None,
                batch,
            )
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
    # Warmup slices alias reserved block 0 and have no live cache rows, so the
    # padding semantics of the parent stream must survive the slicing.
    result.is_s_padded = bool(getattr(attn, "is_s_padded", False))
    padding_offset = getattr(attn, "padding_offset", None)
    for name in ("input_lengths", "prefix_lengths", "sequence_lengths",
                 "kv_cache_block_id", "kv_cache_block_id_device",
                 "kv_cache_kernel_block_id", "kv_cache_kernel_block_id_device",
                 "sequence_lengths_plus_1_device"):
        value = getattr(attn, name)
        if value is not None and value.numel():
            result_value = value[r0:r1]
            setattr(result, name, result_value)
    if padding_offset is not None and padding_offset.numel():
        total = int(cu_seqlens[-1])
        if padding_offset.numel() == total:
            result.padding_offset = padding_offset[t0:t1]
        elif padding_offset.numel() >= r1:
            result.padding_offset = padding_offset[r0:r1]
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
    # Outputs are contract-sized: PyWrappedModel compares hidden_states.size(0)
    # against the full combo token count, so these two buffers cannot be shrunk
    # to one group. Keep them as the only large allocations and log their size,
    # because that number is what the engine's warmup shape has to fit into.
    # `_mtp_hidden_buffer` is written by the layer stack, but only the slice of the
    # group currently running has to be valid. A per-group scratch keeps this at
    # `max_group_tokens` rows instead of the whole warmup batch; PyWrappedModel only
    # reads `mtp_target_hidden_states` when it is defined, so a discarded warmup
    # stream does not need the full-band MTP target tensor.
    max_group_tokens = max(t1 - t0 for _, _, t0, t1 in groups)
    mtp_scratch = None
    workspace = PrefillWorkspace(input_ids.device, q_rows=v4._prefill_ws_q_rows,
                                 q_dim=v4._prefill_ws_q_dim, reserve_cp=False)
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
            if fixed is not None and (t1 - t0) > fixed.size(0):
                if mtp_scratch is None:
                    mtp_scratch = fixed.new_empty((max_group_tokens, fixed.size(1)))
                v4._mtp_hidden_buffer = mtp_scratch[: t1 - t0]
            else:
                v4._mtp_hidden_buffer = fixed
            v4._mtp_last_hidden_buffer = None if last is None else last[r0:r1]
            output = forward_layers(
                v4, kv_cache, input_ids[t0:t1], positions[t0:t1],
                cu_seqlens[r0:r1 + 1] - t0,
                None if block_tables is None else {tag: table[r0:r1] for tag, table in block_tables.items()},
                attn_inputs=primary, attention_inputs=sliced, workspace=workspace)
            if hidden is None:
                hidden = output.new_empty((input_ids.numel(), output.size(1)))
                if hidden.is_cuda:
                    out_bytes = hidden.numel() * hidden.element_size()
                    out_bytes += 0 if mtp_scratch is None else mtp_scratch.numel() * mtp_scratch.element_size()
                    logging.info("[PPU_PREFILL_WARMUP] output buffers=%d MiB free=%d MiB",
                                 out_bytes // (1 << 20),
                                 int(torch.cuda.mem_get_info(hidden.device)[0]) // (1 << 20))
            hidden[t0:t1].copy_(output)
            del output
    finally:
        v4._mtp_hidden_buffer = fixed
        v4._mtp_hidden_valid_tokens = fixed_valid
        v4._mtp_last_hidden_buffer = last
        v4._mtp_last_hidden_valid_tokens = last_valid
        del workspace
        del mtp_scratch
    return PyModelOutputs(hidden)
