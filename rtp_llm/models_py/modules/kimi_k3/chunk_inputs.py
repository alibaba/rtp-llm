"""Rebuild one eager text-prefill round using main's group-local cache ABI."""

from collections.abc import Mapping

import torch

from rtp_llm.ops.compute_ops import PyModelInputs


def _host(values, device):
    return torch.tensor(values, dtype=torch.int32, pin_memory=device.type == "cuda")


def _packed_rows(value, plan, padding):
    if value is None or value.numel() == 0:
        return value
    pieces = [value.narrow(0, s.source_start, s.new_length) for s in plan.slices]
    if padding:
        pieces.append(value.new_zeros((padding, *value.shape[1:])))
    return torch.cat(pieces, dim=0)


def _table_rows(table, indices, padding):
    if table is None or table.numel() == 0:
        return table
    if table.ndim != 2:
        raise ValueError(
            "Main chunk prefill expects group-local [request, block] tables"
        )
    selected = table.index_select(
        0, torch.tensor(indices, device=table.device, dtype=torch.long)
    )
    if padding:
        selected = torch.cat((selected, table.new_zeros((1, table.shape[1]))))
    if table.device.type == "cpu" and table.is_pinned():
        selected = selected.pin_memory()
    return selected


def build_chunk_inputs(inputs, plan, tp_size):
    if tp_size <= 0 or not plan.slices:
        raise ValueError(
            "Chunk input construction requires a nonempty plan and positive TP"
        )
    device = inputs.input_ids.device
    padding = (-plan.token_count) % tp_size
    indices = [s.original_batch_idx for s in plan.slices]
    lengths = [s.new_length for s in plan.slices]
    prefixes = [s.absolute_start for s in plan.slices]
    if padding:
        lengths.append(padding)
        prefixes.append(0)
    host_lengths, host_prefixes = _host(lengths, device), _host(prefixes, device)
    device_lengths = host_lengths.to(device, non_blocking=True)
    device_prefixes = host_prefixes.to(device, non_blocking=True)
    cu, cu_kv = [0], [0]
    for length, prefix in zip(lengths, prefixes):
        cu.append(cu[-1] + length)
        cu_kv.append(cu_kv[-1] + length + prefix)
    host_cu = _host(cu, device)
    physical = plan.token_count + padding
    positions = _packed_rows(inputs.combo_position_ids, plan, padding)

    def attention(original):
        chunk = original.for_prefill_chunk(
            host_lengths, host_prefixes, device_lengths, device_prefixes
        )
        # Main identifies eager prefill with no decode sequence-length rows.
        chunk.sequence_lengths = _host([], device)
        chunk.sequence_lengths_plus_1_device = device_prefixes + 1
        chunk.cu_seqlens = host_cu
        chunk.cu_seqlens_device = host_cu.to(device, non_blocking=True)
        chunk.cu_kv_seqlens_device = _host(cu_kv, device).to(device)
        chunk.context_total_kv_length = cu_kv[-1]
        chunk.total_tokens = physical
        chunk.logical_request_count = len(indices)
        chunk.physical_request_count = len(lengths)
        chunk.logical_token_count = plan.token_count
        chunk.physical_token_count = physical
        chunk.is_s_padded = bool(padding)
        chunk.valid_token_mask = (
            torch.arange(physical, device=device) < plan.token_count
        )
        chunk.combo_position_ids = positions
        max_length = max(lengths)
        offsets = [i * max_length - cu[i] for i in range(len(lengths))]
        chunk.padding_offset = torch.repeat_interleave(
            torch.tensor(offsets, dtype=torch.int32, device=device),
            device_lengths,
            output_size=physical,
        )
        for field in (
            "kv_cache_kernel_block_id",
            "kv_cache_kernel_block_id_device",
            "kv_cache_block_id",
            "kv_cache_block_id_device",
        ):
            setattr(
                chunk, field, _table_rows(getattr(original, field), indices, padding)
            )
        return chunk

    result = PyModelInputs()
    result.input_ids = _packed_rows(inputs.input_ids, plan, padding)
    result.input_hiddens = _packed_rows(inputs.input_hiddens, plan, padding)
    result.combo_position_ids = positions
    tagged = inputs.attention_inputs
    result.attention_inputs = (
        {tag: attention(value) for tag, value in tagged.items()}
        if isinstance(tagged, Mapping)
        else attention(tagged)
    )
    return result
