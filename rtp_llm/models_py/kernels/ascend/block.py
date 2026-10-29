import torch


def _load_initial_state_from_block_map_npu(
    prefix_lengths: torch.Tensor,
    block_map: torch.Tensor,
    conv_states: torch.Tensor,
    initial_states: torch.Tensor,
    seq_size_per_block: int,
) -> None:
    batch, max_block_size = block_map.shape
    assert prefix_lengths.shape[0] == batch

    if batch == 0:
        return
    if max_block_size == 0 or conv_states.shape[0] == 0:
        initial_states.zero_()
        return

    block_positions = torch.div(
        prefix_lengths - 1,
        seq_size_per_block,
        rounding_mode="floor",
    ).clamp_min_(0)
    batch_indices = torch.arange(batch, device=block_map.device)
    block_indices = block_map[batch_indices, block_positions.to(dtype=torch.long)].to(
        dtype=torch.long
    )

    # A zero-length prefix must not consult block_map.  Selecting block zero
    # here keeps the gather valid; torch.where below discards the value.
    block_indices = torch.where(
        prefix_lengths == 0,
        torch.zeros_like(block_indices),
        block_indices,
    )
    # The paged cache is V-first so recurrent decode can consume it directly,
    # whereas the historical chunk-prefill interface exposes K-first states.
    loaded_states = (
        conv_states.index_select(0, block_indices).transpose(-1, -2).contiguous()
    )
    if loaded_states.shape != initial_states.shape:
        raise ValueError(
            "initial_states must use the K-first transpose of the paged cache"
        )
    has_prefix = (prefix_lengths != 0).reshape(
        batch, *((1,) * (initial_states.ndim - 1))
    )
    initial_states.copy_(
        torch.where(has_prefix, loaded_states, torch.zeros_like(loaded_states))
    )


def load_initial_state_from_block_map(
    prefix_lengths: torch.Tensor,
    block_map: torch.Tensor,
    conv_states: torch.Tensor,
    initial_states: torch.Tensor,
    seq_size_per_block: int,
    block_v: int = 64,
):
    return _load_initial_state_from_block_map_npu(
        prefix_lengths,
        block_map,
        conv_states,
        initial_states,
        seq_size_per_block,
    )


def _copy_states_to_blocks(
    ssm_states: torch.Tensor,
    block_indices: torch.Tensor,
    source_states: torch.Tensor,
) -> None:
    valid_positions = torch.nonzero(block_indices > 0, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        return

    valid_blocks = block_indices.index_select(0, valid_positions).to(dtype=torch.long)
    valid_sources = (
        source_states.index_select(0, valid_positions)
        .transpose(-1, -2)
        .contiguous()
        .to(dtype=ssm_states.dtype)
    )
    ssm_states.index_copy_(0, valid_blocks, valid_sources)


def _store_ssm_state_to_block_map_npu(
    h: torch.Tensor,
    final_states: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    block_map: torch.Tensor,
    ssm_states: torch.Tensor,
    seq_size_per_block: int,
    chunk_size: int,
) -> None:
    assert (
        h.dtype == torch.float32 and final_states.dtype == torch.float32
    ), "h and final_states must be float32"

    batch = prefix_lengths.shape[0]
    if batch == 0:
        return

    head_count, value_dim, key_dim = ssm_states.shape[1:]
    source_shape = (head_count, key_dim, value_dim)
    h_states = h.reshape(-1, *source_shape)
    final_states_flat = final_states.reshape(batch, *source_shape)

    input_lengths = cu_seqlens[1 : batch + 1] - cu_seqlens[:batch]
    chunks_per_sequence = torch.div(
        input_lengths + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
    ).to(dtype=torch.long)
    sequence_indices = torch.arange(batch, device=cu_seqlens.device)
    chunk_batches = torch.repeat_interleave(sequence_indices, chunks_per_sequence)
    if chunk_batches.numel() == 0:
        return

    chunk_offsets = torch.cumsum(chunks_per_sequence, dim=0) - chunks_per_sequence
    local_chunks = torch.arange(
        chunk_batches.shape[0], device=cu_seqlens.device
    ) - torch.repeat_interleave(chunk_offsets, chunks_per_sequence)
    chunk_input_lengths = input_lengths.index_select(0, chunk_batches)
    chunk_prefix_lengths = prefix_lengths.index_select(0, chunk_batches)
    is_last_chunk = (local_chunks + 1) * chunk_size >= chunk_input_lengths
    is_middle_block_end = (
        (~is_last_chunk)
        & (local_chunks > 0)
        & (((local_chunks + 1) * chunk_size) % seq_size_per_block == 0)
    )

    # The intermediate state h[i] is the state at the start of chunk i.
    # Therefore an aligned end of global chunk i is stored from h[i + 1].
    middle_positions = torch.nonzero(is_middle_block_end, as_tuple=False).flatten()
    if middle_positions.numel() != 0:
        middle_batches = chunk_batches.index_select(0, middle_positions)
        middle_local_chunks = local_chunks.index_select(0, middle_positions)
        middle_prefixes = chunk_prefix_lengths.index_select(0, middle_positions)
        middle_block_positions = torch.div(
            middle_prefixes + (middle_local_chunks + 1) * chunk_size - 1,
            seq_size_per_block,
            rounding_mode="floor",
        ).to(dtype=torch.long)
        middle_blocks = block_map[middle_batches, middle_block_positions].to(
            dtype=torch.long
        )
        middle_global_chunks = middle_positions + 1
        middle_sources = h_states.index_select(0, middle_global_chunks)
        _copy_states_to_blocks(ssm_states, middle_blocks, middle_sources)

    # Write final states after intermediate states.  In the unusual case that
    # both resolve to the same physical block, the final state must win.
    last_positions = torch.nonzero(is_last_chunk, as_tuple=False).flatten()
    if last_positions.numel() != 0:
        last_batches = chunk_batches.index_select(0, last_positions)
        last_prefixes = chunk_prefix_lengths.index_select(0, last_positions)
        last_input_lengths = chunk_input_lengths.index_select(0, last_positions)
        last_block_positions = torch.div(
            last_prefixes + last_input_lengths - 1,
            seq_size_per_block,
            rounding_mode="floor",
        ).to(dtype=torch.long)
        last_blocks = block_map[last_batches, last_block_positions].to(dtype=torch.long)
        last_sources = final_states_flat.index_select(0, last_batches)
        _copy_states_to_blocks(ssm_states, last_blocks, last_sources)


def store_ssm_state_to_block_map(
    h: torch.Tensor,
    final_states: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    block_map: torch.Tensor,
    ssm_states: torch.Tensor,
    seq_size_per_block: int,
    chunk_size: int,
    block_v: int = 64,
):
    return _store_ssm_state_to_block_map_npu(
        h,
        final_states,
        prefix_lengths,
        cu_seqlens,
        block_map,
        ssm_states,
        seq_size_per_block,
        chunk_size,
    )
