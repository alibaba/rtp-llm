"""Ascend causal convolution wrappers for Qwen3.5.

The Ascend implementation adapts RTP-LLM's paged convolution cache to the
layout expected by ``fla_npu``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

PAD_SLOT_ID = -1


@dataclass
class CausalConv1dMetadata:
    """Metadata compatible with the legacy Triton causal-convolution API.

    AscendC does not need the Triton launch metadata, so all three fields are
    empty for an NPU request.
    """

    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor
    total: int


def _activation_name(activation: Union[bool, str, None]) -> Union[str, None]:
    if activation is None or activation is False:
        return None
    if activation is True or activation in ("silu", "swish"):
        return "silu"
    raise ValueError("activation must be None, False, True, 'silu', or 'swish'")


def _load_npu_causal_conv1d():
    from fla_npu.ops.ascendc import npu_causal_conv1d

    return npu_causal_conv1d


def _load_npu_causal_conv1d_update():
    from fla_npu.ops.ascendc import npu_causal_conv1d_update

    return npu_causal_conv1d_update


def _causal_conv1d_update_device(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Union[bool, str, None],
    block_map: torch.Tensor,
    sequence_lengths: torch.Tensor,
    seq_size_per_block: int,
    original_dtype: torch.dtype,
    squeeze_token_axis: bool,
) -> torch.Tensor:
    """Single-token conv update with fully device-side metadata.

    Same AscendC operator family as the host-metadata path (single consumer):
    the write page per sequence is gathered from ``block_map`` on device and
    the cross-block conv-state seed uses block-level page-row copies, so the
    step is aclgraph-capturable.  ``conv_state`` is the raw paged pool view
    ``(pages, state_len, dim)``; the operator mutates it in place.
    """

    from rtp_llm.models_py.kernels.ascend.paged_state import (
        decode_state_indices,
        seed_state_segment,
    )

    npu_causal_conv1d_update = _load_npu_causal_conv1d_update()
    read_idx, write_idx = decode_state_indices(
        block_map, sequence_lengths, seq_size_per_block
    )
    seed_state_segment(conv_state, read_idx, write_idx)
    # feed the op a dedicated out= buffer: its fallback path copies the result
    # into the x argument, which would clobber the caller's tensor
    x_work = x.to(conv_state.dtype)
    out = torch.empty_like(x_work)
    npu_causal_conv1d_update(
        x=x_work,
        conv_state=conv_state,
        weight=weight.t().contiguous(),
        bias=bias,
        activation=_activation_name(activation),
        conv_state_indices=write_idx.to(torch.int32),
        null_block_id=0,
        out=out,
    )
    out = out.to(original_dtype)
    # caller contract: (batch, dim) in, (batch, dim) out; otherwise
    # (batch, dim, 1)
    return out if squeeze_token_axis else out.unsqueeze(-1)


def _activation_mode(activation: Union[bool, str, None]) -> int:
    if activation is None or activation is False:
        return 0
    if activation is True or activation in ("silu", "swish"):
        return 1
    raise ValueError("activation must be None, False, True, 'silu', or 'swish'")


def _as_int_list(values) -> list[int]:
    if isinstance(values, (list, tuple)):
        return [int(value) for value in values]
    return [
        int(value) for value in values.detach().to(dtype=torch.int64).cpu().tolist()
    ]


def _as_int_rows(values) -> list[list[int]]:
    if isinstance(values, (list, tuple)):
        return [[int(value) for value in row] for row in values]
    return [
        [int(value) for value in row]
        for row in values.detach().to(dtype=torch.int64).cpu().tolist()
    ]


def _mapped_page(
    block_rows: list[list[int]],
    sequence_index: int,
    block_index: int,
    pad_slot_id: int,
) -> int:
    if block_index < 0 or sequence_index >= len(block_rows):
        return pad_slot_id
    row = block_rows[sequence_index]
    if block_index >= len(row):
        return pad_slot_id
    return int(row[block_index])


def prepare_causal_conv1d_metadata(
    query_start_loc: torch.Tensor,
    device: torch.device,
) -> CausalConv1dMetadata:
    """Return the no-op launch metadata expected by the shared model."""

    empty = torch.empty(0, dtype=torch.int32, device=device)
    return CausalConv1dMetadata(empty, empty, 0)


def _gather_pages_from_block_map(
    block_map: torch.Tensor,
    block_indices: torch.Tensor,
    pad_slot_id: int,
) -> torch.Tensor:
    """Device-side page lookup per sequence (equivalent of ``_mapped_page``).

    ``block_map``: (batch, max_blocks) — 3-D group-prefixed tables take
    group 0.  ``block_indices``: (batch,) logical block index.  Returns
    (batch,) int32 device pages; out-of-range entries are masked to
    ``pad_slot_id`` so callers rely on a uniform sentinel.
    """

    if block_map.dim() == 3:
        block_map = block_map[0]
    elif block_map.dim() != 2:
        raise ValueError(
            f"block_map must be 2-D or 3-D, got shape {tuple(block_map.shape)}"
        )
    block_indices = block_indices.to(block_map.device)
    max_col = block_map.shape[1]
    in_range = block_indices < max_col
    safe_indices = block_indices.clamp(min=0, max=max_col - 1)
    pages = block_map.gather(1, safe_indices.unsqueeze(1)).squeeze(1)
    pages = torch.where(
        in_range & (block_indices >= 0), pages, torch.full_like(pages, pad_slot_id)
    )
    return pages.to(torch.int32)


def _gather_prefill_states_device(
    x: torch.Tensor,
    conv_states: Optional[torch.Tensor],
    block_map: Optional[torch.Tensor],
    prefix_lengths: torch.Tensor,
    seq_size_per_block: int,
    state_len: int,
    pad_slot_id: int,
) -> torch.Tensor:
    """Gather each sequence's prefix-ending state on device.

    Produces the flat (batch, state_len, dim) initial-state buffer the
    FLA-NPU prefill op requires from the paged (pages, dim, state) view.
    """

    batch = prefix_lengths.shape[0]
    dim = x.shape[0]  # x: (dim, total_tokens)
    initial_states = torch.zeros(
        (batch, state_len, dim), dtype=x.dtype, device=x.device
    )
    if conv_states is None or block_map is None or state_len == 0:
        return initial_states

    # Prefill is eager-only (the ACL graph runner is decode-only): the
    # boolean-mask indexing below is not graph-capture safe and must not be
    # copied into the decode path.
    prefix_positive = prefix_lengths > 0
    block_indices = (prefix_lengths - 1).clamp(min=0) // seq_size_per_block
    page_indices = _gather_pages_from_block_map(block_map, block_indices, pad_slot_id)

    read_mask = prefix_positive & (page_indices != pad_slot_id)
    valid_pages = page_indices[read_mask].long()
    # conv_states: (pages, dim, state) in RTP layout; FLA wants (state, dim)
    gathered = conv_states.index_select(0, valid_pages).transpose(1, 2)
    initial_states[read_mask] = gathered
    return initial_states


def _history_ending_at(
    initial_state: torch.Tensor,
    sequence_x: torch.Tensor,
    end: int,
) -> torch.Tensor:
    """Return the fixed-width input history ending before ``end``."""

    state_len = initial_state.shape[0]
    if state_len == 0:
        return initial_state
    if end >= state_len:
        return sequence_x[end - state_len : end]
    return torch.cat((initial_state[end:], sequence_x[:end]), dim=0)


def _scatter_prefill_states_host(
    x: torch.Tensor,
    conv_states: Optional[torch.Tensor],
    block_map: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    prefix_lengths: torch.Tensor,
    seq_size_per_block: int,
    initial_states: torch.Tensor,
    state_len: int,
    pad_slot_id: int,
) -> None:
    """Host-side multi-block cache snapshot write-back for prefill.

    Writes a conv_state snapshot at every crossed block edge and at each
    sequence end.  Intentionally not graph-safe (``.tolist()`` + Python
    loops) — prefill is never graph-captured (the ACL graph runner is
    decode-only).
    """

    if conv_states is None or block_map is None or state_len == 0:
        return

    query_starts = [int(v) for v in query_start_loc.detach().cpu().tolist()]
    prefix_values = [int(v) for v in prefix_lengths.detach().cpu().tolist()]
    block_map_2d = block_map[0] if block_map.dim() == 3 else block_map
    block_rows = [
        [int(p) for p in row.detach().cpu().tolist()] for row in block_map_2d
    ]

    for sequence_index, prefix_length in enumerate(prefix_values):
        token_start = query_starts[sequence_index]
        token_end = query_starts[sequence_index + 1]
        sequence_x = x[:, token_start:token_end].transpose(0, 1)
        sequence_len = token_end - token_start

        for local_end in range(1, sequence_len + 1):
            absolute_end = prefix_length + local_end
            if absolute_end % seq_size_per_block != 0 and local_end != sequence_len:
                continue

            block_index = (absolute_end - 1) // seq_size_per_block
            if (
                block_index >= len(block_rows[sequence_index])
                or block_index < 0
            ):
                continue
            page_index = block_rows[sequence_index][block_index]
            if page_index == pad_slot_id:
                continue

            history = _history_ending_at(
                initial_states[sequence_index], sequence_x, local_end
            )
            conv_states[page_index, :, :state_len].copy_(history.transpose(0, 1))


def _load_npu_causal_conv1d_fn():
    from fla_npu.ops.ascendc import npu_causal_conv1d_fn

    return npu_causal_conv1d_fn


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    block_map: Optional[torch.Tensor],
    prefix_lengths: torch.Tensor,
    seq_size_per_block: int,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata: Optional[CausalConv1dMetadata] = None,
    validate_data=False,
):
    """Run varlen causal convolution and update the paged cache in place.

    Uses the FLA-NPU ``npu_causal_conv1d_fn`` entry with device-tensor
    metadata (``query_start_loc`` / ``has_initial_state`` int32); initial
    states are gathered device-side from the paged cache.  **Eager-only** —
    the multi-block snapshot write-back keeps a host loop by design (the ACL
    graph runner is decode-only).  Do not call from a capture region.
    """

    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("NPU prefill expects x=(dim, tokens), weight=(dim, width)")
    if seq_size_per_block <= 0:
        raise ValueError("seq_size_per_block must be positive")

    original_dtype = x.dtype
    x_work = x.to(weight.dtype)
    dim, _ = x_work.shape
    weight_dim, width = weight.shape
    if dim != weight_dim:
        raise ValueError("x and weight feature dimensions must match")

    batch = query_start_loc.shape[0] - 1
    if prefix_lengths.shape[0] != batch:
        raise ValueError(
            f"prefix_lengths must contain one value per sequence, got "
            f"{prefix_lengths.shape[0]} vs expected batch={batch}"
        )

    state_len = width - 1
    initial_states = _gather_prefill_states_device(
        x_work,
        conv_states,
        block_map,
        prefix_lengths,
        seq_size_per_block,
        state_len,
        pad_slot_id,
    )
    has_initial_state = (prefix_lengths > 0).to(torch.int32)

    # FLA-NPU layout: conv_states (batch, state_len, dim); weight (width, dim);
    # x (tokens, dim).  The op writes the sequence-end state back into the
    # conv_states buffer in place — pass a copy so the pristine initial
    # states remain available to the snapshot write-back below.
    npu_states = initial_states.clone()
    npu_weight = weight.transpose(0, 1).contiguous()
    npu_x = x_work.transpose(0, 1).contiguous()

    npu_causal_conv1d_fn = _load_npu_causal_conv1d_fn()
    output = npu_causal_conv1d_fn(
        x=npu_x,
        weight=npu_weight,
        bias=bias,
        conv_states=npu_states,
        query_start_loc=query_start_loc.to(torch.int32),
        has_initial_state=has_initial_state,
        activation=_activation_name(activation),
        pad_slot_id=pad_slot_id,
        validate_data=False,  # keep off the D2H path
    )

    if conv_states is not None and block_map is not None and state_len > 0:
        _scatter_prefill_states_host(
            x=x_work,
            conv_states=conv_states,
            block_map=block_map,
            query_start_loc=query_start_loc,
            prefix_lengths=prefix_lengths,
            seq_size_per_block=seq_size_per_block,
            initial_states=initial_states,
            state_len=state_len,
            pad_slot_id=pad_slot_id,
        )

    return output.transpose(0, 1).to(original_dtype)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """Decode one or more tokens while preserving RTP-LLM's paged state.

    ``conv_state`` is the raw paged pool view ``(pages, state_len, dim)`` —
    the layout produced by ``typed_storage_view`` and consumed directly by the
    AscendC operator (note the different contract from ``causal_conv1d_fn``,
    which keeps the transposed ``(page, dim, state)`` view).

    Single-token decode runs the device-metadata path shared by eager and
    aclgraph capture; multi-token speculative decode keeps the host-metadata
    path (eager only — target-verify never runs inside the decode graph).
    """

    if seq_size_per_block <= 0:
        raise ValueError("seq_size_per_block must be positive")
    if block_map is None or sequence_lengths is None:
        raise ValueError("block_map and sequence_lengths are required on NPU")
    if cache_seqlens is not None or query_start_loc is not None:
        raise NotImplementedError(
            "Ascend paged decode does not support cache_seqlens or varlen "
            "query_start_loc"
        )

    original_dtype = x.dtype
    squeeze_token_axis = x.dim() == 2
    if squeeze_token_axis:
        x = x.unsqueeze(-1)
    if x.dim() != 3:
        raise ValueError("NPU decode expects x=(batch, dim, tokens)")

    batch, dim, token_count = x.shape
    if weight.dim() != 2 or weight.shape[0] != dim:
        raise ValueError("weight must have shape (dim, width)")

    # Standard decode (one token per sequence): device-metadata path.
    if (
        token_count == 1
        and block_map.ndim == 2
        and block_map.shape[0] == batch
    ):
        return _causal_conv1d_update_device(
            x[:, :, 0],
            conv_state,
            weight,
            bias,
            activation,
            block_map,
            sequence_lengths,
            int(seq_size_per_block),
            original_dtype,
            squeeze_token_axis,
        )

    x_work = x.to(conv_state.dtype)

    npu_causal_conv1d = _load_npu_causal_conv1d()
    npu_states = conv_state
    npu_weight = weight.transpose(0, 1).contiguous()
    output_tokens = []

    current_lengths = _as_int_list(sequence_lengths)
    if len(current_lengths) != batch:
        raise ValueError("sequence_lengths must contain one value per sequence")
    block_rows = _as_int_rows(block_map)

    read_pages = []
    write_block_starts = []
    for sequence_index, first_total_length in enumerate(current_lengths):
        if first_total_length <= 0:
            raise ValueError("decode sequence lengths must be positive")
        read_block = max(first_total_length - 2, 0) // seq_size_per_block
        read_page = _mapped_page(block_rows, sequence_index, read_block, pad_slot_id)
        read_pages.append(read_page)
        write_block_starts.append((first_total_length - 1) // seq_size_per_block)

    for token_index in range(token_count):
        cache_indices = []
        for sequence_index in range(batch):
            # Match the CUDA continuous-batching contract: every speculative
            # token is snapshotted into a consecutive block-map entry, even if
            # its logical position has not crossed a normal cache-block edge.
            target_block = write_block_starts[sequence_index] + token_index
            target_page = _mapped_page(
                block_rows, sequence_index, target_block, pad_slot_id
            )

            source_page = (
                read_pages[sequence_index]
                if token_index == 0
                else _mapped_page(
                    block_rows,
                    sequence_index,
                    target_block - 1,
                    pad_slot_id,
                )
            )
            if (
                source_page != pad_slot_id
                and target_page != pad_slot_id
                and source_page != target_page
            ):
                conv_state[target_page].copy_(conv_state[source_page])
            cache_indices.append(target_page)

        token_output = npu_causal_conv1d(
            x=x_work[:, :, token_index],
            weight=npu_weight,
            bias=bias,
            conv_states=npu_states,
            cache_indices=cache_indices,
            activation_mode=_activation_mode(activation),
            pad_slot_id=pad_slot_id,
            run_mode=1,
            head_num=0,
        )
        output_tokens.append(token_output)

    if output_tokens:
        output = torch.stack(output_tokens, dim=-1)
    else:
        output = torch.empty_like(x_work)
    if squeeze_token_axis:
        output = output.squeeze(-1)
    return output.to(original_dtype)
