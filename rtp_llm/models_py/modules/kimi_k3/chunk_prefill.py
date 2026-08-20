"""Whole-model chunk Prefill planning and round-input construction for Kimi K3.

Round planning and packed-input construction live in the Python model; the
C++ executor observes each planned round through a
``mtp_chunk_prefill_round_hook`` callback and only assembles the mirrored
draft-model input.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs


@dataclass(frozen=True)
class KimiK3ChunkSlice:
    original_batch_idx: int
    source_start: int
    source_end: int
    prefix_length: int
    processed_length: int
    new_length: int
    absolute_start: int
    absolute_end: int
    terminal: bool


@dataclass(frozen=True)
class KimiK3ChunkRound:
    slices: tuple[KimiK3ChunkSlice, ...]

    @property
    def token_count(self) -> int:
        return sum(item.new_length for item in self.slices)


def _source_offsets(lengths: Sequence[int]) -> list[int]:
    offsets = [0]
    for length in lengths:
        if int(length) <= 0:
            raise ValueError(f"K3 chunk input lengths must be positive, got {length}")
        offsets.append(offsets[-1] + int(length))
    return offsets


def plan_kimi_k3_chunk_rounds(
    input_lengths: Sequence[int],
    prefix_lengths: Sequence[int],
    *,
    chunk_budget: int,
    page_size: int,
) -> tuple[KimiK3ChunkRound, ...]:
    """Split a packed Prefill batch at absolute MLA page boundaries."""

    lengths = [int(value) for value in input_lengths]
    prefixes = [int(value) for value in prefix_lengths]
    if len(lengths) != len(prefixes):
        raise ValueError(
            "K3 chunk input/prefix batch sizes differ: "
            f"input={len(lengths)} prefix={len(prefixes)}"
        )
    if not lengths:
        raise ValueError("K3 chunk planner requires at least one request")
    if chunk_budget <= 0:
        raise ValueError(f"K3 chunk budget must be positive, got {chunk_budget}")
    if page_size <= 0:
        raise ValueError(f"K3 MLA page size must be positive, got {page_size}")
    if any(prefix < 0 for prefix in prefixes):
        raise ValueError(f"K3 prefix lengths must be non-negative, got {prefixes}")
    if any(prefix % page_size for prefix in prefixes):
        raise ValueError(
            "whole-model K3 chunk Prefill requires page-aligned prefixes: "
            f"prefixes={prefixes} page={page_size}"
        )
    if chunk_budget < page_size and any(length > chunk_budget for length in lengths):
        raise ValueError(
            "K3 chunk budget must cover one MLA page when a request spans "
            f"rounds: budget={chunk_budget}, page={page_size}"
        )

    source_offsets = _source_offsets(lengths)
    processed = [0] * len(lengths)
    rounds: list[KimiK3ChunkRound] = []
    while any(done < total for done, total in zip(processed, lengths)):
        available = chunk_budget
        round_slices: list[KimiK3ChunkSlice] = []
        for request_idx, total_length in enumerate(lengths):
            done = processed[request_idx]
            remaining = total_length - done
            if remaining <= 0 or available <= 0:
                continue

            terminal = remaining <= available
            if terminal:
                take = remaining
            else:
                absolute_start = prefixes[request_idx] + done
                aligned_end = ((absolute_start + available) // page_size) * page_size
                take = aligned_end - absolute_start
                if take <= 0:
                    continue

            absolute_start = prefixes[request_idx] + done
            absolute_end = absolute_start + take
            if not terminal and absolute_end % page_size:
                raise AssertionError(
                    "non-terminal K3 chunk slice is not page aligned: "
                    f"request={request_idx} end={absolute_end} page={page_size}"
                )
            source_start = source_offsets[request_idx] + done
            round_slices.append(
                KimiK3ChunkSlice(
                    original_batch_idx=request_idx,
                    source_start=source_start,
                    source_end=source_start + take,
                    prefix_length=prefixes[request_idx],
                    processed_length=done,
                    new_length=take,
                    absolute_start=absolute_start,
                    absolute_end=absolute_end,
                    terminal=terminal,
                )
            )
            processed[request_idx] += take
            available -= take

        if not round_slices:
            pending = [
                idx
                for idx, (done, total) in enumerate(zip(processed, lengths))
                if done < total
            ]
            raise RuntimeError(
                "K3 chunk budget cannot advance any pending request to an "
                "MLA page boundary: "
                f"budget={chunk_budget}, page={page_size}, pending={pending}"
            )
        rounds.append(KimiK3ChunkRound(tuple(round_slices)))
    return tuple(rounds)


def validate_whole_chunk_prefill(
    inputs: PyModelInputs,
    chunk_tokens: int,
    *,
    tp_size: int,
    ep_size: int,
    page_size: Optional[int],
) -> None:
    """Reject unsupported whole-chunk modes before any cache mutation."""

    attention_inputs = inputs.attention_inputs
    if attention_inputs is None:
        raise RuntimeError("whole-model K3 Prefill requires attention inputs")
    if page_size is None:
        raise RuntimeError("whole-model K3 Prefill requires an initialized cache")
    if page_size <= 0 or page_size % 64:
        raise RuntimeError(
            "whole-model K3 Prefill requires a positive cache page size "
            "divisible by the cuLA checkpoint step 64; "
            f"page_size={page_size}"
        )
    if chunk_tokens <= 0 or chunk_tokens % tp_size:
        raise RuntimeError(
            "KIMI_K3_PREFILL_CHUNK_TOKENS must be divisible by attention TP; "
            f"chunk={chunk_tokens}, TP={tp_size}"
        )
    if ep_size != tp_size:
        raise RuntimeError(
            "whole-model K3 Prefill requires TP == EP Sequence Parallel; "
            f"TP={tp_size}, EP={ep_size}"
        )
    if bool(getattr(attention_inputs, "is_target_verify", False)):
        raise RuntimeError("whole-model K3 Prefill does not support target verify")
    if bool(getattr(attention_inputs, "is_cuda_graph", False)):
        raise RuntimeError("whole-model K3 Prefill does not support CUDA Graph")
    if getattr(attention_inputs, "context_parallel_info", None) is not None:
        raise RuntimeError(
            "whole-model K3 Prefill does not support framework Prefill CP"
        )
    multimodal = inputs.multimodal_inputs
    if multimodal.multimodal_features or (
        multimodal.mm_features_locs_host is not None
        and multimodal.mm_features_locs_host.numel()
    ):
        raise RuntimeError("whole-model K3 Prefill does not support multimodal input")


def host_lengths(value: torch.Tensor, name: str) -> list[int]:
    if value is None or not value.numel():
        raise RuntimeError(f"whole-model K3 Prefill requires {name}")
    source = value if value.device.type == "cpu" else value.detach().cpu()
    return [int(item) for item in source.tolist()]


def _select_batch_rows(
    value: torch.Tensor,
    indices: list[int],
    *,
    batch_dim: int = 0,
) -> torch.Tensor:
    if value is None or not value.numel():
        return value
    if batch_dim < 0 or batch_dim >= value.ndim:
        raise RuntimeError(
            "whole-model K3 block-table batch dimension is invalid: "
            f"shape={tuple(value.shape)} batch_dim={batch_dim}"
        )
    if min(indices) < 0 or max(indices) >= int(value.shape[batch_dim]):
        raise RuntimeError(
            "whole-model K3 active row is outside block table: "
            f"shape={tuple(value.shape)} batch_dim={batch_dim} indices={indices}"
        )
    index = torch.tensor(indices, dtype=torch.long, device=value.device)
    return value.index_select(batch_dim, index).contiguous()


def _select_group_batch_rows(
    values: Sequence[torch.Tensor], indices: list[int]
) -> list[torch.Tensor]:
    return [_select_batch_rows(value, indices) for value in values]


def build_chunk_attention_inputs(
    attention_inputs: PyAttentionInputs,
    *,
    round_plan: KimiK3ChunkRound,
    device: torch.device,
) -> PyAttentionInputs:
    """Rebuild packed attention and block-table metadata for one round."""

    lengths = [item.new_length for item in round_plan.slices]
    prefixes = [item.absolute_start for item in round_plan.slices]
    sequence_lengths = [item.absolute_end for item in round_plan.slices]
    batch_indices = [item.original_batch_idx for item in round_plan.slices]
    total_tokens = sum(lengths)
    chunk = copy.copy(attention_inputs)
    cu_seqlens = [0]
    cu_kv_seqlens = [0]
    for length, sequence_length in zip(lengths, sequence_lengths):
        cu_seqlens.append(cu_seqlens[-1] + length)
        cu_kv_seqlens.append(cu_kv_seqlens[-1] + sequence_length)
    chunk.cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    chunk.cu_kv_seqlens = torch.tensor(
        cu_kv_seqlens, dtype=torch.int32, device=device
    )
    chunk.input_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    chunk.prefix_lengths = torch.tensor(prefixes, dtype=torch.int32, device=device)
    chunk.sequence_lengths = torch.tensor(
        sequence_lengths, dtype=torch.int32, device=device
    )
    chunk.sequence_lengths_plus_1_d = chunk.sequence_lengths + 1
    max_length = max(lengths)
    padding_offset: list[int] = []
    cumulative_padding = 0
    for length in lengths:
        padding_offset.extend([cumulative_padding] * length)
        cumulative_padding += max_length - length
    chunk.padding_offset = torch.tensor(
        padding_offset, dtype=torch.int32, device=device
    )
    chunk.cu_seqlens_host = torch.tensor(cu_seqlens, dtype=torch.int32)
    chunk.input_lengths_host = torch.tensor(lengths, dtype=torch.int32)
    chunk.prefix_lengths_host = torch.tensor(prefixes, dtype=torch.int32)
    chunk.sequence_lengths_host = torch.tensor(sequence_lengths, dtype=torch.int32)
    chunk.total_tokens = int(total_tokens)
    chunk.context_total_kv_length = int(sum(sequence_lengths))
    chunk.is_prefill = True
    chunk.is_cuda_graph = False
    chunk.cache_store_inputs = None

    for name in (
        "kv_cache_block_id_host",
        "kv_cache_kernel_block_id_host",
        "kv_cache_kernel_block_id_device",
    ):
        value = getattr(attention_inputs, name)
        batch_dim = (
            1
            if name == "kv_cache_block_id_host"
            and value is not None
            and value.ndim == 3
            else 0
        )
        selected = _select_batch_rows(value, batch_indices, batch_dim=batch_dim)
        if selected is not None:
            setattr(chunk, name, selected)
    chunk.kv_cache_block_id_host_by_group = _select_group_batch_rows(
        attention_inputs.kv_cache_block_id_host_by_group, batch_indices
    )
    chunk.kv_cache_kernel_block_id_host_by_group = _select_group_batch_rows(
        attention_inputs.kv_cache_kernel_block_id_host_by_group, batch_indices
    )
    chunk.kv_cache_kernel_block_id_device_by_group = _select_group_batch_rows(
        attention_inputs.kv_cache_kernel_block_id_device_by_group, batch_indices
    )
    return chunk


def build_chunk_model_inputs(
    input_ids: torch.Tensor,
    attention_inputs: PyAttentionInputs,
    *,
    round_plan: KimiK3ChunkRound,
) -> PyModelInputs:
    chunk = PyModelInputs()
    chunk.input_ids = torch.cat(
        [
            input_ids.narrow(0, item.source_start, item.new_length)
            for item in round_plan.slices
        ],
        dim=0,
    )
    chunk.attention_inputs = build_chunk_attention_inputs(
        attention_inputs,
        round_plan=round_plan,
        device=input_ids.device,
    )
    return chunk


def prepare_round_fmha(fmha_impl: Any, attention_inputs: PyAttentionInputs) -> None:
    prepare = getattr(fmha_impl, "prepare", None)
    if not callable(prepare):
        raise RuntimeError(
            "whole-model K3 Prefill requires an FMHA implementation "
            "that can be replanned for each internal round"
        )
    prepare(attention_inputs)


def kda_materialized_block_maps(
    attention_inputs: PyAttentionInputs,
    *,
    layer_group_ids: Optional[Sequence[int]],
    kda_layer_indices: Sequence[int],
) -> Optional[tuple[torch.Tensor, ...]]:
    """Select active host block maps used to compact recurrent stores."""

    maps_by_group = getattr(
        attention_inputs, "kv_cache_kernel_block_id_host_by_group", None
    )
    if not maps_by_group or layer_group_ids is None:
        return None
    try:
        group_ids = sorted({int(layer_group_ids[index]) for index in kda_layer_indices})
    except IndexError as error:
        raise RuntimeError("KDA layer/group map does not cover every KDA layer") from error
    if any(group_id < 0 or group_id >= len(maps_by_group) for group_id in group_ids):
        raise RuntimeError("KDA cache group is outside host kernel block maps")
    return tuple(maps_by_group[group_id] for group_id in group_ids)


def kda_round_state_mapping(
    round_plan: Optional[KimiK3ChunkRound],
) -> tuple[Optional[list[int]], Optional[list[bool]]]:
    if round_plan is None:
        return None, None
    return (
        [item.original_batch_idx for item in round_plan.slices],
        [item.processed_length > 0 for item in round_plan.slices],
    )


__all__ = [
    "KimiK3ChunkRound",
    "KimiK3ChunkSlice",
    "build_chunk_model_inputs",
    "host_lengths",
    "kda_materialized_block_maps",
    "kda_round_state_mapping",
    "plan_kimi_k3_chunk_rounds",
    "prepare_round_fmha",
    "validate_whole_chunk_prefill",
]
