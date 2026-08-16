"""Kimi K3 linear-cache state storage kernel."""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class KimiKDARecurrentCheckpointMetadata:
    checkpoint_offsets: torch.Tensor
    store_checkpoint_indices: torch.Tensor
    store_sequence_indices: torch.Tensor
    store_page_indices: torch.Tensor
    final_checkpoint_indices: torch.Tensor
    total_checkpoints: int


def prepare_kimi_kda_recurrent_checkpoint_metadata(
    input_lengths_host: torch.Tensor,
    prefix_lengths_host: torch.Tensor,
    page_size: int,
    device: torch.device,
    *,
    materialized_block_maps_host: Optional[Sequence[torch.Tensor]] = None,
) -> KimiKDARecurrentCheckpointMetadata:
    host_lengths = (input_lengths_host, prefix_lengths_host)
    if any(
        tensor.ndim != 1
        or tensor.numel() == 0
        or tensor.device.type != "cpu"
        for tensor in host_lengths
    ) or input_lengths_host.shape != prefix_lengths_host.shape:
        raise ValueError(
            "KDA checkpoint metadata requires matching CPU "
            "input_lengths/prefix_lengths=[N]"
        )
    if page_size <= 0:
        raise ValueError(f"KDA checkpoint page size must be positive, got {page_size}")
    lengths = [int(value) for value in input_lengths_host.tolist()]
    prefixes = [int(value) for value in prefix_lengths_host.tolist()]
    if any(length <= 0 for length in lengths):
        raise ValueError(f"KDA checkpoint lengths must be positive, got {lengths}")
    if any(prefix < 0 or prefix % page_size for prefix in prefixes):
        raise ValueError(
            "KDA checkpoint prefixes must be non-negative and page-aligned, "
            f"got {prefixes}"
        )
    block_maps = list(materialized_block_maps_host or ())
    for block_map in block_maps:
        if (
            block_map.device.type != "cpu"
            or block_map.ndim != 2
            or block_map.shape[0] != len(lengths)
        ):
            raise ValueError(
                "KDA materialized block maps must be CPU [N,pages] tensors"
            )
    counts = [(length + page_size - 1) // page_size for length in lengths]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    store_sources: list[int] = []
    store_sequences: list[int] = []
    store_pages: list[int] = []
    final_sources: list[int] = []
    for sequence_idx, (length, prefix, count) in enumerate(
        zip(lengths, prefixes, counts)
    ):
        final_sources.append(offsets[sequence_idx + 1] - 1)
        for local_checkpoint in range(count):
            local_end = min((local_checkpoint + 1) * page_size, length)
            absolute_end = prefix + local_end
            page = (absolute_end - 1) // page_size
            materialized = not block_maps or all(
                page < int(block_map.shape[1])
                and int(block_map[sequence_idx, page]) > 0
                for block_map in block_maps
            )
            if materialized:
                store_sources.append(offsets[sequence_idx] + local_checkpoint)
                store_sequences.append(sequence_idx)
                store_pages.append(page)
    return KimiKDARecurrentCheckpointMetadata(
        checkpoint_offsets=torch.tensor(
            offsets, dtype=torch.int32, device=device
        ),
        store_checkpoint_indices=torch.tensor(
            store_sources, dtype=torch.int32, device=device
        ),
        store_sequence_indices=torch.tensor(
            store_sequences, dtype=torch.int32, device=device
        ),
        store_page_indices=torch.tensor(
            store_pages, dtype=torch.int32, device=device
        ),
        final_checkpoint_indices=torch.tensor(
            final_sources, dtype=torch.int64, device=device
        ),
        total_checkpoints=offsets[-1],
    )


@triton.jit(do_not_specialize=["max_block_count", "physical_block_count"])
def _load_recurrent_state_from_block_map_kernel(
    prefix_lengths,
    block_map,
    ssm_cache,
    output,
    stride_bm_b,
    stride_bm_page,
    stride_s_block,
    stride_s_h,
    stride_s_k,
    stride_s_v,
    stride_o_b,
    stride_o_h,
    stride_o_k,
    stride_o_v,
    max_block_count,
    physical_block_count,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i_b = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    state_elements = HEADS * K * V
    valid_element = offsets < state_elements
    i_v = offsets % V
    i_k = (offsets // V) % K
    i_h = offsets // (K * V)

    prefix = tl.load(prefix_lengths + i_b).to(tl.int64)
    page_raw = (prefix - 1) // PAGE_SIZE
    page_valid = (prefix > 0) & (page_raw >= 0) & (page_raw < max_block_count)
    page = tl.where(page_valid, page_raw, 0)
    block_id = tl.load(
        block_map + i_b * stride_bm_b + page * stride_bm_page,
        mask=page_valid,
        other=0,
    ).to(tl.int64)
    valid_state = (
        valid_element
        & page_valid
        & (block_id > 0)
        & (block_id < physical_block_count)
    )
    value = tl.load(
        ssm_cache
        + block_id * stride_s_block
        + i_h * stride_s_h
        + i_k * stride_s_k
        + i_v * stride_s_v,
        mask=valid_state,
        other=0,
    )
    tl.store(
        output
        + i_b * stride_o_b
        + i_h * stride_o_h
        + i_k * stride_o_k
        + i_v * stride_o_v,
        value,
        mask=valid_element,
    )


@triton.jit(do_not_specialize=["max_block_count", "physical_block_count"])
def _store_selected_recurrent_checkpoints_kernel(
    checkpoints,
    checkpoint_indices,
    sequence_indices,
    page_indices,
    block_map,
    ssm_cache,
    stride_c_n,
    stride_c_h,
    stride_c_k,
    stride_c_v,
    stride_bm_b,
    stride_bm_page,
    stride_s_block,
    stride_s_h,
    stride_s_k,
    stride_s_v,
    max_block_count,
    physical_block_count,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    store_index = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    checkpoint_index = tl.load(checkpoint_indices + store_index).to(tl.int64)
    i_b = tl.load(sequence_indices + store_index).to(tl.int64)
    page_raw = tl.load(page_indices + store_index).to(tl.int64)

    state_elements = HEADS * K * V
    valid_element = offsets < state_elements
    i_v = offsets % V
    i_k = (offsets // V) % K
    i_h = offsets // (K * V)

    page_valid = (page_raw >= 0) & (page_raw < max_block_count)
    page = tl.where(page_valid, page_raw, 0)
    block_id = tl.load(
        block_map + i_b * stride_bm_b + page * stride_bm_page,
        mask=page_valid,
        other=0,
    ).to(tl.int64)
    mask = (
        page_valid
        & valid_element
        & (block_id > 0)
        & (block_id < physical_block_count)
    )
    value = tl.load(
        checkpoints
        + checkpoint_index * stride_c_n
        + i_h * stride_c_h
        + i_k * stride_c_k
        + i_v * stride_c_v,
        mask=mask,
        other=0,
    )
    tl.store(
        ssm_cache
        + block_id * stride_s_block
        + i_h * stride_s_h
        + i_k * stride_s_k
        + i_v * stride_s_v,
        value,
        mask=mask,
    )


@torch.compiler.disable
def kimi_kda_load_recurrent_state(
    prefix_lengths: torch.Tensor,
    linear_block_map: torch.Tensor,
    ssm_cache: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Gather one contiguous FP32 recurrent initial state per sequence."""

    if (
        prefix_lengths.ndim != 1
        or linear_block_map.ndim != 2
        or prefix_lengths.numel() != linear_block_map.shape[0]
        or ssm_cache.ndim != 4
        or linear_block_map.shape[1] == 0
    ):
        raise ValueError("invalid KDA recurrent gather shapes")
    tensors = (prefix_lengths, linear_block_map, ssm_cache)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("KDA recurrent gather requires CUDA tensors")
    if ssm_cache.dtype != torch.float32:
        raise ValueError(f"KDA recurrent cache must be FP32, got {ssm_cache.dtype}")
    if prefix_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("KDA recurrent prefix lengths must be int32/int64")
    if linear_block_map.dtype not in (torch.int32, torch.int64):
        raise ValueError("KDA recurrent LINEAR block map must be int32/int64")
    if page_size <= 0:
        raise ValueError(f"KDA recurrent page size must be positive, got {page_size}")
    batch = int(prefix_lengths.numel())
    output = torch.empty(
        (batch, *ssm_cache.shape[1:]),
        dtype=ssm_cache.dtype,
        device=ssm_cache.device,
    )
    heads, key_dim, value_dim = ssm_cache.shape[1:]
    block = 256
    grid = (batch, triton.cdiv(heads * key_dim * value_dim, block))
    _load_recurrent_state_from_block_map_kernel[grid](
        prefix_lengths,
        linear_block_map,
        ssm_cache,
        output,
        linear_block_map.stride(0),
        linear_block_map.stride(1),
        ssm_cache.stride(0),
        ssm_cache.stride(1),
        ssm_cache.stride(2),
        ssm_cache.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        linear_block_map.shape[1],
        ssm_cache.shape[0],
        HEADS=heads,
        K=key_dim,
        V=value_dim,
        PAGE_SIZE=page_size,
        BLOCK=block,
        num_warps=4,
    )
    return output


@torch.compiler.disable
def kimi_kda_store_recurrent_checkpoints(
    checkpoints: torch.Tensor,
    metadata: KimiKDARecurrentCheckpointMetadata,
    linear_block_map: torch.Tensor,
    ssm_cache: torch.Tensor,
) -> None:
    """Scatter explicitly selected cuLA checkpoints into physical SSM blocks."""

    if checkpoints.ndim == 5:
        if checkpoints.shape[0] != 1:
            raise ValueError("packed KDA checkpoints require physical batch 1")
        checkpoints = checkpoints.squeeze(0)
    if (
        checkpoints.ndim != 4
        or checkpoints.shape[0] != metadata.total_checkpoints
        or ssm_cache.ndim != 4
        or tuple(checkpoints.shape[1:]) != tuple(ssm_cache.shape[1:])
        or linear_block_map.ndim != 2
        or linear_block_map.shape[1] == 0
        or metadata.checkpoint_offsets.numel() != linear_block_map.shape[0] + 1
        or metadata.final_checkpoint_indices.numel() != linear_block_map.shape[0]
        or metadata.store_checkpoint_indices.shape
        != metadata.store_sequence_indices.shape
        or metadata.store_checkpoint_indices.shape
        != metadata.store_page_indices.shape
    ):
        raise ValueError("invalid packed KDA recurrent checkpoint shapes")
    tensors = (
        checkpoints,
        metadata.store_checkpoint_indices,
        metadata.store_sequence_indices,
        metadata.store_page_indices,
        linear_block_map,
        ssm_cache,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("KDA recurrent checkpoint store requires CUDA tensors")
    if checkpoints.dtype != torch.float32 or ssm_cache.dtype != torch.float32:
        raise ValueError("KDA recurrent checkpoints/cache must be FP32")
    if linear_block_map.dtype not in (torch.int32, torch.int64):
        raise ValueError("KDA recurrent LINEAR block map must be int32/int64")
    if metadata.store_checkpoint_indices.numel() == 0:
        return
    heads, key_dim, value_dim = checkpoints.shape[1:]
    block = 256
    grid = (
        metadata.store_checkpoint_indices.numel(),
        triton.cdiv(heads * key_dim * value_dim, block),
    )
    _store_selected_recurrent_checkpoints_kernel[grid](
        checkpoints,
        metadata.store_checkpoint_indices,
        metadata.store_sequence_indices,
        metadata.store_page_indices,
        linear_block_map,
        ssm_cache,
        checkpoints.stride(0),
        checkpoints.stride(1),
        checkpoints.stride(2),
        checkpoints.stride(3),
        linear_block_map.stride(0),
        linear_block_map.stride(1),
        ssm_cache.stride(0),
        ssm_cache.stride(1),
        ssm_cache.stride(2),
        ssm_cache.stride(3),
        linear_block_map.shape[1],
        ssm_cache.shape[0],
        HEADS=heads,
        K=key_dim,
        V=value_dim,
        BLOCK=block,
        num_warps=4,
    )


__all__ = [
    "KimiKDARecurrentCheckpointMetadata",
    "kimi_kda_load_recurrent_state",
    "kimi_kda_store_recurrent_checkpoints",
    "prepare_kimi_kda_recurrent_checkpoint_metadata",
]
