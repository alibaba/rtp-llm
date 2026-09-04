"""Checkpoint metadata and cache store for fused KDA prefill."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class KDARecurrentCheckpointMetadata:
    checkpoint_offsets: torch.Tensor
    store_checkpoint_indices: torch.Tensor
    store_sequence_indices: torch.Tensor
    store_page_indices: torch.Tensor
    total_checkpoints: int


def prepare_kda_recurrent_checkpoint_metadata(
    input_lengths_host: torch.Tensor,
    prefix_lengths_host: torch.Tensor,
    page_size: int,
    device: torch.device,
) -> KDARecurrentCheckpointMetadata:
    """Map cuLA's packed page checkpoints to request cache pages."""

    if (
        input_lengths_host.device.type != "cpu"
        or prefix_lengths_host.device.type != "cpu"
        or input_lengths_host.ndim != 1
        or input_lengths_host.shape != prefix_lengths_host.shape
        or input_lengths_host.numel() == 0
    ):
        raise ValueError("KDA checkpoint lengths must be matching CPU vectors")
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

    counts = [(length + page_size - 1) // page_size for length in lengths]
    offsets = [0]
    checkpoint_indices = []
    sequence_indices = []
    page_indices = []
    for sequence_idx, (length, prefix, count) in enumerate(
        zip(lengths, prefixes, counts)
    ):
        offsets.append(offsets[-1] + count)
        for checkpoint_idx in range(count):
            local_end = min((checkpoint_idx + 1) * page_size, length)
            checkpoint_indices.append(offsets[-2] + checkpoint_idx)
            sequence_indices.append(sequence_idx)
            page_indices.append((prefix + local_end - 1) // page_size)

    return KDARecurrentCheckpointMetadata(
        checkpoint_offsets=torch.tensor(offsets, dtype=torch.int32, device=device),
        store_checkpoint_indices=torch.tensor(
            checkpoint_indices, dtype=torch.int32, device=device
        ),
        store_sequence_indices=torch.tensor(
            sequence_indices, dtype=torch.int32, device=device
        ),
        store_page_indices=torch.tensor(page_indices, dtype=torch.int32, device=device),
        total_checkpoints=offsets[-1],
    )


@triton.jit(do_not_specialize=["max_block_count", "physical_block_count"])
def _store_kda_checkpoints_kernel(
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
    sequence_index = tl.load(sequence_indices + store_index).to(tl.int64)
    page = tl.load(page_indices + store_index).to(tl.int64)

    state_elements = HEADS * K * V
    valid_element = offsets < state_elements
    value_index = offsets % V
    key_index = (offsets // V) % K
    head_index = offsets // (K * V)
    page_valid = (page >= 0) & (page < max_block_count)
    safe_page = tl.where(page_valid, page, 0)
    block_id = tl.load(
        block_map + sequence_index * stride_bm_b + safe_page * stride_bm_page,
        mask=page_valid,
        other=0,
    ).to(tl.int64)
    mask = (
        page_valid & valid_element & (block_id > 0) & (block_id < physical_block_count)
    )
    value = tl.load(
        checkpoints
        + checkpoint_index * stride_c_n
        + head_index * stride_c_h
        + key_index * stride_c_k
        + value_index * stride_c_v,
        mask=mask,
        other=0,
    )
    tl.store(
        ssm_cache
        + block_id * stride_s_block
        + head_index * stride_s_h
        + key_index * stride_s_k
        + value_index * stride_s_v,
        value,
        mask=mask,
    )


@torch.compiler.disable
def store_kda_recurrent_checkpoints(
    checkpoints: torch.Tensor,
    metadata: KDARecurrentCheckpointMetadata,
    block_map: torch.Tensor,
    ssm_cache: torch.Tensor,
) -> None:
    """Scatter cuLA checkpoints only to materialized physical cache blocks."""

    if checkpoints.ndim == 5:
        if checkpoints.shape[0] != 1:
            raise ValueError("packed KDA checkpoints require physical batch 1")
        checkpoints = checkpoints.squeeze(0)
    if (
        checkpoints.ndim != 4
        or checkpoints.shape[0] != metadata.total_checkpoints
        or tuple(checkpoints.shape[1:]) != tuple(ssm_cache.shape[1:])
        or block_map.ndim != 2
        or metadata.checkpoint_offsets.numel() != block_map.shape[0] + 1
    ):
        raise ValueError("invalid packed KDA recurrent checkpoint shapes")
    if checkpoints.dtype != torch.float32 or ssm_cache.dtype != torch.float32:
        raise ValueError("KDA recurrent checkpoints/cache must be FP32")
    if metadata.store_checkpoint_indices.numel() == 0:
        return

    heads, key_dim, value_dim = checkpoints.shape[1:]
    block = 256
    _store_kda_checkpoints_kernel[
        (
            metadata.store_checkpoint_indices.numel(),
            triton.cdiv(heads * key_dim * value_dim, block),
        )
    ](
        checkpoints,
        metadata.store_checkpoint_indices,
        metadata.store_sequence_indices,
        metadata.store_page_indices,
        block_map,
        ssm_cache,
        checkpoints.stride(0),
        checkpoints.stride(1),
        checkpoints.stride(2),
        checkpoints.stride(3),
        block_map.stride(0),
        block_map.stride(1),
        ssm_cache.stride(0),
        ssm_cache.stride(1),
        ssm_cache.stride(2),
        ssm_cache.stride(3),
        block_map.shape[1],
        ssm_cache.shape[0],
        HEADS=heads,
        K=key_dim,
        V=value_dim,
        BLOCK=block,
        num_warps=4,
    )


__all__ = [
    "KDARecurrentCheckpointMetadata",
    "prepare_kda_recurrent_checkpoint_metadata",
    "store_kda_recurrent_checkpoints",
]
