"""Fused CUDA builders for DSV4 context-parallel metadata."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_MAX_BATCH = 64
_BLOCK_SIZE = 256
_FORWARD_B64_LOOP_UNROLL = 2


if _TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["batch_size"])
    def _cp_forward_metadata_step(
        lengths_ptr,
        chunk_lengths_ptr,
        prefixes_ptr,
        offsets,
        local_mask,
        unpad_mask,
        relative,
        batch_size,
        req_idx,
        cp_size,
        local_start,
        padded_start,
        real_start,
        padded_position,
        local_position,
        global_position,
        request_id,
        restore_position,
        cu_value,
    ):
        req_valid = req_idx < batch_size
        req_len = tl.load(lengths_ptr + req_idx, mask=req_valid, other=0).to(tl.int64)
        req_chunk = tl.load(chunk_lengths_ptr + req_idx, mask=req_valid, other=0).to(
            tl.int64
        )
        prefix = tl.load(prefixes_ptr + req_idx, mask=req_valid, other=0).to(tl.int64)
        padded_len = req_chunk * cp_size

        is_local_request = (
            local_mask
            & req_valid
            & (offsets >= local_start)
            & (offsets < local_start + req_chunk)
        )
        max_real_position = tl.maximum(req_len - 1, 0)
        clamped_relative = tl.minimum(relative, max_real_position)
        padded_position = tl.where(
            is_local_request, padded_start + relative, padded_position
        )
        local_position = tl.where(is_local_request, clamped_relative, local_position)
        global_position = tl.where(
            is_local_request,
            prefix + clamped_relative,
            global_position,
        )
        request_id = tl.where(is_local_request, req_idx, request_id)

        is_unpad_request = (
            unpad_mask
            & req_valid
            & (offsets >= real_start)
            & (offsets < real_start + req_len)
        )
        restore_position = tl.where(
            is_unpad_request,
            padded_start + offsets - real_start,
            restore_position,
        )
        cu_value += tl.where(req_valid & (req_idx < offsets), req_len, 0)
        return (
            local_start + req_chunk,
            padded_start + padded_len,
            real_start + req_len,
            padded_position,
            local_position,
            global_position,
            request_id,
            restore_position,
            cu_value,
        )

    @triton.jit(
        do_not_specialize=[
            "total_tokens",
            "total_local_kv",
            "cp_size",
            "owner_block_size",
        ]
    )
    def _cp_restore_b1_kernel(
        out_ptr,
        total_tokens,
        total_local_kv,
        cp_size,
        owner_block_size,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(
            tl.int64
        )
        mask = offsets < total_tokens
        block_idx = offsets // owner_block_size
        owner = block_idx % cp_size
        local_block_idx = block_idx // cp_size
        local_pos = local_block_idx * owner_block_size + offsets % owner_block_size
        restore = owner * total_local_kv + local_pos
        tl.store(out_ptr + offsets, restore, mask=mask)

    @triton.jit(
        do_not_specialize=[
            "total_tokens",
            "total_local_kv",
            "cp_size",
            "owner_block_size",
            "batch_size",
        ]
    )
    def _cp_restore_varlen_kernel(
        lengths_ptr,
        out_ptr,
        total_tokens,
        total_local_kv,
        cp_size,
        owner_block_size,
        batch_size,
        BATCH_BLOCK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(
            tl.int64
        )
        mask = offsets < total_tokens
        global_start = tl.zeros((BLOCK,), dtype=tl.int64)
        local_start = tl.zeros((BLOCK,), dtype=tl.int64)
        position = tl.zeros((BLOCK,), dtype=tl.int64)
        request_local_start = tl.zeros((BLOCK,), dtype=tl.int64)
        virtual_block_size = cp_size * owner_block_size

        for req_id in tl.static_range(0, BATCH_BLOCK):
            req_valid = req_id < batch_size
            req_len = tl.load(lengths_ptr + req_id, mask=req_valid, other=0).to(
                tl.int64
            )
            is_request = (
                mask
                & req_valid
                & (offsets >= global_start)
                & (offsets < global_start + req_len)
            )
            position = tl.where(is_request, offsets - global_start, position)
            request_local_start = tl.where(is_request, local_start, request_local_start)
            padded_local_len = (
                (req_len + virtual_block_size - 1) // virtual_block_size
            ) * owner_block_size
            global_start += req_len
            local_start += padded_local_len

        block_idx = position // owner_block_size
        owner = block_idx % cp_size
        local_block_idx = block_idx // cp_size
        local_pos = local_block_idx * owner_block_size + position % owner_block_size
        restore = owner * total_local_kv + request_local_start + local_pos
        tl.store(out_ptr + offsets, restore, mask=mask)

    @triton.jit(do_not_specialize=["total_tokens"])
    def _cp_positions_b1_kernel(
        prefixes_ptr,
        positions_ptr,
        request_ids_ptr,
        total_tokens,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(
            tl.int64
        )
        mask = offsets < total_tokens
        prefix = tl.load(prefixes_ptr).to(tl.int64)
        tl.store(positions_ptr + offsets, prefix + offsets, mask=mask)
        tl.store(request_ids_ptr + offsets, 0, mask=mask)

    @triton.jit(do_not_specialize=["total_tokens", "batch_size"])
    def _cp_positions_varlen_kernel(
        lengths_ptr,
        prefixes_ptr,
        positions_ptr,
        request_ids_ptr,
        total_tokens,
        batch_size,
        BATCH_BLOCK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(
            tl.int64
        )
        mask = offsets < total_tokens
        global_start = tl.zeros((BLOCK,), dtype=tl.int64)
        position = tl.zeros((BLOCK,), dtype=tl.int64)
        request_id = tl.zeros((BLOCK,), dtype=tl.int64)

        for req_idx in tl.static_range(0, BATCH_BLOCK):
            req_valid = req_idx < batch_size
            req_len = tl.load(lengths_ptr + req_idx, mask=req_valid, other=0).to(
                tl.int64
            )
            prefix = tl.load(prefixes_ptr + req_idx, mask=req_valid, other=0).to(
                tl.int64
            )
            is_request = (
                mask
                & req_valid
                & (offsets >= global_start)
                & (offsets < global_start + req_len)
            )
            position = tl.where(is_request, prefix + offsets - global_start, position)
            request_id = tl.where(is_request, req_idx, request_id)
            global_start += req_len

        tl.store(positions_ptr + offsets, position, mask=mask)
        tl.store(request_ids_ptr + offsets, request_id, mask=mask)

    @triton.jit(
        do_not_specialize=[
            "chunk_length",
            "seq_len_full",
            "cp_size",
            "batch_size",
        ]
    )
    def _cp_forward_metadata_kernel(
        lengths_ptr,
        chunk_lengths_ptr,
        prefixes_ptr,
        padding_mask_ptr,
        restore_indices_ptr,
        shuffle_indices_ptr,
        relative_positions_ptr,
        global_positions_ptr,
        request_ids_ptr,
        local_is_real_ptr,
        unpad_restore_ptr,
        cu_seqlens_ptr,
        prefixes_out_ptr,
        chunk_length,
        seq_len_full,
        cp_size,
        batch_size,
        BATCH_BLOCK: tl.constexpr,
        B64_LOOP_UNROLL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(
            tl.int64
        )
        local_mask = offsets < chunk_length
        unpad_mask = offsets < seq_len_full

        local_start = tl.zeros((BLOCK,), dtype=tl.int64)
        padded_start = tl.zeros((BLOCK,), dtype=tl.int64)
        real_start = tl.zeros((BLOCK,), dtype=tl.int64)
        padded_position = tl.zeros((BLOCK,), dtype=tl.int64)
        local_position = tl.zeros((BLOCK,), dtype=tl.int64)
        global_position = tl.zeros((BLOCK,), dtype=tl.int64)
        request_id = tl.zeros((BLOCK,), dtype=tl.int32)
        restore_position = tl.zeros((BLOCK,), dtype=tl.int64)
        cu_value = tl.zeros((BLOCK,), dtype=tl.int64)
        relative = tl.load(shuffle_indices_ptr + offsets, mask=local_mask, other=0).to(
            tl.int64
        )

        if BATCH_BLOCK == 64:
            for req_idx in tl.range(
                0,
                batch_size,
                loop_unroll_factor=B64_LOOP_UNROLL,
            ):
                (
                    local_start,
                    padded_start,
                    real_start,
                    padded_position,
                    local_position,
                    global_position,
                    request_id,
                    restore_position,
                    cu_value,
                ) = _cp_forward_metadata_step(
                    lengths_ptr,
                    chunk_lengths_ptr,
                    prefixes_ptr,
                    offsets,
                    local_mask,
                    unpad_mask,
                    relative,
                    batch_size,
                    req_idx,
                    cp_size,
                    local_start,
                    padded_start,
                    real_start,
                    padded_position,
                    local_position,
                    global_position,
                    request_id,
                    restore_position,
                    cu_value,
                )
        else:
            for req_idx in tl.static_range(0, BATCH_BLOCK):
                (
                    local_start,
                    padded_start,
                    real_start,
                    padded_position,
                    local_position,
                    global_position,
                    request_id,
                    restore_position,
                    cu_value,
                ) = _cp_forward_metadata_step(
                    lengths_ptr,
                    chunk_lengths_ptr,
                    prefixes_ptr,
                    offsets,
                    local_mask,
                    unpad_mask,
                    relative,
                    batch_size,
                    req_idx,
                    cp_size,
                    local_start,
                    padded_start,
                    real_start,
                    padded_position,
                    local_position,
                    global_position,
                    request_id,
                    restore_position,
                    cu_value,
                )

        is_real = tl.load(padding_mask_ptr + padded_position, mask=local_mask, other=0)
        restore = tl.load(
            restore_indices_ptr + restore_position,
            mask=unpad_mask,
            other=0,
        ).to(tl.int64)
        tl.store(
            relative_positions_ptr + offsets,
            padded_position,
            mask=local_mask,
        )
        tl.store(
            global_positions_ptr + offsets,
            global_position,
            mask=local_mask,
        )
        tl.store(request_ids_ptr + offsets, request_id, mask=local_mask)
        tl.store(local_is_real_ptr + offsets, is_real != 0, mask=local_mask)
        tl.store(unpad_restore_ptr + offsets, restore, mask=unpad_mask)
        tl.store(
            cu_seqlens_ptr + offsets,
            cu_value.to(tl.int32),
            mask=offsets < batch_size + 1,
        )
        prefix_mask = offsets < batch_size
        prefix_value = tl.load(prefixes_ptr + offsets, mask=prefix_mask, other=0).to(
            tl.int64
        )
        tl.store(prefixes_out_ptr + offsets, prefix_value, mask=prefix_mask)


def cp_metadata_fusion_supported() -> bool:
    return _TRITON_AVAILABLE


def _batch_block(batch_size: int) -> int:
    """Return the bounded Triton specialization used by a runtime batch."""
    return int(triton.next_power_of_2(int(batch_size)))


def _supported(lengths: torch.Tensor) -> bool:
    return (
        cp_metadata_fusion_supported()
        and lengths.is_cuda
        and lengths.dim() == 1
        and 0 < int(lengths.numel()) <= _MAX_BATCH
    )


def try_build_cp_restore_indices(
    lengths: torch.Tensor,
    *,
    cp_size: int,
    owner_block_size: int,
    total_tokens: int,
    total_local_kv: int,
) -> Optional[torch.Tensor]:
    """Return a one-launch restore tensor, or ``None`` for the torch fallback."""
    if not _supported(lengths):
        return None
    total_tokens = int(total_tokens)
    if total_tokens == 0:
        return torch.empty(0, dtype=torch.int64, device=lengths.device)
    out = torch.empty(total_tokens, dtype=torch.int64, device=lengths.device)
    grid = (triton.cdiv(total_tokens, _BLOCK_SIZE),)
    batch_size = int(lengths.numel())
    if batch_size == 1:
        _cp_restore_b1_kernel[grid](
            out,
            total_tokens,
            int(total_local_kv),
            int(cp_size),
            int(owner_block_size),
            BLOCK=_BLOCK_SIZE,
            num_warps=4,
        )
    else:
        _cp_restore_varlen_kernel[grid](
            lengths,
            out,
            total_tokens,
            int(total_local_kv),
            int(cp_size),
            int(owner_block_size),
            batch_size,
            BATCH_BLOCK=_batch_block(batch_size),
            BLOCK=_BLOCK_SIZE,
            num_warps=4,
        )
    return out


def try_build_cp_full_prefill_positions(
    lengths: torch.Tensor,
    prefixes: torch.Tensor,
    *,
    total_tokens: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return fused ``(positions, request_ids)``, or ``None`` as fallback."""
    if (
        not _supported(lengths)
        or not prefixes.is_cuda
        or prefixes.dim() != 1
        or prefixes.numel() != lengths.numel()
    ):
        return None
    total_tokens = int(total_tokens)
    positions = torch.empty(total_tokens, dtype=torch.int64, device=lengths.device)
    request_ids = torch.empty_like(positions)
    if total_tokens == 0:
        return positions, request_ids
    grid = (triton.cdiv(total_tokens, _BLOCK_SIZE),)
    batch_size = int(lengths.numel())
    if batch_size == 1:
        _cp_positions_b1_kernel[grid](
            prefixes,
            positions,
            request_ids,
            total_tokens,
            BLOCK=_BLOCK_SIZE,
            num_warps=4,
        )
    else:
        _cp_positions_varlen_kernel[grid](
            lengths,
            prefixes,
            positions,
            request_ids,
            total_tokens,
            batch_size,
            BATCH_BLOCK=_batch_block(batch_size),
            BLOCK=_BLOCK_SIZE,
            num_warps=4,
        )
    return positions, request_ids


def try_build_cp_forward_metadata(
    lengths: torch.Tensor,
    chunk_lengths: torch.Tensor,
    prefixes: torch.Tensor,
    padding_mask: torch.Tensor,
    restore_indices: torch.Tensor,
    shuffle_indices: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
    chunk_length: int,
    seq_len_full: int,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """Build all pre-embedding CP metadata with one CUDA launch."""
    tensors = (
        lengths,
        chunk_lengths,
        prefixes,
        padding_mask,
        restore_indices,
        shuffle_indices,
    )
    batch_size = int(lengths.numel())
    if (
        not cp_metadata_fusion_supported()
        or batch_size <= 0
        or batch_size > _MAX_BATCH
        or any(not tensor.is_cuda or tensor.dim() != 1 for tensor in tensors)
        or any(tensor.device != lengths.device for tensor in tensors[1:])
        or int(chunk_lengths.numel()) != batch_size
        or int(prefixes.numel()) < batch_size
        or int(shuffle_indices.numel()) != int(chunk_length)
        or int(padding_mask.numel()) != int(cp_size) * int(chunk_length)
        or int(restore_indices.numel()) != int(padding_mask.numel())
        or int(seq_len_full) > int(padding_mask.numel())
        or int(cp_size) <= 1
        or int(cp_rank) < 0
        or int(cp_rank) >= int(cp_size)
    ):
        return None

    chunk_length = int(chunk_length)
    seq_len_full = int(seq_len_full)
    if chunk_length < 0 or seq_len_full < 0:
        return None
    relative_positions = torch.empty(
        chunk_length, dtype=torch.int64, device=lengths.device
    )
    global_positions = torch.empty_like(relative_positions)
    request_ids = torch.empty(chunk_length, dtype=torch.int32, device=lengths.device)
    local_is_real = torch.empty(chunk_length, dtype=torch.bool, device=lengths.device)
    unpad_restore = torch.empty(seq_len_full, dtype=torch.int64, device=lengths.device)
    cu_seqlens = torch.empty(batch_size + 1, dtype=torch.int32, device=lengths.device)
    prefixes_out = torch.empty(batch_size, dtype=torch.int64, device=lengths.device)
    total_rows = max(chunk_length, seq_len_full, batch_size + 1)
    if total_rows == 0:
        return (
            relative_positions,
            global_positions,
            request_ids,
            local_is_real,
            unpad_restore,
            cu_seqlens,
            prefixes_out,
        )
    batch_block = _batch_block(batch_size)
    grid = (triton.cdiv(total_rows, _BLOCK_SIZE),)
    _cp_forward_metadata_kernel[grid](
        lengths,
        chunk_lengths,
        prefixes,
        padding_mask,
        restore_indices,
        shuffle_indices,
        relative_positions,
        global_positions,
        request_ids,
        local_is_real,
        unpad_restore,
        cu_seqlens,
        prefixes_out,
        chunk_length,
        seq_len_full,
        int(cp_size),
        batch_size,
        BATCH_BLOCK=batch_block,
        B64_LOOP_UNROLL=_FORWARD_B64_LOOP_UNROLL,
        BLOCK=_BLOCK_SIZE,
        num_warps=4,
    )
    return (
        relative_positions,
        global_positions,
        request_ids,
        local_is_real,
        unpad_restore,
        cu_seqlens,
        prefixes_out,
    )
