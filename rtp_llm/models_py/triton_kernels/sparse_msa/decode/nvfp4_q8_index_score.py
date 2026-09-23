"""Fast MiniMax-M3.1 Q8KV4 IndexScore over RTP paged cache."""

import torch
import triton
import triton.language as tl

from .nvfp4_q8_math import E2M1X4_ASM_HI, E2M1X4_ASM_LO, packed4_to_fp8


@triton.jit
def _load_index_page_fp8(
    packed_ptr,
    scale_ptr,
    page,
    valid_page,
    packed_page_stride,
    scale_page_stride,
    MMA_SCALE_LAYOUT: tl.constexpr,
):
    token = tl.arange(0, 128)
    byte = tl.arange(0, 64)
    group = tl.arange(0, 8)
    packed = tl.load(
        packed_ptr + page * packed_page_stride + token[:, None] * 64 + byte[None, :],
        mask=valid_page,
        other=0,
    )
    if MMA_SCALE_LAYOUT:
        scale_offset = (
            (group[None, :] // 4) * 512
            + (token[:, None] % 32) * 16
            + (token[:, None] // 32) * 4
            + group[None, :] % 4
        )
    else:
        scale_offset = token[:, None] * 8 + group[None, :]
    scales = tl.load(
        scale_ptr + page * scale_page_stride + scale_offset,
        mask=valid_page,
        other=0,
    )
    scales = tl.reshape(tl.broadcast_to(scales[:, :, None], [128, 8, 8]), [128, 64])
    low = tl.reshape(packed4_to_fp8(packed, scales, E2M1X4_ASM_LO), [128, 16, 4])
    high = tl.reshape(packed4_to_fp8(packed, scales, E2M1X4_ASM_HI), [128, 16, 4])
    values = tl.reshape(tl.permute(tl.join(low, high), (0, 1, 3, 2)), [128, 128])
    return values.to(tl.float8e4nv, bitcast=True)


@triton.jit
def _q8kv4_index_score_kernel(
    q_ptr,
    packed_ptr,
    scale_ptr,
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    num_phys_pages,
    max_blocks,
    packed_page_stride,
    scale_page_stride,
    q_stride_batch,
    q_stride_head,
    q_stride_dim,
    table_stride_batch,
    out_stride_head,
    out_stride_batch,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    SM_SCALE: tl.constexpr,
    MMA_SCALE_LAYOUT: tl.constexpr,
    PAD_TILE: tl.constexpr,
):
    batch = tl.program_id(0)
    chunk = tl.program_id(1)
    seq_len = tl.minimum(tl.load(seq_lens_ptr + batch), max_blocks * 128)
    num_blocks = (seq_len + 127) // 128
    num_chunks = tl.num_programs(1)
    blocks_per_chunk = (num_blocks + num_chunks - 1) // num_chunks
    first_block = chunk * blocks_per_chunk
    iterations = tl.minimum(blocks_per_chunk, num_blocks - first_block)

    # CUDA Graph buckets reuse the score tensor. Clear every padded element in
    # this launch so a shorter request cannot inherit a previous replay's tail.
    padded_block = chunk * PAD_TILE + tl.arange(0, PAD_TILE)
    padded_head = tl.arange(0, 4)
    tl.store(
        out_ptr
        + padded_head[:, None] * out_stride_head
        + batch * out_stride_batch
        + padded_block[None, :],
        float("-inf"),
        mask=(padded_block[None, :] >= num_blocks)
        & (padded_block[None, :] < max_blocks),
    )
    if iterations <= 0:
        return

    rows = tl.arange(0, 64)
    head = rows % 4
    token_lane = rows // 4
    dim = tl.arange(0, 128)
    q = tl.load(
        q_ptr
        + batch * q_stride_batch
        + head[:, None] * q_stride_head
        + dim[None, :] * q_stride_dim,
        mask=token_lane[:, None] == 0,
        other=0.0,
    )
    key_token = tl.arange(0, 128)
    local_start = tl.maximum(0, num_blocks - local_blocks)
    for offset in tl.range(iterations, num_stages=1):
        logical_block = first_block + offset
        physical_page = tl.load(
            block_table_ptr + batch * table_stride_batch + logical_block
        ).to(tl.int64)
        valid_page = (physical_page >= 0) & (physical_page < num_phys_pages)
        safe_page = tl.where(valid_page, physical_page, 0)
        key = _load_index_page_fp8(
            packed_ptr,
            scale_ptr,
            safe_page,
            valid_page,
            packed_page_stride,
            scale_page_stride,
            MMA_SCALE_LAYOUT,
        )
        score = tl.dot(q, tl.trans(key), out_dtype=tl.float32)
        score = tl.where(
            (logical_block * 128 + key_token)[None, :] < seq_len,
            score,
            float("-inf"),
        )
        maximum = tl.max(score, axis=1) * SM_SCALE
        maximum = tl.where(valid_page, maximum, float("-inf"))
        maximum = tl.where(valid_page & (logical_block >= local_start), 1.0e29, maximum)
        maximum = tl.where(
            valid_page & (logical_block < init_blocks) & (logical_block < local_start),
            1.0e30,
            maximum,
        )
        tl.store(
            out_ptr + head * out_stride_head + batch * out_stride_batch + logical_block,
            maximum,
            mask=token_lane == 0,
        )


@torch.no_grad()
def q8kv4_index_score(
    q: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    *,
    init_blocks: int,
    local_blocks: int,
    sm_scale: float,
    mma_scale_layout: bool = False,
) -> torch.Tensor:
    """Score packed index-K pages without materializing historical rows."""
    if q.dtype != torch.float8_e4m3fn or tuple(q.shape[1:]) != (4, 128):
        raise ValueError("Q8KV4 IndexScore requires E4M3 Q [batch,4,128]")
    if packed.dtype != torch.uint8 or scales.dtype not in (
        torch.uint8,
        torch.float8_e4m3fn,
    ):
        raise ValueError("packed index-K and E4M3 scales must be byte tensors")
    if packed.ndim != 2 or scales.ndim != 2:
        raise ValueError("packed index-K and scales must be [physical_page,bytes]")
    if packed.stride(0) < 128 * 64 or scales.stride(0) < 128 * 8:
        raise ValueError("index-K page strides are smaller than the Q8KV4 ABI")
    if packed.data_ptr() % 4 or packed.stride(0) % 4:
        raise ValueError("packed index-K page base/stride must be 4-byte aligned")
    if block_table.dtype != torch.int32 or block_table.stride(1) != 1:
        raise ValueError("block_table must be contiguous int32 [batch,max_blocks]")
    if seq_lens.dtype != torch.int32 or seq_lens.stride(0) != 1:
        raise ValueError("seq_lens must be contiguous int32 [batch]")
    expected = (4, q.shape[0], block_table.shape[1])
    if (
        out.dtype != torch.float32
        or tuple(out.shape) != expected
        or not out.is_contiguous()
    ):
        raise ValueError(f"score output must be contiguous float32 {expected}")
    pad_tile = triton.next_power_of_2(triton.cdiv(block_table.shape[1], 256))
    _q8kv4_index_score_kernel[(q.shape[0], 256)](
        q,
        packed,
        scales.view(torch.uint8),
        block_table,
        seq_lens,
        out,
        packed.shape[0],
        block_table.shape[1],
        packed.stride(0),
        scales.stride(0),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        SM_SCALE=sm_scale,
        MMA_SCALE_LAYOUT=bool(mma_scale_layout),
        PAD_TILE=pad_tile,
        num_warps=4,
    )
    return out
