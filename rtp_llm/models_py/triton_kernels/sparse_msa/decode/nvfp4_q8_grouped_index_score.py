"""Grouped Q8KV4 target-verify IndexScore over the existing RTP cache ABI."""

import torch
import triton
import triton.language as tl

from .nvfp4_q8_index_score import _load_index_page_fp8


@triton.jit
def _grouped_index_score_kernel(
    Q,
    K,
    S,
    TABLE,
    LENS,
    OUT,
    NP,
    MB: tl.constexpr,
    KS: tl.constexpr,
    SS: tl.constexpr,
    B: tl.constexpr,
    W: tl.constexpr,
    CHUNKS: tl.constexpr,
    PAD: tl.constexpr,
    MMA: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    INIT: tl.constexpr,
    LOCAL: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    req = tl.program_id(0)
    chunk = tl.program_id(1)
    rows = tl.arange(0, 64)
    tok = rows // 4
    head = rows % 4
    valid = tok < W
    lens = tl.load(LENS + req * W + tok, mask=valid, other=0)
    lens = tl.minimum(lens, MB * 128)
    nblocks = (lens + 127) // 128
    max_blocks = tl.max(nblocks, 0)
    blocks_per = (max_blocks + CHUNKS - 1) // CHUNKS
    first = chunk * blocks_per
    iters = tl.minimum(blocks_per, max_blocks - first)
    pad = chunk * PAD + tl.arange(0, PAD)
    output_row = head * B * W * MB + (req * W + tok) * MB
    tl.store(
        OUT + output_row[:, None] + pad[None, :],
        float("-inf"),
        mask=valid[:, None] & (pad[None, :] < MB) & (pad[None, :] >= nblocks[:, None]),
    )
    if iters <= 0:
        return
    d = tl.arange(0, 128)
    query = tl.load(
        Q + (req * W + tok[:, None]) * 512 + head[:, None] * 128 + d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    key_tok = tl.arange(0, 128)
    for off in tl.range(iters, num_stages=1):
        block = first + off
        page = tl.load(TABLE + req * TABLE_STRIDE + block).to(tl.int64)
        page_ok = (page >= 0) & (page < NP)
        key = _load_index_page_fp8(
            K, S, tl.where(page_ok, page, 0), page_ok, KS, SS, MMA
        )
        dot = tl.dot(query, tl.trans(key), out_dtype=tl.float32)
        dot = tl.where(
            block * 128 + key_tok[None, :] < lens[:, None], dot, float("-inf")
        )
        score = tl.max(dot, 1) * SM_SCALE
        visible = page_ok & (block < nblocks) & valid
        local = tl.maximum(0, nblocks - LOCAL)
        score = tl.where(visible, score, float("-inf"))
        score = tl.where(visible & (block >= local), 1.0e29, score)
        score = tl.where(visible & (block < INIT) & (block < local), 1.0e30, score)
        # Padded elements are owned exclusively by the padding-store grid;
        # shorter queries must not race its stores while another query scans on.
        tl.store(OUT + output_row + block, score, mask=valid & (block < nblocks))


@triton.jit
def _ragged_grouped_index_score_kernel(
    Q,
    K,
    S,
    TABLE,
    LENS,
    CU_SEQLENS,
    OUT,
    NP,
    MB: tl.constexpr,
    KS: tl.constexpr,
    SS: tl.constexpr,
    TOKENS: tl.constexpr,
    MAX_W: tl.constexpr,
    CHUNKS: tl.constexpr,
    PAD: tl.constexpr,
    MMA: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    INIT: tl.constexpr,
    LOCAL: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    """Ragged counterpart: request offsets replace a uniform query width."""
    req = tl.program_id(0)
    chunk = tl.program_id(1)
    start = tl.load(CU_SEQLENS + req)
    stop = tl.load(CU_SEQLENS + req + 1)
    live = tl.maximum(0, tl.minimum(stop - start, MAX_W))
    rows = tl.arange(0, 64)
    tok = rows // 4
    head = rows % 4
    row = start + tok
    valid = (tok < live) & (row >= 0) & (row < TOKENS)
    lens = tl.load(LENS + row, mask=valid, other=0)
    lens = tl.minimum(lens, MB * 128)
    nblocks = (lens + 127) // 128
    max_blocks = tl.max(nblocks, 0)
    blocks_per = (max_blocks + CHUNKS - 1) // CHUNKS
    first = chunk * blocks_per
    iters = tl.minimum(blocks_per, max_blocks - first)
    pad = chunk * PAD + tl.arange(0, PAD)
    output_row = head * TOKENS * MB + row * MB
    tl.store(
        OUT + output_row[:, None] + pad[None, :],
        float("-inf"),
        mask=valid[:, None] & (pad[None, :] < MB) & (pad[None, :] >= nblocks[:, None]),
    )
    if iters <= 0:
        return
    d = tl.arange(0, 128)
    query = tl.load(
        Q + row[:, None] * 512 + head[:, None] * 128 + d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    key_tok = tl.arange(0, 128)
    for off in tl.range(iters, num_stages=1):
        block = first + off
        # Address preparation expands the request page table into every compact
        # token row. Read the first live row once for the whole request group.
        page = tl.load(TABLE + start * TABLE_STRIDE + block).to(tl.int64)
        page_ok = (start >= 0) & (start < TOKENS) & (page >= 0) & (page < NP)
        key = _load_index_page_fp8(
            K, S, tl.where(page_ok, page, 0), page_ok, KS, SS, MMA
        )
        dot = tl.dot(query, tl.trans(key), out_dtype=tl.float32)
        dot = tl.where(
            block * 128 + key_tok[None, :] < lens[:, None], dot, float("-inf")
        )
        score = tl.max(dot, 1) * SM_SCALE
        visible = page_ok & (block < nblocks) & valid
        local = tl.maximum(0, nblocks - LOCAL)
        score = tl.where(visible, score, float("-inf"))
        score = tl.where(visible & (block >= local), 1.0e29, score)
        score = tl.where(visible & (block < INIT) & (block < local), 1.0e30, score)
        tl.store(OUT + output_row + block, score, mask=valid & (block < nblocks))


@torch.no_grad()
def q8kv4_grouped_index_score(
    q,
    packed,
    scales,
    block_table,
    seq_lens,
    out,
    *,
    query_width,
    init_blocks,
    local_blocks,
    sm_scale,
    mma_scale_layout=False,
):
    """Score grouped verify rows, sharing each index-K load across queries.

    block_table remains expanded [requests * query_width, max_blocks]. Rows
    within one request must refer to the same physical pages. Causal lengths
    and mandatory local pages remain independently defined for every query.
    """
    if not 2 <= query_width <= 16 or q.shape[0] % query_width:
        raise ValueError(
            "grouped Q8KV4 requires width 2..16 and complete request groups"
        )
    if (
        q.dtype != torch.float8_e4m3fn
        or q.shape[1:] != (4, 128)
        or not q.is_contiguous()
    ):
        raise ValueError("grouped Q must be contiguous E4M3 [tokens,4,128]")
    if (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.stride(1) != 1
        or block_table.shape[0] != q.shape[0]
    ):
        raise ValueError("block_table must be expanded int32 [tokens,blocks]")
    if (
        seq_lens.dtype != torch.int32
        or seq_lens.shape != (q.shape[0],)
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("seq_lens must be contiguous int32 [tokens]")
    if (
        packed.dtype != torch.uint8
        or packed.ndim != 2
        or scales.ndim != 2
        or scales.dtype not in (torch.uint8, torch.float8_e4m3fn)
    ):
        raise ValueError("packed/scales must be two-dimensional byte planes")
    if (
        packed.stride(0) < 8192
        or scales.stride(0) < 1024
        or packed.stride(1) != 1
        or scales.stride(1) != 1
        or packed.shape[0] != scales.shape[0]
    ):
        raise ValueError("invalid packed/scales page strides")
    if packed.data_ptr() % 4 or packed.stride(0) % 4:
        raise ValueError("packed page base and stride must be 4-byte aligned")
    batch, blocks = q.shape[0] // query_width, block_table.shape[1]
    if (
        out.dtype != torch.float32
        or out.shape != (4, q.shape[0], blocks)
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous float32 [4,tokens,blocks]")
    chunks = 128
    _grouped_index_score_kernel[(batch, chunks)](
        q,
        packed,
        scales.view(torch.uint8),
        block_table,
        seq_lens,
        out,
        packed.shape[0],
        blocks,
        packed.stride(0),
        scales.stride(0),
        batch,
        query_width,
        chunks,
        triton.next_power_of_2(triton.cdiv(blocks, chunks)),
        mma_scale_layout,
        block_table.stride(0) * query_width,
        init_blocks,
        local_blocks,
        sm_scale,
        num_warps=4,
    )
    return out


@torch.no_grad()
def q8kv4_ragged_grouped_index_score(
    q,
    packed,
    scales,
    block_table,
    seq_lens,
    out,
    *,
    cu_seqlens,
    max_query_width,
    init_blocks,
    local_blocks,
    sm_scale,
    mma_scale_layout=False,
):
    """Score compact request groups without padding them back to dense width.

    ``cu_seqlens`` partitions the compact token rows by request. Every request
    must contain 1..``max_query_width`` rows and the expanded block-table rows
    within a group must be identical. These invariants are produced by the
    DSpark compact target-addressing kernel and remain device-resident for
    CUDA Graph replay.
    """
    if not 2 <= max_query_width <= 16:
        raise ValueError("ragged grouped Q8KV4 requires max width 2..16")
    if (
        q.dtype != torch.float8_e4m3fn
        or q.shape[1:] != (4, 128)
        or not q.is_contiguous()
    ):
        raise ValueError("ragged grouped Q must be contiguous E4M3 [tokens,4,128]")
    if (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.stride(1) != 1
        or block_table.shape[0] != q.shape[0]
    ):
        raise ValueError("block_table must be expanded int32 [tokens,blocks]")
    if (
        seq_lens.dtype != torch.int32
        or seq_lens.shape != (q.shape[0],)
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("seq_lens must be contiguous int32 [tokens]")
    if (
        cu_seqlens.dtype != torch.int32
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 2
        or not cu_seqlens.is_cuda
        or not cu_seqlens.is_contiguous()
        or cu_seqlens.device != q.device
    ):
        raise ValueError("cu_seqlens must be contiguous CUDA int32 [requests+1]")
    if (
        packed.dtype != torch.uint8
        or packed.ndim != 2
        or scales.ndim != 2
        or scales.dtype not in (torch.uint8, torch.float8_e4m3fn)
    ):
        raise ValueError("packed/scales must be two-dimensional byte planes")
    if (
        packed.stride(0) < 8192
        or scales.stride(0) < 1024
        or packed.stride(1) != 1
        or scales.stride(1) != 1
        or packed.shape[0] != scales.shape[0]
    ):
        raise ValueError("invalid packed/scales page strides")
    if packed.data_ptr() % 4 or packed.stride(0) % 4:
        raise ValueError("packed page base and stride must be 4-byte aligned")
    blocks = block_table.shape[1]
    if (
        out.dtype != torch.float32
        or out.shape != (4, q.shape[0], blocks)
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous float32 [4,tokens,blocks]")
    chunks = 128
    _ragged_grouped_index_score_kernel[(cu_seqlens.numel() - 1, chunks)](
        q,
        packed,
        scales.view(torch.uint8),
        block_table,
        seq_lens,
        cu_seqlens,
        out,
        packed.shape[0],
        blocks,
        packed.stride(0),
        scales.stride(0),
        q.shape[0],
        max_query_width,
        chunks,
        triton.next_power_of_2(triton.cdiv(blocks, chunks)),
        mma_scale_layout,
        block_table.stride(0),
        init_blocks,
        local_blocks,
        sm_scale,
        num_warps=4,
    )
    return out
