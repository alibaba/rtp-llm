"""Paged sparse GQA attention for Qwen4 decode and context-style prefill.

The QSA indexer returns request-local absolute token indices.  This kernel maps
each selected token through the main MHA cache block table and performs GQA
attention without materialising gathered K/V tensors.  It is intentionally
Qwen4-specific: BF16 HND cache layout, token-level selections, and block id zero
reserved as the allocator sentinel.

The same B/S-shaped primitive serves ordinary decode, target verification, and
the correctness-first MTP draft incremental-prefill bridge (which invokes it
once per ragged request with ``B=1``).  Model-level routing remains explicitly
gated until indexer side-pool scoring/writes and prefix semantics are connected.
"""

import torch
import triton
import triton.language as tl

_BLOCK_K = 64


@triton.jit
def _sparse_paged_gqa_kernel(
    q_ptr,
    cache_ptr,
    block_table_ptr,
    kv_lens_ptr,
    selected_ptr,
    out_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_cache_page,
    stride_cache_kv,
    stride_cache_h,
    stride_cache_token,
    stride_bt_b,
    stride_bt_block,
    stride_len_b,
    stride_len_s,
    stride_sel_b,
    stride_sel_s,
    stride_out_b,
    stride_out_h,
    stride_out_s,
    selected_width,
    max_blocks,
    cache_blocks,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_SELECTED_BLOCKS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    query_idx = tl.program_id(2)
    kv_head = q_head // (H_Q // H_KV)
    rd = tl.arange(0, D)

    q = tl.load(
        q_ptr
        + batch_idx * stride_q_b
        + q_head * stride_q_h
        + query_idx * stride_q_s
        + rd
    ).to(tl.float32)
    row_base = selected_ptr + batch_idx * stride_sel_b + query_idx * stride_sel_s
    kv_len = tl.load(kv_lens_ptr + batch_idx * stride_len_b + query_idx * stride_len_s)

    max_score = tl.full((1,), float("-inf"), dtype=tl.float32)
    for selected_block in tl.static_range(MAX_SELECTED_BLOCKS):
        selected_offset = selected_block * BLOCK_K + tl.arange(0, BLOCK_K)
        in_width = selected_offset < selected_width
        token_idx = tl.load(row_base + selected_offset, mask=in_width, other=-1)
        logical_block = token_idx // PAGE_SIZE
        in_table = logical_block < max_blocks
        physical_block = tl.load(
            block_table_ptr + batch_idx * stride_bt_b + logical_block * stride_bt_block,
            mask=in_width & (token_idx >= 0) & (token_idx < kv_len) & in_table,
            other=0,
        )
        valid = (
            in_width
            & (token_idx >= 0)
            & (token_idx < kv_len)
            & in_table
            & (physical_block > 0)
            & (physical_block < cache_blocks)
        )
        token_offset = token_idx % PAGE_SIZE
        k_ptr = (
            cache_ptr
            + physical_block[:, None] * stride_cache_page
            + kv_head * stride_cache_h
            + token_offset[:, None] * stride_cache_token
            + rd[None, :]
        )
        k = tl.load(k_ptr, mask=valid[:, None], other=0.0).to(tl.float32)
        score = tl.sum(q[None, :] * k, axis=1) * SCALE
        score = tl.where(valid, score, float("-inf"))
        max_score = tl.maximum(max_score, tl.max(score, axis=0))

    normalizer = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((D,), dtype=tl.float32)
    for selected_block in tl.static_range(MAX_SELECTED_BLOCKS):
        selected_offset = selected_block * BLOCK_K + tl.arange(0, BLOCK_K)
        in_width = selected_offset < selected_width
        token_idx = tl.load(row_base + selected_offset, mask=in_width, other=-1)
        logical_block = token_idx // PAGE_SIZE
        in_table = logical_block < max_blocks
        physical_block = tl.load(
            block_table_ptr + batch_idx * stride_bt_b + logical_block * stride_bt_block,
            mask=in_width & (token_idx >= 0) & (token_idx < kv_len) & in_table,
            other=0,
        )
        valid = (
            in_width
            & (token_idx >= 0)
            & (token_idx < kv_len)
            & in_table
            & (physical_block > 0)
            & (physical_block < cache_blocks)
        )
        token_offset = token_idx % PAGE_SIZE
        k_ptr = (
            cache_ptr
            + physical_block[:, None] * stride_cache_page
            + kv_head * stride_cache_h
            + token_offset[:, None] * stride_cache_token
            + rd[None, :]
        )
        v_ptr = k_ptr + stride_cache_kv
        k = tl.load(k_ptr, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr, mask=valid[:, None], other=0.0).to(tl.float32)
        score = tl.sum(q[None, :] * k, axis=1) * SCALE
        score = tl.where(valid, score, float("-inf"))
        safe_delta = tl.where(valid, score - max_score, float("-inf"))
        weight = tl.exp(safe_delta)
        normalizer += tl.sum(weight)
        acc += tl.sum(weight[:, None] * v, axis=0)

    output = tl.where(normalizer > 0, acc / normalizer, 0.0)
    tl.store(
        out_ptr
        + batch_idx * stride_out_b
        + q_head * stride_out_h
        + query_idx * stride_out_s
        + rd,
        output.to(q_ptr.dtype.element_ty),
    )


def sparse_paged_gqa_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    kv_head_num: int | None = None,
    block_k: int = _BLOCK_K,
) -> torch.Tensor:
    """Attend to request-local token indices in a paged main MHA cache.

    Args:
        q: ``[B, H_q, S, D]`` BF16 CUDA queries after RoPE.
        kv_cache: HND cache ``[pages, 2, H_kv, page_size, D]`` or its packed
            two-dimensional HybridPool view.
        block_table: ``[B, max_blocks]`` int32 CUDA logical-to-physical map.
        kv_lens: ``[B, S]`` int32 CUDA visible main-KV lengths.
        selected: ``[B, S, K]`` int32 CUDA request-local token indices; ``-1``
            is padding.
        page_size: tokens per main MHA cache page.
        kv_head_num: local KV head count. Required for a packed HybridPool view,
            whose row stride may be padded by a larger cache group and therefore
            cannot be inferred from ``kv_cache.shape[1]``.
    """
    if q.dim() != 4 or q.dtype != torch.bfloat16 or not q.is_cuda:
        raise ValueError("q must be a rank-4 BF16 CUDA tensor")
    batch, q_heads, query_len, head_dim = q.shape
    if q_heads <= 0 or head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError("query heads must be positive and head_dim a power of two")
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError(f"page_size must be a positive power of two, got {page_size}")
    if block_k <= 0 or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a positive power of two, got {block_k}")
    if (
        block_table.dim() != 2
        or tuple(block_table.shape[:1]) != (batch,)
        or block_table.dtype != torch.int32
    ):
        raise ValueError(f"block_table must be int32 [{batch}, max_blocks]")
    if kv_lens.shape != (batch, query_len) or kv_lens.dtype != torch.int32:
        raise ValueError(f"kv_lens must be int32 [{batch}, {query_len}]")
    if selected.dim() != 3 or selected.shape[:2] != (batch, query_len):
        raise ValueError(f"selected must have leading shape [{batch}, {query_len}]")
    if selected.dtype != torch.int32:
        raise ValueError("selected must be int32")

    if kv_cache.dim() == 2:
        if kv_head_num is None or kv_head_num <= 0:
            raise ValueError("kv_head_num is required for a packed HybridPool view")
        required_width = 2 * kv_head_num * page_size * head_dim
        if int(kv_cache.shape[1]) < required_width:
            raise ValueError(
                f"packed kv_cache width {kv_cache.shape[1]} is smaller than "
                f"the required main-cache width {required_width}"
            )
        if kv_cache.stride(1) != 1:
            raise ValueError("packed HybridPool rows must have contiguous elements")
        elem_stride = kv_cache.stride(1)
        kv_cache = torch.as_strided(
            kv_cache,
            (int(kv_cache.shape[0]), 2, kv_head_num, page_size, head_dim),
            (
                kv_cache.stride(0),
                kv_head_num * page_size * head_dim * elem_stride,
                page_size * head_dim * elem_stride,
                head_dim * elem_stride,
                elem_stride,
            ),
        )
    if kv_cache.dim() != 5 or int(kv_cache.shape[1]) != 2:
        raise ValueError("kv_cache must be [pages, 2, H_kv, page_size, D]")
    cache_blocks, _, kv_heads, cache_page_size, cache_head_dim = kv_cache.shape
    if kv_head_num is not None and kv_heads != kv_head_num:
        raise ValueError(
            f"kv_cache has {kv_heads} KV heads, expected kv_head_num={kv_head_num}"
        )
    if cache_page_size != page_size or cache_head_dim != head_dim:
        raise ValueError(
            "kv_cache page/head geometry does not match q: "
            f"cache={tuple(kv_cache.shape)}, page_size={page_size}, D={head_dim}"
        )
    if kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError(
            f"query heads {q_heads} must be divisible by KV heads {kv_heads}"
        )
    tensors = (kv_cache, block_table, kv_lens, selected)
    if any(not tensor.is_cuda or tensor.device != q.device for tensor in tensors):
        raise ValueError(
            "q, kv_cache, block_table, kv_lens, and selected must share CUDA"
        )
    if kv_cache.dtype != torch.bfloat16:
        raise ValueError("qwen4 sparse paged GQA requires a BF16 main KV cache")
    expected_inner_strides = (
        kv_heads * page_size * head_dim,
        page_size * head_dim,
        head_dim,
        1,
    )
    if tuple(kv_cache.stride()[1:]) != expected_inner_strides:
        raise ValueError(
            "kv_cache must use HND inner layout; got strides "
            f"{tuple(kv_cache.stride())}"
        )
    if int(block_table.shape[1]) == 0:
        raise ValueError("block_table must contain at least one logical block")
    if bool(torch.any(kv_lens < 0).item()) or bool(
        torch.any(kv_lens > int(block_table.shape[1]) * page_size).item()
    ):
        raise ValueError("kv_lens exceed the block-table token capacity")
    if selected.numel():
        if int(selected.min().item()) < -1:
            raise ValueError("selected indices may only use -1 as padding")
        invalid_visible = (selected >= 0) & (selected >= kv_lens.unsqueeze(-1))
        if bool(invalid_visible.any().item()):
            raise ValueError("selected contains an index outside its row's kv_len")
        logical_blocks = torch.clamp_min(selected, 0) // page_size
        if int(logical_blocks.max().item()) >= int(block_table.shape[1]):
            raise ValueError("selected index exceeds the block-table width")
        physical = torch.gather(
            block_table.unsqueeze(1).expand(batch, query_len, -1),
            2,
            logical_blocks.to(torch.long),
        )
        if bool(((selected >= 0) & (physical <= 0)).any().item()):
            raise ValueError("selected index resolves to an unallocated cache block")
    max_physical = int(block_table.max().item()) if block_table.numel() else 0
    if max_physical >= cache_blocks:
        raise ValueError(
            f"block table physical id {max_physical} exceeds cache blocks {cache_blocks}"
        )

    q = q.contiguous()
    block_table = block_table.contiguous()
    kv_lens = kv_lens.contiguous()
    selected = selected.contiguous()
    output = torch.empty_like(q)
    if batch == 0 or query_len == 0:
        return output

    selected_width = int(selected.shape[2])
    if selected_width == 0:
        output.zero_()
        return output
    grid = (batch, q_heads, query_len)
    _sparse_paged_gqa_kernel[grid](
        q,
        kv_cache,
        block_table,
        kv_lens,
        selected,
        output,
        *q.stride()[:3],
        *kv_cache.stride()[:4],
        *block_table.stride(),
        *kv_lens.stride(),
        *selected.stride()[:2],
        *output.stride()[:3],
        selected_width,
        int(block_table.shape[1]),
        cache_blocks,
        H_Q=q_heads,
        H_KV=kv_heads,
        PAGE_SIZE=page_size,
        D=head_dim,
        SCALE=1.0 / float(head_dim) ** 0.5,
        BLOCK_K=block_k,
        MAX_SELECTED_BLOCKS=triton.cdiv(selected_width, block_k),
    )
    return output
