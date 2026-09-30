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

import os

import torch
import triton
import triton.language as tl

_BLOCK_K = 64


def _validate_sparse_paged_indices(
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    cache_blocks: int,
) -> None:
    """Check paged reads with one device-to-host verdict on valid inputs."""
    table_width = int(block_table.shape[1])
    invalid_lengths = (kv_lens < 0) | (kv_lens > table_width * page_size)
    invalid_pool = block_table >= cache_blocks
    if selected.numel():
        invalid_negative = selected < -1
        invalid_visible = (selected >= 0) & (selected >= kv_lens.unsqueeze(-1))
        logical_blocks = torch.clamp_min(selected, 0) // page_size
        invalid_logical = logical_blocks >= table_width
        # An invalid selected index must never reach gather before its error
        # is reported. Clamp only the lookup; invalid_logical retains the
        # original value for the verdict and the diagnostic below.
        safe_logical_blocks = logical_blocks.clamp(max=table_width - 1)
        physical = torch.gather(
            block_table.unsqueeze(1).expand(-1, selected.shape[1], -1),
            2,
            safe_logical_blocks.to(torch.long),
        )
        invalid_missing = (selected >= 0) & (physical <= 0)
        invalid_selection = (
            invalid_negative | invalid_visible | invalid_logical | invalid_missing
        )
        failed = invalid_lengths.any() | invalid_pool.any() | invalid_selection.any()
    else:
        failed = invalid_lengths.any() | invalid_pool.any()
    if not bool(failed.item()):
        return

    # Error paths may synchronize again to preserve the existing specific
    # diagnostics. No main-cache write has happened at this point.
    if bool(invalid_lengths.any().item()):
        raise ValueError("kv_lens exceed the block-table token capacity")
    if selected.numel():
        if bool(invalid_negative.any().item()):
            raise ValueError("selected indices may only use -1 as padding")
        if bool(invalid_visible.any().item()):
            raise ValueError("selected contains an index outside its row's kv_len")
        if bool(invalid_logical.any().item()):
            raise ValueError("selected index exceeds the block-table width")
        if bool(invalid_missing.any().item()):
            raise ValueError("selected index resolves to an unallocated cache block")
    max_physical = int(block_table.max().item())
    raise ValueError(
        f"block table physical id {max_physical} exceeds cache blocks {cache_blocks}"
    )


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


@triton.jit
def _sparse_paged_online_split_kernel(
    q_ptr,
    cache_ptr,
    block_table_ptr,
    kv_lens_ptr,
    selected_ptr,
    partial_ptr,
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
    selected_width,
    max_blocks,
    cache_blocks,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCKS_PER_SPLIT: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    query = tl.program_id(2) // NUM_SPLITS
    split = tl.program_id(2) % NUM_SPLITS
    kv_head = head // (H_Q // H_KV)
    rd = tl.arange(0, D)
    q = tl.load(
        q_ptr + batch * stride_q_b + head * stride_q_h + query * stride_q_s + rd
    ).to(tl.float32)
    row = selected_ptr + batch * stride_sel_b + query * stride_sel_s
    kv_len = tl.load(kv_lens_ptr + batch * stride_len_b + query * stride_len_s)
    maximum = tl.full((), float("-inf"), tl.float32)
    normalizer = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    for block in range(BLOCKS_PER_SPLIT):
        # Interleave chunks so a short valid prefix is not concentrated in
        # one partition when the selected tensor has a fixed wide capacity.
        offset = (block * NUM_SPLITS + split) * BLOCK_K + tl.arange(0, BLOCK_K)
        index = tl.load(row + offset, mask=offset < selected_width, other=-1)
        logical = index // PAGE_SIZE
        candidate = (offset < selected_width) & (index >= 0) & (index < kv_len)
        physical = tl.load(
            block_table_ptr + batch * stride_bt_b + logical * stride_bt_block,
            mask=candidate & (logical < max_blocks),
            other=0,
        )
        valid = (
            candidate
            & (logical < max_blocks)
            & (physical > 0)
            & (physical < cache_blocks)
        )
        key_ptr = (
            cache_ptr
            + physical[:, None] * stride_cache_page
            + kv_head * stride_cache_h
            + (index % PAGE_SIZE)[:, None] * stride_cache_token
            + rd[None, :]
        )
        key = tl.load(key_ptr, mask=valid[:, None], other=0.0).to(tl.float32)
        score = tl.sum(q[None, :] * key, axis=1) * SCALE
        score = tl.where(valid, score, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(score, axis=0))
        alpha = tl.where(next_max == float("-inf"), 0.0, tl.exp(maximum - next_max))
        weights = tl.where(valid, tl.exp(score - next_max), 0.0)
        value = tl.load(key_ptr + stride_cache_kv, mask=valid[:, None], other=0.0).to(
            tl.float32
        )
        acc = acc * alpha + tl.sum(weights[:, None] * value, axis=0)
        normalizer = normalizer * alpha + tl.sum(weights)
        maximum = next_max
    output_row = (batch * H_Q + head) * QUERY_LEN + query
    if NUM_SPLITS == 1:
        out = tl.where(normalizer > 0, acc / normalizer, 0.0)
        tl.store(partial_ptr + output_row * D + rd, out.to(q_ptr.dtype.element_ty))
    else:
        base = partial_ptr + (output_row * NUM_SPLITS + split) * (D + 2)
        tl.store(base + rd, acc)
        tl.store(base + D, maximum)
        tl.store(base + D + 1, normalizer)


@triton.jit
def _merge_sparse_paged_splits_kernel(
    partial_ptr,
    out_ptr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    splits = tl.arange(0, NUM_SPLITS)
    rd = tl.arange(0, D)
    base = partial_ptr + row * NUM_SPLITS * (D + 2)
    maxima = tl.load(base + splits * (D + 2) + D)
    sums = tl.load(base + splits * (D + 2) + D + 1)
    maximum = tl.max(maxima, axis=0)
    # Empty partitions contribute exactly zero, including an all-empty row.
    scale = tl.where(sums > 0, tl.exp(maxima - maximum), 0.0)
    partial = tl.load(base + splits[:, None] * (D + 2) + rd[None, :])
    numerator = tl.sum(partial * scale[:, None], axis=0)
    denominator = tl.sum(sums * scale, axis=0)
    out = tl.where(denominator > 0, numerator / denominator, 0.0)
    tl.store(out_ptr + row * D + rd, out.to(out_ptr.dtype.element_ty))


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
    graph_capture: bool = False,
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
        graph_capture: capture only. The caller must validate the initial
            metadata before capture; replayed values are masked by the kernel.
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
    if graph_capture:
        warmup = os.environ.get("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD") == "1"
        if not torch.cuda.is_current_stream_capturing() and not warmup:
            raise RuntimeError(
                "qwen4 sparse paged GQA graph_capture requires an active capture or graph warmup"
            )
    else:
        _validate_sparse_paged_indices(
            block_table,
            kv_lens,
            selected,
            page_size=page_size,
            cache_blocks=cache_blocks,
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
    online = os.environ.get("RTP_LLM_QWEN4_PAGED_ATTENTION_ONLINE", "0").lower()
    if online in ("1", "true", "on"):
        requested = int(os.environ.get("RTP_LLM_QWEN4_PAGED_ATTENTION_SPLITS", "8"))
        if requested not in (1, 2, 4, 8, 16):
            raise ValueError("paged attention splits must be one of 1, 2, 4, 8, 16")
        splits = min(
            requested, triton.next_power_of_2(triton.cdiv(selected_width, block_k))
        )
        partial = (
            torch.empty(
                (batch, q_heads, query_len, splits, head_dim + 2),
                dtype=torch.float32,
                device=q.device,
            )
            if splits > 1
            else output
        )
        _sparse_paged_online_split_kernel[(batch, q_heads, query_len * splits)](
            q,
            kv_cache,
            block_table,
            kv_lens,
            selected,
            partial,
            *q.stride()[:3],
            *kv_cache.stride()[:4],
            *block_table.stride(),
            *kv_lens.stride(),
            *selected.stride()[:2],
            selected_width,
            int(block_table.shape[1]),
            cache_blocks,
            H_Q=q_heads,
            H_KV=kv_heads,
            QUERY_LEN=query_len,
            PAGE_SIZE=page_size,
            D=head_dim,
            SCALE=1.0 / float(head_dim) ** 0.5,
            BLOCK_K=block_k,
            NUM_SPLITS=splits,
            BLOCKS_PER_SPLIT=triton.cdiv(selected_width, block_k * splits),
        )
        if splits > 1:
            _merge_sparse_paged_splits_kernel[(batch * q_heads * query_len,)](
                partial, output, D=head_dim, NUM_SPLITS=splits
            )
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
