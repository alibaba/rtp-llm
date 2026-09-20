"""Quantize rotated QKV and write an HND paged FP8 cache in one launch.

This is an independent Triton implementation of the fusion described by SGLang's
``attention/fused_fp8_qkv_kv_cache.cuh``. No SGLang implementation is copied.
The input is already rotated; this kernel does not apply RoPE or normalize Q/K.
"""

import math
from numbers import Real
from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.offset import linear_offset_64


@triton.jit
def _clear_value_page_tail(
    cache,
    block_table,
    batch,
    seq_len,
    cache_page_stride: tl.constexpr,
    cache_kv_stride: tl.constexpr,
    cache_head_stride: tl.constexpr,
    cache_token_stride: tl.constexpr,
    table_row_stride: tl.constexpr,
    table_col_stride: tl.constexpr,
    KV_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # TRTLLM Gen masks attention scores, but its P@V still reads the final
    # page's V tail: 0 * NaN contaminates valid outputs. Cache blocks can be
    # reused, so initialization of the pool alone is insufficient. Clear only
    # positions outside this sequence, in the caller's existing launch.
    tail = seq_len % PAGE_SIZE
    page_col = seq_len // PAGE_SIZE
    if (seq_len > 0) & (tail != 0) & (page_col < MAX_PAGES):
        page = tl.load(
            block_table
            + linear_offset_64(batch, table_row_stride)
            + page_col * table_col_stride
        )
        if (page >= 0) & (page < CACHE_PAGES):
            cols = tl.arange(0, BLOCK)
            for begin in range(tl.cdiv(PAGE_SIZE * KV_DIM, BLOCK)):
                flat = begin * BLOCK + cols
                token = (flat // HEAD_DIM) % PAGE_SIZE
                head = flat // (PAGE_SIZE * HEAD_DIM)
                offset = (
                    linear_offset_64(page, cache_page_stride)
                    + cache_kv_stride
                    + head.to(tl.int64) * cache_head_stride
                    + token.to(tl.int64) * cache_token_stride
                    + flat % HEAD_DIM
                )
                tl.store(
                    cache + offset,
                    0.0,
                    mask=(flat < PAGE_SIZE * KV_DIM) & (token >= tail),
                )


@triton.jit
def _fused_fp8_qkv_cache_kernel(
    qkv,
    q_out,
    cache,
    block_table,
    cu_seqlens,
    prefix_lengths,
    qkv_row_stride: tl.constexpr,
    cache_page_stride: tl.constexpr,
    cache_kv_stride: tl.constexpr,
    cache_head_stride: tl.constexpr,
    cache_token_stride: tl.constexpr,
    table_row_stride: tl.constexpr,
    table_col_stride: tl.constexpr,
    Q_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    TOKEN_COUNT: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    Q_SCALE: tl.constexpr,
    K_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local_pos = tl.program_id(0)
    batch = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    # Cover the entire output allocation, including graph-padding rows which
    # are absent from cu_seqlens. Do this in the same launch as the cache write.
    padding_row = batch * tl.num_programs(0) + local_pos
    total_valid = tl.load(cu_seqlens + BATCH_SIZE)
    if (padding_row >= total_valid) & (padding_row < TOKEN_COUNT):
        tl.store(
            q_out + linear_offset_64(padding_row, Q_DIM) + cols,
            0.0,
            mask=cols < Q_DIM,
        )
    start = tl.load(cu_seqlens + batch)
    end = tl.load(cu_seqlens + batch + 1)
    row = start + local_pos
    # A padded CUDA graph batch entry can have zero query tokens. It must not
    # read QKV or the page table, or overwrite another sequence's cache.
    if (local_pos < end - start) & (row >= 0) & (row < TOKEN_COUNT):
        qkv_mask = cols < Q_DIM + 2 * KV_DIM
        values = tl.load(
            qkv + linear_offset_64(row, qkv_row_stride) + cols,
            mask=qkv_mask,
            other=0,
        ).to(tl.float32)
        scales = tl.where(
            cols < Q_DIM,
            Q_SCALE,
            tl.where(cols < Q_DIM + KV_DIM, K_SCALE, V_SCALE),
        )
        # Scales are dequantization scales: stored FP8 = value / scale.
        # Saturate explicitly: a plain e4m3fn conversion overflows to NaN.
        quantized = tl.minimum(
            tl.maximum(
                tl.div_rn(values, scales), -448.0, propagate_nan=tl.PropagateNan.ALL
            ),
            448.0,
            propagate_nan=tl.PropagateNan.ALL,
        )
        quantized = quantized.to(q_out.dtype.element_ty)
        tl.store(
            q_out + linear_offset_64(row, Q_DIM) + cols,
            quantized,
            mask=cols < Q_DIM,
        )

        prefix = tl.load(prefix_lengths + batch)
        position = prefix + local_pos
        page_col = position // PAGE_SIZE
        valid_position = (position >= 0) & (page_col < MAX_PAGES)
        page = tl.load(
            block_table
            + linear_offset_64(batch, table_row_stride)
            + page_col * table_col_stride,
            mask=valid_position,
            other=-1,
        )
        valid_page = valid_position & (page >= 0) & (page < CACHE_PAGES)
        kv_cols = tl.maximum(cols - Q_DIM, 0)
        within_kv = kv_cols % KV_DIM
        cache_offset = (
            linear_offset_64(page, cache_page_stride)
            + (kv_cols // KV_DIM).to(tl.int64) * cache_kv_stride
            + (within_kv // HEAD_DIM).to(tl.int64) * cache_head_stride
            + (position % PAGE_SIZE).to(tl.int64) * cache_token_stride
            + within_kv % HEAD_DIM
        )
        tl.store(
            cache + cache_offset,
            quantized,
            mask=valid_page & (cols >= Q_DIM) & qkv_mask,
        )
        if local_pos == end - start - 1:
            _clear_value_page_tail(
                cache,
                block_table,
                batch,
                prefix + end - start,
                cache_page_stride,
                cache_kv_stride,
                cache_head_stride,
                cache_token_stride,
                table_row_stride,
                table_col_stride,
                KV_DIM,
                HEAD_DIM,
                PAGE_SIZE,
                MAX_PAGES,
                CACHE_PAGES,
                BLOCK,
            )


def is_supported(
    qkv: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lengths: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_query_len: int,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Check the metadata-only support boundary, without synchronizing CUDA.

    The caller must provide non-overlapping HND cache storage, valid cumulative
    query lengths and exclusive ownership of each writable final partial page
    (shared prefixes require copy-on-write before appending). ``max_query_len`` must bound
    the query length of every batch entry, including on subsequent graph replays.
    Invalid page IDs are ignored. No GPU metadata is copied to the host here.
    """
    dims = (num_q_heads, num_kv_heads, head_dim, page_size)
    if any(not isinstance(dim, int) or dim <= 0 for dim in dims):
        return False
    if not isinstance(max_query_len, int) or max_query_len < 0:
        return False
    if any(
        not isinstance(scale, Real) or not math.isfinite(scale) or scale <= 0
        for scale in (q_scale, k_scale, v_scale)
    ):
        return False
    tensors = (qkv, kv_cache, block_table, cu_seqlens, prefix_lengths)
    if any(not tensor.is_cuda or tensor.device != qkv.device for tensor in tensors):
        return False
    if torch.version.hip is not None:
        return False
    if torch.cuda.get_device_capability(qkv.device) < (8, 9):
        return False
    if qkv.dtype != torch.bfloat16 or kv_cache.dtype != torch.float8_e4m3fn:
        return False
    packed_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
    if (
        qkv.ndim != 2
        or qkv.shape[1] != packed_dim
        or qkv.stride(1) != 1
        or qkv.stride(0) < packed_dim
        or packed_dim > 65536
    ):
        return False
    if (
        kv_cache.ndim != 5
        or tuple(kv_cache.shape[1:]) != (2, num_kv_heads, page_size, head_dim)
        or kv_cache.stride(4) != 1
        or any(stride <= 0 for stride in kv_cache.stride())
    ):
        return False
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        return False
    batch_size = block_table.shape[0]
    if batch_size == 0 and qkv.shape[0] != 0:
        return False
    for tensor, length in (
        (cu_seqlens, batch_size + 1),
        (prefix_lengths, batch_size),
    ):
        if (
            tensor.ndim != 1
            or tensor.numel() != length
            or tensor.dtype not in (torch.int32, torch.int64)
            or not tensor.is_contiguous()
        ):
            return False
    if out is not None and (
        out.device != qkv.device
        or out.dtype != torch.float8_e4m3fn
        or tuple(out.shape) != (qkv.shape[0], num_q_heads, head_dim)
        or not out.is_contiguous()
    ):
        return False
    return True


def fused_fp8_qkv_cache(
    qkv: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lengths: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_query_len: int,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return FP8 Q [T, Hq, D] and write FP8 K/V directly into paged storage.

    ``qkv`` is a BF16 [T, (Hq + 2 * Hkv) * D] tensor after RoPE. Row padding is
    supported. ``kv_cache`` is FP8 [pages, 2, Hkv, page_size, D] in HND order;
    its strides are respected, including hybrid-pool physical page padding.
    ``block_table`` contains logical kernel page IDs, after any physical-page
    expansion by the caller. For sequence b, query row ``cu_seqlens[b] + i``
    writes cache position ``prefix_lengths[b] + i``.

    Each positive static scale denotes a dequantization multiplier. Quantizing
    uses float32 division, finite saturation and E4M3FN rounding. Invalid page
    IDs, out-of-range positions and zero-length sequences do not write cache.
    The unused V tail of each final partial page is zeroed in the same launch,
    preventing masked P@V from reading NaNs left by a previous cache owner.
    Rows beyond cu_seqlens[-1] are filled with zero in the same launch.
    The caller selects the attention output dtype independently of the FP8 Q.

    Call ``is_supported`` to select an existing fallback at the integration
    boundary. The wrapper raises for unsupported shapes instead of silently
    taking a multi-launch path inside a CUDA graph.
    """
    kwargs = dict(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_query_len=max_query_len,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        out=out,
    )
    if not is_supported(
        qkv, kv_cache, block_table, cu_seqlens, prefix_lengths, **kwargs
    ):
        raise ValueError("Unsupported fused FP8 QKV/cache layout, dtype or scale")
    if out is None:
        out = torch.empty(
            (qkv.shape[0], num_q_heads, head_dim),
            dtype=torch.float8_e4m3fn,
            device=qkv.device,
        )
    if qkv.shape[0] == 0:
        return out
    packed_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
    batch_size = block_table.shape[0]
    grid_query_len = max(max_query_len, triton.cdiv(qkv.shape[0], batch_size))
    _fused_fp8_qkv_cache_kernel[(grid_query_len, batch_size)](
        qkv,
        out,
        kv_cache,
        block_table,
        cu_seqlens,
        prefix_lengths,
        qkv.stride(0),
        *kv_cache.stride()[:4],
        *block_table.stride(),
        Q_DIM=num_q_heads * head_dim,
        KV_DIM=num_kv_heads * head_dim,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        MAX_PAGES=block_table.shape[1],
        CACHE_PAGES=kv_cache.shape[0],
        TOKEN_COUNT=qkv.shape[0],
        BATCH_SIZE=batch_size,
        Q_SCALE=float(q_scale),
        K_SCALE=float(k_scale),
        V_SCALE=float(v_scale),
        BLOCK=triton.next_power_of_2(packed_dim),
        num_warps=4 if packed_dim <= 8192 else 8,
    )
    return out


@triton.jit
def _quantize_fp8_query_kernel(
    query,
    output,
    cache,
    block_table,
    seq_lens,
    row_stride: tl.constexpr,
    head_stride: tl.constexpr,
    cache_page_stride: tl.constexpr,
    cache_kv_stride: tl.constexpr,
    cache_head_stride: tl.constexpr,
    cache_token_stride: tl.constexpr,
    table_row_stride: tl.constexpr,
    table_col_stride: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    CLEAR_VALUE_TAIL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    values = tl.load(
        query
        + linear_offset_64(row, row_stride)
        + (cols // HEAD_DIM) * head_stride
        + cols % HEAD_DIM,
        mask=cols < Q_DIM,
        other=0,
    ).to(tl.float32)
    quantized = tl.minimum(
        tl.maximum(tl.div_rn(values, SCALE), -448.0, propagate_nan=tl.PropagateNan.ALL),
        448.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    tl.store(
        output + linear_offset_64(row, Q_DIM) + cols,
        quantized.to(output.dtype.element_ty),
        mask=cols < Q_DIM,
    )
    if CLEAR_VALUE_TAIL:
        if row < BATCH_SIZE:
            seq_len = tl.load(seq_lens + row)
            _clear_value_page_tail(
                cache,
                block_table,
                row,
                seq_len,
                cache_page_stride,
                cache_kv_stride,
                cache_head_stride,
                cache_token_stride,
                table_row_stride,
                table_col_stride,
                KV_DIM,
                HEAD_DIM,
                PAGE_SIZE,
                MAX_PAGES,
                CACHE_PAGES,
                BLOCK,
            )


def is_query_supported(
    query: torch.Tensor,
    scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Metadata-only support gate for the strided-Q quantization fallback."""
    if (
        not query.is_cuda
        or torch.version.hip is not None
        or query.dtype != torch.bfloat16
        or query.ndim not in (2, 3)
        or query.stride(-1) != 1
        or any(stride <= 0 for stride in query.stride())
        or not isinstance(scale, Real)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        return False
    if torch.cuda.get_device_capability(query.device) < (8, 9):
        return False
    width = math.prod(query.shape[1:])
    if width < 1 or width > 65536:
        return False
    return out is None or (
        out.device == query.device
        and out.dtype == torch.float8_e4m3fn
        and out.shape == query.shape
        and out.is_contiguous()
    )


def quantize_fp8_query(
    query: torch.Tensor,
    scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    *,
    kv_cache: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Saturating FP8 conversion of strided BF16 Q, without a contiguous copy.

    Supports [T, H, D] and [T, H * D]. The positive static ``scale`` is the
    dequantization multiplier. The FP8 output has the same shape as ``query``.
    This is the decode path when RoPE already writes the K/V cache. With cache
    metadata, it also zeros unused V page tails in this same kernel launch.
    Each request must exclusively own its final partial page before writing.
    """
    if not is_query_supported(query, scale, out):
        raise ValueError("Unsupported FP8 query layout, dtype or scale")
    clear_tail = kv_cache is not None
    if clear_tail:
        if (
            block_table is None
            or seq_lens is None
            or kv_cache.ndim != 5
            or kv_cache.shape[1] != 2
            or kv_cache.dtype != torch.float8_e4m3fn
            or kv_cache.device != query.device
            or kv_cache.stride(-1) != 1
            or any(stride <= 0 for stride in kv_cache.stride())
            or block_table.ndim != 2
            or block_table.dtype != torch.int32
            or block_table.device != query.device
            or seq_lens.ndim != 1
            or seq_lens.dtype != torch.int32
            or seq_lens.device != query.device
            or not seq_lens.is_contiguous()
            or seq_lens.numel() != block_table.shape[0]
            or query.shape[0] < seq_lens.numel()
            or (query.ndim == 3 and query.shape[-1] != kv_cache.shape[-1])
            or math.prod(query.shape[1:]) % kv_cache.shape[-1] != 0
        ):
            raise ValueError("Unsupported FP8 query/cache tail metadata")
    elif block_table is not None or seq_lens is not None:
        raise ValueError("Cache storage is required with tail metadata")
    if out is None:
        out = torch.empty(query.shape, dtype=torch.float8_e4m3fn, device=query.device)
    if query.shape[0] == 0:
        return out
    q_dim = math.prod(query.shape[1:])
    _quantize_fp8_query_kernel[(query.shape[0],)](
        query,
        out,
        kv_cache if clear_tail else out,
        block_table if clear_tail else out,
        seq_lens if clear_tail else out,
        query.stride(0),
        (
            query.stride(1)
            if query.ndim == 3
            else (kv_cache.shape[-1] if clear_tail else query.shape[-1])
        ),
        *(kv_cache.stride()[:4] if clear_tail else (0, 0, 0, 0)),
        *(block_table.stride() if clear_tail else (0, 0)),
        HEAD_DIM=kv_cache.shape[-1] if clear_tail else query.shape[-1],
        Q_DIM=q_dim,
        KV_DIM=kv_cache.shape[2] * kv_cache.shape[4] if clear_tail else 0,
        PAGE_SIZE=kv_cache.shape[3] if clear_tail else 1,
        MAX_PAGES=block_table.shape[1] if clear_tail else 0,
        CACHE_PAGES=kv_cache.shape[0] if clear_tail else 0,
        BATCH_SIZE=seq_lens.numel() if clear_tail else 0,
        CLEAR_VALUE_TAIL=clear_tail,
        SCALE=float(scale),
        BLOCK=triton.next_power_of_2(q_dim),
        num_warps=4 if q_dim <= 8192 else 8,
    )
    return out
